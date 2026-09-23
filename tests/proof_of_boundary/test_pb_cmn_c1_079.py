"""Proof-of-Boundary tests — CMN-C1-079 MultiLangIntentRoutingAgent"""

import ast
import json
import os

from framework.schemas.invocation_context import InvocationContext, TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider

from src.graph.graph import MultiLangIntentRoutingGraph
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.post_process_node import PostProcessNode

_VALID_OP_CONFIG = json.dumps({})
_SECRETS = InMemoryProvider(
    {
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
    }
)


def _mock_llm(intent_id="general_inquiry", confidence=0.90):
    class _MockLLM:
        def complete(self, messages: list) -> dict:
            return {
                "content": json.dumps(
                    {
                        "business_relevant": True,
                        "intent_class": intent_id,
                        "confidence": confidence,
                        "candidates": [{"id": intent_id, "label": intent_id, "score": confidence}],
                    }
                )
            }

    return _MockLLM()


def _build_agent(llm_mock=None) -> MultiLangIntentRoutingGraph:
    agent = MultiLangIntentRoutingGraph(config={"llm": llm_mock})
    agent.compile()
    return agent


def _ctx() -> InvocationContext:
    # INTERNAL: submitting operator_config requires INTERNAL trust per
    # the framework's Channel 4 Rule 2 (caller-supplied-claim input channel).
    return InvocationContext(
        caller_id="test",
        caller_trust_level=TrustLevel.INTERNAL,
    )


def _run_graph(raw_message, operator_config=None, llm_mock=None):
    agent = _build_agent(llm_mock=llm_mock)
    with bound_secrets(_SECRETS):
        return agent.invoke(
            user_input=raw_message,
            input_context={"raw_message": raw_message, "operator_config": operator_config or _VALID_OP_CONFIG},
            ctx=_ctx(),
        )


def _output(result: dict) -> str:
    """formatted_output/result["output"] is a Markdown report string, not a
    dict or JSON — PostProcessNode assembles it via "\n".join(sections)."""
    return str(result.get("output") or "")


# ---------------------------------------------------------------------------
# PB-1  Output is a Markdown string (SDK output shape) — no non-serializable types
# ---------------------------------------------------------------------------
def test_pb1_state_primitives_after_pipeline():
    llm = _mock_llm("account_inquiry", 0.88)
    result = _run_graph("I need to check my account.", llm_mock=llm)
    assert result["status"] == "success"

    # formatted_output (SDK result["output"]) is a plain Markdown string
    # (PostProcessNode assembles it via "\n".join(sections)) — state must
    # stay a flat TypedDict of msgpack-safe primitives, so the envelope is
    # a string, not a dict, before it ever enters State (ADR-005).
    out = result.get("output")
    assert out is not None
    assert isinstance(out, str)
    assert out.startswith("# Customer Service Routing Decision")


# ---------------------------------------------------------------------------
# PB-2  PII not echoed in output
# ---------------------------------------------------------------------------
def test_pb2_pii_not_echoed_in_output():
    pii = "Customer name is Taro Yamada, phone 090-1234-5678, complaint about invoice."
    llm = _mock_llm("complaint", 0.91)
    result = _run_graph(pii, llm_mock=llm)

    assert result["status"] == "success"
    output_str = _output(result)
    assert (
        pii not in output_str
    ), "PII content from raw_message appeared verbatim in output — S-3 non-echo gate violated"


# ---------------------------------------------------------------------------
# PB-3  Non-allowlisted routing target → error
# ---------------------------------------------------------------------------
def test_pb3_non_allowlisted_target_rejected():
    llm = _mock_llm("general_inquiry", 0.90)
    config = json.dumps({"routing_allowlist": ["queue_escalation", "queue_human_review"]})
    result = _run_graph("I have a question.", operator_config=config, llm_mock=llm)
    assert result["status"] == "error"
    assert result.get("output") is None or result.get("output") == {}


# ---------------------------------------------------------------------------
# PB-4  Import isolation: no agenticstar Level 0 in src/
# ---------------------------------------------------------------------------
def test_pb4_import_isolation():
    src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    violations = []
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                source = f.read()
            try:
                tree = ast.parse(source, filename=fpath)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("agenticstar"):
                            violations.append(f"{fpath}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("agenticstar"):
                        violations.append(f"{fpath}: from {node.module} import ...")
    assert violations == [], "Level 0 import violations:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# PB-5  Injection in message → error, no output
# ---------------------------------------------------------------------------
def test_pb5_injection_blocked():
    result = _run_graph("normal start\x00injected null byte")
    assert result["status"] == "error"
    assert result.get("output") is None or result.get("output") == {}


# ---------------------------------------------------------------------------
# PB-6  Security gate execution order: _run_input_gates before _run_output_gate
#
# Exercises the real invoke path (_build_agent()/MultiLangIntentRoutingGraph)
# and monkeypatches the actual gate methods on PreProcessNode (S-1/S-2/S-5
# input gates) and PostProcessNode (S-3 output gate) — the two SDK-slot nodes
# that now own these gates after the LangDetectNode/ResponseValidateNode
# migration — proving gate ordering on the live pipeline, not on dead code.
# ---------------------------------------------------------------------------
def test_pb6_security_gate_execution_order():
    call_log = []

    original_input_gate = PreProcessNode._run_input_gates
    original_output_gate = PostProcessNode._run_output_gate

    def patched_input_gate(self, raw_message, operator_config, caller_trust_level):
        call_log.append("_run_input_gates")
        return original_input_gate(self, raw_message, operator_config, caller_trust_level)

    def patched_output_gate(self, output_str, raw_message=""):
        call_log.append("_run_output_gate")
        return original_output_gate(self, output_str, raw_message)

    PreProcessNode._run_input_gates = patched_input_gate
    PostProcessNode._run_output_gate = patched_output_gate

    try:
        llm = _mock_llm("general_inquiry", 0.90)
        result = _run_graph("Check my account balance please.", llm_mock=llm)
    finally:
        PreProcessNode._run_input_gates = original_input_gate
        PostProcessNode._run_output_gate = original_output_gate

    assert "_run_input_gates" in call_log, "_run_input_gates was never called"
    assert "_run_output_gate" in call_log, "_run_output_gate was never called"
    assert call_log.index("_run_input_gates") < call_log.index(
        "_run_output_gate"
    ), "_run_input_gates must precede _run_output_gate"
    assert result["status"] == "success"
