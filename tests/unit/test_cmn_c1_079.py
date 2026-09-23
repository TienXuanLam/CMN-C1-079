"""Unit tests — CMN-C1-079 MultiLangIntentRoutingAgent"""

import ast
import json
import os

from framework.schemas.invocation_context import InvocationContext, TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider

from src.graph.graph import MultiLangIntentRoutingGraph
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.post_process_node import PostProcessNode
from src.schemas.state import MultiLangIntentRoutingState

# "trust_level" here is inert domain data now, kept only because
# routing_decision.py's default operator_config shape doesn't require it —
# authorization is decided from state["caller_trust_level"] (see
# TestOperatorConfigAuthorization below), never from a field the caller
# controls inside this JSON blob.
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
                        "candidates": [
                            {"id": intent_id, "label": intent_id, "score": confidence},
                            {"id": "general_inquiry", "label": "General Inquiry", "score": round(1 - confidence, 2)},
                        ],
                    }
                )
            }

    return _MockLLM()


def _build_agent(llm_mock=None) -> MultiLangIntentRoutingGraph:
    agent = MultiLangIntentRoutingGraph(config={"llm": llm_mock})
    agent.compile()
    return agent


def _ctx(trust_level: TrustLevel = TrustLevel.INTERNAL) -> InvocationContext:
    # INTERNAL by default: submitting operator_config at all requires
    # INTERNAL per the framework's Channel 4 Rule 2 (caller-supplied-claim
    # input channel) — most tests exercise the ordinary pipeline with a default
    # operator_config and don't care about the trust boundary itself; the
    # dedicated TestOperatorConfigAuthorization tests below vary this.
    return InvocationContext(
        caller_id="test",
        caller_trust_level=trust_level,
    )


def _run(raw_message, operator_config=None, llm_mock=None, trust_level: TrustLevel = TrustLevel.INTERNAL):
    """Run via agent.invoke(). Marketplace populates user_input; input_context
    is passed too, only as PreProcessNode's compatibility fallback."""
    agent = _build_agent(llm_mock=llm_mock)
    with bound_secrets(_SECRETS):
        return agent.invoke(
            user_input=raw_message,
            input_context={"raw_message": raw_message, "operator_config": operator_config or _VALID_OP_CONFIG},
            ctx=_ctx(trust_level),
        )


def _output(result: dict) -> str:
    """formatted_output/result["output"] is a Markdown report string, not a
    dict or JSON — PostProcessNode assembles it via "\n".join(sections)."""
    return str(result.get("output") or "")


# ---------------------------------------------------------------------------
# TC-01  State fields are Optional primitives (TypedDict check)
# ---------------------------------------------------------------------------
def test_tc01_state_fields_are_optional_primitives():
    import typing

    hints = typing.get_type_hints(MultiLangIntentRoutingState)
    allowed = (str, int, float, bool)
    # formatted_output was previously excluded here because it was declared
    # dict[str, Any] — a genuine ADR-005 violation this test should have
    # caught but didn't, since it was carved out instead of fixed. It is now
    # a plain Markdown string (src/nodes/post_process_node.py) and belongs in
    # the checked set like every other domain field.
    excluded_sdk_fields = {
        "status",
        "node_history",
        "error_log",
        "execution_time",
        "input_context",
        "messages",
        "hitl_feedback",
        "hitl_metadata",
        "hitl_status",
        "hitl_draft",
        "hitl_count",
        "hitl_allowed",
        "subgraph_thread_id",
        "response_metadata",
        "validated_input",
        "enriched_context",
        "result",
        "llm_response",
        "context",
        "intent",
    }
    for field, hint in hints.items():
        if field in excluded_sdk_fields:
            continue
        args = getattr(hint, "__args__", (hint,))
        for arg in args:
            if arg is type(None):
                continue
            assert arg in allowed, f"Field {field!r} has non-primitive type {arg!r}"


# ---------------------------------------------------------------------------
# TC-02  S-1: empty raw_message
#
# build_input_guidance() returns a successful, safe Markdown response (not a
# fatal error) so the caller gets actionable guidance instead of a bare
# error — PreProcessNode marks input_validation_failed="true" and
# PostProcessNode returns that guidance text as formatted_output.
# ---------------------------------------------------------------------------
def test_tc02_s1_empty_input():
    result = _run("")
    assert result["status"] == "success"
    assert "Customer service message required" in str(result.get("output") or "")


# ---------------------------------------------------------------------------
# TC-03  S-1: oversized input
# ---------------------------------------------------------------------------
def test_tc03_s1_input_too_long():
    result = _run("x" * 10001)
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# TC-04  S-1: binary blob
# ---------------------------------------------------------------------------
def test_tc04_s1_binary_input():
    blob = "A" * 44 + "AA=="
    result = _run(blob)
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# TC-05  S-1: injection detection
# ---------------------------------------------------------------------------
def test_tc05_s1_injection_detected():
    result = _run("hello\x00world")
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# TC-06  S-2: credential pattern in operator_config
# ---------------------------------------------------------------------------
def test_tc06_s2_credential_detected():
    config = json.dumps({"notes": "api_key=sk-supersecret123"})
    result = _run("normal message", operator_config=config)
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# TC-07  operator_config is optional — an omitted/empty value defaults to
# "{}" (every node's own built-in defaults apply) rather than being an S-5
# violation. Requiring it unconditionally would make the agent unusable by
# its own declared VERIFIED_EXTERNAL entry point for the ordinary case.
# ---------------------------------------------------------------------------
def test_tc07_missing_operator_config_uses_defaults():
    llm = _mock_llm("general_inquiry", 0.90)
    result = _run("hello", operator_config="", llm_mock=llm, trust_level=TrustLevel.VERIFIED_EXTERNAL)
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# TC-08  S-5: caller_trust_level below INTERNAL cannot submit operator_config
# ---------------------------------------------------------------------------
def test_tc08_s5_caller_trust_level_insufficient():
    # A self-declared "trust_level" field inside operator_config is inert
    # data now — this is the exact bypass the fix closes: a VERIFIED_EXTERNAL
    # caller cannot grant itself operator/admin behavior by claiming it in
    # the caller-controlled JSON body.
    result = _run("hello", trust_level=TrustLevel.VERIFIED_EXTERNAL)
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# TC-09  Fatal error propagation: a genuine fatal S1 code (not empty input,
# which is guidance-success — see TC-02) → status=error, no output.
# ---------------------------------------------------------------------------
def test_tc09_error_propagation():
    result = _run("hello\x00world")  # S1_INJECTION_DETECTED is fatal
    assert result["status"] == "error"
    assert result.get("output") is None or result.get("output") == {}


# ---------------------------------------------------------------------------
# TC-10  Happy path: Japanese complaint → queue_escalation
# ---------------------------------------------------------------------------
def test_tc10_happy_path_ja_complaint():
    llm = _mock_llm("complaint", 0.92)
    result = _run("商品が届かなくて非常に困っています。至急対応してください。", llm_mock=llm)
    assert result["status"] == "success"
    out = _output(result)
    assert "`queue_escalation`" in out
    assert "**Detected language:** `ja`" in out
    assert "**Intent:** `complaint`" in out


# ---------------------------------------------------------------------------
# TC-11  Happy path: English account inquiry
# ---------------------------------------------------------------------------
def test_tc11_happy_path_en_account_inquiry():
    llm = _mock_llm("account_inquiry", 0.88)
    result = _run("I need to check my account balance.", llm_mock=llm)
    assert result["status"] == "success"
    out = _output(result)
    assert "**Detected language:** `en`" in out
    assert "**Intent:** `account_inquiry`" in out
    assert ("`queue_selfservice`" in out) or ("`queue_ops`" in out)


# ---------------------------------------------------------------------------
# TC-12  Happy path: Vietnamese general inquiry
# ---------------------------------------------------------------------------
def test_tc12_happy_path_vi_general():
    llm = _mock_llm("general_inquiry", 0.85)
    result = _run("Tôi cần hỏi về dịch vụ khách hàng.", llm_mock=llm)
    assert result["status"] == "success"
    out = _output(result)
    assert "**Detected language:** `vi`" in out


# ---------------------------------------------------------------------------
# TC-13  Low confidence → human_review_required=True in output
# ---------------------------------------------------------------------------
def test_tc13_low_confidence_human_review():
    llm = _mock_llm("general_inquiry", 0.50)
    result = _run("Some ambiguous message.", llm_mock=llm)
    assert result["status"] == "success"
    out = _output(result)
    assert "**Human review:** Required" in out


# ---------------------------------------------------------------------------
# TC-14  Routing target not in allowlist → error
# ---------------------------------------------------------------------------
def test_tc14_routing_not_allowlisted():
    llm = _mock_llm("general_inquiry", 0.90)
    config = json.dumps({"routing_allowlist": ["queue_escalation"]})
    result = _run("I need help.", operator_config=config, llm_mock=llm)
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# TC-15  PII not echoed in output
# ---------------------------------------------------------------------------
def test_tc15_pii_not_echoed():
    pii_message = "My name is John and I have a complaint about my order."
    llm = _mock_llm("complaint", 0.88)
    result = _run(pii_message, llm_mock=llm)
    assert result["status"] == "success"
    output_str = _output(result)
    assert pii_message not in output_str


# ---------------------------------------------------------------------------
# TC-16  LLM auth/permission error → fatal status=error (S2_LLM_ERROR).
#
# A plain/timeout-flavoured exception is intentionally *not* fatal anymore
# (IntentClassifyNode._is_timeout_error() routes it to a safe guidance
# response instead — see TC-16b) — only an auth/permission failure, which
# _is_auth_or_permission_error() recognises, is treated as a fatal error.
# ---------------------------------------------------------------------------
def test_tc16_llm_error():
    class _ErrorLLM:
        def complete(self, messages: list) -> dict:
            raise RuntimeError("401 Unauthorized: invalid api key")

    result = _run("Please help me.", llm_mock=_ErrorLLM())
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# TC-16b  LLM timeout → non-fatal guidance response (status=success), not
# indistinguishable from a genuine input-validation failure — see
# input_guidance.build_input_guidance()'s docstring.
# ---------------------------------------------------------------------------
def test_tc16b_llm_timeout_returns_guidance_not_fatal_error():
    class _TimeoutLLM:
        def complete(self, messages: list) -> dict:
            raise RuntimeError("LLM timeout")

    result = _run("Please help me.", llm_mock=_TimeoutLLM())
    assert result["status"] == "success"
    assert "Classification timed out" in str(result.get("output") or "")


# ---------------------------------------------------------------------------
# TC-17  Import isolation: no agenticstar in src/
# ---------------------------------------------------------------------------
def test_tc17_import_isolation():
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
# TC-18  execute() contract: no _invoke_impl in src/
# ---------------------------------------------------------------------------
def test_tc18_execute_contract():
    src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    violations = []
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                source = f.read()
            if "_invoke_impl" in source:
                violations.append(fpath)
    assert violations == [], "_invoke_impl found (must use execute()):\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# TC-19  Graph compile + invoke smoke test
# ---------------------------------------------------------------------------
def test_tc19_graph_compile_and_invoke():
    llm = _mock_llm("general_inquiry", 0.85)
    agent = _build_agent(llm_mock=llm)
    assert agent._compiled is not None

    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input="I have a question.",
            input_context={"raw_message": "I have a question.", "operator_config": _VALID_OP_CONFIG},
            ctx=_ctx(),
        )
    assert result["status"] == "success"
    assert result.get("output") is not None


# ---------------------------------------------------------------------------
# TC-20  Unknown language — Cyrillic → detected_language=unknown
# ---------------------------------------------------------------------------
def test_tc20_unknown_language():
    llm = _mock_llm("general_inquiry", 0.80)
    result = _run("Привет мир тест сообщение", llm_mock=llm)
    assert result["status"] == "success"
    out = _output(result)
    assert "**Detected language:** `unknown`" in out


# ---------------------------------------------------------------------------
# TC-S1  _run_input_gates() returns error dict on S-1 violation
# ---------------------------------------------------------------------------
def test_tcs1_run_input_gates_non_bypassable():
    node = PreProcessNode()
    result = node._run_input_gates(
        raw_message="", operator_config=_VALID_OP_CONFIG, caller_trust_level=TrustLevel.INTERNAL.value
    )
    assert result is not None
    assert result.get("error_code") == "S1_EMPTY_INPUT"


# ---------------------------------------------------------------------------
# TC-S2  _run_input_gates() returns error dict on S-5 (missing config)
# ---------------------------------------------------------------------------
def test_tcs2_run_input_gates_returns_error_on_s5():
    node = PreProcessNode()
    result = node._run_input_gates(
        raw_message="hello", operator_config="", caller_trust_level=TrustLevel.INTERNAL.value
    )
    assert result is not None
    assert result.get("error_code") == "S5_PERMISSION_DENIED"


# ---------------------------------------------------------------------------
# TC-S3  _run_output_gate() returns error dict on PII echo
# ---------------------------------------------------------------------------
def test_tcs3_run_output_gate_non_bypassable():
    node = PostProcessNode()
    pii = "verbatim user content 12345678"
    output_containing_pii = json.dumps({"result": pii})
    result = node._run_output_gate(output_containing_pii, pii)
    assert result is not None
    assert result.get("error_code") == "S3_BLOCKED"


# ---------------------------------------------------------------------------
# TC-S4  required_trust_level declared on PreProcessNode
# ---------------------------------------------------------------------------
def test_tcs4_required_trust_level_declared():
    node = PreProcessNode()
    assert hasattr(node, "required_trust_level")
    assert node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL


# ---------------------------------------------------------------------------
# TC-S5  Graph entry point declares required_trust_level, matching the manifest
# ---------------------------------------------------------------------------
def test_tcs5_graph_required_trust_level_declared():
    assert MultiLangIntentRoutingGraph.required_trust_level == TrustLevel.VERIFIED_EXTERNAL


# ---------------------------------------------------------------------------
# TestOperatorConfigAuthorization — regression coverage for the S-5 fix:
# operator_config authorization must come from state["caller_trust_level"]
# (framework-established), never from a field inside the caller-controlled
# operator_config JSON body itself.
# ---------------------------------------------------------------------------
class TestOperatorConfigAuthorization:
    def test_self_declared_trust_level_field_is_ignored(self):
        """A caller claiming {"trust_level": "admin"} inside operator_config
        must NOT be granted operator_config privileges — only a real
        caller_trust_level of INTERNAL may."""
        node = PreProcessNode()
        config = json.dumps({"trust_level": "admin", "routing_allowlist": ["queue_anything"]})
        state = {"caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        result = node._run_input_gates(
            raw_message="hello", operator_config=config, caller_trust_level=state["caller_trust_level"]
        )
        assert result is not None
        assert result.get("error_code") == "S5_PERMISSION_DENIED"

    def test_empty_operator_config_needs_no_elevated_trust(self):
        """An empty {} customises nothing, so any caller (even ANONYMOUS) may
        submit it — only a NON-EMPTY operator_config requires INTERNAL."""
        node = PreProcessNode()
        config = json.dumps({})
        result = node._run_input_gates(
            raw_message="hello", operator_config=config, caller_trust_level=TrustLevel.ANONYMOUS.value
        )
        assert result is None

    def test_internal_caller_trust_level_permitted_for_non_empty_config(self):
        node = PreProcessNode()
        config = json.dumps({"confidence_threshold": 0.9})
        result = node._run_input_gates(
            raw_message="hello", operator_config=config, caller_trust_level=TrustLevel.INTERNAL.value
        )
        assert result is None

    def test_anonymous_caller_denied_for_non_empty_config(self):
        node = PreProcessNode()
        config = json.dumps({"confidence_threshold": 0.9})
        result = node._run_input_gates(
            raw_message="hello", operator_config=config, caller_trust_level=TrustLevel.ANONYMOUS.value
        )
        assert result is not None
        assert result.get("error_code") == "S5_PERMISSION_DENIED"

    def test_verified_external_caller_denied_for_non_empty_config(self):
        """VERIFIED_EXTERNAL is an authenticated caller, but not sufficient
        to submit a NON-EMPTY operator_config — Channel 4 Rule 2 requires
        INTERNAL as the compensating control since routing_rules/
        routing_allowlist/intent_taxonomy/confidence_threshold are all
        reachable through it."""
        node = PreProcessNode()
        config = json.dumps({"confidence_threshold": 0.9})
        result = node._run_input_gates(
            raw_message="hello", operator_config=config, caller_trust_level=TrustLevel.VERIFIED_EXTERNAL.value
        )
        assert result is not None
        assert result.get("error_code") == "S5_PERMISSION_DENIED"

    def test_caller_cannot_self_allowlist_arbitrary_routing_target(self):
        """End-to-end: even if a VERIFIED_EXTERNAL caller supplies both a
        custom routing_rules pointing at an arbitrary queue AND adds that
        queue to routing_allowlist in the same operator_config payload, the
        S-5 gate rejects the whole request before either field is ever read."""
        llm = _mock_llm("general_inquiry", 0.90)
        malicious_config = json.dumps(
            {
                "routing_rules": {"general_inquiry+MEDIUM": "queue_attacker_controlled"},
                "routing_allowlist": ["queue_attacker_controlled"],
            }
        )
        result = _run(
            "I need help.",
            operator_config=malicious_config,
            llm_mock=llm,
            trust_level=TrustLevel.VERIFIED_EXTERNAL,
        )
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# TC-R3-7  BaseLLM used as the type contract; a concrete provider is only
# constructed inside the lazy _build_llm() resolver, never at module scope.
# ---------------------------------------------------------------------------
def test_tcr37_base_llm_used_not_direct_provider():
    src_file = os.path.join(os.path.dirname(__file__), "..", "..", "src", "nodes", "intent_classify.py")
    with open(src_file, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)

    # Imports nested inside any function are exempt: shared.services.llm.
    # openai_client is imported inside _build_llm() specifically so it is
    # only ever loaded when a real client must be lazily constructed
    # (per the framework's LLM-injection guidance), not at module import time.
    nested_import_ids = {
        id(n) for fn in ast.walk(tree) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(fn)
    }

    base_llm_imports = []
    module_level_provider_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "shared.services.llm.base_llm":
                for alias in node.names:
                    if alias.name == "BaseLLM":
                        base_llm_imports.append(node.module)
            if node.module in ("shared.services.llm.openai_client", "shared.services.llm.anthropic_client"):
                if id(node) not in nested_import_ids:
                    module_level_provider_imports.append(node.module)

    assert base_llm_imports, "BaseLLM not imported from shared.services.llm.base_llm"
    assert (
        not module_level_provider_imports
    ), f"Direct provider import at module scope detected: {module_level_provider_imports}"
