# CMN-C1-079 — Integration smoke test: full pipeline compile + invoke

import json

from framework.schemas.invocation_context import InvocationContext, TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider

from src.graph.graph import MultiLangIntentRoutingGraph

_SECRETS = InMemoryProvider(
    {
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
    }
)
_VALID_OP_CONFIG = json.dumps({})


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


def _ctx() -> InvocationContext:
    # INTERNAL: submitting operator_config requires INTERNAL trust per
    # the framework's Channel 4 Rule 2 (caller-supplied-claim input channel) — a
    # self-declared "trust_level" field inside operator_config is inert data
    # now, not an authorization signal.
    return InvocationContext(caller_id="smoke-test", caller_trust_level=TrustLevel.INTERNAL)


def _build_agent(llm_mock=None) -> MultiLangIntentRoutingGraph:
    agent = MultiLangIntentRoutingGraph(config={"llm": llm_mock})
    agent.compile()
    return agent


def _output(result: dict) -> str:
    """formatted_output/result["output"] is a Markdown report string, not a
    dict or JSON — PostProcessNode assembles it via "\n".join(sections)."""
    return str(result.get("output") or "")


def test_graph_compiles_successfully():
    agent = _build_agent(llm_mock=_mock_llm())
    assert agent._compiled is not None


def test_full_pipeline_success_english():
    agent = _build_agent(llm_mock=_mock_llm("account_inquiry", 0.88))
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input="I need to check my account balance.",
            input_context={"raw_message": "I need to check my account balance.", "operator_config": _VALID_OP_CONFIG},
            ctx=_ctx(),
        )
    assert result["status"] == "success"
    assert result.get("output") is not None
    out = _output(result)
    assert "**Detected language:** `en`" in out
    assert ("`queue_selfservice`" in out) or ("`queue_ops`" in out)


def test_full_pipeline_success_japanese():
    agent = _build_agent(llm_mock=_mock_llm("complaint", 0.92))
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input="商品が届きません。",
            input_context={"raw_message": "商品が届きません。", "operator_config": _VALID_OP_CONFIG},
            ctx=_ctx(),
        )
    assert result["status"] == "success"
    out = _output(result)
    assert "**Detected language:** `ja`" in out


def test_full_pipeline_guidance_on_empty_input():
    # build_input_guidance() returns a successful, safe Markdown response
    # (not a fatal error) for empty raw_message — PreProcessNode marks
    # input_validation_failed="true" and PostProcessNode surfaces that
    # guidance text as formatted_output.
    agent = _build_agent(llm_mock=_mock_llm())
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input="",
            input_context={"raw_message": "", "operator_config": _VALID_OP_CONFIG},
            ctx=_ctx(),
        )
    assert result["status"] == "success"
    assert "Customer service message required" in str(result.get("output") or "")


def test_full_pipeline_succeeds_with_missing_operator_config():
    # operator_config is optional — an ordinary VERIFIED_EXTERNAL caller
    # omitting it entirely must still succeed using every node's built-in
    # defaults, not be rejected by the S-5 gate.
    agent = _build_agent(llm_mock=_mock_llm("general_inquiry", 0.90))
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input="hello",
            input_context={"raw_message": "hello", "operator_config": ""},
            ctx=InvocationContext(caller_id="smoke-test", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
        )
    assert result["status"] == "success"


def test_node_history_contains_all_pipeline_nodes():
    agent = _build_agent(llm_mock=_mock_llm("general_inquiry", 0.85))
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input="I have a question.",
            input_context={"raw_message": "I have a question.", "operator_config": _VALID_OP_CONFIG},
            ctx=_ctx(),
        )
    assert result["status"] == "success"
    history = result.get("node_history", [])
    assert "InitializeNode" in history
    assert "PreProcessNode" in history
    assert "MainNode" in history
    assert "PostProcessNode" in history
    assert "FinalizeNode" in history
