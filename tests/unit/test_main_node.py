# CMN-C1-079 — Unit Tests: MainNode (IntentClassify + UrgencyScore + RoutingDecision)

import json

from framework.schemas.agent_status import AgentStatus
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider

from src.nodes.main_node import MainNode

_SECRETS = InMemoryProvider(
    {
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
    }
)
_VALID_OP_CONFIG = json.dumps({"trust_level": "operator"})


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


def _base_state(**overrides) -> dict:
    state = {
        "user_input": "test",
        "detected_language": "en",
        "normalised_message": "I need help with my account.",
        "raw_message": "I need help with my account.",
        "operator_config": _VALID_OP_CONFIG,
        "input_context": {},
        "correlation_id": "test-corr",
        "session_id": "test-session",
        "thread_id": "test-thread",
        "trace_id": "",
        "caller_trust_level": "verified_external",
        "caller_id": "",
        "hitl_allowed": True,
        "node_history": [],
        "error_log": [],
        "error_code": None,
    }
    state.update(overrides)
    return state


class TestMainNodeSuccessPath:
    def test_success_path_sets_intent_and_routing(self):
        node = MainNode(llm_client=_mock_llm("account_inquiry", 0.88))
        with bound_secrets(_SECRETS):
            result = node.execute(_base_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result.get("intent_class") == "account_inquiry"
        assert result.get("routing_target") in ("queue_selfservice", "queue_ops")
        assert result.get("urgency_level") in ("LOW", "MEDIUM", "HIGH")

    def test_success_path_complaint_routes_to_escalation(self):
        node = MainNode(llm_client=_mock_llm("complaint", 0.92))
        state = _base_state(normalised_message="至急対応してください。")
        with bound_secrets(_SECRETS):
            result = node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result.get("routing_target") == "queue_escalation"

    def test_low_confidence_sets_human_review_required(self):
        node = MainNode(llm_client=_mock_llm("general_inquiry", 0.50))
        with bound_secrets(_SECRETS):
            result = node.execute(_base_state())
        assert result.get("human_review_required") is True
        assert result.get("error_code") == "LOW_CONFIDENCE_ADVISORY"


class TestMainNodeErrorPaths:
    def test_no_llm_client_returns_s2_llm_error(self):
        node = MainNode(llm_client=None)
        with bound_secrets(_SECRETS):
            result = node.execute(_base_state())
        assert result["status"] == AgentStatus.ERROR.value
        assert result.get("error_code") == "S2_LLM_ERROR"

    def test_llm_exception_returns_s2_llm_error(self):
        # An auth/permission-flavoured exception is fatal — a plain/timeout
        # exception is intentionally not (IntentClassifyNode._is_timeout_error()
        # routes it to a safe guidance response instead).
        class _ErrorLLM:
            def complete(self, messages):
                raise RuntimeError("401 Unauthorized: invalid api key")

        node = MainNode(llm_client=_ErrorLLM())
        with bound_secrets(_SECRETS):
            result = node.execute(_base_state())
        assert result["status"] == AgentStatus.ERROR.value
        assert result.get("error_code") == "S2_LLM_ERROR"

    def test_fatal_error_code_in_state_is_propagated(self):
        node = MainNode(llm_client=_mock_llm())
        with bound_secrets(_SECRETS):
            result = node.execute(_base_state(error_code="S1_EMPTY_INPUT"))
        assert result == {}

    def test_routing_not_allowlisted_returns_error(self):
        llm = _mock_llm("general_inquiry", 0.90)
        node = MainNode(llm_client=llm)
        config = json.dumps({"trust_level": "operator", "routing_allowlist": ["queue_escalation"]})
        with bound_secrets(_SECRETS):
            result = node.execute(_base_state(operator_config=config))
        assert result.get("error_code") == "S3_ROUTING_NOT_ALLOWLISTED"


class TestMainNodeContract:
    """Node contract: must implement execute(state), not _invoke_impl."""

    def test_execute_method_signature(self):
        import inspect

        assert hasattr(MainNode, "execute"), "MainNode must implement execute()"
        sig = inspect.signature(MainNode.execute)
        params = list(sig.parameters.keys())
        assert len(params) >= 2
        assert params[1] == "state"
        assert "_invoke_impl" not in MainNode.__dict__
