"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Composes IntentClassifyNode → UrgencyScoreNode → RoutingDecisionNode
#    sequentially inside the `main` SDK slot.
#  - Returns the merged state update from all three sub-steps.

from typing import Any, Optional, cast

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.services.llm.base_llm import BaseLLM
from shared.utils.audit_logger import emit_trace_event

from src.nodes.intent_classify import IntentClassifyNode
from src.nodes.urgency_score import UrgencyScoreNode
from src.nodes.routing_decision import RoutingDecisionNode
from src.schemas.state import MultiLangIntentRoutingState


class MainNode(FunctionNode):
    """Compose IntentClassify → UrgencyScore → RoutingDecision in one main slot."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        llm_client: Optional[BaseLLM] = None,
        timeout_s: float = 20.0,
        max_retry: int = 0,
    ) -> None:
        self._intent_classify = IntentClassifyNode(
            llm_client=llm_client,
            timeout_s=timeout_s,
            max_retry=max_retry,
        )
        self._urgency_score = UrgencyScoreNode()
        self._routing_decision = RoutingDecisionNode()

    def execute(self, state: MultiLangIntentRoutingState, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        emit_trace_event(
            event_type="main_node_started",
            payload={"input_validation_failed": state.get("input_validation_failed") == "true"},
            state=state,
        )

        if state.get("input_validation_failed") == "true":
            return {}

        # Propagate fatal error from upstream node (e.g. PreProcessNode)
        error_code = state.get("error_code")
        if error_code and error_code not in ("LOW_CONFIDENCE_ADVISORY",):
            return {}

        accumulated: dict[str, Any] = {}

        # Step 1: classify intent. {**state, **accumulated} is a plain dict at
        # the type level even though every key it can carry is a valid
        # MultiLangIntentRoutingState field at runtime — cast() documents
        # that invariant instead of loosening each sub-node's own signature.
        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Classifying the customer service intent with Azure OpenAI.",
            metadata={"stage": "intent_classification"},
        )
        result = self._intent_classify.execute(cast(MultiLangIntentRoutingState, {**state, **accumulated}))
        accumulated.update(result)
        if accumulated.get("input_validation_failed") == "true":
            return accumulated
        if accumulated.get("error_code") and accumulated["error_code"] not in ("LOW_CONFIDENCE_ADVISORY",):
            return accumulated

        # Step 2: score urgency
        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Scoring urgency and escalation requirements.",
            metadata={"stage": "urgency_scoring"},
        )
        result = self._urgency_score.execute(cast(MultiLangIntentRoutingState, {**state, **accumulated}))
        accumulated.update(result)
        if accumulated.get("error_code") and accumulated["error_code"] not in ("LOW_CONFIDENCE_ADVISORY",):
            return accumulated

        # Step 3: routing decision
        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Selecting an allowlisted support queue.",
            metadata={"stage": "routing_decision"},
        )
        result = self._routing_decision.execute(cast(MultiLangIntentRoutingState, {**state, **accumulated}))
        accumulated.update(result)

        if "status" not in accumulated:
            accumulated["status"] = AgentStatus.SUCCESS.value

        return accumulated
