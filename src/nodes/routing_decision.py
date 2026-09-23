"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - S-3 allowlist gate kept as private helper (non-deletable)
#  - Never import from mediator/, api/, or other agents

import json
from typing import Any, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import MultiLangIntentRoutingState

_DEFAULT_ROUTING_RULES: dict[str, str] = {
    "complaint+HIGH": "queue_escalation",
    "complaint+MEDIUM": "queue_complaints",
    "complaint+LOW": "queue_complaints",
    "escalation+HIGH": "queue_escalation",
    "escalation+MEDIUM": "queue_escalation",
    "account_inquiry+LOW": "queue_selfservice",
    "account_inquiry+MEDIUM": "queue_ops",
    "transaction_request+MEDIUM": "queue_ops",
    "transaction_request+HIGH": "queue_escalation",
    "general_inquiry+LOW": "queue_selfservice",
    "default": "queue_general",
}

_DEFAULT_ROUTING_ALLOWLIST: list[str] = [
    "queue_escalation",
    "queue_complaints",
    "queue_selfservice",
    "queue_ops",
    "queue_general",
    "queue_human_review",
]


class RoutingDecisionNode(FunctionNode):
    """Resolve routing target from intent × urgency with S-3 allowlist enforcement."""

    required_trust_level = TrustLevel.ANONYMOUS

    def execute(self, state: MultiLangIntentRoutingState, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        error_code = state.get("error_code")
        if error_code and error_code not in ("LOW_CONFIDENCE_ADVISORY",):
            return {}

        operator_config = self._load_operator_config(state)
        routing_rules_raw = operator_config.get("routing_rules", _DEFAULT_ROUTING_RULES)
        routing_allowlist_raw = operator_config.get("routing_allowlist", _DEFAULT_ROUTING_ALLOWLIST)

        # A malformed routing_rules/routing_allowlist (wrong container type,
        # or a routing_rules value that isn't itself a string) previously
        # crashed with AttributeError/TypeError at routing_rules.get() or
        # the `in` membership check below instead of a structured error.
        if not isinstance(routing_rules_raw, dict) or not all(isinstance(v, str) for v in routing_rules_raw.values()):
            return {
                "error_code": "S1_INVALID_OPERATOR_CONFIG",
                "error_message": "operator_config.routing_rules must be an object mapping strings to strings.",
                "status": AgentStatus.ERROR.value,
            }
        if not isinstance(routing_allowlist_raw, list) or not all(isinstance(v, str) for v in routing_allowlist_raw):
            return {
                "error_code": "S1_INVALID_OPERATOR_CONFIG",
                "error_message": "operator_config.routing_allowlist must be a list of strings.",
                "status": AgentStatus.ERROR.value,
            }
        routing_rules = routing_rules_raw
        routing_allowlist = routing_allowlist_raw

        human_review_required = state.get("human_review_required", False)
        intent_class = state.get("intent_class", "general_inquiry")
        urgency_level = state.get("urgency_level", "MEDIUM")

        # human_review_required is non-suppressible — bypass allowlist check
        if human_review_required:
            emit_trace_event(
                event_type="routing_decision_complete",
                payload={
                    "intent_class": intent_class,
                    "urgency_level": urgency_level,
                    "routing_target": "queue_human_review",
                },
                state=state,
            )
            return {
                "routing_target": "queue_human_review",
                "status": AgentStatus.SUCCESS.value,
            }

        lookup_key = f"{intent_class}+{urgency_level}"
        routing_target = routing_rules.get(lookup_key, routing_rules.get("default", "queue_general"))

        # S-3 allowlist gate
        allowlist_error = self._check_allowlist(routing_target, routing_allowlist)
        if allowlist_error is not None:
            return allowlist_error

        emit_trace_event(
            event_type="routing_decision_complete",
            payload={
                "intent_class": intent_class,
                "urgency_level": urgency_level,
                "routing_target": routing_target,
            },
            state=state,
        )

        return {
            "routing_target": routing_target,
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _check_allowlist(routing_target: str, routing_allowlist: list[str]) -> Optional[dict[str, Any]]:
        """S-3 gate: reject routing targets not in the operator allowlist."""
        if routing_target not in routing_allowlist:
            return {
                "error_code": "S3_ROUTING_NOT_ALLOWLISTED",
                "error_message": (
                    f"routing_target '{routing_target}' is not in routing_allowlist. " "Request blocked by S-3 gate."
                ),
                "status": AgentStatus.ERROR.value,
            }
        return None

    @staticmethod
    def _load_operator_config(state: dict[str, Any]) -> dict[str, Any]:
        raw = state.get("operator_config", "{}")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (json.JSONDecodeError, TypeError):
            return {}
        # Defense-in-depth: see urgency_score.py's identical comment.
        return parsed if isinstance(parsed, dict) else {}
