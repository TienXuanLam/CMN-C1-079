"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Never import from mediator/, api/, or other agents

import json
from typing import Any, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import MultiLangIntentRoutingState

_DEFAULT_INTENT_TAXONOMY = [
    {"id": "account_inquiry", "urgency_default": "LOW"},
    {"id": "transaction_request", "urgency_default": "MEDIUM"},
    {"id": "complaint", "urgency_default": "HIGH"},
    {"id": "general_inquiry", "urgency_default": "LOW"},
    {"id": "escalation", "urgency_default": "HIGH"},
]

# docs/01_proposal.md declares Chinese/Korean/Vietnamese/Thai as supported
# detected_language values (see pre_process_node.py's zh/ko/vi/th branches), but
# this keyword list previously covered only Japanese/English — urgency
# escalation (a direct routing input) silently never fired for a genuine
# emergency/legal/fraud message in the other four declared languages. Terms
# below are common urgency/escalation/legal/fraud vocabulary per language;
# same precision tradeoff as the existing JA/EN lists — a keyword match is a
# signal, not a guarantee, and a missed idiom in any language degrades to
# the intent-taxonomy default rather than crashing.
_HIGH_URGENCY_KEYWORDS = [
    # Japanese
    "緊急",
    "至急",
    "危険",
    "法的",
    "訴訟",
    "詐欺",
    "解約",
    # English
    "urgent",
    "emergency",
    "danger",
    "escalate",
    "legal",
    "lawsuit",
    "fraud",
    "cancel",
    # Chinese (Simplified/Traditional)
    "紧急",
    "緊急",
    "危险",
    "危險",
    "法律",
    "诉讼",
    "訴訟",
    "诈骗",
    "詐騙",
    "取消",
    # Korean
    "긴급",
    "위험",
    "법적",
    "소송",
    "사기",
    "취소",
    # Vietnamese
    "khẩn cấp",
    "nguy hiểm",
    "pháp lý",
    "kiện",
    "lừa đảo",
    "hủy",
    # Thai
    "ด่วน",
    "ฉุกเฉิน",
    "อันตราย",
    "กฎหมาย",
    "ฟ้องร้อง",
    "หลอกลวง",
    "ยกเลิก",
]


class UrgencyScoreNode(FunctionNode):
    """Score urgency level from intent taxonomy default and keyword signals."""

    required_trust_level = TrustLevel.ANONYMOUS

    def execute(self, state: MultiLangIntentRoutingState, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        error_code = state.get("error_code")
        if error_code and error_code not in ("LOW_CONFIDENCE_ADVISORY",):
            return {}

        operator_config = self._load_operator_config(state)
        taxonomy = operator_config.get("intent_taxonomy", _DEFAULT_INTENT_TAXONOMY)

        intent_class = state.get("intent_class", "general_inquiry")
        urgency_level = "MEDIUM"
        for intent in taxonomy:
            if intent.get("id") == intent_class:
                urgency_level = intent.get("urgency_default", "MEDIUM").upper()
                break

        normalised_message = (state.get("normalised_message") or "").lower()
        for kw in _HIGH_URGENCY_KEYWORDS:
            if kw.lower() in normalised_message:
                urgency_level = "HIGH"
                break

        emit_trace_event(
            event_type="urgency_score_complete",
            payload={
                "urgency_level": urgency_level,
                "intent_class": intent_class,
            },
            state=state,
        )

        return {
            "urgency_level": urgency_level,
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _load_operator_config(state: dict[str, Any]) -> dict[str, Any]:
        raw = state.get("operator_config", "{}")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (json.JSONDecodeError, TypeError):
            return {}
        # Defense-in-depth: PreProcessNode's S-5 gate already rejects a
        # non-dict operator_config before this state field is ever set, but
        # this helper is called independently and must not assume that gate
        # ran — a bare list/scalar would otherwise reach the `.get()` calls
        # below and raise AttributeError.
        return parsed if isinstance(parsed, dict) else {}
