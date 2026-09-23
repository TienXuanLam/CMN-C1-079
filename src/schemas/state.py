"""AgentCore Platform v1.0"""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption. Extend AgentState with agent-specific
# fields only. Do NOT add credentials, secrets, or Pydantic models.

from typing import Optional

from framework.schemas.agent_state import AgentState


class MultiLangIntentRoutingState(AgentState):
    """Agent state for CMN-C1-079 MultiLangIntentRoutingAgent.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.

    Field naming convention:
      - Input:      raw_message, operator_config
      - Processing: detected_language, normalised_message,
                    intent_class, intent_candidates, confidence_score,
                    human_review_required, urgency_level, routing_target
      - Output:     final_output
      - Errors:     error_code, error_message
    """

    # ---------- Input ----------
    raw_message: Optional[str]
    operator_config: Optional[str]
    input_validation_failed: Optional[str]

    # ---------- Processing ----------
    detected_language: Optional[str]
    normalised_message: Optional[str]
    intent_class: Optional[str]
    intent_candidates: Optional[str]
    confidence_score: Optional[float]
    human_review_required: Optional[bool]
    urgency_level: Optional[str]
    routing_target: Optional[str]

    # ---------- Output ----------
    final_output: Optional[str]
    formatted_output: Optional[str]  # Plain Markdown returned as result["output"]

    # ---------- Error propagation ----------
    error_code: Optional[str]
    error_message: Optional[str]
