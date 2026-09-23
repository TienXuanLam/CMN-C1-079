"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - S-3 output gate in private helper (no SDK name conflicts)
#  - Sets formatted_output (SDK get_output() reads this for result["output"])
#  - Never import from mediator/, api/, or other agents

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.security import detect_credentials
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import MultiLangIntentRoutingState

_RE_INTERNAL_URL = re.compile(
    r"https?://(?:localhost|127\.\d+\.\d+\.\d+|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)"
)


class PostProcessNode(FunctionNode):
    """Assemble final output envelope with S-3 PII non-echo and credential scan.

    Sets formatted_output so SDK get_output() exposes it as result["output"].
    """

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: MultiLangIntentRoutingState, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        emit_trace_event(
            event_type="post_process_started",
            payload={"input_validation_failed": state.get("input_validation_failed") == "true"},
            state=state,
        )

        if state.get("input_validation_failed") == "true":
            guidance = str(state.get("formatted_output") or "# Customer service message required")
            return {
                "final_output": guidance,
                "formatted_output": guidance,
                "status": AgentStatus.SUCCESS.value,
            }

        error_code = state.get("error_code")
        if error_code and error_code not in ("LOW_CONFIDENCE_ADVISORY",):
            return {}

        raw_message = state.get("raw_message") or ""
        routing_target = state.get("routing_target", "")
        intent_class = state.get("intent_class", "")
        confidence_score = state.get("confidence_score")
        detected_language = state.get("detected_language", "unknown")
        urgency_level = state.get("urgency_level", "MEDIUM")
        human_review_required = state.get("human_review_required", False)
        intent_candidates_raw = state.get("intent_candidates", "[]")

        try:
            intent_candidates = json.loads(intent_candidates_raw) if intent_candidates_raw else []
        except (json.JSONDecodeError, TypeError):
            intent_candidates = []

        message_ref_hash = hashlib.sha256(raw_message.encode("utf-8")).hexdigest()

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        confidence_text = (
            f"{float(confidence_score):.1%}" if isinstance(confidence_score, (int, float)) else "Not available"
        )
        review_text = "Required" if human_review_required else "Not required"
        candidate_lines = []
        for candidate in intent_candidates[:3]:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("id") or "unknown")
            label = str(candidate.get("label") or candidate_id)
            score = candidate.get("score")
            score_text = f"{float(score):.1%}" if isinstance(score, (int, float)) else "Not available"
            candidate_lines.append(f"- `{candidate_id}` — {label} ({score_text})")
        if not candidate_lines:
            candidate_lines.append("- No additional candidates were returned.")

        sections = [
            "# Customer Service Routing Decision",
            "",
            "## Classification",
            "",
            f"- **Detected language:** `{detected_language}`",
            f"- **Intent:** `{intent_class}`",
            f"- **Confidence:** {confidence_text}",
            f"- **Urgency:** **{urgency_level}**",
            f"- **Human review:** {review_text}",
            "",
            "## Routing",
            "",
            f"Route this request to **`{routing_target}`**.",
            "",
            "## Candidate Intents",
            "",
            *candidate_lines,
        ]
        if error_code == "LOW_CONFIDENCE_ADVISORY":
            sections.extend(["", "## Advisory", "", str(state.get("error_message") or "Human review is required.")])
        sections.extend(
            [
                "",
                "## Audit",
                "",
                f"- **Message reference:** `{message_ref_hash}`",
                f"- **Timestamp:** `{timestamp}`",
                "",
                "> The original customer message is intentionally excluded from this report.",
            ]
        )
        output_str = "\n".join(sections)

        gate_error = self._run_output_gate(output_str, raw_message)
        if gate_error is not None:
            return gate_error

        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Preparing the final routing decision.",
            metadata={"stage": "output_assembly"},
        )

        emit_trace_event(
            event_type="post_process_complete",
            payload={
                "routing_target": routing_target,
                "intent_class": intent_class,
                "detected_language": detected_language,
                "human_review_required": bool(human_review_required),
                "message_ref_hash": message_ref_hash,
            },
            state=state,
        )

        return {
            "final_output": output_str,
            "formatted_output": output_str,
            "status": AgentStatus.SUCCESS.value,
        }

    def _run_output_gate(self, output_str: str, raw_message: str = "") -> Optional[dict[str, Any]]:
        """S-3 output gate: PII non-echo + credential/internal-URL scan."""
        if len(raw_message) >= 8 and raw_message in output_str:
            return {
                "error_code": "S3_BLOCKED",
                "error_message": "S-3 PII non-echo gate: raw_message content detected in output envelope.",
                "status": AgentStatus.ERROR.value,
            }
        if detect_credentials(output_str) or _RE_INTERNAL_URL.search(output_str):
            return {
                "error_code": "S3_BLOCKED",
                "error_message": "S-3 gate: credential or internal URL pattern detected in output envelope.",
                "status": AgentStatus.ERROR.value,
            }
        return None
