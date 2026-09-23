"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - S-1/S-2/S-5 security logic in private helpers (no SDK name conflicts)
#  - raw_message comes from framework user_input; input_context is a local-adapter fallback
#  - Never import from mediator/, api/, or other agents

import json
import re
import unicodedata
from typing import Any, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.security import detect_credentials
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import MultiLangIntentRoutingState
from src.services.input_guidance import build_input_guidance

_RE_BINARY = re.compile(r"^([A-Za-z0-9+/]{4}){10,}([A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$")
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_VIET = re.compile(r"[Ạ-ỹđơưƯƠôƣƱ]")


class PreProcessNode(FunctionNode):
    """S-1/S-2/S-5 input gates, language detection, and text normalisation.

    Reads the customer message from the framework ``user_input`` field used by
    Marketplace. ``input_context.raw_message`` remains a compatibility fallback
    for standalone/local callers. Operator configuration stays in input_context.
    """

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    @staticmethod
    def _reset_transient_state() -> dict[str, Any]:
        """Clear values that belong only to the previous conversation turn.

        Marketplace may invoke the graph repeatedly with the same session and
        checkpointed state.  Every domain result must therefore be explicitly
        cleared before the new message enters the pipeline; omitting a key from
        a node update leaves its previous value intact in LangGraph state.
        """

        return {
            "input_validation_failed": None,
            "detected_language": None,
            "normalised_message": None,
            "intent_class": None,
            "intent_candidates": None,
            "confidence_score": None,
            "human_review_required": None,
            "urgency_level": None,
            "routing_target": None,
            "final_output": None,
            "formatted_output": None,
            "error_code": None,
            "error_message": None,
        }

    def execute(self, state: MultiLangIntentRoutingState, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        transient_reset = self._reset_transient_state()
        emit_trace_event(
            event_type="pre_process_started",
            payload={"has_input_context": bool(state.get("input_context"))},
            state=state,
        )

        # Marketplace places conversational text in user_input. The standalone
        # FastAPI adapter also mirrors it to input_context.raw_message, so retain
        # that field only as a compatibility fallback. Reading input_context alone
        # made valid Marketplace messages appear empty.
        input_context_value = state.get("input_context")
        input_context = input_context_value if isinstance(input_context_value, dict) else {}
        raw_message = str(state.get("user_input") or input_context.get("raw_message") or "")
        # operator_config is OPTIONAL — a caller with no customisation need
        # (the overwhelming majority: any ordinary VERIFIED_EXTERNAL request)
        # omits it entirely and gets every node's built-in defaults
        # (_DEFAULT_ROUTING_RULES, _DEFAULT_INTENT_TAXONOMY, etc.). Only a
        # caller that actually wants to customise routing/taxonomy/thresholds
        # needs to submit operator_config at all, and doing so requires
        # INTERNAL (see _run_input_gates below) — requiring it unconditionally
        # would have made the agent unusable by its own declared
        # VERIFIED_EXTERNAL entry point for the common case.
        operator_config = input_context.get("operator_config") or "{}"

        # S-5 / S-2 / S-1 validation. caller_trust_level comes from state,
        # which the framework populates from InvocationContext (established
        # by AuthMiddleware / the standalone entry-point's bearer-token
        # resolver) — never from operator_config, which is caller-supplied
        # JSON a requester fully controls. Trusting a self-declared
        # "trust_level": "operator" field there let any VERIFIED_EXTERNAL
        # caller grant themselves operator/admin privilege with zero
        # authentication.
        caller_trust_level = state.get("caller_trust_level", TrustLevel.ANONYMOUS.value)
        gate_error = self._run_input_gates(raw_message, operator_config, caller_trust_level)
        if gate_error is not None:
            if gate_error.get("error_code") == "S1_EMPTY_INPUT":
                return {**transient_reset, **build_input_guidance("The message is empty.")}
            return {**transient_reset, **gate_error}

        normalised = unicodedata.normalize("NFC", raw_message)
        normalised = _RE_CONTROL.sub("", normalised)
        detected_language = self._detect_language(normalised)

        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Detecting the message language and validating the request.",
            metadata={"stage": "language_detection"},
        )

        emit_trace_event(
            event_type="pre_process_complete",
            payload={
                "detected_language": detected_language,
                "message_length": len(normalised),
            },
            state=state,
        )

        return {
            **transient_reset,
            "raw_message": raw_message,
            "operator_config": operator_config,
            "detected_language": detected_language,
            "normalised_message": normalised,
        }

    def _run_input_gates(
        self, raw_message: str, operator_config: str, caller_trust_level: str
    ) -> Optional[dict[str, Any]]:
        """S-5 trust check, S-2 credential scan, S-1 input validation.

        Returns an error state update dict on failure, or None if all checks pass.
        """
        try:
            op_cfg = json.loads(operator_config) if isinstance(operator_config, str) else operator_config
        except (json.JSONDecodeError, TypeError):
            return {
                "error_code": "S5_PERMISSION_DENIED",
                "error_message": "operator_config is not valid JSON — S5_PERMISSION_DENIED",
                "status": AgentStatus.ERROR.value,
            }

        if not isinstance(op_cfg, dict):
            # A syntactically valid JSON value that isn't an object (e.g.
            # operator_config="[]" or "42") passes the HTTP layer's `str`
            # field type and json.loads() cleanly, but every downstream
            # `.get()` call this file and intent_classify.py/
            # routing_decision.py/urgency_score.py make on the parsed value
            # assumes a mapping — AttributeError otherwise.
            return {
                "error_code": "S5_PERMISSION_DENIED",
                "error_message": f"operator_config must be a JSON object, got {type(op_cfg).__name__} — S5_PERMISSION_DENIED",
                "status": AgentStatus.ERROR.value,
            }

        # S-5: authorization to submit a NON-EMPTY operator_config requires
        # INTERNAL trust, established by the framework from a real
        # authentication channel (AuthMiddleware / entry-point bearer-token
        # resolver) — not a self-declared field inside the caller-controlled
        # JSON payload itself. VERIFIED_EXTERNAL is not sufficient here: any
        # authenticated external caller could otherwise write operator_config
        # (routing_rules, routing_allowlist, intent_taxonomy, confidence
        # thresholds) and grant themselves operator/admin behavior. An empty
        # `{}` (the default when the caller omits the field) needs no
        # elevated trust — it customises nothing.
        if op_cfg and caller_trust_level != TrustLevel.INTERNAL.value:
            return {
                "error_code": "S5_PERMISSION_DENIED",
                "error_message": f"caller_trust_level '{caller_trust_level}' cannot submit operator_config — S5_PERMISSION_DENIED",
                "status": AgentStatus.ERROR.value,
            }

        if detect_credentials(operator_config):
            return {
                "error_code": "S2_CREDENTIAL_DETECTED",
                "error_message": "Credential pattern detected in operator_config — request rejected.",
                "status": AgentStatus.ERROR.value,
            }

        if not raw_message:
            return {
                "error_code": "S1_EMPTY_INPUT",
                "error_message": "raw_message is empty or None.",
                "status": AgentStatus.ERROR.value,
            }
        if len(raw_message) > 10000:
            return {
                "error_code": "S1_INPUT_TOO_LONG",
                "error_message": f"raw_message exceeds 10,000 chars (got {len(raw_message)}).",
                "status": AgentStatus.ERROR.value,
            }
        if _RE_BINARY.match(raw_message.strip()):
            return {
                "error_code": "S1_BINARY_INPUT",
                "error_message": "raw_message appears to be a binary/base64 blob.",
                "status": AgentStatus.ERROR.value,
            }
        if _RE_CONTROL.search(raw_message):
            return {
                "error_code": "S1_INJECTION_DETECTED",
                "error_message": "Control character injection detected in raw_message.",
                "status": AgentStatus.ERROR.value,
            }

        return None

    @staticmethod
    def _detect_language(text: str) -> str:
        has_cjk = any("一" <= ch <= "鿿" for ch in text)
        has_hiragana = any("぀" <= ch <= "ゟ" for ch in text)
        has_katakana = any("゠" <= ch <= "ヿ" for ch in text)
        has_hangul = any(("가" <= ch <= "힯") or ("ᄀ" <= ch <= "ᇿ") for ch in text)
        has_thai = any("฀" <= ch <= "๿" for ch in text)
        has_viet = bool(_RE_VIET.search(text))

        if has_hangul:
            return "ko"
        if has_thai:
            return "th"
        if has_viet:
            return "vi"
        if has_cjk and (has_hiragana or has_katakana):
            return "ja"
        if has_hiragana or has_katakana:
            return "ja"
        if has_cjk:
            return "zh"

        printable_ascii = sum(1 for ch in text if 0x20 <= ord(ch) <= 0x7E)
        if len(text) > 0 and (printable_ascii / len(text)) >= 0.80:
            return "en"

        return "unknown"
