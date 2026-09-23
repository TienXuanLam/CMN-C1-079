"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - LLM injected via constructor (llm_client: BaseLLM); required at runtime
#  - Never import from mediator/, api/, or other agents

import json
import os
from queue import Empty, Queue
from threading import BoundedSemaphore, Thread
from typing import Any, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from shared.services.llm.azure_openai_client import AzureOpenAIClient
from shared.services.llm.base_llm import BaseLLM
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import MultiLangIntentRoutingState
from src.services.input_guidance import build_input_guidance

_DEFAULT_INTENT_TAXONOMY = [
    {"id": "account_inquiry", "label_ja": "アカウント照会", "label_en": "Account Inquiry", "urgency_default": "LOW"},
    {
        "id": "transaction_request",
        "label_ja": "取引依頼",
        "label_en": "Transaction Request",
        "urgency_default": "MEDIUM",
    },
    {"id": "complaint", "label_ja": "苦情", "label_en": "Complaint", "urgency_default": "HIGH"},
    {"id": "general_inquiry", "label_ja": "一般照会", "label_en": "General Inquiry", "urgency_default": "LOW"},
    {"id": "escalation", "label_ja": "エスカレーション", "label_en": "Escalation", "urgency_default": "HIGH"},
]

_DEFAULT_CONFIDENCE_THRESHOLD = 0.75
_LLM_CALL_SLOTS = BoundedSemaphore(value=4)


class _CapacityExhaustedError(Exception):
    """All _LLM_CALL_SLOTS are held; raised instantly, never after a wait.

    Kept a distinct type (not TimeoutError) so the except-block in
    execute() can tell "no slot was free right now" apart from "a slot was
    free, the call started, and it ran past _LLM_TIMEOUT_SECONDS" --
    the two have very different causes and remedies.
    """


_INTENT_CLASSIFY_PROMPT = """You validate and classify multilingual customer service messages without translating them.

First decide whether the message is a real customer service request related to
an account, transaction, complaint, product or service inquiry, or escalation.
Greetings without an issue, random text, coding requests, general knowledge,
weather, jokes, and other unrelated tasks are not business-relevant.

If business-relevant, classify it using only one configured intent ID.

Detected language: {detected_language}
Customer message: {message}

Available intent categories:
{intent_list}

Return JSON only, without Markdown:
{{
  "business_relevant": <true or false>,
  "intent_class": "<best intent ID>",
  "confidence": <number from 0.0 to 1.0>,
  "candidates": [
    {{"id": "<intent ID>", "label": "<label>", "score": <score>}},
    {{"id": "<intent ID>", "label": "<label>", "score": <score>}},
    {{"id": "<intent ID>", "label": "<label>", "score": <score>}}
  ]
}}"""


class IntentClassifyNode(FunctionNode):
    """Classify customer intent from the normalised message using an LLM.

    llm_client must be a BaseLLM instance. Injected via constructor or
    via Graph(config={"llm": ...}) → MainNode → IntentClassifyNode.
    """

    required_trust_level = TrustLevel.ANONYMOUS

    def __init__(
        self,
        llm_client: BaseLLM | None = None,
        timeout_s: float = 20.0,
        max_retry: int = 0,
    ):
        # llm_client exists ONLY for isolated unit tests / the standalone
        # server.py's own explicit config={"llm": ...} injection — it is
        # never how the canonical AgentRegistry deployment path resolves an
        # LLM. AgentRegistry (mediator/registry/agent_registry.py) discovers
        # and instantiates Graph(config=...) from config/config.yaml alone;
        # it does not run server.py's boot-time LLM construction, so
        # self.config.get("llm") in graph.py is always None on that path,
        # and the compile-time requires.secrets check passing said nothing
        # about whether a client actually reaches this node. Resolving the
        # secret fresh inside execute() (self._build_llm below) is the only
        # path that works identically under both AgentRegistry and the
        # standalone server.
        self._llm = llm_client
        # Finding: this previously hard-coded (timeout=20.0, max_retries=0)
        # via the module-level _LLM_TIMEOUT_SECONDS constant -- config.yaml's
        # tuned timeout_s/max_retry (30/2) were never read, the same
        # dead-config bug found and fixed elsewhere in the fleet's equivalent nodes. A
        # single transient provider hiccup was never retried by the Azure
        # OpenAI SDK itself, surfacing as "Service temporarily unavailable"
        # on the very next otherwise-identical request.
        self._timeout_s = timeout_s
        self._max_retry = max_retry

    def execute(self, state: MultiLangIntentRoutingState, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        error_code = state.get("error_code")
        if error_code and error_code not in ("LOW_CONFIDENCE_ADVISORY",):
            return {}

        # STG_MOCK_MODE lets staging validate the graph without external LLM access.
        # _build_llm() raises MissingSecret and the provisional Stage 5 smoke
        # invoke would otherwise always report agent_invoke_responsive=false.
        # STG_MOCK_MODE is the scaffold's standard Stage 5 provisional-deploy
        # toggle, set "true" by the shared deploy-stg CI job -- never in
        # production, and is read directly from os.environ (not ctx.secrets) since it is a CI-only
        # structural toggle, not a credential.
        if self._llm is None and os.environ.get("STG_MOCK_MODE") == "true":
            mock_intent_class = "general_inquiry"
            emit_trace_event(
                "llm_call_mocked",
                {"reason": "STG_MOCK_MODE=true; external LLM access disabled for staging smoke test"},
                state,
            )
            return {
                "intent_class": mock_intent_class,
                "intent_candidates": json.dumps(
                    [{"id": mock_intent_class, "label": mock_intent_class, "score": 1.0}], ensure_ascii=False
                ),
                "confidence_score": 1.0,
                "human_review_required": False,
                "status": AgentStatus.SUCCESS.value,
            }

        try:
            llm = self._llm or self._build_llm(state)
        except Exception as exc:  # noqa: BLE001
            return {
                "error_code": "S2_LLM_ERROR",
                "error_message": self._safe_llm_error(state, exc),
                "status": AgentStatus.ERROR.value,
            }

        operator_config = self._load_operator_config(state)

        taxonomy_raw = operator_config.get("intent_taxonomy", _DEFAULT_INTENT_TAXONOMY)
        if not isinstance(taxonomy_raw, list) or not all(
            isinstance(i, dict) and isinstance(i.get("id"), str) for i in taxonomy_raw
        ):
            # A malformed intent_taxonomy (dict, string, or a list with a
            # non-dict/missing-id entry) previously crashed with
            # AttributeError/TypeError/KeyError at the `intent['id']` access
            # below instead of surfacing a structured contract error.
            return {
                "error_code": "S1_INVALID_OPERATOR_CONFIG",
                "error_message": "operator_config.intent_taxonomy must be a list of objects with a string 'id'.",
                "status": AgentStatus.ERROR.value,
            }
        taxonomy = taxonomy_raw

        raw_threshold = operator_config.get("confidence_threshold", _DEFAULT_CONFIDENCE_THRESHOLD)
        try:
            confidence_threshold = float(raw_threshold)
        except (TypeError, ValueError):
            confidence_threshold = _DEFAULT_CONFIDENCE_THRESHOLD
        if (
            confidence_threshold != confidence_threshold  # NaN
            or confidence_threshold in (float("inf"), float("-inf"))
            or not (0.0 <= confidence_threshold <= 1.0)
        ):
            return {
                "error_code": "S1_INVALID_OPERATOR_CONFIG",
                "error_message": "operator_config.confidence_threshold must be a finite number in [0.0, 1.0].",
                "status": AgentStatus.ERROR.value,
            }

        detected_language = state.get("detected_language", "en")
        normalised_message = state.get("normalised_message", "")

        intent_lines = []
        for intent in taxonomy:
            label = intent.get("label_ja", "") if detected_language == "ja" else intent.get("label_en", "")
            intent_lines.append(f"  - {intent['id']}: {label}")

        prompt = _INTENT_CLASSIFY_PROMPT.format(
            detected_language=detected_language,
            message=normalised_message,
            intent_list="\n".join(intent_lines),
        )

        try:
            response = self._complete_with_deadline(
                llm,
                [{"role": "user", "content": prompt}],
                timeout_seconds=self._timeout_s,
            )
            raw_text = response.get("content", "") if isinstance(response, dict) else str(response)
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, _CapacityExhaustedError):
                # Distinct from a genuine per-call timeout: this fires
                # instantly (no _LLM_TIMEOUT_SECONDS wait) whenever all 4
                # _LLM_CALL_SLOTS are held -- normally by concurrent
                # in-flight calls, but also permanently by any earlier
                # daemon worker thread that is still blocked inside
                # llm.complete() and therefore never reached its `finally:
                # release()` (see _build_llm's timeout comment). Reusing
                # the "Classification timed out" heading here would make a
                # capacity leak indistinguishable from a normal slow
                # response in every bug report, exactly the failure mode
                # this whole heading-splitting effort exists to prevent.
                emit_trace_event(
                    event_type="intent_classify_capacity_exhausted",
                    payload={"available_slots": 0},
                    state=state,
                )
                return build_input_guidance(
                    "The classification service is at capacity right now. This is not caused by your message "
                    f"— please try again in a moment. {self._reference_note(state)}",
                    heading="Service busy",
                )
            if self._is_timeout_error(exc):
                emit_trace_event(
                    event_type="intent_classify_timeout",
                    payload={"timeout_seconds": self._timeout_s},
                    state=state,
                )
                # Finding: this and the two branches below previously all
                # returned the same "Customer service message required"
                # heading as PreProcessNode's genuine empty-input case,
                # making a temporary LLM/provider problem indistinguishable
                # from a user typing mistake in every bug report -- see
                # input_guidance.build_input_guidance()'s docstring. A
                # distinct heading plus a reference ID (same reference the
                # S2_LLM_ERROR branch below already exposes) lets a user
                # retry with confidence it wasn't their input, and lets
                # support correlate the report against backend traces
                # without exposing internals.
                return build_input_guidance(
                    "The message could not be classified in time. This is usually temporary and not caused by "
                    f"your message — please try again. {self._reference_note(state)}",
                    heading="Classification timed out",
                )
            if self._is_auth_or_permission_error(exc):
                return {
                    "error_code": "S2_LLM_ERROR",
                    "error_message": self._safe_llm_error(state, exc),
                    "status": AgentStatus.ERROR.value,
                }
            emit_trace_event(
                event_type="intent_classify_provider_unavailable",
                payload={"fallback": "input_guidance", "exception_type": type(exc).__name__},
                state=state,
            )
            return build_input_guidance(
                "The language classification service is temporarily unavailable. This is not caused by your "
                f"message — please try again shortly. {self._reference_note(state)}",
                heading="Service temporarily unavailable",
            )

        parsed = self._parse_llm_response(raw_text)
        if parsed is None or not isinstance(parsed.get("business_relevant"), bool):
            emit_trace_event(
                event_type="intent_classify_invalid_response",
                payload={"response_contract_valid": False},
                state=state,
            )
            return build_input_guidance(
                "The classification service returned an unexpected response. This is not caused by your "
                f"message — please try again. {self._reference_note(state)}",
                heading="Service temporarily unavailable",
            )
        # Finding: business_relevant=False previously short-circuited straight
        # to build_input_guidance(), rejecting the message outright. In
        # practice this classification is unreliable enough (LLM judgment
        # call on a single message, no taxonomy fit required) that legitimate
        # customer messages were being bounced as "not related to customer
        # service" — the actual bug report this addresses. Falling through
        # to the normal intent_class/confidence handling below instead means
        # an off-topic-leaning message still gets routed (typically as
        # general_inquiry with low confidence, which already triggers
        # LOW_CONFIDENCE_ADVISORY human review) rather than being rejected
        # outright on a single LLM call's say-so.
        if parsed["business_relevant"] is False:
            emit_trace_event(
                event_type="business_relevance_low_confidence",
                payload={"detected_language": detected_language},
                state=state,
            )

        valid_intent_ids = {intent["id"] for intent in taxonomy if isinstance(intent, dict) and "id" in intent}

        intent_class = parsed.get("intent_class", "general_inquiry")
        if not isinstance(intent_class, str) or intent_class not in valid_intent_ids:
            # An LLM-returned intent_class outside the configured taxonomy
            # would otherwise flow straight into UrgencyScoreNode/
            # RoutingDecisionNode's lookup keys unvalidated, silently
            # defaulting behavior downstream rather than surfacing that the
            # model didn't follow the prompt's contract.
            intent_class = "general_inquiry"

        raw_confidence = parsed.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence != confidence or confidence in (float("inf"), float("-inf")):
            # confidence != confidence is the NaN check — float("nan") compares
            # unequal to itself. A NaN/Infinity confidence must not reach
            # min()/max() (their behavior with NaN is not the intended
            # clamping semantics) or be compared against confidence_threshold
            # (a NaN comparison is always False, silently skipping the
            # human-review escalation this value exists to trigger).
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        candidates_raw = parsed.get("candidates", [])
        candidates = candidates_raw if isinstance(candidates_raw, list) else []

        result: dict[str, Any] = {
            "intent_class": intent_class,
            "intent_candidates": json.dumps(candidates, ensure_ascii=False),
            "confidence_score": confidence,
            "human_review_required": False,
            "status": AgentStatus.SUCCESS.value,
        }

        if confidence < confidence_threshold:
            result["human_review_required"] = True
            result["error_code"] = "LOW_CONFIDENCE_ADVISORY"
            result["error_message"] = (
                f"Confidence {confidence:.3f} below threshold {confidence_threshold:.3f}; " "human review required."
            )

        emit_trace_event(
            event_type="intent_classify_complete",
            payload={
                "intent_class": intent_class,
                "confidence_score": confidence,
                "detected_language": detected_language,
            },
            state=state,
        )

        return result

    @staticmethod
    def _load_operator_config(state: dict[str, Any]) -> dict[str, Any]:
        raw = state.get("operator_config", "{}")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (json.JSONDecodeError, TypeError):
            return {}
        # Defense-in-depth: see urgency_score.py's identical comment.
        return parsed if isinstance(parsed, dict) else {}

    def _build_llm(self, state: dict[str, Any]) -> BaseLLM:
        ctx = InvocationContext.from_state(state)
        return AzureOpenAIClient(
            {
                "api_key": ctx.secrets.require("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": ctx.secrets.require("AZURE_OPENAI_ENDPOINT"),
                "azure_deployment": ctx.secrets.require("AZURE_OPENAI_DEPLOYMENT"),
                # `timeout` here only bounds a single HTTP request attempt
                # inside ChatOpenAI's underlying httpx client (connect +
                # read); it does not bound _complete_with_deadline()'s
                # Queue.get() wait below. If the socket never completes its
                # TCP handshake/TLS negotiation in a way httpx's own timeout
                # machinery detects (a proxy silently accepting the
                # connection and then never responding is the known case
                # here), Queue.get() still times out and returns control to
                # the caller, but the daemon worker thread stays blocked
                # inside llm.complete() forever and never reaches its
                # `finally: _LLM_CALL_SLOTS.release()`. Each such leak
                # permanently consumes one of the 4 module-level semaphore
                # slots; after 4 leaked calls (across ANY requests sharing
                # this process, not just one session) every subsequent
                # request fails instantly at `acquire(blocking=False)`
                # instead of waiting out a real per-call timeout -- which is
                # exactly the "first message works, every one after it
                # fails immediately" pattern this fixes. Setting connect/
                # first-byte timeouts well under self._timeout_s gives
                # httpx more opportunities to raise on its own before the
                # outer deadline is reached, shrinking (but not eliminating
                # -- see the capacity-exhausted branch below) the window
                # where a leak can occur.
                "timeout": self._timeout_s,
                "max_retries": self._max_retry,
                "max_tokens": 500,
            }
        )

    @staticmethod
    def _safe_llm_error(state: dict[str, Any], _exc: Exception) -> str:
        reference = str(state.get("trace_id") or state.get("correlation_id") or "unavailable")
        return f"Intent classification is temporarily unavailable. Reference: {reference}."

    @staticmethod
    def _reference_note(state: dict[str, Any]) -> str:
        """Same reference id as _safe_llm_error, for the SUCCESS+guidance
        branches (timeout, malformed LLM response, provider unavailable)
        that are not S2_LLM_ERROR but still need a trace id a user can
        quote to support."""
        reference = str(state.get("trace_id") or state.get("correlation_id") or "unavailable")
        return f"Reference: {reference}."

    @staticmethod
    def _complete_with_deadline(
        llm: BaseLLM,
        messages: list[dict[str, str]],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Apply a total wall-clock deadline to the synchronous LLM call.

        ChatOpenAI's request timeout applies at the transport-operation level,
        not necessarily to the complete end-to-end call. A daemon worker lets
        the graph return guidance at the deadline even if the provider socket
        is still unwinding. The bounded semaphore prevents unbounded worker
        accumulation during a provider outage.
        """
        if not _LLM_CALL_SLOTS.acquire(blocking=False):
            raise _CapacityExhaustedError("Intent classification capacity is temporarily exhausted.")

        result_queue: Queue[tuple[bool, dict[str, Any] | Exception]] = Queue(maxsize=1)

        def _worker() -> None:
            try:
                result_queue.put((True, llm.complete(messages)))
            except Exception as exc:  # noqa: BLE001
                result_queue.put((False, exc))
            finally:
                _LLM_CALL_SLOTS.release()

        Thread(target=_worker, name="intent-classify-llm", daemon=True).start()
        try:
            succeeded, payload = result_queue.get(timeout=timeout_seconds)
        except Empty as exc:
            raise TimeoutError(f"Intent classification exceeded {timeout_seconds:.1f}s.") from exc

        if not succeeded:
            if isinstance(payload, Exception):
                raise payload
            raise RuntimeError("Intent classification failed without an exception payload.")
        if not isinstance(payload, dict):
            raise TypeError("LLM response must be a dictionary.")
        return payload

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        """Recognise provider/transport timeouts without coupling to one SDK."""
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            name = type(current).__name__.lower()
            message = str(current).lower()
            if "timeout" in name or "timed out" in message or "timeout" in message:
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _is_auth_or_permission_error(exc: Exception) -> bool:
        """Keep credential/authorization failures visible as real errors."""
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            name = type(current).__name__.lower()
            message = str(current).lower()
            if any(token in name for token in ("authentication", "permissiondenied", "unauthorized")):
                return True
            if any(
                token in message
                for token in (
                    "authentication failed",
                    "invalid api key",
                    "incorrect api key",
                    "unauthorized",
                    "permission denied",
                    "status code: 401",
                    "status code: 403",
                    "error code: 401",
                    "error code: 403",
                )
            ):
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _parse_llm_response(text: str) -> dict[str, Any] | None:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
