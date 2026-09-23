"""Safe, reusable guidance for inputs that cannot enter the routing workflow."""

from typing import Any

from framework.schemas.agent_status import AgentStatus


def build_input_guidance(reason: str, *, heading: str = "Customer service message required") -> dict[str, Any]:
    """Build a successful Markdown response without echoing the user input.

    Finding: every caller previously got the same fixed heading regardless
    of *why* the pipeline stopped short of a routing decision. PreProcessNode
    calls this for a genuinely empty/invalid message, but IntentClassifyNode
    also calls it for an LLM timeout, a malformed LLM response, an
    off-topic message, and a provider outage -- four situations with
    nothing in common except "no routing decision was made", yet the UI
    showed "Customer service message required" for all of them. That
    reads as "you typed something wrong" even when the actual cause was a
    timing-out or misbehaving provider, making a real infrastructure
    problem (e.g. a proxy silently swallowing the LLM call) look
    indistinguishable from a user input mistake in every bug report.
    `heading` lets each call site say what actually happened; the default
    preserves PreProcessNode's original wording for its own genuine
    input-validation case.
    """

    output = (
        f"# {heading}\n\n"
        f"{reason}\n\n"
        "Please enter a customer request that describes the service issue and the help needed.\n\n"
        "## Example\n\n"
        "> I was charged twice for the same order and need an urgent refund."
    )
    return {
        "input_validation_failed": "true",
        "final_output": output,
        "formatted_output": output,
        "error_code": None,
        "error_message": None,
        "status": AgentStatus.SUCCESS.value,
    }
