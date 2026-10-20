"""The errors a Conversation raises, and the classification of the SDK's structured error messages into them.

Exception hierarchy: BackendError
    + PermanentError (terminal error of the conversation; escapes send())
    + AgentError (fatal SDK subprocess error; escapes send())

A retryable provider condition never crosses this boundary as an exception: the Conversation recovers it itself and never resends the interrupted request (#3047).
"""
from __future__ import annotations


from claude_agent_sdk import AssistantMessage, ResultError, ResultMessage, TextBlock


from aibuildai_conversation.errors import BackendError, PermanentError


class _RetryableError(BackendError):
    """PRIVATE -- the adapter's own. Every non-conclusive provider/transport error: rate limit, quota, an expired credential, 5xx, overload, socket drop, safety refusal, or any text the classifier does not recognize. The Conversation catches it, opens the outage pause, and ends the request without completing it; it never crosses the neutral boundary (#1253, #2502, #3047).

    Carries only the raw provider error text -- no reset-time estimate. The three-outcome contract polls at a flat interval until recovery (or the configured ceiling), so nothing predicts when the provider recovers."""


# The structured provider codes that terminate a run; everything else, free
# text included, retries (#1253). The membership test is "can a retry of the
# identical request ever succeed": an overflowed reply ceiling (#2123), a
# missing account permission, or an invalid request cannot, while an expired
# credential (`authentication_failed`) waits for the operator's login (#2502).
_CONCLUSIVE_CODES = {
    "billing_error": PermanentError.Reason.BILLING,
    "invalid_request": PermanentError.Reason.BAD_REQUEST,
    "max_output_tokens": PermanentError.Reason.REPLY_TOO_LONG,
}

# The CLI ends a turn with ``terminal_reason`` ``aborted_streaming`` /
# ``aborted_tools`` only after ``interrupt()``: that is the product's own
# cancellation, never a provider outage and never a failure, so the
# conversation ends the turn on it and it never reaches classification.
CANCELLED_REASONS = frozenset({"aborted_streaming", "aborted_tools"})
# The CLI's own loop ended this conversation (its turn cap, its cost cap, its
# structured-output retries); the identical request cannot continue in that
# session, so the conversation is over for this Agent.
_EXHAUSTED_SUBTYPES = frozenset(
    {
        "error_max_turns",
        "error_max_budget_usd",
        "error_max_structured_output_retries",
    }
)


def classify_assistant_error(message: AssistantMessage) -> BackendError:
    """Classify an ``AssistantMessage.error`` code: a conclusive code is permanent, everything else retries (#1253, #2502)."""
    text = " ".join(
        b.text for b in (message.content or []) if isinstance(b, TextBlock)
    )
    reason = _CONCLUSIVE_CODES.get(message.error or "")
    if reason is not None:
        return PermanentError(reason, provider_code=message.error, message=text)
    return _RetryableError(text)


def classify_result_error(result: ResultMessage | ResultError) -> BackendError:
    """Classify an error result the CLI ended a turn with, or the ``ResultError`` the SDK raises when the CLI exits on one.

    The structured fields decide; ``errors`` and ``result`` are diagnostic prose carried in the message and never choose the outcome.
    """
    said = (
        "; ".join(result.errors or ())
        or result.result
        or result.subtype
        or "error result"
    )
    if result.api_error_status == 400:
        return PermanentError(
            PermanentError.Reason.BAD_REQUEST,
            provider_code=str(result.api_error_status),
            message=said,
        )
    if result.subtype in _EXHAUSTED_SUBTYPES:
        return PermanentError(
            PermanentError.Reason.SESSION_LOST,
            provider_code=result.subtype,
            message=said,
        )
    return _RetryableError(said)
