"""Errors raised by a conversation."""
from enum import Enum

__all__ = ("BackendError", "PermanentError", "AgentError")

class BackendError(Exception):
    """Base for all backend exceptions."""


class PermanentError(BackendError):
    """Caller must stop after a backend-recognized terminal provider error.

    The Reason enum gives engine code a stable taxonomy independent of any backend's native error codes. The provider_code carries the raw native code, when the provider gave one, for diagnostic display (Web failure_code field, transcript writer); a reason the Conversation itself decided carries none.
    """

    class Reason(Enum):
        BILLING = "billing"
        SESSION_LOST = "session_lost"
        # The provider stayed unavailable for the whole configured outage
        # ceiling (``llm.retry_ceiling_s``, #2502): the Conversation's own
        # decision to stop waiting, not a provider code.
        RETRY_CEILING = "retry_ceiling"
        # Issue #306: the backend rejected the request itself as invalid
        # (HTTP 400) — e.g. a model name the active account cannot serve.
        # Permanent: retrying the identical request cannot succeed, the
        # operator must fix the config. Distinct from BILLING (403, account
        # permission).
        BAD_REQUEST = "bad_request"
        # Issue #2123: the model's reply did not fit in the reply-size ceiling,
        # so nothing usable came back. Permanent: the identical request
        # overflows the same ceiling again, so retrying only burns the clock —
        # one role spent its whole hour on six such replies and returned
        # nothing.
        REPLY_TOO_LONG = "reply_too_long"

    def __init__(
        self,
        reason: "PermanentError.Reason",
        *,
        provider_code: str | None = None,
        message: str = "",
    ) -> None:
        super().__init__(message or reason.value)
        self.reason = reason
        self.provider_code = provider_code


class AgentError(BackendError):
    """Public wrapper for SDK-level errors, propagated unchanged.

    Raised when a backend SDK subprocess itself fails. Its message preserves the native exception type and diagnostic output, so a caller does not replace those details with a generic message.

    ``oom_killed`` answers the one question the message cannot: the SDK reports a dead subprocess as an exit code, and a negative one says only that some signal killed it, so nothing in the text separates "the memory cap" from "our own stop wait" (#2232). The kernel counts OOM kills per cgroup, and the unit's Confinement owns that cgroup, so the backend asks it across the launch and carries the answer here. ``PermanentError`` already keeps this shape: a fact the engine can act on, beside the third party's own text.
    """

    def __init__(self, message: str, *, oom_killed: bool) -> None:
        super().__init__(message)
        self.oom_killed = oom_killed

    def __reduce__(self) -> tuple[object, ...]:
        # The durable step records a failed attempt by pickling its error,
        # and a resume reads it back. The default exception reduction
        # replays only ``args``, so a keyword-only field is dropped and the
        # rebuild raises TypeError inside the resume itself — the run then
        # dies with "an unexpected error occurred" instead of resuming.
        return (_rebuild_agent_error, (str(self), self.oom_killed))


def _rebuild_agent_error(message: str, oom_killed: bool) -> "AgentError":
    return AgentError(message, oom_killed=oom_killed)
