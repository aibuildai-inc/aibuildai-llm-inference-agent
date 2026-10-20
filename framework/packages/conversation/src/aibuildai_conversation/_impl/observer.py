"""Named facts observed during one conversation send."""

from __future__ import annotations
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .spec import Message


class ConversationObserver:
    """Observe selected facts from one Conversation send.

    Each method does nothing unless the caller overrides it. Callback failures propagate to the caller.

    Each method names one semantic fact of the conversation and carries that fact's own values, nothing else: no SDK type comes up through it, and no Event, journal, or run state goes down through it. The observer belongs to one ``send``: every fact of a conversation is reported inside the interaction that caused it, so a caller that records each interaction separately hands ``send`` that interaction's own recorder, and a caller with nothing to record passes none.
    """

    def invocation_started(self, *, message: Message) -> None:
        """The Conversation is about to hand the complete composed message to the SDK. Includes notifications and recovery or repair input. Reported before the send, so a request with no reported completion leaves its trace (#3047)."""

    def session_started(self, *, session_id: str, model: str, cwd: str, tools: tuple[str, ...], mcp_server_names: tuple[str, ...], launcher_path: str | None) -> None:
        """The provider opened a session under ``model``, the accepted semantic model (its family checked against the requested one). ``cwd`` is the SDK subprocess working directory, not the run's output directory (#17); ``mcp_server_names`` are the servers the provider mounted; ``launcher_path`` is the exact SDK executable this Conversation launched."""

    def session_replaced(self, *, session_id: str) -> None:
        """The provider replaced this conversation under the same caller (Claude ``ConversationResetMessage``) and continues it as ``session_id``; the request in flight ends without completing (#3047)."""

    def text_delta(self, text: str, *, parent_tool_use_id: str | None, message_id: str | None = None, model: str | None = None) -> None:
        """One text block of an assistant message."""

    def reasoning_delta(self, text: str, *, parent_tool_use_id: str | None, message_id: str | None = None, model: str | None = None) -> None:
        """One thinking block of an assistant message: its text."""

    def tool_use_start(self, *, tool_use_id: str, name: str, input: dict, parent_tool_use_id: str | None, message_id: str | None = None, model: str | None = None) -> None:
        """The model called one tool, a server-side tool included."""

    def tool_use_end(self, *, tool_use_id: str, output: str | list | None, is_error: bool, parent_tool_use_id: str | None) -> None:
        """One tool call returned."""

    def tool_denied(self, *, tool_use_id: str | None, name: str, reason: str, source: Literal["permission", "unavailable"], agent_id: str | None = None, agent_type: str | None = None, parent_tool_use_id: str | None = None) -> None:
        """A tool was refused by permission policy or reported unavailable by the CLI. Missing scope IDs mean they were not reported. A handler's own failure is not a denial."""

    def usage_delta(self, *, input_tokens: int, output_tokens: int, cached_input_tokens: int, cache_creation_input_tokens: int | None, web_search_requests: int, speed: str, model: str, cumulative: bool, message_id: str | None, parent_tool_use_id: str | None) -> None:
        """Token usage. ``cumulative=False`` is one assistant model call's own usage, keyed by ``message_id``: one real turn arrives as several per-block AssistantMessages sharing the id, and the settled stream figure overwrites the snapshot (#1327). ``cumulative=True`` is the whole-invocation total of Claude's ResultMessage: the billing total, which must neither be added on top of the per-call figures nor mistaken for one call's context occupancy. Only the cumulative usage carries ``web_search_requests`` and ``speed``; a per-call usage reports 0 / "" (probed 2026-06-02)."""

    def turn_complete(self, *, duration_s: float | None, duration_api_s: float | None, cost_usd: float | None, stop_reason: str, num_turns: int, usage_by_model: dict[str, dict[str, float]] | None, structured_output: dict[str, Any] | None) -> None:  # Values follow the caller's recursive JSON schema.
        """One native request ended; it can contain several assistant messages. ``usage_by_model`` is the turn's settled usage per canonical model, in the product's own usage vocabulary (``input_tokens``, ``output_tokens``, ``cache_read_input_tokens``, ``cache_creation_input_tokens``, ``web_search_requests``, ``cost_usd``), covering every model call of the turn, the provider's hidden side queries included; the caller settles it by endpoint (#275, #262). ``structured_output`` is the provider's one terminal structured candidate for a spec with an output schema, judged by the caller's own gates (#3047)."""

    def backend_log(self, *, level: Literal["debug", "info", "warn", "error"], source: str, text: str) -> None:
        """A provider diagnostic with no product content, such as a CLI status notification."""

    def rate_limit_paused(self) -> None:
        """The provider stopped answering; the caller's Local Time pauses."""

    def rate_limit_resumed(self) -> None:
        """The provider answers again; the paused Local Time resumes."""
