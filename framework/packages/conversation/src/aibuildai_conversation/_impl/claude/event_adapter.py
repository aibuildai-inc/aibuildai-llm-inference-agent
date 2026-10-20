"""Native Claude SDK messages -> semantic observations of one provider request.

``RequestStream`` reads every native SDK message of one provider request and calls the ``ConversationObserver`` method that names what the provider did: a session that started, a text block, a tool use, a usage report, a completed turn. There is no intermediate message object between the two: the observer's arguments are the facts themselves, so ``engine`` never sees a Claude SDK type and this package never sees an engine Event.

Reading is one-to-N: an AssistantMessage with mixed blocks reports reasoning, text, tool-start, and usage observations; a ResultMessage reports the cumulative usage and the turn completion; the init system message reports the session start; a UserMessage's tool results report tool ends. ``parent_tool_use_id`` travels from the SDK message onto every observation: None at top level, the spawning Agent tool's id for a sub-agent's messages (with ``forward_subagent_text`` its text and thinking arrive the same way), which routes them to the Task tool's transcript sub-block.

Dispatch is by product meaning, not by SDK class inventory. A system message is classified by its ``subtype`` against an explicit list; a subtype, a content block, or a top-level message class this module has never classified fails loud, because it may carry state the product cannot see.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from claude_agent_sdk import (
    AssistantMessage,
    Message,
    ModelUsage,
    ResultMessage,
    ServerToolResultBlock,
    ServerToolUseBlock,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from aibuildai_conversation._impl.names import canonical_model_name
from aibuildai_conversation._impl.observer import ConversationObserver


_API_ERROR_PREFIX = "API Error:"

# Bookkeeping the bundled CLI 2.1.251 writes as system messages: turn and
# session state notes, and the sub-agent task lifecycle that the spawning
# Agent tool's own tool_use / tool_result blocks already attribute. None of
# them carries content, usage, or a lifecycle change the product acts on.
# ``commands_changed`` is one of these: the CLI re-announces the session's
# slash commands when the available skill set changes under a running session.
# It is purely informational, but classifying it loud tore down a whole
# 10-step run at step 5 -- the same failure shape the permission_denied
# branch below was added to prevent.
_BOOKKEEPING_SUBTYPES = frozenset(
    {
        "api_retry",
        "commands_changed",
        "compact_boundary",
        "informational",
        "session_state_changed",
        "status",
        "thinking_tokens",
        "turn_duration",
        "turn_starting",
        "task_started",
        "task_progress",
        "task_notification",
        "task_updated",
    }
)




# The per-model usage facts of a Claude ResultMessage, named in the product's
# own usage vocabulary. The SDK's per-model capability metadata (its
# contextWindow and maxOutputTokens, recomputed from the model id on every
# response and wrong for a model outside its registry, #1332) is not a fact
# of this run and is not read; infra.model_info.context_window_for_model is the
# one source for a model's context window.
_MODEL_USAGE_FIELDS: dict[str, str] = {
    "inputTokens": "input_tokens",
    "outputTokens": "output_tokens",
    "cacheReadInputTokens": "cache_read_input_tokens",
    "cacheCreationInputTokens": "cache_creation_input_tokens",
    "webSearchRequests": "web_search_requests",
    "costUSD": "cost_usd",
}


def _model_family(model_id: str) -> str:
    """The family a Claude model id names, across its aliases, dated snapshots, and the ``[1m]`` context suffix."""
    base = model_id.lower().removesuffix("[1m]")
    for family in ("opus", "sonnet", "haiku"):
        if family in base:
            return family
    return base


def _usage_by_model(raw: Mapping[str, ModelUsage]) -> dict[str, dict[str, float]]:
    """The turn's usage per canonical model, from the SDK's per-model records.

    ``canonicalModel`` is the SDK's pricing identity for an entry; the raw key (an alias, a dated snapshot, a provider id) names it only when the SDK gives none. An alias and its dated snapshot are one model, so their usage adds."""
    out: dict[str, dict[str, float]] = {}
    for raw_key, record in raw.items():
        entry = cast(dict[str, Any], record)  # Native payloads contain nested JSON values with different types.
        model = canonical_model_name(entry.get("canonicalModel") or raw_key)
        missing = sorted(set(_MODEL_USAGE_FIELDS) - set(entry))
        if missing:
            raise AssertionError(
                f"Claude ResultMessage.model_usage[{raw_key!r}] lacks {missing}; "
                f"keys={sorted(entry)}. Wire shape changed."
            )
        totals = out.setdefault(model, {})
        for sdk_key, field in _MODEL_USAGE_FIELDS.items():
            totals[field] = totals.get(field, 0) + entry[sdk_key]
    return out


def _tool_result_is_error(block: ToolResultBlock) -> bool:
    if block.is_error is True:
        return True
    content = block.content
    if isinstance(content, str):
        return content.lstrip().startswith(_API_ERROR_PREFIX)
    if isinstance(content, list):
        return any(
            isinstance(item, dict)
            and isinstance(item.get("text"), str)
            and item["text"].lstrip().startswith(_API_ERROR_PREFIX)
            for item in content
        )
    return False


class RequestStream:
    """The native SDK messages of one provider request, read into observations for the Conversation's observer and into the request's own outcome: whether a turn completed, its top-level text, and its terminal structured candidate."""

    def __init__(self, observers: tuple[ConversationObserver, ...], *, launcher_path: str | None, requested_model: str, observed_hooks: set[str], previous_result: ResultMessage | None = None) -> None:
        self._observers = observers
        self._observed_hooks = observed_hooks
        self._tool_names: dict[str, str] = {}
        self._previous_result = previous_result
        self.result: ResultMessage | None = None
        self._launcher_path = launcher_path
        self._requested_model = requested_model
        # The in-flight message per (session_id, parent_tool_use_id) lane of
        # the partial-message stream, from its message_start (#1327).
        self._open_lanes: dict[tuple[str, str | None], tuple[str, str, dict[str, Any]]] = {}  # Native payloads contain nested JSON values with different types.
        self._text: list[str] = []
        self.completed = False
        self.output: dict[str, Any] | None = None  # Native payloads contain nested JSON values with different types.

    @property
    def text(self) -> str:
        """The completed turn's top-level assistant text so far."""
        return "".join(self._text)

    def read(self, ev: Message) -> None:
        """Read one native SDK message of this request."""
        if isinstance(ev, StreamEvent):
            self._read_stream(ev)
        elif isinstance(ev, SystemMessage):
            self._read_system(ev)
        elif isinstance(ev, AssistantMessage):
            self._read_assistant(ev)
        elif isinstance(ev, UserMessage):
            self._read_user(ev)
        elif isinstance(ev, ResultMessage):
            self._read_result(ev)
        else:
            # The conversation consumes the rate-limit and conversation-reset
            # messages before they reach this reader.
            raise AssertionError(
                f"unclassified Claude SDK message {type(ev).__name__} reached the request reader"
            )

    def _read_system(self, ev: SystemMessage) -> None:
        data = ev.data or {}
        if ev.subtype == "init":
            sid = data.get("session_id")
            if sid is None:
                raise AssertionError(
                    f"Claude SystemMessage(init).data lacks session_id; "
                    f"data keys={sorted(data.keys())}. Wire shape changed."
                )
            model = canonical_model_name(data["model"])
            # The provider's naming is checked here, where the names are the
            # provider's: the session must serve the requested model's family.
            if _model_family(self._requested_model) != _model_family(model):
                raise AssertionError(
                    f"model mismatch: requested {self._requested_model!r}, "
                    f"backend reported {model!r}"
                )
            tools_fact = tuple(data.get('tools') or ())
            mcp_server_names_fact = tuple((str(server['name']) for server in data.get('mcp_servers') or ()))
            for observer in self._observers:
                observer.session_started(
                    session_id=sid,
                    model=model,
                    cwd=data["cwd"],
                    tools=tools_fact,
                    mcp_server_names=mcp_server_names_fact,
                    launcher_path=self._launcher_path,
                )
        elif ev.subtype == "notification":
            # A CLI status-line notice: its own text, with a color that marks a
            # warning (a blocked agent MCP server, a restricted sub-agent model).
            level_fact = 'warn' if data.get('color') == 'warning' else 'info'
            text_fact = f"Claude notification: {data.get('text', '')}"
            for observer in self._observers:
                observer.backend_log(
                    level=level_fact,
                    source="claude-sdk",
                    text=text_fact,
                )
        elif ev.subtype == "permission_denied":
            # The CLI refused a tool call on its own authority: a safetyCheck
            # (an `rm` whose target is a possibly-empty shell variable, say) that
            # no permission rule, bypassPermissions included, may auto-allow.
            # The call is already settled for the model, which sees the refusal
            # as that tool_use's error result and re-plans on its next turn; the
            # product only records why. Failing loud here instead tore down the
            # whole run (every unit of the search) on the first such refusal.
            # The CLI's schema requires tool_name, tool_use_id and message;
            # decision_reason_type and decision_reason are optional.
            text_fact = f"Claude denied {data['tool_name']} call {data['tool_use_id']} ({data.get('decision_reason_type', 'unspecified')}): {data.get('decision_reason') or data['message']}"
            for observer in self._observers:
                observer.backend_log(
                    level="warn",
                    source="claude-sdk",
                    text=text_fact,
                )
        elif ev.subtype not in _BOOKKEEPING_SUBTYPES:
            # Hook events and session-store mirror errors need options the
            # product never sets; anything else is a subtype never classified.
            raise AssertionError(
                f"unclassified Claude system message {ev.subtype!r}: {data}"
            )

    def _read_assistant(self, ev: AssistantMessage) -> None:
        parent = ev.parent_tool_use_id
        for block in (ev.content or []):
            if isinstance(block, TextBlock):
                if parent is None:
                    self._text.append(block.text)
                for observer in self._observers:
                    observer.text_delta(block.text, parent_tool_use_id=parent, message_id=ev.message_id, model=ev.model)
            elif isinstance(block, ThinkingBlock):
                for observer in self._observers:
                    observer.reasoning_delta(block.thinking, parent_tool_use_id=parent, message_id=ev.message_id, model=ev.model)
            elif isinstance(block, (ToolUseBlock, ServerToolUseBlock)):
                if isinstance(block, ToolUseBlock):
                    self._tool_names[block.id] = block.name
                # A server-side tool (web search / fetch, advisor) ran inside the
                # API call; it is a tool use of this turn all the same.
                input_fact = dict(block.input)
                for observer in self._observers:
                    observer.tool_use_start(
                        tool_use_id=block.id,
                        name=block.name,
                        input=input_fact,
                        parent_tool_use_id=parent, message_id=ev.message_id, model=ev.model,
                    )
            elif isinstance(block, ServerToolResultBlock):
                output_fact = [block.content]
                is_error_fact = str(block.content.get('type', '')).endswith('_error')
                for observer in self._observers:
                    observer.tool_use_end(
                        tool_use_id=block.tool_use_id,
                        output=output_fact,
                        is_error=is_error_fact,
                        parent_tool_use_id=parent,
                    )
            else:
                # A ToolResultBlock arrives on the UserMessage, never here.
                raise AssertionError(
                    f"unclassified Claude content block {type(block).__name__} "
                    f"in an AssistantMessage"
                )
        if ev.usage:
            self._usage(
                ev.usage,
                model=canonical_model_name(ev.model or ""),
                parent_tool_use_id=parent,
                message_id=ev.message_id,
            )

    def _read_user(self, ev: UserMessage) -> None:
        content = ev.content
        if not isinstance(content, list):
            # str content (prompt echo, not a tool result) carries no facts.
            return
        parent = ev.parent_tool_use_id
        for block in content:
            if isinstance(block, ToolResultBlock):
                name = self._tool_names.pop(block.tool_use_id, None)
                # A real handler passes the mandatory hook, even if its error
                # text matches the CLI diagnostic. Unavailable tools do not.
                if (name is not None and block.tool_use_id not in self._observed_hooks
                        and _tool_result_is_error(block) and isinstance(block.content, str)
                        and block.content.startswith(f"<tool_use_error>Error: No such tool available: {name}")):
                    for observer in self._observers:
                        observer.tool_denied(tool_use_id=block.tool_use_id, name=name,
                            reason=block.content, source="unavailable", parent_tool_use_id=parent)
                is_error_fact = _tool_result_is_error(block)
                for observer in self._observers:
                    observer.tool_use_end(
                        tool_use_id=block.tool_use_id,
                        output=block.content,
                        is_error=is_error_fact,
                        parent_tool_use_id=parent,
                    )
            # A user turn's own text blocks are the prompt the product sent (or
            # the parent's brief to a sub-agent), which the product already has.

    def _usage(
        self,
        usage: dict,
        *,
        model: str = "",
        cumulative: bool = False,
        message_id: str | None = None,
        parent_tool_use_id: str | None = None,
    ) -> None:
        """Report one SDK usage dict by its actual shape.

        Only the cumulative ResultMessage.usage carries ``server_tool_use`` and ``speed``; a per-call or stream usage reports their honest absence (0 / "") rather than an invented value.
        """
        server_tools = usage.get("server_tool_use") or {}
        input_tokens_fact = int(usage.get('input_tokens') or 0)
        output_tokens_fact = int(usage.get('output_tokens') or 0)
        cached_input_tokens_fact = int(usage.get('cache_read_input_tokens') or 0)
        cache_creation_input_tokens_fact = usage.get('cache_creation_input_tokens')
        web_search_requests_fact = int(server_tools.get('web_search_requests', 0) or 0)
        speed_fact = str(usage.get('speed', '') or '')
        for observer in self._observers:
            observer.usage_delta(
                input_tokens=input_tokens_fact,
                output_tokens=output_tokens_fact,
                cached_input_tokens=cached_input_tokens_fact,
                cache_creation_input_tokens=cache_creation_input_tokens_fact,
                web_search_requests=web_search_requests_fact,
                speed=speed_fact,
                model=model,
                cumulative=cumulative,
                message_id=message_id,
                parent_tool_use_id=parent_tool_use_id,
            )

    def _read_result(self, ev: ResultMessage) -> None:
        self.result = ev
        cost = ev.total_cost_usd
        models = _usage_by_model(ev.model_usage) if ev.model_usage else None
        # Cost and per-model totals span this native client. Result.usage is
        # already scoped to this query. Report only this query's added totals.
        if self._previous_result is not None:
            previous_cost = self._previous_result.total_cost_usd
            cost = None if cost is None or previous_cost is None else cost - previous_cost
            if models is not None:
                previous_models = _usage_by_model(self._previous_result.model_usage) if self._previous_result.model_usage else None
                models = None if previous_models is None else {
                    model: {name: value - previous_models.get(model, {}).get(name, 0) for name, value in values.items()}
                    for model, values in models.items()
                }
        if (cost is not None and cost < 0) or any(value < 0 for values in (models or {}).values() for value in values.values()):
            raise AssertionError("native client billing counters decreased")
        # ResultMessage.usage is the whole-invocation total (summed across
        # every internal turn), not a single turn's occupancy. Mark it so the
        # consumer feeds totals from it without clobbering last_turn_*.
        self._usage(ev.usage or {}, cumulative=True)
        # The provider's terminal structured candidate (#3047). The caller's
        # own gates judge it; reading requires only the declared object shape.
        candidate = None if ev.is_error else ev.structured_output
        if candidate is not None and not isinstance(candidate, dict):
            raise AssertionError(
                "ResultMessage.structured_output must be an object, "
                f"got {type(candidate).__name__}"
            )
        duration_s_fact = ev.duration_ms / 1000.0
        duration_api_s_fact = ev.duration_api_ms / 1000.0 if ev.duration_api_ms is not None else None
        stop_reason_fact = (ev.terminal_reason or ev.subtype if ev.is_error else ev.stop_reason) or 'end_turn'
        for observer in self._observers:
            observer.turn_complete(
                duration_s=duration_s_fact,
                duration_api_s=duration_api_s_fact,
                cost_usd=cost,
                stop_reason=stop_reason_fact,
                num_turns=ev.num_turns,
                usage_by_model=models,
                structured_output=candidate,
            )
        self.output = candidate
        self.completed = True

    def _read_stream(self, ev: StreamEvent) -> None:
        """Bind stream usage to its message and report both the initial and updated snapshots.

        The CLI's AssistantMessage snapshots usage at message_start, where the OpenAI-adapter route can only report zeros (the OpenAI protocol reveals usage at stream end) and the Anthropic route holds a placeholder output count. The initial snapshot is reported before content so an interruption cannot discard it. The raw message_delta carries the settled numbers on BOTH routes; the delta event itself has no message id, so the lane tracks the current message from the preceding message_start. Downstream folds key on message_id and let the later figure overwrite the earlier snapshot.
        """
        raw = ev.event
        kind = raw.get("type")
        lane = (ev.session_id, ev.parent_tool_use_id)
        if kind == "message_start":
            message = raw.get("message") or {}
            message_id = message.get("id")
            if message_id is None:
                raise AssertionError(
                    f"stream message_start lacks message.id; "
                    f"keys={sorted(message.keys())}. Wire shape changed."
                )
            self._open_lanes[lane] = (
                message_id,
                canonical_model_name(message.get("model") or ""),
                dict(message.get("usage") or {}),
            )
            if message.get("usage"):
                self._usage(message["usage"], model=self._open_lanes[lane][1],
                            parent_tool_use_id=ev.parent_tool_use_id, message_id=message_id)
            return
        if kind != "message_delta":
            return  # other partial-stream kinds carry no settlement data
        usage = raw.get("usage")
        if not usage:
            return
        bound = self._open_lanes.get(lane)
        if bound is None:
            raise AssertionError(
                f"stream message_delta with usage arrived on lane {lane} "
                f"with no preceding message_start. Wire shape changed."
            )
        message_id, model, settled = bound
        settled.update(usage)
        self._usage(
            settled,
            model=model,
            parent_tool_use_id=ev.parent_tool_use_id,
            message_id=message_id,
        )
