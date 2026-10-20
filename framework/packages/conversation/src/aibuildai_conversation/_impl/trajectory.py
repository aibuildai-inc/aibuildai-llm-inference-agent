"""JSON output types and request-local fact association."""
from __future__ import annotations

from copy import deepcopy
from typing import Literal, cast
from typing_extensions import TypedDict
from pydantic import JsonValue, TypeAdapter, ConfigDict, with_config


ObservationKind = Literal["input", "request_input", "text", "thinking", "tool_call", "tool_result", "tool_denied", "usage", "request_started", "request_completed", "session_started", "session_replaced", "rejection", "log", "rate_limit_paused", "rate_limit_resumed"]


@with_config(ConfigDict(extra="forbid", strict=True))
class _Position(TypedDict):
    sequence: int
    ts: float
    call: int | None
    request: int | None
    parent_tool_use_id: str | None
    parent_sequence: int | None
    message_id: str | None
    model: str | None
    tool_call_sequence: int | None
    candidate_sequence: int | None


@with_config(ConfigDict(extra="forbid", strict=True))
class _TextPart(TypedDict):
    type: Literal["text"]
    text: str


@with_config(ConfigDict(extra="forbid", strict=True))
class _ImagePart(TypedDict):
    type: Literal["image"]
    mime_type: str
    data: str


@with_config(ConfigDict(extra="forbid", strict=True))
class _Input(TypedDict):
    content: list[_TextPart | _ImagePart]


class InputObservation(_Position):
    kind: Literal["input", "request_input"]
    data: _Input


@with_config(ConfigDict(extra="forbid", strict=True))
class _Text(TypedDict):
    text: str


class TextObservation(_Position):
    kind: Literal["text", "thinking"]
    data: _Text


@with_config(ConfigDict(extra="forbid", strict=True))
class _ToolCall(TypedDict):
    tool_use_id: str
    name: str
    input: dict[str, JsonValue]


class ToolCallObservation(_Position):
    kind: Literal["tool_call"]
    data: _ToolCall


@with_config(ConfigDict(extra="forbid", strict=True))
class _ToolResult(TypedDict):
    tool_use_id: str
    output: str | list[JsonValue] | None
    is_error: bool


class ToolResultObservation(_Position):
    kind: Literal["tool_result"]
    data: _ToolResult


@with_config(ConfigDict(extra="forbid", strict=True))
class _ToolDenied(TypedDict):
    tool_use_id: str | None
    name: str
    reason: str
    source: Literal["permission", "unavailable"]
    agent_id: str | None
    agent_type: str | None


class ToolDeniedObservation(_Position):
    kind: Literal["tool_denied"]
    data: _ToolDenied


@with_config(ConfigDict(extra="forbid", strict=True))
class _Usage(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int | None
    web_search_requests: int
    speed: str
    model: str
    cumulative: bool
    message_id: str | None


class UsageObservation(_Position):
    kind: Literal["usage"]
    data: _Usage


@with_config(ConfigDict(extra="forbid", strict=True))
class _Completion(TypedDict):
    duration_s: float | None
    duration_api_s: float | None
    cost_usd: float | None
    cost_source: Literal["provider"] | None
    stop_reason: str
    num_turns: int
    usage_by_model: dict[str, dict[str, float]] | None
    structured_output: dict[str, JsonValue] | None


class CompletionObservation(_Position):
    kind: Literal["request_completed"]
    data: _Completion


@with_config(ConfigDict(extra="forbid", strict=True))
class _Session(TypedDict):
    session_id: str
    model: str
    cwd: str
    tools: list[str]
    mcp_server_names: list[str]
    launcher_path: str | None


class SessionObservation(_Position):
    kind: Literal["session_started"]
    data: _Session


@with_config(ConfigDict(extra="forbid", strict=True))
class _Replacement(TypedDict):
    session_id: str


class ReplacementObservation(_Position):
    kind: Literal["session_replaced"]
    data: _Replacement


@with_config(ConfigDict(extra="forbid", strict=True))
class _Rejection(TypedDict):
    reason: str


class RejectionObservation(_Position):
    kind: Literal["rejection"]
    data: _Rejection


@with_config(ConfigDict(extra="forbid", strict=True))
class _Log(TypedDict):
    level: str
    source: str
    text: str


class LogObservation(_Position):
    kind: Literal["log"]
    data: _Log


@with_config(ConfigDict(extra="forbid", strict=True))
class _Lifecycle(TypedDict):
    pass


class LifecycleObservation(_Position):
    kind: Literal["request_started", "rate_limit_paused", "rate_limit_resumed"]
    data: _Lifecycle


Observation = InputObservation | TextObservation | ToolCallObservation | ToolResultObservation | ToolDeniedObservation | UsageObservation | CompletionObservation | SessionObservation | ReplacementObservation | RejectionObservation | LogObservation | LifecycleObservation


class _TokenCounts(TypedDict):
    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None
    web_search_requests: int | None


class _SourceUsage(_TokenCounts):
    parent_tool_use_id: str | None
    model: str
    speed: str


class AssistantMessage(TypedDict):
    """One reported assistant message; usage is its latest snapshot, never a request total."""
    request: int
    message_id: str
    parent_tool_use_id: str | None
    model: str | None
    observations: list[int]
    usage: _Usage | None


class Request(TypedDict):
    """One observed provider request; absent completion and usage remain None.

    usage is a reported cumulative observation, or the sum of stable message
    snapshots. usage_by_source preserves each model and parent separately.
    Unidentified message snapshots remain in observations and make totals unknown.
    """
    index: int
    call: int | None
    start: int
    completion: int | None
    usage: _TokenCounts | None
    usage_by_source: list[_SourceUsage]


class Trajectory(TypedDict):
    """Detached JSON data for a complete observed prefix.

    name and started_at are caller facts; started_at is an ISO timezone string.
    description holds only explicitly supplied system_prompt, user_message and
    input values. observations preserves every callback in arrival order.
    messages identifies reported assistant messages and their latest usage snapshots.
    requests provides resolved completion and usage associations so consumers
    do not implement message snapshot or cumulative semantics themselves.

    Each observation has sequence (local integer), kind (callback fact), ts
    (original seconds), call/request (local integer or None), parent_tool_use_id
    (reported string or None), parent_sequence (observed dispatch index or None),
    message_id/model (reported strings or None), tool_call_sequence and candidate_sequence (distinct relation indexes or None), and data (the fact payload).

    Payloads retain these callback fields:
    - input: caller intention. request_input: the complete native request input.
      Both use ordered text or base64 image content parts.
    - text/thinking: text. log: level, source, text.
    - tool_call: tool_use_id, name, input. tool_result: tool_use_id, output, is_error.
    - tool_denied: tool_use_id, name, reason, source, agent_id, agent_type.
    - usage: input_tokens, output_tokens, cache_read_input_tokens,
      cache_creation_input_tokens, web_search_requests, speed, model,
      cumulative, message_id. Missing cache creation remains None.
    - request_completed: duration_s, duration_api_s, cost_usd, cost_source,
      stop_reason, num_turns, usage_by_model, structured_output. Missing amounts
      and candidates remain None; cost_source is provider only when reported.
    - session_started: session_id, model, cwd, tools, mcp_server_names, launcher_path.
    - session_replaced: session_id. rejection: reason. Start and rate-limit
      facts have empty payloads and retain their order, request and time.

    Each request has index/call, start (observation index), completion (index
    or None), usage (resolved totals or None), and usage_by_source (model,
    parent_tool_use_id, speed and token/request counts). Unknown totals do not
    erase the original usage observations. A missing completion means incomplete,
    including failure or cancellation before a terminal fact was observed.
    """
    name: str
    started_at: str
    description: dict
    observations: list[Observation]
    requests: list[Request]
    messages: list[AssistantMessage]


def organize(name: str, started_at: str, description: dict,
             observations: list[Observation]) -> Trajectory:
    records = deepcopy(observations)
    requests: list[Request] = []
    calls: dict[tuple[int | None, str | None, str | None], int] = {(item["request"], item["parent_tool_use_id"], item["data"]["tool_use_id"]): item["sequence"]
        for item in records if item["kind"] == "tool_call"}
    candidates: dict[int | None, int] = {}
    snapshots: dict[tuple[int, str | None, str], _Usage] = {}
    unidentified: set[int] = set()
    for item in records:
        request = item["request"]
        if item["kind"] == "usage" and not item["data"]["cumulative"]:
            item["message_id"], item["model"] = item["data"]["message_id"], item["data"]["model"] or None
        if item["kind"] == "request_started":
            requests.append(Request(index=len(requests), call=item["call"], start=item["sequence"],
                completion=None, usage=None, usage_by_source=[]))
        item["parent_sequence"] = next((record["sequence"] for record in reversed(records[:item["sequence"]])
            if record["request"] == request and record["kind"] == "tool_call"
            and record["data"]["tool_use_id"] == item["parent_tool_use_id"]), None)
        key = (request, item["parent_tool_use_id"], item["data"].get("tool_use_id", ""))
        if item["kind"] == "tool_call":
            calls[key] = item["sequence"]
        elif item["kind"] == "tool_result" or item["kind"] == "tool_denied":
            item["tool_call_sequence"] = calls.get(key)
        elif item["kind"] == "rejection":
            item["candidate_sequence"] = candidates.pop(item["call"], None)
        elif item["kind"] == "request_completed" and request is not None:
            requests[request]["completion"] = item["sequence"]
            if item["data"]["structured_output"] is not None:
                candidates[item["call"]] = item["sequence"]
        elif item["kind"] == "usage" and request is not None:
            if item["data"]["cumulative"]:
                requests[request]["usage"] = cast(_TokenCounts, item["data"])
            elif item["data"]["message_id"] is None:
                unidentified.add(request)
            else:
                snapshots[request, item["parent_tool_use_id"], item["data"]["message_id"]] = item["data"]
    fields = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "web_search_requests")
    for request in requests:
        sources: dict[tuple[str | None, str, str], _SourceUsage] = {}
        for (index, parent, _message), usage in snapshots.items():
            if index != request["index"]:
                continue
            key = parent, usage["model"], usage["speed"]
            total = sources.setdefault(key, cast(_SourceUsage, dict(parent_tool_use_id=parent, model=usage["model"],
                speed=usage["speed"], **dict.fromkeys(fields, 0))))
            for field in fields:
                total[field] = None if total[field] is None or usage[field] is None else cast(int, total[field]) + cast(int, usage[field])
        request["usage_by_source"] = list(sources.values())
        if request["usage"] is None and sources and request["index"] not in unidentified:
            request["usage"] = cast(_TokenCounts, {field: None if any(u[field] is None for u in sources.values())
                else sum(cast(int, u[field]) for u in sources.values()) for field in fields})
    messages_by_id: dict[tuple[int, str | None, str], AssistantMessage] = {}
    for item in records:
        if item["request"] is None or item["message_id"] is None:
            continue
        identity = item["request"], item["parent_tool_use_id"], item["message_id"]
        message = messages_by_id.setdefault(identity, AssistantMessage(request=item["request"],
            message_id=item["message_id"], parent_tool_use_id=item["parent_tool_use_id"],
            model=item["model"], usage=snapshots.get(identity), observations=[]))
        message["observations"].append(item["sequence"])
    messages = list(messages_by_id.values())
    return Trajectory(name=name, started_at=started_at, description=deepcopy(description), observations=records, requests=requests, messages=messages)


observation_adapter = TypeAdapter(Observation)
