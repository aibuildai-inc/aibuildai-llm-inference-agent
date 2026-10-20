"""Record semantic observations without a file or display policy."""
from __future__ import annotations

import base64
import copy
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from aibuildai_conversation import ConversationObserver, Image, Message
from .trajectory import Trajectory, Observation, ObservationKind, organize, observation_adapter


class Transcript(ConversationObserver):
    """Collect facts and return detached, JSON-compatible trajectory snapshots.

    Times are caller-provided seconds, or seconds from the supplied clock.
    No files, model clients, pricing rules, or product Action state are owned here.
    """

    def __init__(self, name: str, *, started_at: datetime,
                 clock: Callable[[], float] | None = None) -> None:
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError("started_at must include a timezone")
        origin = time.monotonic()
        self._clock = clock if clock is not None else lambda: time.monotonic() - origin
        self._name = name
        self._started_at = started_at.isoformat()
        self._description: dict = {}
        self._observations: list[Observation] = []
        self._call: int | None = None
        self._request: int | None = None

    def describe(self, *, system_prompt: str | None = None,
                 user_message: str | None = None, input_payload: dict | None = None) -> None:
        """Attach explicit description facts; repeated matching values are allowed."""
        for key, value in (("system_prompt", system_prompt), ("user_message", user_message), ("input", input_payload)):
            if value is None:
                continue
            if key in self._description and self._description[key] != value:
                raise ValueError(f"Conflicting transcript description: {key}")
            self._description[key] = copy.deepcopy(value)

    @property
    def trajectory(self) -> Trajectory:
        """Return a detached snapshot, directly serializable with json.dumps.

        Observation sequence numbers are local indexes, never provider IDs.
        Request usage replaces repeated message snapshots with the latest value;
        a reported cumulative value supersedes that request's message sum.
        """
        return organize(self._name, self._started_at, self._description, self._observations)

    def begin_call(self, message: Message, *, ts: float | None = None) -> int:
        """Record caller input and return its local call index."""
        self._call = 0 if self._call is None else self._call + 1
        self._request = None
        self._append("input", ts, {"content": _content(message)})
        return self._call

    def reject_output(self, reason: str, *, ts: float | None = None) -> None:
        """Record the caller's rejection of its most recent candidate."""
        self._append("rejection", ts, {"reason": reason})

    def invocation_started(self, *, message: Message | None = None, ts: float | None = None) -> None:
        self._request = sum(item["kind"] == "request_started" for item in self._observations)
        self._append("request_started", ts, {})
        if message is not None:
            self._append("request_input", ts, {"content": _content(message)})

    def session_started(self, *, session_id: str, model: str, cwd: str,
                        tools: tuple[str, ...], mcp_server_names: tuple[str, ...],
                        launcher_path: str | None, ts: float | None = None) -> None:
        self._append("session_started", ts, dict(session_id=session_id, model=model, cwd=cwd,
            tools=list(tools), mcp_server_names=list(mcp_server_names), launcher_path=launcher_path))

    def session_replaced(self, *, session_id: str, ts: float | None = None) -> None:
        self._append("session_replaced", ts, {"session_id": session_id})

    def text_delta(self, text: str, *, parent_tool_use_id: str | None, message_id: str | None = None, model: str | None = None, ts: float | None = None) -> None:
        self._append("text", ts, {"text": text}, parent_tool_use_id, message_id=message_id, model=model)

    def reasoning_delta(self, text: str, *, parent_tool_use_id: str | None, message_id: str | None = None, model: str | None = None, ts: float | None = None) -> None:
        self._append("thinking", ts, {"text": text}, parent_tool_use_id, message_id=message_id, model=model)

    def tool_use_start(self, *, tool_use_id: str, name: str, input: dict,
                       parent_tool_use_id: str | None, message_id: str | None = None, model: str | None = None, ts: float | None = None) -> None:
        self._append("tool_call", ts, dict(tool_use_id=tool_use_id, name=name, input=input), parent_tool_use_id, message_id=message_id, model=model)

    def tool_use_end(self, *, tool_use_id: str, output: str | list | None, is_error: bool,
                     parent_tool_use_id: str | None, ts: float | None = None) -> None:
        self._append("tool_result", ts, dict(tool_use_id=tool_use_id, output=output, is_error=is_error), parent_tool_use_id)

    def tool_denied(self, *, tool_use_id: str | None, name: str, reason: str,
                    source: Literal["permission", "unavailable"], agent_id: str | None = None,
                    agent_type: str | None = None, parent_tool_use_id: str | None = None,
                    ts: float | None = None) -> None:
        self._append("tool_denied", ts, dict(tool_use_id=tool_use_id, name=name, reason=reason,
            source=source, agent_id=agent_id, agent_type=agent_type), parent_tool_use_id)

    def usage_delta(self, *, input_tokens: int, output_tokens: int, cached_input_tokens: int,
                    cache_creation_input_tokens: int | None, web_search_requests: int, speed: str,
                    model: str, cumulative: bool, message_id: str | None,
                    parent_tool_use_id: str | None, ts: float | None = None) -> None:
        self._append("usage", ts, dict(input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_input_tokens=cached_input_tokens, cache_creation_input_tokens=cache_creation_input_tokens,
            web_search_requests=web_search_requests, speed=speed, model=model, cumulative=cumulative,
            message_id=message_id), parent_tool_use_id)

    def turn_complete(self, *, duration_s: float | None, duration_api_s: float | None,
                      cost_usd: float | None, stop_reason: str, num_turns: int,
                      usage_by_model: dict[str, dict[str, float]] | None,
                      structured_output: dict[str, Any] | None, ts: float | None = None) -> None:
        # Any: structured output is the caller's arbitrary JSON schema.
        self._append("request_completed", ts, dict(duration_s=duration_s, duration_api_s=duration_api_s,
            cost_usd=cost_usd, cost_source="provider" if cost_usd is not None else None,
            stop_reason=stop_reason, num_turns=num_turns, usage_by_model=usage_by_model,
            structured_output=structured_output))

    def backend_log(self, *, level: str, source: str, text: str, ts: float | None = None) -> None:
        self._append("log", ts, dict(level=level, source=source, text=text))

    def rate_limit_paused(self, *, ts: float | None = None) -> None:
        self._append("rate_limit_paused", ts, {})

    def rate_limit_resumed(self, *, ts: float | None = None) -> None:
        self._append("rate_limit_resumed", ts, {})

    def _append(self, kind: ObservationKind, ts: float | None, data: dict,
                parent: str | None = None, *, message_id: str | None = None, model: str | None = None) -> None:
        self._observations.append(observation_adapter.validate_python(dict(sequence=len(self._observations), kind=kind,
            ts=self._clock() if ts is None else ts, call=self._call, request=self._request,
            parent_tool_use_id=parent, data=copy.deepcopy(data), tool_call_sequence=None, candidate_sequence=None, parent_sequence=None, message_id=message_id, model=model)))


def _content(message: Message) -> list[dict]:
    content = [({"type": "image", "mime_type": part.mime_type,
                 "data": base64.b64encode(part.data).decode("ascii")} if isinstance(part, Image)
                else {"type": "text", "text": part})
               for part in ([message] if isinstance(message, str) else message)]
    return content
