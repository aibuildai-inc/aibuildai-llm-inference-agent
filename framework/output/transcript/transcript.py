"""Read saved product events through the public conversation document API."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from aibuildai_conversation.transcript import Transcript
from aibuildai_exec.files import atomic_write_text
import json
import base64
from aibuildai_conversation import Image
from urllib.parse import quote
from engine.event.events import ActionStarted, ExecutionCreated
from engine.work_unit.events import (
    BackendLog, InvocationStarted, ReasoningDelta, SessionReplaced, SessionStarted,
    StructuredOutputRejected, TextDelta, ToolUseEnd, ToolUseStart, TurnComplete,
    UsageDelta, RateLimitPaused, RateLimitResumed,
)

if TYPE_CHECKING:
    from engine.event.base import Event
    from engine.work_unit.agent.base import Agent


class TranscriptProjection:
    """Map one recorded event prefix to documents and Action display boundaries."""

    def __init__(self, run_home: str | Path, *, write_files: bool = False) -> None:
        self._write_files = write_files
        self._workspace = Path(run_home) / "workspace"
        self._paths: set[str] = set()
        self._transcripts: dict[str, Transcript] = {}
        self._boundaries: dict[tuple[str, int], int] = {}

    def for_agent(self, agent: Agent) -> Transcript | None:
        return self._transcripts.get(agent.path)

    def query_boundary(self, path: str, ordinal: int) -> int | None:
        return self._boundaries.get((path, ordinal))

    def resume_all(self, events: Iterable[object]) -> None:
        """Rebuild derived output once after a complete recorded prefix."""
        inputs: dict[str, dict] = {}
        for event in events:
            self.apply(event, export=False)
            if isinstance(event, ExecutionCreated):
                path = event.uid if event.target is None else f"{event.target}/{event.uid}"
                inputs[path] = event.input
        for path, transcript in self._transcripts.items():
            if path in inputs:
                transcript.describe(input_payload=inputs[path])
            if self._write_files:
                self.export(path)

    def apply(self, event: object, *, export: bool = True) -> None:
        # Agent and its product runtime refer to this projection through annotations.
        from engine.durable_execution import resolve_execution_type
        from engine.work_unit.agent.base import Agent

        if isinstance(event, ExecutionCreated):
            if issubclass(resolve_execution_type(event.type_name), Agent):
                self._paths.add(event.uid if event.target is None else f"{event.target}/{event.uid}")
            return
        path = getattr(event, "target", None)
        if path not in self._paths:
            return
        event = cast("Event", event)
        transcript = self._transcripts.get(path)
        if transcript is None:
            if not isinstance(event, (ActionStarted, SessionStarted)):
                return
            transcript = Transcript(path.replace("/", "_"), started_at=event.timestamp)
            self._transcripts[path] = transcript
        if isinstance(event, ActionStarted):
            if event.method != "run" or not isinstance(event.request, (str, type(None))):
                raise AssertionError("Agent Action must contain a run message")
            self._boundaries[path, event.ordinal] = transcript.begin_call(
                event.request or "", ts=event.ts)
        elif isinstance(event, InvocationStarted):
            transcript.invocation_started(ts=event.ts, message=None if event.message is None else [
                part["text"] if part["type"] == "text" else Image(base64.b64decode(part["data"]), part["mime_type"])
                for part in event.message])
        elif isinstance(event, SessionStarted):
            transcript.session_started(session_id=event.session_id, model=event.model, cwd=event.cwd,
                tools=event.tools, mcp_server_names=event.mcp_server_names,
                launcher_path=event.launcher_path, ts=event.ts)
        elif isinstance(event, SessionReplaced):
            transcript.session_replaced(session_id=event.session_id, ts=event.ts)
        elif isinstance(event, TextDelta):
            transcript.text_delta(event.text, parent_tool_use_id=event.parent_tool_use_id, ts=event.ts, message_id=event.message_id, model=event.model)
        elif isinstance(event, ReasoningDelta):
            transcript.reasoning_delta(event.text, parent_tool_use_id=event.parent_tool_use_id, ts=event.ts, message_id=event.message_id, model=event.model)
        elif isinstance(event, ToolUseStart):
            transcript.tool_use_start(tool_use_id=event.tool_use_id, name=event.name, input=event.input,
                                     parent_tool_use_id=event.parent_tool_use_id, ts=event.ts, message_id=event.message_id, model=event.model)
        elif isinstance(event, ToolUseEnd):
            if event.denial_source is not None:
                if event.name is None or event.denial_reason is None:
                    raise ValueError("A denied tool requires its name and reason")
                transcript.tool_denied(tool_use_id=event.tool_use_id, name=event.name, reason=event.denial_reason,
                    source=event.denial_source, agent_id=event.agent_id, agent_type=event.agent_type,
                    parent_tool_use_id=event.parent_tool_use_id, ts=event.ts)
            else:
                if event.tool_use_id is None:
                    raise ValueError("A tool result requires its tool ID")
                transcript.tool_use_end(tool_use_id=event.tool_use_id, output=event.output,
                    is_error=event.is_error, parent_tool_use_id=event.parent_tool_use_id, ts=event.ts)
        elif isinstance(event, UsageDelta):
            transcript.usage_delta(input_tokens=event.input_tokens, output_tokens=event.output_tokens,
                cached_input_tokens=event.cached_input_tokens, cache_creation_input_tokens=event.cache_creation_input_tokens,
                web_search_requests=event.web_search_requests, speed=event.speed, model=event.model,
                cumulative=event.cumulative, message_id=event.message_id,
                parent_tool_use_id=event.parent_tool_use_id, ts=event.ts)
        elif isinstance(event, TurnComplete):
            transcript.turn_complete(duration_s=event.duration_s, duration_api_s=event.duration_api_s,
                cost_usd=event.cost_usd, stop_reason=event.stop_reason, num_turns=event.num_turns,
                usage_by_model=event.usage_by_model, structured_output=event.structured_output, ts=event.ts)
        elif isinstance(event, StructuredOutputRejected):
            transcript.reject_output(event.reason, ts=event.ts)
        elif isinstance(event, BackendLog):
            transcript.backend_log(level=event.level, source=event.source, text=event.text, ts=event.ts)
        elif isinstance(event, RateLimitPaused):
            transcript.rate_limit_paused(ts=event.ts)
        elif isinstance(event, RateLimitResumed):
            transcript.rate_limit_resumed(ts=event.ts)
        if export and self._write_files:
            self.export(path)

    def export(self, path: str) -> None:
        """Write the product's files from the same public trajectory used by Web."""
        # Default Markdown is optional; history and Web only consume trajectory.
        from aibuildai_conversation.transcript import render_markdown

        trajectory = self._transcripts[path].trajectory
        directory = self._workspace / path / "runs" / "run_1"
        directory.mkdir(parents=True, exist_ok=True)
        resources = self.resources(path)
        links = " | ".join(f"[{name}]({name})" for name in resources)
        atomic_write_text(directory / "execution.md", links + "\n\n" + render_markdown(trajectory))
        for name, (content, _mime) in resources.items():
            (directory / name).parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(directory / name, content)
        for identity, ordinal in self._boundaries:
            name = f"output_{ordinal}.json"
            if identity == path and name not in resources:
                (directory / name).unlink(missing_ok=True)

    def resources(self, path: str, ordinal: int | None = None) -> dict[str, tuple[str, str]]:
        """Derive product resource names and bytes from the selected observed prefix."""
        trajectory = self._transcripts[path].trajectory
        following = self._boundaries.get((path, ordinal + 1)) if ordinal is not None else None
        observations = [item for item in trajectory["observations"]
            if following is None or item["call"] is None or item["call"] < following]
        resources: dict[str, tuple[str, str]] = {}
        for key, name in (("input", "input.json"), ("system_prompt", "system_prompt.md"), ("user_message", "user_message.md")):
            if key in trajectory["description"]:
                value = trajectory["description"][key]
                resources[name] = (json.dumps(value, indent=2) if key == "input" else value,
                    "application/json" if key == "input" else "text/plain")
        calls = {call: action for (identity, action), call in self._boundaries.items() if identity == path}
        candidates: dict[int, dict] = {}
        for item in observations:
            call = item["call"]
            if call is not None and item["kind"] == "request_completed" and item["data"]["structured_output"] is not None:
                candidates[call] = item["data"]["structured_output"]
            elif call is not None and item["kind"] == "rejection":
                candidates.pop(call, None)
            if item["kind"] == "tool_call" and item["data"]["name"] in {"Task", "Agent"}:
                identifier = item["data"]["tool_use_id"]
                children = [child for child in observations if child["parent_sequence"] == item["sequence"]]
                result = [child for child in observations if child["tool_call_sequence"] == item["sequence"]]
                content = ["# " + str(item["data"]["input"].get("name") or item["data"]["input"].get("subagent_type") or identifier)]
                content.append("## Task\n\n" + json.dumps(item["data"]["input"], ensure_ascii=False, indent=2))
                for child in sorted([*children, *result], key=lambda fact: fact["sequence"]):
                    kind = child["kind"]
                    if child["kind"] == "text" or child["kind"] == "thinking":
                        content.extend(("## " + ("Assistant" if kind == "text" else "Reasoning"), child["data"]["text"]))
                    elif child["kind"] == "tool_call":
                        content.extend((f"### Tool: {child['data']['name']}", json.dumps(child["data"]["input"], ensure_ascii=False, indent=2)))
                    elif child["kind"] == "tool_result":
                        value = child["data"]["output"]
                        content.extend(("### Tool error" if child["data"]["is_error"] else "### Tool result", value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)))
                    elif child["kind"] == "tool_denied":
                        content.extend((f"### Tool refused: {child['data']['name']}", child["data"]["reason"]))
                    else:
                        content.extend((f"### {kind} · {child['ts']:g}s", json.dumps(child["data"], ensure_ascii=False, indent=2)))
                resources[f"sub-agents/{quote(identifier, safe='')}.md"] = ("\n\n".join(content), "text/plain")

        for call, candidate in candidates.items():
            if call in calls:
                resources[f"output_{calls[call]}.json"] = (json.dumps(candidate, indent=2), "application/json")
        return resources
