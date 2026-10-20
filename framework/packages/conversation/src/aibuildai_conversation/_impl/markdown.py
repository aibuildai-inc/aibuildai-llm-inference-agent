"""Default human-readable conversation record from public trajectory facts."""
from __future__ import annotations

import json
from typing import cast
from .trajectory import Trajectory


def render_markdown(trajectory: Trajectory) -> str:
    """Show speakers, tool actions, results and incomplete requests in arrival order."""
    lines = [f"# {trajectory['name']}", f"Started: {trajectory['started_at']}"]
    for key, title in (("system_prompt", "Instructions"), ("user_message", "Caller description"), ("input", "Input data")):
        if key in trajectory["description"]:
            value = trajectory["description"][key]
            lines.extend((f"## {title}", value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)))
    for item in trajectory["observations"]:
        kind = item["kind"]
        when = f"{item['ts']:g}s"
        speaker = "Assistant" if item["parent_tool_use_id"] is None else f"Sub-agent ({item['parent_tool_use_id']})"
        if item["kind"] == "input" or item["kind"] == "request_input":
            lines.append(f"## {'Caller request' if kind == 'input' else 'Request input'} · {when}")
            for part in item["data"]["content"]:
                lines.append(part["text"] if part["type"] == "text" else f"![image](data:{part['mime_type']};base64,{part['data']})")
        elif item["kind"] == "text" or item["kind"] == "thinking":
            lines.extend((f"## {speaker}{' reasoning' if kind == 'thinking' else ''} · {when}", item["data"]["text"]))
        elif item["kind"] == "tool_call":
            lines.extend((f"### {speaker} calls {item['data']['name']} · {when}",
                f"Tool ID: `{item['data']['tool_use_id']}`", "```json", json.dumps(item["data"]["input"], ensure_ascii=False, indent=2), "```"))
        elif item["kind"] == "tool_result":
            lines.append(f"### {'Tool error' if item['data']['is_error'] else 'Tool result'} · {when}")
            lines.append(f"Tool ID: `{item['data']['tool_use_id']}`")
            value = item["data"]["output"]
            if isinstance(value, str):
                lines.append(value)
            elif value is None:
                lines.append("No result content was reported.")
            else:
                for part in value:
                    if isinstance(part, dict) and part.get("type") == "text":
                        lines.append(cast(str, part["text"]))
                    elif isinstance(part, dict) and part.get("type") == "image":
                        mime = part.get("mimeType", part.get("mime_type"))
                        lines.append(f"![tool image](data:{mime};base64,{part['data']})")
                    else:
                        lines.extend(("```json", json.dumps(part, ensure_ascii=False, indent=2), "```"))
        elif item["kind"] == "tool_denied":
            lines.extend((f"### Tool refused: {item['data']['name']} · {when}", item["data"]["reason"]))
        elif item["kind"] == "rejection":
            lines.extend((f"### Caller rejected the answer · {when}", item["data"]["reason"]))
        elif item["kind"] == "request_started":
            lines.append(f"Conversation request started · {when}")
        elif item["kind"] == "request_completed":
            cost = "unknown" if item["data"]["cost_usd"] is None else f"${item['data']['cost_usd']:.6f} (provider reported)"
            lines.append(f"Request ended · {when} · {item['data']['stop_reason']} · Cost: {cost}")
            if item["data"]["structured_output"] is not None:
                lines.extend(("### Candidate answer", "```json", json.dumps(item["data"]["structured_output"], ensure_ascii=False, indent=2), "```"))
        elif item["kind"] == "usage":
            unit = "Request total" if item["data"]["cumulative"] else "Message snapshot"
            cache = "unknown" if item["data"]["cache_creation_input_tokens"] is None else str(item["data"]["cache_creation_input_tokens"])
            lines.append(f"{unit} · {when} · Model: {item['data']['model'] or 'unknown'} · Input: {item['data']['input_tokens']} · Output: {item['data']['output_tokens']} · Cache read: {item['data']['cache_read_input_tokens']} · Cache creation: {cache} · Web searches: {item['data']['web_search_requests']} · Speed: {item['data']['speed'] or 'unknown'}")
        elif item["kind"] == "session_started":
            lines.append(f"Session {item['data']['session_id']} opened · {when} · Model: {item['data']['model']}")
        elif item["kind"] == "session_replaced":
            lines.append(f"Session replaced by {item['data']['session_id']} · {when}")
        elif kind in {"rate_limit_paused", "rate_limit_resumed"}:
            lines.append(f"Provider {'paused' if kind == 'rate_limit_paused' else 'resumed'} · {when}")
        elif item["kind"] == "log":
            lines.append(f"{item['data']['level']} · {item['data']['source']} · {when}: {item['data']['text']}")
        else:
            raise ValueError(f"Unknown trajectory fact: {kind}")
    if any(request["completion"] is None for request in trajectory["requests"]):
        lines.append("An observed request has no reported completion. Its result and final cost remain unknown.")
    return "\n\n".join(lines) + "\n"
