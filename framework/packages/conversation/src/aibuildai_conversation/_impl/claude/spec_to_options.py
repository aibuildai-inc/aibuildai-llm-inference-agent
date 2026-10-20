"""AgentSpec launch facts -> ClaudeAgentOptions."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from fnmatch import fnmatchcase
from typing import Any

from mcp.client import Client

from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    HookMatcher,
    McpSdkServerConfig,
)

from aibuildai_conversation._impl.spec import AgentSpec, HookSpec, Confinement


from aibuildai_conversation.model_tools import _claude_schema


def make_command_cgroup_reaper_spec(confinement: Confinement, pending: set[str]) -> HookSpec:
    """Ask the physical owner to remove the completed command scope."""

    async def command_cgroup_reaper(
        input_data: dict, tool_use_id: "str | None", context: object
    ) -> dict:
        if tool_use_id in pending:
            confinement.reap_command(tool_use_id)
            pending.discard(tool_use_id)
        return {}

    return HookSpec(callback=command_cgroup_reaper, tool_name_glob="Bash")


def make_command_self_enroll_spec(confinement: Confinement, pending: set[str]) -> HookSpec:
    """Wrap a Bash command through its declared physical owner before execution. A native tool-use ID is required."""

    async def command_self_enroll(
        input_data: dict, tool_use_id: "str | None", context: object
    ) -> dict:
        command = input_data["tool_input"].get("command", "")
        if not tool_use_id:
            raise AssertionError(
                f"PreToolUse Bash hook requires a tool_use_id (SDK contract); got {tool_use_id!r}"
            )
        wrapped = confinement.wrap_command(command, tool_use_id)
        pending.add(tool_use_id)
        return {
            "hookSpecificOutput": {
                "hookEventName": input_data["hook_event_name"],
                "updatedInput": {
                    **input_data["tool_input"],
                    "command": wrapped,
                },
            }
        }


    return HookSpec(callback=command_self_enroll, tool_name_glob="Bash")


def _hook_matchers(hooks: Sequence[HookSpec]) -> list[HookMatcher]:
    """One HookMatcher per tool-name pattern, in first-seen order, carrying every callback registered for it."""
    grouped: dict[str, list] = {}
    for hook in hooks:
        grouped.setdefault(hook.tool_name_glob, []).append(hook.callback)
    return [HookMatcher(matcher=glob, hooks=callbacks) for glob, callbacks in grouped.items()]


async def spec_to_options(
    spec: AgentSpec,
    *,
    environment: Mapping[str, str],
    denials: deque[dict],
    observed_hooks: set[str],
    dispatch_inputs: dict[str, asyncio.Future[dict]],
    pending_commands: set[str],
    resume_session_id: str | None,
    stderr_callback: Callable[[str], None],
) -> ClaudeAgentOptions:
    """Build ClaudeAgentOptions from the launch facts plus this session's extras.

    ``environment`` is the snapshot resolved by this Conversation. The compiler adds the required native controls. Conversation attaches its generated launcher path before it creates the SDK client. ``setting_sources=[]`` excludes ambient settings; ``resume`` and ``stderr`` belong to this session.

    ``spec.plugins`` mounts as ``ClaudeAgentOptions(plugins=...)`` and, when non-empty, ``skills=`` lists the role-scoped ``<plugin>:<skill>`` handles the system prompt's ``## Skills`` section advertises (both come from the same role filter, so they never drift). The SDK ``skills`` list is a context filter: only the listed handles resolve, and the Skill tool rejects any other; without both fields it returns ``Unknown skill`` for every handle the prompt advertises, because ``setting_sources=[]`` suppresses the CLI's own discovery.
    """
    server_names = [server.name for server in spec.mcp_servers]
    if len(set(server_names)) != len(server_names):
        raise ValueError("duplicate MCP server names")
    children = {child.name: child for child in spec.sub_agents}
    if len(children) != len(spec.sub_agents):
        raise ValueError("duplicate sub-agent names")
    published: set[str] = set()
    for server in spec.mcp_servers:
        async with Client(server, cache=None) as client:
            cursor = None
            while True:
                page = await client.list_tools(cursor=cursor)
                for tool in page.tools:
                    name = f"mcp__{server.name}__{tool.name}"
                    if name in published:
                        raise ValueError(f"duplicate MCP tool {name!r}")
                    published.add(name)
                cursor = page.next_cursor
                if cursor is None:
                    break
    grants: dict[str | None, set[str]] = {}
    for actor, declared in ((None, spec.tools), *((child.name, child.tools) for child in spec.sub_agents)):
        allowed: set[str] = set()
        for tool in declared:
            if tool.startswith("mcp__"):
                matched = {name for name in published if fnmatchcase(name, tool)}
                if not matched:
                    raise ValueError(f"{actor or spec.name}: MCP grant {tool!r} matches no mounted tool")
                allowed.update(matched)
            else:
                allowed.add(tool)
        grants[actor] = allowed
    if spec.output_schema is not None:
        grants[None].add("StructuredOutput")
    for actor, allowed in grants.items():
        if actor is not None and not allowed.issubset(grants[None]):
            raise ValueError(f"{actor}: child tools must be granted to the parent")

    async def enforce_grant(input_data: dict, tool_use_id: str | None, context: object) -> dict:
        name = input_data.get("tool_name") if isinstance(input_data, dict) else None
        reason = ""
        if isinstance(tool_use_id, str):
            observed_hooks.add(tool_use_id)
        if not isinstance(name, str) or not name or input_data.get("hook_event_name") != "PreToolUse" or not isinstance(input_data.get("tool_input"), dict) or not isinstance(tool_use_id, str) or not tool_use_id:
            name = name if isinstance(name, str) else "<malformed>"
            reason = "malformed tool authority input"
        else:
            agent_id = input_data.get("agent_id")
            actor = input_data.get("agent_type")
            if not isinstance(actor, (str, type(None))) or actor == "" or (agent_id is None) != (actor is None) or (agent_id is not None and (not isinstance(agent_id, str) or not agent_id)):
                reason = "malformed tool actor"
            elif name not in grants.get(actor, set()):
                reason = f"{name} is not granted to {actor or spec.name}"
            elif name in ("Agent", "Task"):
                # Hook input can already have lost run_in_background. The SDK's
                # AssistantMessage retains the original declaration.
                pending = dispatch_inputs.setdefault(tool_use_id, asyncio.get_running_loop().create_future())
                try:
                    # The SDK matcher allows 60 s; reserve half for a denied response.
                    requested = await asyncio.wait_for(asyncio.shield(pending), timeout=30)
                except TimeoutError:
                    reason = "original sub-agent dispatch input was not received"
                else:
                    child_name = requested.get("subagent_type")
                    child = children.get(child_name) if isinstance(child_name, str) else None
                    if child is None:
                        reason = "the requested sub-agent is not declared"
                    elif requested.get("run_in_background", False) is not False:
                        reason = "only foreground sub-agent execution is supported"
                    elif requested.get("isolation") is not None:
                        reason = "sub-agent worktree isolation is not supported"
                    elif requested.get("model") not in (None, child.model or "inherit", spec.model if child.model in (None, "inherit") else child.model):
                        reason = "the requested model conflicts with the sub-agent declaration"
                finally:
                    dispatch_inputs.pop(tool_use_id, None)
        if not reason:
            return {}
        scope = input_data if isinstance(input_data, dict) else {}
        denials.append(dict(tool_use_id=tool_use_id, name=name, reason=reason, source="permission",
            agent_id=scope.get("agent_id") if isinstance(scope.get("agent_id"), str) else None,
            agent_type=scope.get("agent_type") if isinstance(scope.get("agent_type"), str) else None))
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason}}

    # ``tools`` is the availability knob (#302): left at the SDK default every
    # role would expose the full CLI built-in set. MCP availability is governed
    # by mcp_servers, so mcp__* stays out of the built-in list.
    builtin_tools = [t for t in spec.tools if not t.startswith("mcp__")]
    env = dict(environment)
    if spec.max_output_tokens is not None:
        ceiling = str(spec.max_output_tokens)
        if env.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS", ceiling) != ceiling:
            raise ValueError("max_output_tokens conflicts with CLAUDE_CODE_MAX_OUTPUT_TOKENS")
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = ceiling
    # The CLI would hand a Kb or Kaggle call that runs past 120 s back to the
    # model as a background receipt; its backgroundTasksDisabled read of this
    # variable short-circuits that deadline (bundled CLI 2.1.251 source), so
    # the call returns its result. Sub-agents stay in the foreground through
    # AgentDefinition(background=False), and the process entry scrubs every
    # inherited CLAUDE* feature flag (aibuildai_conversation.environment).
    env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
    # The CLI otherwise replaces an explicit budget with adaptive thinking.
    env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = (
        "1" if spec.max_thinking_tokens is not None and spec.max_thinking_tokens > 0 else "0"
    )

    kwargs: dict[str, Any] = dict(  # Native option values have different SDK-defined types.
        system_prompt={"type": "preset", "preset": "claude_code"} if spec.use_default_system_prompt else None,
        tools=builtin_tools,
        allowed_tools=sorted(grants[None]),
        model=spec.model,
        extra_args={"name": spec.name},
        add_dirs=list(spec.add_dirs),
        cwd=spec.cwd,
        env=env,
        permission_mode="bypassPermissions",
        setting_sources=[],
        # Only the servers this options object names exist for the CLI; it
        # reads no MCP configuration from the filesystem.
        strict_mcp_config=True,
        # Foreground sub-agents stream their text and thinking to the parent
        # session under their spawning Agent tool's id (parent_tool_use_id),
        # so the nested transcript holds what the sub-agent said, not only
        # what it called.
        forward_subagent_text=True,
        resume=resume_session_id,
        stderr=stderr_callback,
    )
    # Reasoning effort: set it ONLY when explicitly configured (spec.effort not
    # None). When unset we send no effort, so the binary's own default applies
    # and a run that does not set llm.default.effort is unchanged (Claude included).
    if spec.effort is not None:
        kwargs["effort"] = spec.effort
    # None lets the model choose thinking depth; effort guides that choice.
    # Request summary text so the transcript can record more than a signature.
    if spec.max_thinking_tokens == 0:
        kwargs["thinking"] = {"type": "disabled"}
    elif spec.max_thinking_tokens is None:
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
    else:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": spec.max_thinking_tokens,
            "display": "summarized",
        }
    if spec.plugins:
        # Startup validated each mounted plugin's manifest; here a resolved
        # directory is simply a native local plugin mount.
        kwargs["plugins"] = [{"type": "local", "path": path} for path in spec.plugins.values()]
        # Role-scoped skill handles, the set the system prompt's ## Skills
        # section lists; the SDK list is a context filter, so every other
        # <plugin>:<skill> stays hidden.
        kwargs["skills"] = list(spec.skills)
    # The role's final JSON Schema becomes the CLI's structured output format.
    # The terminal ResultMessage.structured_output is the one candidate the
    # conversation reports with the turn completion (#3047).
    if spec.output_schema is not None:
        kwargs["output_format"] = {
            "type": "json_schema",
            "schema": _claude_schema(spec.output_schema),
        }
    pre = (HookSpec(callback=enforce_grant, tool_name_glob=".*"), *spec.pre_tool_hooks)
    post = spec.post_tool_hooks
    conf = spec.confinement
    reaper = None
    if conf is not None and conf.delegates_commands():
        pre = (*pre, make_command_self_enroll_spec(conf, pending_commands))
        reaper = make_command_cgroup_reaper_spec(conf, pending_commands)
        post = (*post, reaper)
    hooks = {
        key: _hook_matchers(registered)
        for key, registered in (("PreToolUse", pre), ("PostToolUse", post))
        if registered
    }
    if reaper is not None:
        hooks["PostToolUseFailure"] = _hook_matchers((reaper,))
    kwargs["hooks"] = hooks
    if spec.mcp_servers:
        # Each standard server is served to the CLI over the SDK's native
        # in-memory bridge. The CLI controls which result content reaches the model.
        kwargs["mcp_servers"] = {
            server.name: McpSdkServerConfig(type="sdk", name=server.name, instance=server)
            for server in spec.mcp_servers
        }
    if spec.sub_agents:
        # The child reaches the parent's live MCP mounts through ``tools``: the
        # SDK AgentDefinition has no allowed_tools, so tools IS the availability
        # allowlist. Server-qualified MCP patterns select the granted mounts.
        # It carries no ``mcpServers``: under strict_mcp_config the CLI 2.1.251
        # discards every agent-scoped MCP entry and emits an "agent MCP server
        # blocked" notification per spawn, while the parent's mounts still
        # serve the child (paid probe, PR #2765). background=False is the
        # public statement that the sub-agent runs in the foreground and
        # returns its findings to the parent.
        kwargs["agents"] = {
            sub.name: AgentDefinition(
                description=sub.description,
                prompt=sub.prompt,
                tools=sorted(grants[sub.name]),
                model=sub.model,
                background=False,
            )
            for sub in spec.sub_agents
        }
    if spec.max_turns is not None:
        kwargs["max_turns"] = spec.max_turns
    # Partial-message streaming carries each message's SETTLED usage on the raw
    # message_delta event -- the only per-turn settlement channel that exists on
    # BOTH the Anthropic route and the owned OpenAI adapter route (#1327; the
    # ResultMessage usage.iterations breakdown is EMPTY on the adapter route).
    # The event adapter's RequestStream reads these StreamEvents.
    kwargs["include_partial_messages"] = True
    # The SDK reads the CLI stream one JSON message at a time and refuses any
    # message longer than max_buffer_size (default 1 MiB). One Read of a 191 KB
    # PNG produced a longer message and ended a run as CLIJSONDecodeError
    # (#3127). 64 MiB holds an image tool result and the largest text result
    # the tools can return.
    kwargs["max_buffer_size"] = 64 * 1024 * 1024
    return ClaudeAgentOptions(**kwargs)
