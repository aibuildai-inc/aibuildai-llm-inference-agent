"""AgentSpec: the product launch facts of one model conversation.

``Conversation`` opens its client from an AgentSpec. Every field is a fact the product decides on its own, independently of Claude: who the role is, what it may call, the final JSON Schema of its answer, where and under what confinement it runs, and which resolved extensions it was granted. The Claude implementation compiles these facts into native options; nothing here mirrors that option surface, so engine, memory, audit, and sibling code construct and read this value without an SDK type, and a denied tool, hook point, or extension is simply absent rather than present-and-disabled.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from aibuildai_conversation.confinement import Confinement
from mcp.server import Server

@dataclass(frozen=True)
class Image:
    """One image with its original bytes and declared MIME type."""

    data: bytes
    mime_type: str

    def __post_init__(self) -> None:
        # Load codecs only when the caller supplies an image.
        from PIL import Image as PillowImage

        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("image data must be non-empty bytes")
        if self.mime_type not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
            raise ValueError(f"unsupported image MIME type {self.mime_type!r}")
        try:
            with PillowImage.open(BytesIO(self.data)) as decoded:
                if PillowImage.MIME.get(decoded.format or "") != self.mime_type:
                    raise ValueError("image bytes do not match the declared MIME type")
                decoded.verify()
            with PillowImage.open(BytesIO(self.data)) as decoded:
                decoded.load()
        except (OSError, SyntaxError, PillowImage.DecompressionBombError) as exc:
            raise ValueError("invalid image data") from exc


Message = str | Sequence[str | Image]
"""One text message, or ordered text and image parts."""


@dataclass(frozen=True)
class SubAgent:
    """One resolved sub-agent a role may dispatch: the product facts the provider's agent definition needs, and nothing of the root launch (no confinement, output schema, hooks, cwd, or plugin mounts). ``model`` None inherits the dispatching role's model."""

    name: str
    description: str
    prompt: str
    tools: tuple[str, ...]
    model: str | None


@dataclass(frozen=True)
class HookSpec:
    """One tool-hook registration: the async callback and the tool-name pattern it matches. Whether it runs before or after the tool is where the launch value lists it, ``pre_tool_hooks`` or ``post_tool_hooks``."""

    callback: Callable[..., Awaitable[dict]]
    tool_name_glob: str


@dataclass(frozen=True)
class AgentSpec:
    """The launch facts of one model conversation.

    ``tools`` is the resolved grant of tool names; MCP tools appear as their ``mcp__<server>__<tool>`` names. ``output_schema`` is the final JSON-native schema of the structured result, resolved by the owner that knows the Output type, or None for a conversation that answers in text. ``mcp_servers`` are standard in-process MCP servers (``aibuildai_conversation.mcp.tool_server`` builds one), each mounted under its own name. ``sub_agents`` are the resolved sub-agent definitions the plugin owner produced. ``plugins`` maps each mounted plugin name to its absolute directory and ``skills`` lists the role-scoped ``<plugin>:<skill>`` handles the system prompt advertises, so the Skill tool resolves exactly those. ``max_output_tokens`` limits each provider reply, independently of any caller total-output limit. ``effort`` and ``max_thinking_tokens`` are the role's reasoning profile. An unset effort uses the provider default. An unset thinking budget lets the model choose thinking depth; zero disables thinking, and a positive value sets the budget. ``use_default_system_prompt`` keeps the backend's own agent base prompt in front of ``instructions``; False makes ``instructions`` the entire system prompt, for a caller that is not one of the product's coding Agents. ``env``, ``add_dirs``, and ``cwd`` are the final launch environment, read roots, and working directory whether or not the launch is confined. ``confinement`` is the unit's exact host confinement, built by the same owner that built the rest of this value; None for a conversation that needs no host confinement.
    """

    name: str
    instructions: str
    model: str
    tools: tuple[str, ...]
    cwd: str
    use_default_system_prompt: bool
    output_schema: dict[str, Any] | None = None  # Schema values include nested JSON and literals.
    mcp_servers: tuple[Server, ...] = ()
    sub_agents: tuple[SubAgent, ...] = ()
    pre_tool_hooks: tuple[HookSpec, ...] = ()
    post_tool_hooks: tuple[HookSpec, ...] = ()
    max_turns: int | None = None
    add_dirs: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    plugins: dict[str, str] = field(default_factory=dict)
    skills: tuple[str, ...] = ()
    effort: str | None = None
    max_thinking_tokens: int | None = None
    max_output_tokens: int | None = None
    confinement: Confinement | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cwd, str) or not self.cwd:
            raise ValueError("AgentSpec.cwd must name the working directory")
        if not isinstance(self.use_default_system_prompt, bool):
            raise TypeError("AgentSpec.use_default_system_prompt must be True or False")
        if self.max_output_tokens is not None and (
            type(self.max_output_tokens) is not int or self.max_output_tokens <= 0
        ):
            raise ValueError("AgentSpec.max_output_tokens must be a positive integer or None")
