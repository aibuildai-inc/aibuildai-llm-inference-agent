"""Construct standard MCP servers from declared tools and handlers."""

from collections.abc import Awaitable, Callable, Sequence

import jsonschema
from jsonschema.validators import validator_for
from mcp.server import Server, ServerRequestContext
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent, Tool

ToolHandler = Callable[[dict], Awaitable[CallToolResult]]
"""An async function from tool arguments to a standard MCP result."""


def tool_result(text: str, is_error: bool = False) -> CallToolResult:
    """One tool's whole answer to the model: what it reads, and whether that tool failed.

    This is the only place a product tool's answer becomes an MCP result, so a caller that
    produces an answer without knowing MCP -- the durable crossing, which must not -- hands its
    text and failure flag to this and stays generic."""
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=is_error)


def tool_server(name: str, tools: Sequence[tuple[Tool, ToolHandler]]) -> Server:
    """Publish and dispatch the same standard declarations.

    Return each handler's CallToolResult without changing its content. Invalid inputs and handler failures become tool errors; invalid declarations fail at assembly.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("an MCP server requires a non-empty name")
    by_name = {tool.name: (tool, handler) for tool, handler in tools}
    if len(by_name) != len(tools):
        raise ValueError(f"MCP server {name!r} declares duplicate tool names")
    for tool, handler in by_name.values():
        if not tool.name or not callable(handler):
            raise ValueError(f"invalid tool declaration in MCP server {name!r}")
        validator_for(tool.input_schema).check_schema(tool.input_schema)

    async def on_list_tools(
        ctx: ServerRequestContext, params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        return ListToolsResult(tools=[tool for tool, _handler in by_name.values()])

    async def on_call_tool(
        ctx: ServerRequestContext, params: CallToolRequestParams
    ) -> CallToolResult:
        entry = by_name.get(params.name)
        if entry is None:
            return tool_result(f"Tool '{params.name}' not found", True)
        tool, handler = entry
        arguments = dict(params.arguments or {})
        try:
            jsonschema.validate(instance=arguments, schema=tool.input_schema)
            return await handler(arguments)
        except jsonschema.ValidationError as exc:
            return tool_result(f"Input validation error: {exc.message}", True)
        except Exception as exc:  # noqa: BLE001 - the model reads tool failures
            return tool_result(f"{type(exc).__name__}: {exc}", True)

    return Server(name, on_list_tools=on_list_tools, on_call_tool=on_call_tool)



__all__ = ('ToolHandler', 'tool_server', 'tool_result')
