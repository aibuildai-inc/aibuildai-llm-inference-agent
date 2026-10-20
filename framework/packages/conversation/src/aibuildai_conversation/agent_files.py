"""Read explicit agent file declarations."""
import yaml

__all__ = ("parse_frontmatter", "text_frontmatter_field", "parse_agent_file")

def parse_frontmatter(content: str) -> tuple[dict, str] | None:
    """Parse one Markdown file's YAML frontmatter and body."""
    if not content.startswith("---\n"):
        return None
    rest = content[len("---\n") :]
    end = rest.find("\n---\n")
    if end == -1:
        if not rest.endswith("\n---"):
            return None
        end = len(rest) - len("\n---")
        body_start = len(rest)
    else:
        body_start = end + len("\n---\n")
    try:
        meta = yaml.safe_load(rest[:end])
    except yaml.YAMLError:
        return None
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        return None
    return meta, rest[body_start:].lstrip("\n")


def text_frontmatter_field(meta: dict, field: str, source: str) -> str:
    """Read a trimmed text field; reject non-text values."""
    if field not in meta:
        return ""
    value = meta[field]
    if not isinstance(value, str):
        raise ValueError(f"{source}: {field} must be a string")
    return value.strip()


def _list_frontmatter_field(meta: dict, field: str, source: str) -> list[str]:
    if field not in meta:
        return []
    value = meta[field]
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        items = value
    else:
        raise ValueError(f"{source}: {field} must be a string or list of strings")
    if any(not item.strip() for item in items):
        raise ValueError(f"{source}: {field} entries must not be blank")
    return [item.strip() for item in items]


def parse_agent_file(content: str, default_name: str) -> dict | None:
    """Parse one agent markdown file's YAML frontmatter and body.

    Frontmatter delimiter is the literal ``---`` line. The first ``---`` line opens the block; the second closes it; everything after is body.

    Returns ``None`` if the file lacks YAML frontmatter or it cannot be parsed. Otherwise returns a dict with keys:
        - ``name`` (defaults to ``default_name`` if frontmatter omits it)
        - ``description`` (string, empty if absent)
        - ``model`` (string or None)
        - ``disallowed_tools`` (list of tool names, empty if absent)
        - ``mcp_servers`` (list of MCP server names, empty if absent)
        - ``prompt`` (the markdown body, leading newlines stripped)

    The ``disallowedTools`` frontmatter field accepts either a comma-separated string (``Write, Edit``) or a YAML list of strings.

    The ``mcpServers`` frontmatter field (camelCase, following the SDK convention) is converted to the snake_case ``mcp_servers`` key. It accepts a YAML inline list (``[arxiv, scholar]``), a YAML block list, or a comma-separated string. Other values fail at startup. Full inline server definitions are not supported here.
    """
    parsed = parse_frontmatter(content)
    if parsed is None:
        return None
    meta, body = parsed
    source = f"agent {default_name!r}"
    supported = {"name", "description", "model", "disallowedTools", "mcpServers"}
    unknown = sorted(
        repr(field)
        for field in meta
        if not isinstance(field, str) or field not in supported
    )
    if unknown:
        raise ValueError(f"{source}: unknown frontmatter field(s): {unknown}")
    name = text_frontmatter_field(meta, "name", source)
    if "name" in meta and not name:
        raise ValueError(f"{source}: name must not be blank")

    return {
        "name": name or default_name,
        "description": text_frontmatter_field(meta, "description", source),
        "model": text_frontmatter_field(meta, "model", source) or None,
        "disallowed_tools": _list_frontmatter_field(meta, "disallowedTools", source),
        "mcp_servers": _list_frontmatter_field(meta, "mcpServers", source),
        "prompt": body,
    }
