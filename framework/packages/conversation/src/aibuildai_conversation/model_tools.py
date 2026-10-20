"""Model tool declarations shared by runtime and training."""
from __future__ import annotations
from collections.abc import Sequence
from copy import deepcopy
from typing import Any
from urllib.parse import quote, unquote
from jsonschema import Draft7Validator
from mcp.types import Tool

__all__ = ("STRUCTURED_OUTPUT_TOOL_NAME", "model_tool_schemas")

def _claude_schema(schema: dict[str, Any]) -> dict[str, Any]:  # Values follow the caller's recursive JSON schema.
    """Compile positional tuples without changing properties, literal data, or constraints.

    The locked CLI accepts draft-07 positional items. Pydantic tuples have no explicit dialect and use prefixItems. A declared incompatible dialect is refused, not relabelled.
    """
    Draft7Validator.check_schema(schema)
    result = deepcopy(schema)
    locations: dict[str, str] = {}
    references: list[tuple[dict[str, Any], str, str]] = []  # Values follow the caller's recursive JSON schema.

    def visit(node: dict[str, Any] | bool, before: str, after: str, declared: str | None, resource: tuple[str, str]) -> dict[str, Any] | bool:  # Values follow the caller's recursive JSON schema.
        locations[before] = after
        if isinstance(node, bool):
            return node
        if not isinstance(node, dict):
            raise ValueError(f"{before}: expected a schema object or boolean")
        node = dict(node)
        if "$id" in node:
            resource = before, after
        declared = node.get("$schema", declared)
        if declared is not None and declared not in (
            "http://json-schema.org/draft-07/schema#", "http://json-schema.org/draft-07/schema",
        ):
            raise ValueError(f"{before}/$schema: Claude does not support declared dialect {declared!r}")
        unsupported = node.keys() & {
            "dependentRequired", "dependentSchemas", "unevaluatedItems", "unevaluatedProperties",
            "minContains", "maxContains", "$dynamicRef", "$dynamicAnchor", "$recursiveRef", "$recursiveAnchor",
        }
        if unsupported:
            raise ValueError(f"{before}: unsupported Claude schema keywords {sorted(unsupported)}")
        moved: dict[str, str] = {}
        if "prefixItems" in node:
            if declared is not None or "additionalItems" in node:
                raise ValueError(f"{before}/prefixItems: conflicting tuple dialect declarations")
            members = node.pop("prefixItems")
            if not isinstance(members, list) or not members:
                raise ValueError(f"{before}/prefixItems must be a non-empty list of schemas")
            if "items" in node:
                node["additionalItems"] = node.pop("items")
                moved["additionalItems"] = "items"
            node["items"] = members
            moved["items"] = "prefixItems"
        reference = None
        if "$ref" in node:
            # Draft-07 ignores assertions beside $ref; put the reference in allOf.
            assertions = node.keys() - {"$ref", "$id", "$schema", "$defs", "definitions", "$comment", "title", "description", "default", "examples", "readOnly", "writeOnly", "deprecated"}
            reference = {"$ref": node.pop("$ref")} if declared is None and assertions else node
            references.append((reference, *resource))
        for key, value in node.items():
            old = before + "/" + moved.get(key, key)
            new = after + "/" + key
            if key in ("properties", "patternProperties", "$defs", "definitions", "dependencies"):
                node[key] = dict(value)
                for name, child in value.items():
                    if isinstance(child, (dict, bool)):
                        escaped = name.replace("~", "~0").replace("/", "~1")
                        node[key][name] = visit(child, old + "/" + escaped, new + "/" + escaped, declared, resource)
            elif key in ("allOf", "anyOf", "oneOf") or (key == "items" and isinstance(value, list)):
                node[key] = [visit(child, old + f"/{index}", new + f"/{index}", declared, resource)
                             for index, child in enumerate(value)]
            elif key in ("items", "additionalItems", "additionalProperties", "contains", "propertyNames", "not", "if", "then", "else", "contentSchema"):
                node[key] = visit(value, old, new, declared, resource)

        if reference is not None and reference is not node:
            node["allOf"] = [*node.get("allOf", ()), reference]
        return node

    result = visit(result, "", "", None, ("", ""))
    if not isinstance(result, dict):
        raise TypeError("the output schema must be an object")
    for node, before_root, after_root in references:
        reference = node["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#"):
            raise ValueError(f"Claude requires a local schema reference; got {reference!r}")
        if reference != "#" and not reference.startswith("#/"):
            continue  # Local anchors keep their identity when a schema moves.
        pointer = unquote(reference[1:], errors="strict")
        target = before_root + pointer
        if target not in locations:
            raise ValueError(f"unsupported local schema reference {reference!r}")
        relocated = locations[target][len(after_root):]
        if relocated != pointer:
            node["$ref"] = "#" + quote(relocated, safe="/~")
    Draft7Validator.check_schema(result)
    return result


STRUCTURED_OUTPUT_TOOL_NAME = "StructuredOutput"
"""The native CLI tool name used for structured responses."""


def model_tool_schemas(server_name: str, tools: Sequence[Tool], output_schema: dict | None) -> list[dict]:
    """Describe owned tools for training with the locked CLI's names and schema rules."""
    declarations = []
    if output_schema is not None:
        declarations.append({
            "name": STRUCTURED_OUTPUT_TOOL_NAME,
            "description": "Use this tool to return your final response in the requested structured format. You MUST call this tool exactly once at the end of your response to provide the structured output.",
            "parameters": _claude_schema(output_schema),
        })
    if not server_name or len({tool.name for tool in tools}) != len(tools):
        raise ValueError("tool schemas require a server name and unique tool names")
    for tool in tools:
        declarations.append({
            "name": f"mcp__{server_name}__{tool.name}",
            **({"description": tool.description} if tool.description is not None else {}),
            "parameters": deepcopy(tool.input_schema),
        })
    return [{"type": "function", "function": declaration} for declaration in declarations]
