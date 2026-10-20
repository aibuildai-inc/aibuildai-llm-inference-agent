"""Client contract for the remote kb service."""

from __future__ import annotations

import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

AIBUILDAI_KB_BASE_URL = "AIBUILDAI_KB_BASE_URL"
_DERIVED_PATHS = frozenset({"/mcp", "/contribute", "/healthz"})
REQUEST_PAYLOAD_QUERY_PARAM = "aibuildai_request_payload"


class KbServiceConfigError(ValueError):
    """The configured kb service endpoint cannot satisfy the client contract."""


class KbCaller(BaseModel):
    """Agent identity fields recorded with a kb request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    role: str = Field(min_length=1)
    uid: str = Field(min_length=1)


class KbRequestPayload(BaseModel):
    """Run and optional caller attribution sent with a kb request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: str = Field(min_length=1)
    task_name: str = Field(min_length=1)
    caller: KbCaller | None

def attach_request_payload_url(
    url: str,
    payload: KbRequestPayload,
) -> str:
    """Return *url* with the kb request payload attached as one query item."""
    parts = urlsplit(url)
    query_items = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True)
    if any(name == REQUEST_PAYLOAD_QUERY_PARAM for name, _ in query_items):
        raise AssertionError(
            f"{REQUEST_PAYLOAD_QUERY_PARAM} query parameter is already present"
        )
    query_items.append(
        (
            REQUEST_PAYLOAD_QUERY_PARAM,
            payload.model_dump_json(ensure_ascii=True),
        )
    )
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query_items),
            parts.fragment,
        )
    )


def kb_base_url() -> str:
    """Return the configured kb service base URL."""
    return _normalize_base_url(os.environ.get(AIBUILDAI_KB_BASE_URL, "https://32.194.230.84"))


def kb_mcp_url() -> str:
    """Return the kb Streamable-HTTP MCP endpoint URL."""
    return _append_path(kb_base_url(), "/mcp")


def kb_contribute_url() -> str:
    """Return the kb run-upload endpoint URL."""
    return _append_path(kb_base_url(), "/contribute")


def _normalize_base_url(raw: str) -> str:
    try:
        parts = urlsplit(raw)
    except ValueError as exc:
        raise KbServiceConfigError(
            f"{AIBUILDAI_KB_BASE_URL} must be a valid service base URL, got {raw!r}"
        ) from exc
    if not parts.scheme or not parts.netloc:
        raise KbServiceConfigError(
            f"{AIBUILDAI_KB_BASE_URL} must be an absolute service base URL, got {raw!r}"
        )
    if parts.query or parts.fragment:
        raise KbServiceConfigError(
            f"{AIBUILDAI_KB_BASE_URL} must not include query or fragment, got {raw!r}"
        )
    path = parts.path.rstrip("/")
    if any(path == derived or path.endswith(derived) for derived in _DERIVED_PATHS):
        raise KbServiceConfigError(
            f"{AIBUILDAI_KB_BASE_URL} must be the service base URL, not an endpoint: "
            f"{raw!r}"
        )
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _append_path(base_url: str, path: str) -> str:
    parts = urlsplit(base_url)
    base_path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, f"{base_path}{path}", "", ""))
