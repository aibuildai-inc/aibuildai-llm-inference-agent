"""Locate the runtime supplied by the declared Claude SDK dependency."""

import hashlib
import json
import os
from functools import cache
from pathlib import Path

import claude_agent_sdk


def claude_path() -> Path:
    """Return the SDK-owned executable; never search PATH or another cache."""
    path = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"the installed Claude SDK executable is unavailable: {path}")
    return path


@cache
def proxy_path() -> Path:
    """Return the proxy built into this installation and verify its declared bytes."""
    directory = Path(__file__).parent / "_native"
    identity = json.loads((directory / "identity.json").read_text())
    path = directory / "cli-proxy-api"
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"the installed proxy executable is unavailable: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != identity["sha256"]:
        raise RuntimeError(f"the installed proxy differs from its build identity: {identity['commit']}")
    return path


__all__ = ('claude_path', 'proxy_path')
