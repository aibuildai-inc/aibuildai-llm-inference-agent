"""Mirror of the bundled claude CLI's config-dir / state-file resolution.

The CLI keys its on-disk layout on whether ``CLAUDE_CONFIG_DIR`` is set:

  - unset (default layout): the state file is the TOP-LEVEL ``~/.claude.json``; credentials (``.credentials.json``) and ``projects/`` live inside ``~/.claude/``.
  - set (explicit layout): everything, INCLUDING the state file, lives inside ``$CLAUDE_CONFIG_DIR/``.

These functions are the single place that encodes those two rules. Every consumer -- the credential's token read + display email, the SDK-session-log path fallback, the agent-launcher rw-bind, and ``aibuildai whoami`` -- derives its answer from the effective subprocess environment through them, so no consumer keeps a private copy of the rule. ``env`` is the effective environment the subprocess will run under: the operator environment overlaid with the credential's env delta.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def _home(env: Mapping[str, str]) -> Path:
    """The home directory used for ``~`` expansion, read from ``env['HOME']``.

    The claude CLI resolves ``~`` against its own ``HOME``; mirroring that means reading ``HOME`` from the effective environment rather than the process's. A missing ``HOME`` is a broken environment and fails loud."""
    home = env.get("HOME")
    if not home:
        raise AssertionError(
            "cannot resolve the claude config paths: HOME is unset in the "
            "effective environment"
        )
    return Path(home)


def config_dir(env: Mapping[str, str]) -> Path:
    """The claude config dir for ``env``: ``$CLAUDE_CONFIG_DIR`` when set, else ``~/.claude``.

    ``.credentials.json`` and ``projects/`` live inside it under BOTH layout conventions, so the OAuth token read and the SDK-session-log scan resolve through here, as does the agent launcher's rw-bind of the auth dir."""
    cd = env.get("CLAUDE_CONFIG_DIR")
    if cd:
        return Path(cd)
    return _home(env) / ".claude"


def state_file(env: Mapping[str, str]) -> Path:
    """The claude state file for ``env``: ``$CLAUDE_CONFIG_DIR/.claude.json`` in the explicit layout (var set), else the TOP-LEVEL ``~/.claude.json`` in the default layout (var unset).

    The account-email label reads ``oauthAccount.emailAddress`` from this file. Encoding the top-level default here -- instead of implicitly reading ``<config_dir>/.claude.json`` in each consumer -- is what fixes the lone-credential drift: the interactive CLI writes the top-level ``~/.claude.json`` on every login, so the label now tracks the live login."""
    cd = env.get("CLAUDE_CONFIG_DIR")
    if cd:
        return Path(cd) / ".claude.json"
    return _home(env) / ".claude.json"


def session_available(session_id: str, cwd: str, env: Mapping[str, str]) -> bool:
    """Find the native session file in this launch's config root. The CLI validates its contents."""
    # Session lookup needs the provider SDK; credential path lookups stay independent of that heavy import.
    from claude_agent_sdk._internal.sessions import (
        MAX_SANITIZED_LENGTH, _canonicalize_path, _sanitize_path, _validate_uuid,
    )

    identifier = _validate_uuid(session_id)
    if identifier is None:
        return False
    projects = config_dir(env) / "projects"
    name = _sanitize_path(_canonicalize_path(cwd))
    if (projects / name / f"{identifier}.jsonl").is_file():
        return True
    # Bun and the SDK can produce different hashes for the same long path.
    return len(name) > MAX_SANITIZED_LENGTH and any(
        path.is_file()
        for path in projects.glob(f"{name[:MAX_SANITIZED_LENGTH]}-*/{identifier}.jsonl")
    )
