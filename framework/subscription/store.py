"""Local persistence for the user's Clerk session.

State lives at ~/.aibuildai/<name>.json, owned 0600 so other local users cannot read the session token.

Schema note: prior versions cached subscription check results (last_license_check, last_plan, etc.) here because the check had to call the Clerk Backend API and needed an offline grace window. The current session code verifies an id_token JWT locally via JWKS, so the JWT itself is the cache and those fields are gone. Every saved login carries the exact product version that wrote it, and only that exact build reads it back.
"""
from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from aibuildai_version import APP_VERSION
from infra.fs.pathing import (
    _assert_posix_path_semantics,
    atomic_write_owner_only_text,
)

_CONFIG_DIR = Path.home() / ".aibuildai"
_AUTH_FILE = _CONFIG_DIR / "auth.json"
_VERSION_KEY = "app_version"
# The Worker-signed run-grant, cached as the raw compact-JWS string. It is
# self-contained (signature + exp + claims), so it persists as a bare string
# with no dataclass / schema -- the file holds exactly the grant the Worker
# returned.
_GRANT_FILE = _CONFIG_DIR / "run_grant.jwt"
_LOCK_FILE = _CONFIG_DIR / "auth.lock"


@dataclass
class AuthState:
    user_id: str          # Clerk user_xxx, derived from id_token.sub
    id_token: str         # OIDC id_token (RS256 JWT). Carries public_metadata.
    refresh_token: str    # OAuth refresh_token; used to mint a new id_token
                          # when the cached one expires.


def load() -> Optional[AuthState]:
    if not _AUTH_FILE.exists():
        return None
    try:
        raw = json.loads(_AUTH_FILE.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get(_VERSION_KEY) != APP_VERSION:
        return None
    allowed = {f.name for f in AuthState.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    # A stored file that cannot fill every field with a USABLE value is not a
    # session this build can run -- an older release wrote a different schema, a
    # newer one wrote a field this build does not know, or a field is present and
    # empty. The test is non-emptiness on purpose, not key presence: every field
    # of AuthState is required to be non-empty for the session to work at all (an
    # empty refresh_token makes session._refresh_id_token raise before any network
    # call), so a present-but-empty field is exactly as unusable as a missing one
    # and reading it as a session would only move the same dead end one layer
    # deeper. Read it as "not signed in" so the next aibuildai command prompts a
    # fresh login that writes this schema.
    if not all(filtered.get(name) for name in allowed):
        return None
    return AuthState(**filtered)


def save(state: AuthState) -> None:
    payload = asdict(state)
    payload[_VERSION_KEY] = APP_VERSION
    atomic_write_owner_only_text(
        _AUTH_FILE,
        json.dumps(payload, indent=2),
        purpose="aibuildai auth state",
    )


def clear() -> None:
    if _AUTH_FILE.exists():
        _AUTH_FILE.unlink()
    clear_grant()


@contextmanager
def auth_refresh_lock() -> Iterator[None]:
    """Cross-process exclusive lock serializing every Clerk refresh_token rotation.

    Clerk's OAuth refresh_token is single-use and ROTATES on every refresh, and Clerk reuse-detection REVOKES the whole session lineage if an already-rotated token is presented again. Two processes sharing ~/.aibuildai that both read the same refresh_token and both refresh would make the second present a consumed token -> the login is destroyed. This blocking fcntl.flock(LOCK_EX) serializes the read-modify-write so each refresh sees the token the previous one rotated and saved.

    BLOCKING on purpose (no LOCK_NB): a second caller must WAIT for the first to finish, not fail. The kernel releases the flock when the holder process exits, so a crashed holder never wedges a waiter. A fresh fd (open file description) is taken per call so two callers -- threads or processes -- get real flock mutual exclusion. Callers MUST (re-)read the refresh_token from disk INSIDE this lock.
    """
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _assert_posix_path_semantics(_LOCK_FILE.parent, purpose="aibuildai auth state")
    fd = os.open(_LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def save_grant(grant: str) -> None:
    """Cache the Worker-signed run-grant (raw compact-JWS string), 0600."""
    atomic_write_owner_only_text(
        _GRANT_FILE, grant, purpose="aibuildai auth state"
    )


def load_grant() -> Optional[str]:
    """Return the cached run-grant string, or None when none is cached."""
    if not _GRANT_FILE.exists():
        return None
    return _GRANT_FILE.read_text()


def clear_grant() -> None:
    if _GRANT_FILE.exists():
        _GRANT_FILE.unlink()
