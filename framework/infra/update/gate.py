"""Decides whether the current launch should run an auto-update check.

Kept dependency-light (os + version only) so cli can import it on every startup without pulling requests/cryptography; the heavy runner is imported only after this gate passes (deferred-heavy-import policy)."""
import os

from aibuildai_version import is_frozen


def should_check(command: str) -> bool:
    """True only when ALL hold: frozen binary; auto-update not disabled; a heavy command (run/memorize); the outermost process (not the systemd-service re-exec child); and not the post-update re-exec child (loop prevention)."""
    if not is_frozen():
        return False
    if os.environ.get("AIBUILDAI_AUTO_UPDATE") == "0":
        return False
    if command not in ("run", "memorize"):
        return False
    if os.environ.get("AIBUILDAI_IN_SERVICE") == "1":
        return False
    if os.environ.get("AIBUILDAI_UPDATED_THIS_LAUNCH"):
        return False
    return True
