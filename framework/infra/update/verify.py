"""Fail-closed integrity check of a downloaded release: SHA-256 of the tarball against the release's SHA256SUMS. A mismatch never installs."""
import hashlib
from pathlib import Path


def expected_sha256(sha256sums_text: str, tarball_name: str) -> str | None:
    """The expected sha256 hex for ``tarball_name`` parsed from a SHA256SUMS body (``<sha>  <name>`` lines), or None if the file has no line for that name. The pure parse half of ``verify_sha256`` — also the content identity the runner reads from the release SHA256SUMS to compare against the installed one."""
    for line in sha256sums_text.splitlines():
        digest, _, name = line.partition("  ")
        if name == tarball_name:
            return digest
    return None


def verify_sha256(file_path: Path, sha256sums_text: str, tarball_name: str) -> bool:
    expected = expected_sha256(sha256sums_text, tarball_name)
    if expected is None:
        return False
    actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return actual == expected
