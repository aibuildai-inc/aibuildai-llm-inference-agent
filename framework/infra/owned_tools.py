"""Prepare product tools and delegate model runtime assets to Conversation.

Product binaries use the existing identity cache and release staging tree. The installed Conversation package owns the Claude SDK and proxy executables; product setup and lookup use that same owner without making another copy.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Collection
from pathlib import Path

from aibuildai_conversation.assets import claude_path, proxy_path

from aibuildai_version import is_frozen

# Ruff: the pinned Linux x86_64 executable the Meta verification boundary runs.
RUFF_VERSION = "0.15.16"
_RUFF_TARBALL_URL = (
    "https://github.com/astral-sh/ruff/releases/download/"
    f"{RUFF_VERSION}/ruff-x86_64-unknown-linux-musl.tar.gz"
)
_RUFF_TARBALL_SHA256 = "f52d90f8a6b1b3ad7d74301c3c796652e851d8f05b6ba26d139f05f4838cf412"
_RUFF_MEMBER = "ruff-x86_64-unknown-linux-musl/ruff"

# Tectonic: the pinned static musl build (zero shared-library dependencies).
TECTONIC_VERSION = "0.15.0"
_TECTONIC_TARBALL_URL = (
    "https://github.com/tectonic-typesetting/tectonic/releases/download/"
    f"tectonic%40{TECTONIC_VERSION}/"
    f"tectonic-{TECTONIC_VERSION}-x86_64-unknown-linux-musl.tar.gz"
)
_TECTONIC_TARBALL_SHA256 = "dfb82876f2986862996e564fa507a9e576e0c1e3bee63c2c1bd677c2543e6407"

# micromamba: the pinned environment tool that creates the run's Program
# environment. A single static executable (max glibc requirement 2.17, below
# the product's own build floor), so the product never requires a host conda.
MICROMAMBA_VERSION = "2.9.0"
_MICROMAMBA_BINARY_URL = (
    "https://github.com/mamba-org/micromamba-releases/releases/download/"
    f"{MICROMAMBA_VERSION}-0/micromamba-linux-64"
)
_MICROMAMBA_BINARY_SHA256 = (
    "366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3"
)

# Caddy: the pinned Workspace gateway.
CADDY_VERSION = "2.11.4"
_CADDY_TARBALL_URL = (
    "https://github.com/caddyserver/caddy/releases/download/"
    f"v{CADDY_VERSION}/caddy_{CADDY_VERSION}_linux_amd64.tar.gz"
)
_CADDY_TARBALL_SHA256 = "527fbf917c39189a1e3b31d34fa955601680b2d5c8055d2a87b8b9588dec7bb9"

# PostgreSQL: the complete private server and client installation tree.
POSTGRESQL_VERSION = "18.6"
POSTGRESQL_MAJOR = 18
_POSTGRESQL_TARBALL_URL = (
    "https://ftp.postgresql.org/pub/source/"
    f"v{POSTGRESQL_VERSION}/postgresql-{POSTGRESQL_VERSION}.tar.bz2"
)
_POSTGRESQL_TARBALL_SHA256 = (
    "555610c24d53e4316da5b7d3fc25c279d96856d5e0e23ee308c328c5fa881d9f"
)
_POSTGRESQL_REQUIRED = (
    "bin/pg_ctl",
    "bin/postgres",
    "bin/initdb",
    "bin/psql",
    "lib/libpq.so",
    "lib/uuid-ossp.so",
    "share/postgres.bki",
    "share/extension/uuid-ossp.control",
    "share/doc/postgresql/COPYRIGHT",
)

_PROBE_TIMEOUT_S = 30.0

# The shipped tools/ layout of an installed build; also the release staging
# layout (stage_tools). Source mode uses the per-identity cache instead.
_FROZEN_PATHS = {
    "micromamba": "bin/micromamba",
    "ruff": "bin/ruff",
    "tectonic": "bin/tectonic",
    "caddy": "bin/caddy",
}


def _cache_root() -> Path:
    return Path.home() / ".aibuildai" / "tools"


def tools_root() -> Path:
    """The root every owned tool lives under in this mode — the one path a sandbox ro-binds so in-sandbox launchers reach the owned binaries."""
    if is_frozen():
        return Path(sys.executable).resolve().parent / "tools"
    return _cache_root()


def postgresql_root() -> Path:
    """Return the complete prepared PostgreSQL installation tree."""
    if is_frozen():
        root = Path(sys.executable).resolve().parent / "tools" / "postgresql"
    else:
        root = _cache_root() / "postgresql" / POSTGRESQL_VERSION
    if not all((root / relative).is_file() for relative in _POSTGRESQL_REQUIRED):
        action = "reinstall the release" if is_frozen() else "run aibuildai setup"
        raise ValueError(f"the owned PostgreSQL tree is incomplete at {root}; {action}")
    return root


def postgresql_environment() -> dict[str, str]:
    """Return the library path required by the relocated PostgreSQL tree."""
    return {"LD_LIBRARY_PATH": str(postgresql_root() / "lib")}


def micromamba_root_prefix() -> Path:
    """Return the owned micromamba root prefix (its ``pkgs/`` download cache and ``envs/``).

    Mutable per-user state, so it lives beside the tools cache under ``~/.aibuildai`` rather than inside a tool identity directory (which setup deletes and rebuilds whole). Created on first use: sandbox binds may name it before the first environment exists.
    """
    root = Path.home() / ".aibuildai" / "micromamba"
    (root / "pkgs").mkdir(parents=True, exist_ok=True)
    (root / "envs").mkdir(parents=True, exist_ok=True)
    return root


def _source_path(name: str) -> Path:
    """Canonical private-cache path of one tool for this checkout's identity."""
    root = _cache_root()
    if name == "micromamba":
        return root / "micromamba" / MICROMAMBA_VERSION / "micromamba"
    if name == "ruff":
        return root / "ruff" / RUFF_VERSION / "ruff"
    if name == "tectonic":
        return root / "tectonic" / TECTONIC_VERSION / "tectonic"
    if name == "caddy":
        return root / "caddy" / CADDY_VERSION / "caddy"
    raise KeyError(name)


def owned(name: str) -> Path:
    """Return the absolute path to one already-prepared AIBuildAI-owned tool."""
    if name == "claude":
        return claude_path()
    if name == "cli-proxy-api":
        return proxy_path()
    if is_frozen():
        path = Path(sys.executable).resolve().parent / "tools" / _FROZEN_PATHS[name]
    else:
        path = _source_path(name)
    if not path.is_file():
        if is_frozen():
            raise ValueError(
                f"the shipped AIBuildAI tools directory is incomplete: owned "
                f"tool {name!r} is missing at {path}; reinstall the release."
            )
        raise ValueError(
            f"owned tool {name!r} is not prepared at {path}; "
            "setup_owned_tools() must run before product work."
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, sha256: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, dest.open("wb") as output:  # noqa: S310 - pinned https URL
        shutil.copyfileobj(response, output)
    actual = _sha256(dest)
    if actual != sha256:
        raise ValueError(f"sha256 mismatch for {url}: expected {sha256}, got {actual}")


def _install(
    name: str,
    required: tuple[str, ...],
    build: "Callable[[Path], None]",
    target_dir: Path | None = None,
) -> None:
    """Materialize one tool identity directory atomically under a file lock, so concurrent source runs never build twice or observe a half-written tool.

    An identity is ready only when its COMPLETE file set is present; a partial
    directory (an interrupted copy, a hand-deleted file) is removed and rebuilt
    instead of passing setup and failing later inside paid work."""
    target_dir = target_dir or _source_path(name).parent
    parent = target_dir.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (parent / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if all((target_dir / name).is_file() for name in required):
            return
        if target_dir.exists():
            shutil.rmtree(target_dir)
        with tempfile.TemporaryDirectory(dir=parent) as scratch:
            stage = Path(scratch) / "stage"
            stage.mkdir()
            build(stage)
            stage.rename(target_dir)


def _fetch_tar_member(
    url: str, sha256: str, member: str, filename: str
) -> "Callable[[Path], None]":
    """A build step that downloads one pinned release tarball, verifies its sha256, and extracts one executable member as ``filename``."""

    def build(stage: Path) -> None:
        dest = stage / filename
        with tempfile.TemporaryDirectory() as scratch_name:
            scratch = Path(scratch_name)
            archive = scratch / "archive.tar.gz"
            _download(url, sha256, archive)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extract(tar.getmember(member), scratch, filter="data")
            shutil.copyfile(scratch / member, dest)
        dest.chmod(0o755)

    return build


def _fetch_binary(url: str, sha256: str, filename: str) -> "Callable[[Path], None]":
    """A build step that downloads one pinned standalone executable, verifies its sha256, and marks it executable."""

    def build(stage: Path) -> None:
        dest = stage / filename
        _download(url, sha256, dest)
        dest.chmod(0o755)

    return build


def _setup_postgresql(stage: Path) -> None:
    """Build the pinned official PostgreSQL source into one relocatable tree."""
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        archive = scratch / f"postgresql-{POSTGRESQL_VERSION}.tar.bz2"
        _download(_POSTGRESQL_TARBALL_URL, _POSTGRESQL_TARBALL_SHA256, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(scratch, filter="data")
        source = scratch / f"postgresql-{POSTGRESQL_VERSION}"
        commands = (
            (
                str(source / "configure"),
                f"--prefix={stage}",
                "--disable-rpath",
                "--disable-nls",
                "--without-icu",
                "--without-readline",
                "--without-zlib",
                "--without-lz4",
                "--without-zstd",
                "--with-uuid=e2fs",
            ),
            ("make", f"-j{max(1, os.cpu_count() or 1)}"),
            ("make", "install"),
            ("make", "-C", "contrib/uuid-ossp", f"-j{max(1, os.cpu_count() or 1)}"),
            ("make", "-C", "contrib/uuid-ossp", "install"),
        )
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=source,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completed.returncode != 0:
                raise ValueError(
                    f"PostgreSQL build command {command[0]!r} failed:\n"
                    f"{completed.stdout}"
                )
        license_dir = stage / "share" / "doc" / "postgresql"
        license_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "COPYRIGHT", license_dir / "COPYRIGHT")


def setup_owned_tools(required: Collection[str] | None = None) -> None:
    """Prepare and probe only the named tools. With no argument, prepare the full product.

    Source mode uses the existing locked cache and atomic install. Frozen mode checks only the required shipped tools. Neither mode uses executables from PATH.
    """
    available = _FROZEN_PATHS.keys() | {"postgresql", "claude", "cli-proxy-api"}
    selected = available if required is None else set(required)
    if unknown := selected - available:
        raise ValueError(f"unknown owned tools: {sorted(unknown)}")
    if "claude" in selected:
        claude_path()
    if "cli-proxy-api" in selected:
        proxy_path()
    selected = selected - {"claude", "cli-proxy-api"}
    if not is_frozen():
        builds = {
            "ruff": _fetch_tar_member(_RUFF_TARBALL_URL, _RUFF_TARBALL_SHA256, _RUFF_MEMBER, "ruff"),
            "tectonic": _fetch_tar_member(_TECTONIC_TARBALL_URL, _TECTONIC_TARBALL_SHA256, "tectonic", "tectonic"),
            "caddy": _fetch_tar_member(_CADDY_TARBALL_URL, _CADDY_TARBALL_SHA256, "caddy", "caddy"),
            "micromamba": _fetch_binary(_MICROMAMBA_BINARY_URL, _MICROMAMBA_BINARY_SHA256, "micromamba"),
        }
        for name in sorted(selected):
            if name == "postgresql":
                _install(name, _POSTGRESQL_REQUIRED, _setup_postgresql,
                         _cache_root() / name / POSTGRESQL_VERSION)
            else:
                _install(name, (name,), builds[name])
    _probe_owned_tools(selected)


def stage_tools(dest: Path) -> None:
    """Copy the resolved tools of this checkout into a release staging tree at ``dest`` (the shipped ``tools/`` layout). The release thereby ships the SAME bytes the source-owned setup path resolved — no second list of executables exists."""
    setup_owned_tools()
    for name, relative in sorted(_FROZEN_PATHS.items()):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(owned(name), target)
        if relative.startswith("bin/"):
            target.chmod(0o755)
    shutil.copytree(
        postgresql_root(),
        dest / "postgresql",
        symlinks=True,
        dirs_exist_ok=True,
    )


def _probe(
    argv: list[str],
    pattern: str,
    check_exit: bool = True,
    env: dict[str, str] | None = None,
) -> "re.Match[str]":
    """Run one no-side-effect version probe; require ``pattern`` in its output and return the match."""
    try:
        completed = subprocess.run(
            argv,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(
            f"owned tool probe failed for {argv[0]}: {exc}; the cached tool is "
            "corrupt or hung — remove its directory and rerun setup."
        ) from exc
    match = re.search(pattern, completed.stdout or "")
    if (check_exit and completed.returncode != 0) or match is None:
        raise ValueError(
            f"owned tool probe failed for {argv[0]}: exit {completed.returncode}: "
            f"{(completed.stdout or '').strip()!r}"
        )
    return match


def _probe_owned_tools(required: Collection[str]) -> None:
    """Run each required tool's existing binary check."""
    commands = {
        "ruff": ("--version", r"^ruff \d"),
        "tectonic": ("--version", r"^[Tt]ectonic \d"),
        "caddy": ("version", r"^v\d"),
        "micromamba": ("--version", rf"^{re.escape(MICROMAMBA_VERSION)}\s*$"),
    }
    for name in sorted(required):
        if name == "postgresql":
            root = postgresql_root()
            environment = {**os.environ, **postgresql_environment()}
            for binary in ("pg_ctl", "postgres", "initdb", "psql"):
                _probe([str(root / "bin" / binary), "--version"],
                       rf" {POSTGRESQL_VERSION}\b", env=environment)
        else:
            argument, pattern = commands[name]
            _probe([str(owned(name)), argument], pattern)
