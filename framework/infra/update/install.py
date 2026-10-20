"""Locate the installed product and unpack the one installer that mutates it."""
import sys
import tarfile
from pathlib import Path

# Name of the per-version content-identity file: the sha256 (bare hex) of the
# tarball this version was installed from, written into versions/<v>/ by BOTH
# install paths (auto-update runner + manual install.sh). The running binary
# reads its own identity from here so the auto-updater can trigger on a
# same-version content change (a same-tag rebuilt tarball).
IDENTITY_FILENAME = "tarball.sha256"

def _running_version_dir() -> Path | None:
    """The versions/<v>/ dir the running binary lives in, or None if the install layout is not the expected versioned one. resolve() follows the bin symlink -> current -> versions/<v>/aibuildai."""
    exe = Path(sys.executable).resolve()
    if exe.parent.parent.name == "versions":
        return exe.parent
    return None


def resolve_install_root() -> Path | None:
    """The lib root (~/.local/lib/aibuildai) derived from the running binary; None if the layout is not the expected versioned one."""
    vdir = _running_version_dir()
    return vdir.parents[1] if vdir is not None else None


def recorded_identity() -> str | None:
    """The content identity (tarball sha256, bare hex) recorded for the RUNNING install, or None when absent — an older install written before this file existed, a non-versioned layout, or a manual install.sh that could not find its sibling tarball. A None identity makes the same-version check fall back to version-only behaviour (never installs on an unknown identity)."""
    vdir = _running_version_dir()
    if vdir is None:
        return None
    f = vdir / IDENTITY_FILENAME
    if not f.is_file():
        return None
    return f.read_text().strip() or None


def extract_package(tarball: Path, dest: Path) -> Path:
    """Extract ``tarball`` under ``dest`` and return its one package directory."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        top_names = {name.split("/", 1)[0] for name in tf.getnames() if name}
        if len(top_names) != 1:
            raise ValueError("release tarball must contain one package directory")
        tf.extractall(dest, filter="data")
    package_dir = dest / next(iter(top_names))
    if not package_dir.is_dir() or not (package_dir / "install.sh").is_file():
        raise ValueError("release tarball has no install.sh")
    return package_dir
