"""Auto-update orchestration: gate -> check -> (download -> verify -> install -> self-check -> switch -> telemetry -> prune -> re-exec). The whole attempt is a best-effort boundary (authorized by the fail-closed design): a transient failure prints one stderr skip line and the run continues on the CURRENT version. A checksum mismatch is fail-closed and NEVER installs.

The check triggers on a newer version OR on a SAME-version content change (a same-tag rebuilt tarball whose sha256 differs from the installed one); it NEVER downgrades. The same-version content identity is the tarball sha256 read from the release SHA256SUMS — no extra asset."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

from infra import telemetry
from infra.update import install, source, verify
from infra.update.gate import should_check
from aibuildai_version import APP_LINE, APP_VERSION, display_version, release_label


def maybe_auto_update(command: str) -> None:
    if not should_check(command):
        return
    try:
        _attempt()
    except Exception as exc:
        print(f"[auto-update] skipped: {type(exc).__name__}: {exc}", file=sys.stderr)


def _attempt() -> None:
    current = display_version(APP_VERSION)
    release = source.fetch_line_latest_release(source.release_line(current, APP_LINE), timeout=3.0)
    if release is None:
        # No stable release on the running build's line (or a dev build ahead
        # of every published release): nothing to update to.
        return
    cmp = source.compare_versions(source.latest_version(release), current)

    if cmp < 0:
        # The latest public release is OLDER than the running version: never
        # downgrade. The version compare reads only the tag, so this reports
        # correctly even against a release with no assets.
        return
    if cmp == 0 and not _same_version_rebuilt(release):
        # Same version, and no content change (or no recorded identity / no assets
        # to compare against): up to date.
        return

    lib_root = install.resolve_install_root()
    if lib_root is None:
        print("[auto-update] skipped: unversioned install layout", file=sys.stderr)
        return

    # cmp > 0 (a newer version) OR cmp == 0 with a content change. NOW require the
    # release assets; a malformed release raises here and the best-effort boundary
    # skips the update.
    assets = source.select_assets(release)
    reason = "rebuilt" if cmp == 0 else "newer"
    print(f"[auto-update] a {reason} {release_label(assets.version, APP_LINE)} build is "
          f"available; downloading...", file=sys.stderr)
    with tempfile.TemporaryDirectory(dir=str(lib_root), prefix=".update-") as tmp:
        tmpd = Path(tmp)
        package_root = tmpd / "package"
        package_root.mkdir()
        tarball = package_root / assets.tarball_name
        _download(assets.tarball_url, tarball)
        sums = _download_text(assets.sha256sums_url)

        print("[auto-update] verifying the download...", file=sys.stderr)
        if not verify.verify_sha256(tarball, sums, assets.tarball_name):
            raise ValueError("SHA-256 checksum mismatch")

        package_dir = install.extract_package(tarball, package_root)
        print(f"[auto-update] switching to the new {release_label(assets.version, APP_LINE)} "
              f"build...", file=sys.stderr)
        env = {**os.environ, "AIBUILDAI_AUTO_UPDATE": "0", "AIBUILDAI_IN_SERVICE": "1"}
        subprocess.run(["bash", str(package_dir / "install.sh")], check=True, env=env)

    telemetry.track("auto_update", from_version=current, to_version=assets.version,
                    result="success", flush=True)

    os.environ["AIBUILDAI_UPDATED_THIS_LAUNCH"] = current
    new_binary = str(lib_root / "current" / "aibuildai")
    os.execv(new_binary, [new_binary, *sys.argv[1:]])   # never returns


def _same_version_rebuilt(release: dict) -> bool:
    """True iff the latest release (same version as the running binary) ships a DIFFERENT tarball than the one installed.

    Fail-closed: a missing recorded identity or missing assets returns False (treat as up to date — never install on an unknown same-version release)."""
    recorded = install.recorded_identity()
    if recorded is None:
        return False
    try:
        assets = source.select_assets(release)
    except ValueError:
        return False
    sums = _download_text(assets.sha256sums_url)
    latest_sha = verify.expected_sha256(sums, assets.tarball_name)
    if latest_sha is None:
        raise ValueError("SHA256SUMS missing tarball line")
    return latest_sha != recorded


def _download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=(3.0, 60.0)) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def _download_text(url: str) -> str:
    r = requests.get(url, timeout=(3.0, 30.0))
    r.raise_for_status()
    return r.text
