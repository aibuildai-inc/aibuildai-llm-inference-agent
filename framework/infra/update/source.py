"""GitHub release source: among the public repo's releases, find the newest stable release on the RUNNING build's own line, compare its version (from the tag alone), and — only when a newer version exists — select its two assets (tarball, SHA256SUMS). A line is major.minor on a direct line (``v2.5.*``) and the codename on a named line (``science-*``); an install is never offered a release from another line, so several lines can be maintained at once and a direct-line install is never moved onto a named line. Version compare is a strict triple compare, which on a named line orders dates. The compare deliberately does NOT touch the assets, so an up-to-date check succeeds even against a release that carries no assets at all."""
from dataclasses import dataclass

import requests

from aibuildai_version import parse_release_tag


def parse_version(v: str) -> tuple[int, int, int]:
    parts = v.split(".")
    if len(parts) != 3:
        raise ValueError(f"not a 3-part version: {v!r}")
    major, minor, patch = (int(p) for p in parts)  # ValueError on non-int
    return (major, minor, patch)


def compare_versions(latest: str, current: str) -> int:
    """Three-way triple compare: 1 if ``latest`` is newer than ``current``, -1 if older, 0 if equal. The runner needs all three to split newer / same / older — the same-version-but-different-content path triggers an update while the older path must never downgrade, so a plain "is newer" boolean (which collapses same and older into one) is not enough."""
    a, b = parse_version(latest), parse_version(current)
    return (a > b) - (a < b)


def release_line(version: str, line: str | None) -> str:
    """The line an install stays on: the codename of a named line, else the major.minor of ``version``."""
    if line is not None:
        return line
    major, minor, _ = parse_version(version)
    return f"{major}.{minor}"


@dataclass(frozen=True)
class ReleaseAssets:
    tag: str
    version: str          # bare (no leading v)
    tarball_name: str     # aibuildai-linux-x86_64-<tag>.tar.gz (the SHA256SUMS key)
    tarball_url: str
    sha256sums_url: str


def latest_version(release_json: dict) -> str:
    """The bare version of a release, from its ``tag_name``: ``2.5.5`` for ``v2.5.5``, ``2026.8.28`` for ``science-2026-08-28``. Reads only the tag, NOT the assets — so the up-to-date check never depends on which assets the release carries (a release with none still answers "what version is latest")."""
    return parse_release_tag(release_json["tag_name"])[1]


def select_assets(release_json: dict) -> ReleaseAssets:
    tag = release_json["tag_name"]
    by_name = {a["name"]: a["browser_download_url"] for a in release_json["assets"]}
    tarball_name = f"aibuildai-linux-x86_64-{tag}.tar.gz"
    missing = [n for n in (tarball_name, "SHA256SUMS") if n not in by_name]
    if missing:
        raise ValueError(f"release {tag} missing assets: {missing}")
    return ReleaseAssets(
        tag=tag,
        version=latest_version(release_json),
        tarball_name=tarball_name,
        tarball_url=by_name[tarball_name],
        sha256sums_url=by_name["SHA256SUMS"],
    )


def fetch_line_latest_release(current_line: str, timeout: float = 3.0) -> dict | None:
    """GET the public repo's releases and return the newest STABLE release on ``current_line`` (see ``release_line``), or None when the line has no stable release. Resolving per-line (not the single global ``/releases/latest``) keeps each maintained line's users on their own line: a 2.5.x user is never offered a later-published 2.0.x patch, and a Science release does not pull 2.5.x users off their line. Tags that are not a release tag (the ``-latest`` rolling website pointers), pre-releases, and drafts are skipped. Network / HTTP / JSON errors propagate to the runner's best-effort boundary (fail-closed: skip the update, keep running). The caller compares the version via ``latest_version`` and only calls ``select_assets`` once an update is actually warranted."""
    resp = requests.get(
        "https://api.github.com/repos/aibuildai/AI-Build-AI/releases?per_page=100",
        timeout=timeout,
    )
    resp.raise_for_status()
    on_line: list[tuple[tuple[int, int, int], dict]] = []
    for rel in resp.json():
        if rel.get("draft") or rel.get("prerelease"):
            continue
        try:
            line, bare = parse_release_tag(rel["tag_name"])
        except ValueError:
            continue  # not a release tag (e.g. the -latest rolling pointers)
        if release_line(bare, line) == current_line:
            on_line.append((parse_version(bare), rel))
    if not on_line:
        return None
    return max(on_line, key=lambda item: item[0])[1]
