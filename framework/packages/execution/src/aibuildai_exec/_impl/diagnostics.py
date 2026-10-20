"""Read kernel OOM observations without inventing a kill origin."""

from __future__ import annotations

from dataclasses import dataclass

from .tree import own_cgroup_dir


@dataclass(frozen=True)
class KilledCgroup:
    """A node with observed kills and its recorded local limit, not the triggering ancestor."""

    cgroup: str
    memory_max: str | None


@dataclass(frozen=True)
class OomReport:
    """Known victims plus whether all requested kernel observations were available."""

    killed: tuple[KilledCgroup, ...]
    available: bool

    @property
    def oom_killed(self) -> bool:
        return bool(self.killed)


def detect_oom(*, scope_prefix: str) -> OomReport:
    """Read a named application's current cgroup subtree for exit diagnostics."""
    if not scope_prefix:
        raise ValueError("An application scope prefix is required")
    try:
        root = own_cgroup_dir()
        while root != root.parent and not root.name.startswith(scope_prefix):
            root = root.parent
        if not root.name.startswith(scope_prefix):
            return OomReport((), False)
        while root.parent.name.startswith(scope_prefix):
            root = root.parent
        records = sorted(root.rglob("memory.events.local"))
    except (OSError, AssertionError):
        return OomReport((), False)
    available = bool(records)
    killed = []
    for record in records:
        try:
            counters = dict(line.split() for line in record.read_text().splitlines())
            count = int(counters["oom_kill"])
            limit = (record.parent / "memory.max").read_text().strip()
        except (OSError, KeyError, ValueError):
            available = False
            continue
        if count:
            killed.append(KilledCgroup(str(record.parent.relative_to(root)), limit))
    return OomReport(tuple(killed), available)
