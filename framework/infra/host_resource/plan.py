"""Resolve the run-level limits used to build the resource tree.

Each execution level writes its OWN sparse local ceilings when it claims its place in the tree; this plan resolves only the run parent ceiling and the fixed process backstops. No level's limit is sized from a Search-local algorithm parameter.
"""
from __future__ import annotations

import os

RUN_PIDS_MAX = 8192


GIB = 1024 ** 3
FRAMEWORK_PIDS = 1024


def physical_memory_bytes() -> int:
    """Return the host's physical RAM without adding a runtime dependency."""
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
