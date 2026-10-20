"""The first-party Search Definition packages, and the one map that selects them.

``search.kind`` names exactly one package here. That name is the root-selection
authority: nothing scans classes, nothing imports a package the run did not
select, and a kind this build does not offer fails as a user config error."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import features
from config import StartupUserInputError

if TYPE_CHECKING:
    from engine.builtin.aibuildai.search import AIBuildAISearch

# The complete map from a configured kind to the package that implements it.
# Adding a built-in Search kind is one entry here plus one package; there is no
# registry to scan and no import side effect to arrange.
SEARCH_PACKAGES: dict[str, str] = {
    "tree": "engine.builtin.tree",
    "evolutionary": "engine.builtin.evolutionary",
    "linear": "engine.builtin.linear",
    "parallel": "engine.builtin.parallel",
    "nb": "engine.builtin.nb",
    "nb_tree": "engine.builtin.nb_tree",
    "agentic": "engine.builtin.agentic",
    "research_first": "engine.builtin.research_first",
    "autoresearch": "engine.builtin.autoresearch",
    "meta": "engine.builtin.meta",
}


class UnknownSearchKindError(StartupUserInputError):
    """A config's ``search.kind`` names no package this build offers.

    A user config typo, not an internal bug, so the cli boundary reports it
    like a config error -- the value plus the offered kinds, exit 2 -- instead
    of the "bug in aibuildai" pre-run crash banner (#1720)."""

    def __init__(self, kind: str, offered: "list[str]") -> None:
        self.kind = kind
        self.offered = offered
        super().__init__(f"unknown search.kind {kind!r}; offered kinds: {offered}")


def offered_kinds() -> "list[str]":
    """The kinds this build offers, in map order."""
    offered = features.offered_config_values().get("search.kind")
    return [k for k in SEARCH_PACKAGES if offered is None or k in offered]


def search_type_for_kind(kind: str) -> "type[AIBuildAISearch]":
    """Import the one package this kind names and return its ``SEARCH_TYPE``.

    Only the selected package enters ``sys.modules``; a sibling package is
    never imported because a run chose a different kind."""
    if kind not in offered_kinds():
        raise UnknownSearchKindError(kind, offered_kinds())
    package = importlib.import_module(SEARCH_PACKAGES[kind])
    search_type = getattr(package, "SEARCH_TYPE", None)
    if search_type is None:
        raise AssertionError(f"{SEARCH_PACKAGES[kind]} exports no SEARCH_TYPE")
    return search_type


def offered_kind_docs() -> "list[tuple[str, str]]":
    """Each offered kind with its package's own one-paragraph description.

    The starter config renders its ``search.kind`` comment from this, so the
    rendered method list cannot go stale against the packages a build ships
    (#2080). Rendering the starter config is the one operation that reads every
    offered package; a run reads only its own."""
    docs: "list[tuple[str, str]]" = []
    for kind in offered_kinds():
        doc = search_type_for_kind(kind).__dict__.get("description")
        if not doc:
            raise AssertionError(
                f"search kind {kind!r} has no description; the starter "
                "config renders each offered method from its own class"
            )
        docs.append((kind, " ".join(doc.split("\n\n")[0].split())))
    return docs


def load_run_definitions(kind: str) -> None:
    """Import the Definition types one archived run can name, and no others.

    The run's own package plus the Definitions every AIBuildAI run composes.
    A sibling package is not imported: a restored run whose journal names one
    would be a run of a different kind."""
    from aibuildai_version import add_runtime_submodule_paths

    add_runtime_submodule_paths()
    # The two shared verifier Programs belong to no one role, so no role module
    # imports them at load time; a resumed run still has to resolve the type of
    # a child either of them recorded.
    import engine.work_unit.program.attempt_verifier  # noqa: F401
    import engine.work_unit.program.document_verifier  # noqa: F401
    import engine.builtin.aibuildai.programs.score  # noqa: F401

    search_type_for_kind(kind)
