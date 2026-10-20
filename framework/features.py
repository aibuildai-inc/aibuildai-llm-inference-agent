"""The single place to open or close immature features in a shipped build.

A "feature" here is an in-development OR internal-only design we don't want exposed to users in a SHIPPED (frozen) build. Each entry names the feature, lists the surfaces it exposes, and carries one ``exposed`` flag covering ALL its surfaces. Source / dev builds always expose everything (``is_frozen()`` is False), so this registry only takes effect in the frozen binary.

This is capability gating on the BUILD, not on where a config came from: a shipped build must not exercise a hidden feature whatever the source, new user input and archived run configs alike. A frozen build refusing a source-written archive that carries hidden blocks is this rule working, not a defect (#2037).

To open a feature to users in releases, flip its ``exposed`` to True (or delete the entry once the feature is permanently mature). Changing a feature's exposure is a ONE-LINE edit HERE; the enforcement sites only query this table.

A feature can span more than one surface, all governed by its single ``exposed`` flag (so the surfaces can never drift apart):

  - ``config_fields``: dotted ``AgentConfig`` paths. A hidden field is rejected at config load and never rendered into ``aibuildai config`` (both in ``config.py``), and pruned from the run's archived config at save time (``startup/run_config_store.py``) so the build's own archive passes its own strict re-validation on resume. A top-level block is named by its block path ("verifier", "resources.sandbox"); a single field inside a shared/required block by its leaf path ("llm.auto"); a built-in MCP by its registry path ("mcps.kb").
  - ``config_values``: ``"<dotted path>=<value>"``. The field stays settable and stays in ``aibuildai config``; these are the ONLY values a shipped build accepts there, and any other value is refused at config load. An ALLOW-list, not a deny-list, for two reasons: naming what goes can never close an open string like ``search.kind``, and a value added to the model later would ship by default instead of waiting to be listed here. Use it when a field's other values are real choices a customer should keep ("search.kind=tree" stays, every other method goes); use ``config_fields`` when the whole knob goes.
  - ``roles``: agent role names. A hidden role is dropped from the ``work_units`` starter list, so the printed config does not budget time for a role this build never activates, and from the roles ``verifier.enable`` will accept.
  - ``commands``: ``aibuildai <name>`` subcommands, plus the internal arguments ``cli.py`` dispatches before the parser. A hidden command is not registered and a hidden internal argument is not dispatched, so the shipped binary reports either as an unknown command (enforced in ``startup/config_loader.py`` and ``cli.py``).
"""

from typing import NamedTuple

from aibuildai_version import is_frozen


class Feature(NamedTuple):
    name: str  # logical feature name (docs / clarity)
    exposed: bool  # shown to users in a SHIPPED build?
    config_fields: tuple[str, ...] = ()  # dotted AgentConfig paths this hides
    config_values: tuple[str, ...] = ()  # "<dotted path>=<value>" this hides
    roles: tuple[str, ...] = ()  # agent roles this hides
    commands: tuple[str, ...] = ()  # aibuildai subcommands this hides


# THE registry. Edit a row's ``exposed`` to open/close that feature in releases.
FEATURES: tuple[Feature, ...] = (
    # Hidden from shipped builds today:
    # audit: the `aibuildai audit <id>|<dir>` trust-audit command. Hidden from
    # shipped builds until matured; always available in source / dev.
    Feature("audit", exposed=False, commands=("audit",)),
    # submission: spending an external scoring service's submissions. Above 1
    # activates the SUBMITTER role. The shipped Kaggle MCP supplies the provider,
    # so the built-in `kaggle` name and the internal argument that starts that
    # MCP are surfaces of THIS feature and close with it.
    Feature(
        "submission",
        exposed=False,
        config_fields=("submission", "mcps.kaggle"),
        roles=("submitter",),
        commands=("__kaggle_submission_mcp",),
    ),
    # extensions: the user-extension point (#991) -- a directory of user-written
    # Search subclasses imported at startup. It is dev-facing groundwork, not a
    # capability the product documents to users: README describes it nowhere, and
    # a shipped user has no way to learn it exists except by finding it in the
    # generated example. Hidden from shipped builds; source/dev keeps it.
    Feature("extensions", exposed=False, config_fields=("extensions",)),
    # search methods: the Science line ships two, and `search.kind` stays
    # settable because both are real choices. Listing what STAYS is what closes
    # the surface: `search.kind` is an open string, so no list of refused names
    # could. Worker_review is the one role no remaining method activates: the
    # three Worker kinds that compose it are not shipped here. Method-local
    # knobs need no entry: each kind's own Input model already rejects a field it
    # does not declare.
    Feature(
        "search-methods",
        exposed=False,
        config_values=("search.kind=tree", "search.kind=nb_tree"),
        roles=("worker_review",),
    ),
    # research-first: the Research First Search prototype and its three new
    # roles. The kind itself is already closed by the search-methods allow-list
    # above; this row keeps the shipped starter config from budgeting roles no
    # shipped method activates.
    Feature(
        "research-first", exposed=False, roles=("planner", "committee", "reviewer")
    ),
    # Switches in place; currently shipped (exposed=True). Flip exposed=False to
    # hide the feature from releases -- one-line edit, no other code change.
    # writer: the WRITEUP stage that drafts a NeurIPS paper about a finished run
    # and compiles it. Four surfaces, one flag, so they can never drift: the
    # config block, the faithfulness gate keyed under verifier, the standalone
    # command (which runs the stage even with writer.enable off), and the skill
    # the stage reads. Shipped: the stage, its skill and the bundled tectonic are
    # all in the tarball, and writer.enable stays off by default, so this row open
    # offers an opt-in block rather than turning a stage on.
    Feature(
        "writer",
        exposed=True,
        config_fields=(
            "writer",
            "verifier.enable.writer",
            "skills.aibuildai-builtin:research-paper-writing",
        ),
        roles=("writer", "writer_review"),
        commands=("write-paper",),
    ),
    Feature("backend", exposed=True, config_fields=("backend",)),
    Feature("verifier", exposed=True, config_fields=("verifier",)),
    Feature("resources.sandbox", exposed=True, config_fields=("resources.sandbox",)),
    Feature("llm.auto", exposed=True, config_fields=("llm.auto",)),
)


def hidden_config_fields() -> tuple[str, ...]:
    """Dotted ``AgentConfig`` paths hidden from users in THIS build: only when frozen, the config-field surfaces of every not-exposed feature. Empty in source/dev."""
    if not is_frozen():
        return ()
    return tuple(path for f in FEATURES if not f.exposed for path in f.config_fields)


def offered_config_values() -> dict[str, frozenset[str]]:
    """Dotted ``AgentConfig`` path -> the ONLY values of it THIS build accepts. The field itself stays settable; every other value is refused. Empty in source/dev, where a path with no entry here offers everything."""
    offered: dict[str, set[str]] = {}
    if not is_frozen():
        return {}
    for f in FEATURES:
        if f.exposed:
            continue
        for entry in f.config_values:
            path, _, value = entry.partition("=")
            offered.setdefault(path, set()).add(value)
    return {path: frozenset(values) for path, values in offered.items()}


def hidden_roles() -> frozenset[str]:
    """Agent roles hidden from users in THIS build. Empty in source/dev."""
    if not is_frozen():
        return frozenset()
    return frozenset(r for f in FEATURES if not f.exposed for r in f.roles)


def command_hidden(name: str) -> bool:
    """Runtime gate: hide the ``aibuildai <name>`` subcommand from users in THIS build (frozen AND the registry hides it). Source / dev never hides it. An unregistered command is a normal, always-shipped command and is never hidden."""
    return is_frozen() and any(name in f.commands for f in FEATURES if not f.exposed)


def hidden_feature_set_hash() -> str:
    """Stable fingerprint of THIS build's hidden-feature set — the names of every not-exposed feature in FEATURES, sorted and hashed. Registry-level (NOT gated on the current process being frozen), so it is deterministic across a fresh run and its resume in the same build, and CHANGES when a feature is opened or closed. Pinned into the journal genesis RunOpened (#1032) so a resume under a differently-configured build is detectable."""
    import hashlib

    names = sorted(f.name for f in FEATURES if not f.exposed)
    return hashlib.sha256(";".join(names).encode("utf-8")).hexdigest()[:16]
