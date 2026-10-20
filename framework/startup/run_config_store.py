"""Archive the resolved AgentConfig in the Run home for later startup."""

from __future__ import annotations

from pathlib import Path

import features
from config import AgentConfig

RUN_CONFIG_FILENAME = "run_config.json"


def _config_path(run_home: str) -> Path:
    return Path(run_home) / RUN_CONFIG_FILENAME


def _hidden_exclude() -> dict:
    """Pydantic nested-exclude structure for THIS build's hidden-feature fields.

    Turns each dotted ``features.hidden_config_fields()`` path into pydantic's nested exclude mapping (``"verifier"`` -> ``{"verifier": True}``, ``"llm.auto"`` -> ``{"llm": {"auto": True}}``), so a leaf-level hidden field prunes only that field. Empty in source/dev (``hidden_config_fields()`` is empty there).
    """
    exclude: dict = {}
    for path in features.hidden_config_fields():
        *parents, leaf = path.split(".")
        node = exclude
        for name in parents:
            node = node.setdefault(name, {})
        node[leaf] = True
    return exclude


def save_run_config(config: AgentConfig, run_home: str) -> Path:
    # Prune the build's hidden-feature fields from the archive so resume's
    # strict re-validation (``load_run_config`` -> ``_reject_hidden_features``)
    # accepts the product's own archive. A frozen build rejects those fields in
    # user YAML too, so they are always at their defaults at save time and
    # pydantic refills the same defaults on reload -- lossless. In source/dev
    # the exclude is empty, so the dump is byte-identical to a plain one.
    path = _config_path(run_home)
    config.run.require_run_id()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        config.model_dump_json(exclude=_hidden_exclude() or None), encoding="utf-8"
    )
    return path


def load_run_config(run_home: str) -> AgentConfig:
    path = _config_path(run_home)
    if not path.exists():
        raise FileNotFoundError(f"archived run config not found: {path}")
    config = AgentConfig.model_validate_json(path.read_text(encoding="utf-8"))
    config.run.require_run_id()
    return config
