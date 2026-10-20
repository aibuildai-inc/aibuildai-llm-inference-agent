"""infra.process — subprocess running, conda env, and process lifecycle."""
from infra.process.conda_env import (
    conda_root,
    resolve_task_interpreter_or_none,
)
from infra.process.process_lifecycle import ProcessLifecycle

__all__ = [
    "conda_root",
    "resolve_task_interpreter_or_none",
    "ProcessLifecycle",
]
