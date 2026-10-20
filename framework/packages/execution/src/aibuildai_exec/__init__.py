"""Create scopes and execute work through one physical owner."""
from ._impl.executor import Executor, Scope
from ._impl.process import Process
from ._impl.spec import ExecSpec, ExecOutcome, Termination
from ._impl.errors import ExecutionError

__all__ = ("Executor", "Scope", "Process", "ExecSpec", "ExecOutcome", "Termination", "ExecutionError")
