"""Physical process declarations and observed outcomes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Literal


class Termination(Enum):
    """Why execution ended; the observed exit code remains a separate fact."""

    EXITED = "exited"
    MEMORY_LIMIT = "memory_limit"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    TERMINATED = "terminated"


@dataclass(frozen=True)
class ExecSpec:
    """Run argv without an implicit shell; None inherits the parent environment. A sandbox derives PWD from resolved cwd, and an explicit PWD must agree with that directory."""

    argv: tuple[str, ...]
    cwd: str | Path
    env: Mapping[str, str] | None = None
    timeout_s: float | None = None
    stop_wait_seconds: float = 5.0
    capture_limit_bytes: int = 65536
    stdout_path: str | Path | None = None
    stderr_path: str | Path | None = None
    on_output: Callable[[Literal["stdout", "stderr"], bytes], None] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.argv, (str, bytes)) or not self.argv or not self.argv[0]:
            raise ValueError("argv must contain an executable")
        if any(not isinstance(arg, str) or "\0" in arg for arg in self.argv):
            raise ValueError("argv entries must be strings without NUL")
        for name, value in (("timeout_s", self.timeout_s), ("stop_wait_seconds", self.stop_wait_seconds)):
            if (value is None and name != "timeout_s") or (value is not None and (isinstance(value, bool) or not isfinite(value) or value < 0)):
                raise ValueError(f"{name} must be a finite nonnegative number")
        if type(self.capture_limit_bytes) is not int or self.capture_limit_bytes <= 0:
            raise ValueError("capture_limit_bytes must be a positive byte count")
        if self.env is not None and any(not isinstance(k, str) or not isinstance(v, str) or not k or "=" in k or "\0" in k or "\0" in v for k, v in self.env.items()):
            raise ValueError("env must contain valid string names and values")
        if not str(self.cwd):
            raise ValueError("cwd must name a directory")


@dataclass(frozen=True)
class ExecOutcome:
    """Preserve the observed exit code even when a limit ended the process."""

    termination: Termination
    return_code: int | None
    stdout: bytes | None
    stderr: bytes | None
    duration_s: float
    cleanup_s: float
    memory_kills: int | None
    stdout_path: str | Path | None
    stderr_path: str | Path | None
    output_truncated: bool
