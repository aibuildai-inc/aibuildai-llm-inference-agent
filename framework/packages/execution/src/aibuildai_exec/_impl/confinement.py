"""Place one unit in its cgroup and optional bubblewrap file view."""

from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from .environment import make_killable_preamble
from .errors import ExecutionError
from .command_launcher import command_self_enroll_launcher, command_cgroup_dir, stop_command_cgroups
from .tree import enter_cgroup_preamble, oom_kill_count
from .bwrap import build_bwrap_argv
from .environment import environment_preamble

_SHEBANG = "#!/usr/bin/env bash\nset -u\n"


def probe_unprivileged_userns() -> None:
    """Check the two host facts bwrap needs: the binary, and an unprivileged user namespace."""
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise ExecutionError("bwrap is not installed")
    try:
        proc = subprocess.run(
            [str(bwrap), "--ro-bind", "/", "/", "--unshare-user", "--", "/bin/true"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExecutionError(
            "the bwrap user namespace check did not finish in 5 seconds"
        ) from exc
    if proc.returncode != 0:
        raise ExecutionError(
            "unprivileged user namespaces are disabled on this host; "
            f"bwrap exited {proc.returncode}: {proc.stderr.strip()!r}"
        )


# The read-only view of the operating system every sandboxed process needs to
# run at all: its binaries and shared objects, its configuration, and its
# runtime sockets. An assembler adds it explicitly, beside the paths that are
# that launch's own; nothing else is granted by default.
OS_READ_BASELINE: tuple[str, ...] = ("/usr", "/etc", "/run")
"""The operating-system paths included in a sandboxed process view."""


@dataclass(frozen=True)
class Confinement:
    """The exact host confinement of one unit: its cgroups, and under bwrap the exact file and device view. Frozen so a call site cannot mutate it after the producer stamped it; every path here is one the producer named, and nothing is granted by default."""

    cgroup: Path
    apply_bwrap: bool

    # The resolved working directory of a subprocess launch, bound and used as
    # one path (#235). An Agent launch starts where its AgentSpec says.
    cwd: str | None = None

    # The full file and device view for this local launch.
    system_read_paths: tuple[str, ...] = ()
    ro_paths: tuple[str, ...] = ()
    hard_ro: tuple[str, ...] = ()
    rw_paths: tuple[str, ...] = ()
    dev_nodes: tuple[str, ...] = ()
    mask_paths: tuple[str, ...] = ()
    pinned_device_count: int = 0

    # Where an Agent launch keeps its launcher history, beside the model-owned
    # scratch directory.
    step_dir: str = ""

    commands_cgroup: Path | None = None

    def memory_kills(self) -> int | None:
        """Return the cumulative OOM kill count for this unit."""
        return oom_kill_count(self.cgroup)

    def wrap_command(self, command: str, tool_use_id: str) -> str:
        if self.commands_cgroup is None:
            raise AssertionError(f"The launch at {self.cgroup} has no command cgroup")
        commands_root = "/sys/fs/cgroup" if self.apply_bwrap else str(self.commands_cgroup)
        return command_self_enroll_launcher(command, tool_use_id, commands_root)

    def reap_command(self, tool_use_id: str) -> None:
        if self.commands_cgroup is None:
            raise AssertionError("this confinement has no command scope")
        command_cgroup_dir(self.commands_cgroup, tool_use_id).rmdir()

    async def terminate_command(self, tool_use_id: str, *, enforce: bool = True) -> None:
        """Stop and remove one command and its descendant processes."""
        if self.commands_cgroup is None:
            raise AssertionError("this confinement has no command scope")
        await stop_command_cgroups(self.commands_cgroup,
            command_cgroup_dir(self.commands_cgroup, tool_use_id), enforce=enforce)

    def _preamble(self) -> str:
        """Make the payload killable, then put it in its cgroup. In that order: a process that joins a cgroup while still OOM-immune is a process the kernel cannot reap when that cgroup fills."""
        return make_killable_preamble() + "\n" + enter_cgroup_preamble(self.cgroup)

    def _exec_argv(self, inner_argv: "list[str]", setenv: "dict[str, str]") -> list[str]:
        """The one argv a confined process is exec'd with: ``inner_argv`` inside bwrap with this view and ``setenv``, or ``inner_argv`` itself without bwrap."""
        if not self.apply_bwrap:
            return list(inner_argv)
        mask_dirs = tuple(path for path in self.mask_paths if os.path.isdir(path))
        mask_files = tuple(path for path in self.mask_paths if not os.path.isdir(path))
        binds = build_bwrap_argv(
            self.ro_paths,
            list(self.rw_paths),
            hard_ro=[*self.system_read_paths, *self.hard_ro],
            dev_nodes=self.dev_nodes,
            mask_files=mask_files,
            mask_dirs=mask_dirs,
        )
        if self.pinned_device_count > 0:
            # The K bound /dev/nvidia<idx> nodes are in-sandbox indices 0..K-1;
            # an inherited host CUDA_VISIBLE_DEVICES would point CUDA at devices
            # that are absent from, or renumbered inside, the sandbox.
            renumbered = ",".join(str(i) for i in range(self.pinned_device_count))
            binds += ["--setenv", "CUDA_VISIBLE_DEVICES", renumbered]
        for k, v in setenv.items():
            binds += ["--setenv", k, v]
        if self.commands_cgroup is not None:
            # The process sees ONLY its own commands subtree at /sys/fs/cgroup. It
            # can create a cgroup per command and join it -- which is how the
            # kernel later attributes a command's OOM to that exact command -- and
            # it cannot reach the ceiling, which is on the parent, outside the bind.
            binds += ["--bind", str(self.commands_cgroup), "/sys/fs/cgroup"]
        binds += ["--die-with-parent"]
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise ExecutionError("bwrap is not installed")
        return [bwrap, *binds, "--", *inner_argv]

    @staticmethod
    def _exports(setenv: "dict[str, str]") -> str:
        return "".join(f"export {k}={shlex.quote(v)}\n" for k, v in setenv.items())

    def wrap(
        self,
        inner_argv: "list[str]",
        *,
        setenv: "dict[str, str] | None" = None,
        environment_fd: int | None = None,
    ) -> list[str]:
        """The argv the call site spawns. A bash that joins the cgroup and then ``exec``s the confined argv, so no extra process lingers inside the unit's limits and the payload is bounded from its first instruction."""
        setenv = dict(setenv or {})
        exports = "" if self.apply_bwrap else self._exports(setenv)
        argv = self._exec_argv(inner_argv, setenv)
        if environment_fd is not None:
            # Restore the exact payload environment after the shell preamble,
            # before entering bwrap. Values stay in an inherited anonymous FD,
            # never in argv; the helper interpreter stays outside the sandbox.
            entry = "__exec_environment" if getattr(sys, "frozen", False) else str(Path(__file__).with_name("environment.py"))
            argv = [sys.executable, entry, str(environment_fd), *argv]
        return [
            "/bin/bash",
            "-c",
            self._preamble() + "\n" + exports + 'exec "$@"',
            "bash",
            *argv,
        ]

    def as_script(
        self,
        inner_argv: "list[str]",
        *,
        setenv: "dict[str, str] | None" = None,
        reopen_fds: "Mapping[int, str] | None" = None,
        environment_keys: Collection[str] | None = None,
        prelude: str = "",
    ) -> str:
        """Write the same launch as an executable script and return its path: the preamble, the descriptor reopens, ``prelude`` (shell text the caller wants run at every start, outside the sandbox), and ``exec`` of the confined argv with the script's own arguments appended.

        ``environment_keys`` restricts inherited exported variables before the launch. Explicit ``setenv`` values are applied after that restriction.

        Launchers are immutable step history. Exclusive creation makes concurrent starts choose different monotonically increasing names instead of overwriting evidence from an earlier start.
        """
        if not self.step_dir:
            raise AssertionError("agent Confinement requires step_dir")
        launcher_dir = Path(self.step_dir) / "launcher"
        launcher_dir.mkdir(parents=True, exist_ok=True)
        setenv = dict(setenv or {})
        reopen = "".join(
            f"exec {fd}<{shlex.quote(source)}\n"
            for fd, source in (reopen_fds or {}).items()
        )
        exports = "" if self.apply_bwrap else self._exports(setenv)
        argv = self._exec_argv(inner_argv, setenv)
        script = (
            _SHEBANG
            + (environment_preamble(environment_keys) if environment_keys is not None else "")
            + self._preamble()
            + "\n"
            + reopen
            + prelude
            + exports
            + "exec "
            + " \\\n  ".join(shlex.quote(a) for a in argv)
            + ' "$@"\n'
        )

        index = 1
        while True:
            out = launcher_dir / f"{index:04d}.sh"
            try:
                fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755)
            except FileExistsError:
                index += 1
                continue
            with os.fdopen(fd, "w") as stream:
                stream.write(script)
            out.chmod(
                out.stat().st_mode
                | stat.S_IRWXU
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
            return str(out)
