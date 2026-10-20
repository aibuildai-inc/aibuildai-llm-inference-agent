"""One live owner of the existing physical resource tree."""

from __future__ import annotations

import asyncio
import fcntl
import itertools
import math
import os
import tempfile
import threading
import platform
from collections.abc import AsyncIterator, Collection, Mapping
from contextlib import asynccontextmanager
from concurrent.futures import Future
from pathlib import Path
from decimal import Decimal
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from .confinement import Confinement, OS_READ_BASELINE, probe_unprivileged_userns
from .errors import ExecutionError
from .paths import is_within, safe_tag
from .spec import ExecOutcome, ExecSpec
from .tree import CGROUP_MOUNT, Limits, ResourceTree, _write_verify, kill_tree, own_cgroup_dir

if TYPE_CHECKING:
    from .process import Process



def _declared_paths(paths: Collection[str | Path] | None) -> tuple[str, ...] | None:
    if paths is None:
        return None
    return tuple(dict.fromkeys(value for path in paths for value in (str(Path(path).absolute()), str(Path(path).resolve()))))


class Executor:
    """Bind one application-owned delegated root; never re-execute the application.

    The host must provide an exclusive delegated cgroup. An explicit external_root provides no local kernel limits; descendant cleanup is best effort.
    """

    def __init__(self, *, external_root: str | Path | None = None) -> None:
        self._external_root = None if external_root is None else Path(external_root).resolve()
        self._tree: ResourceTree | None = None
        self._guard: int | None = None
        self._closed = False
        self._released = False
        self._pid = os.getpid()
        self._sequence = itertools.count(1)
        self._scopes: set[Scope] = set()
        self._processes: set[Process] = set()
        self._cleanup_lock = threading.Lock()
        self._cleanup_tail: Future[None] = Future()
        self._cleanup_tail.set_result(None)

    async def __aenter__(self) -> Executor:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ExecutionError("This execution package requires Linux x86_64")
        if self._tree is not None or self._closed:
            raise ExecutionError("An Executor can be entered only once")
        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        if not runtime.is_dir():
            raise ExecutionError("Executor requires the host's user runtime directory")
        guards = runtime / "aibuildai-exec"
        guards.mkdir(mode=0o700, exist_ok=True)
        root = self._external_root or own_cgroup_dir()
        if self._external_root is not None:
            root.mkdir(parents=True, exist_ok=True)
        # The inode key identifies the actual root, not its caller's spelling.
        # A marker survives partial bootstrap so a process moved to the control
        # leaf can find the same root on its next attempt. Only flock owns it.
        candidates = (root,) if self._external_root is not None else (root, *root.parents)
        for candidate in candidates:
            if self._external_root is None and not is_within(candidate, CGROUP_MOUNT):
                break
            stat = candidate.stat()
            marker = guards / f"{stat.st_dev}-{stat.st_ino}.lock"
            if marker.is_file() and marker.read_text() == str(candidate):
                root = candidate
                break
        stat = root.stat()
        guard = guards / f"{stat.st_dev}-{stat.st_ino}.lock"
        self._guard = os.open(guard, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        try:
            try:
                fcntl.flock(self._guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ExecutionError(f"An Executor already owns this physical root: {root}") from exc
            os.ftruncate(self._guard, 0)
            os.write(self._guard, str(root).encode())
            self._tree = ResourceTree(root, enforce=self._external_root is None)
            self._tree.bootstrap(
                run_limits=Limits(memory_max_bytes=None, oom_group=False),
                framework_limits=Limits(memory_max_bytes=None, oom_group=False),
            )
        except BaseException as error:
            os.close(self._guard)
            self._guard = None
            self._tree = None
            self._closed = True
            if isinstance(error, OSError):
                raise ExecutionError(f"The delegated root is unavailable: {root}") from error
            raise
        return self

    async def __aexit__(self, _exc_type: type[BaseException] | None, exc: BaseException | None, _traceback: object) -> None:
        try:
            await self.close()
        except BaseException as cleanup_error:
            if exc is not None and cleanup_error is not exc:
                raise exc from cleanup_error
            raise

    @property
    def cpu_limits_available(self) -> bool:
        self._require_open()
        return cast(ResourceTree, self._tree)._controllers.cpu

    def scope(
        self, *, name: str, memory_bytes: int | None = None,
        cpu_cores: float | None = None, pids: int | None = None,
        gpu_devices: tuple[int, ...] | None = None,
        read_paths: Collection[str | Path] | None = None,
        write_paths: Collection[str | Path] | None = None,
        optional_read_paths: Collection[str | Path] | None = None,
        mask_paths: Collection[str | Path] = (),
        sandbox: bool | None = None, attach_host: bool = False,
        control_pids: int | None = None, launcher_dir: str | Path | None = None,
    ) -> Scope:
        """Declare a Scope in this tree; enter it before starting work."""
        self._require_open()
        if self._tree is None:
            raise ExecutionError("Executor has not entered its physical root")
        if attach_host and any(scope._path == self._tree.root for scope in self._scopes):
            raise ExecutionError("The host already has an attached Scope")
        return Scope._create(self, self._tree.root, name=name, memory_bytes=memory_bytes,
                     cpu_cores=cpu_cores, pids=pids, gpu_devices=gpu_devices,
                     read_paths=read_paths, write_paths=write_paths, optional_read_paths=optional_read_paths, mask_paths=mask_paths, sandbox=sandbox,
                     attach_host=attach_host, control_pids=control_pids, launcher_dir=launcher_dir)

    def _require_open(self) -> None:
        if self._pid != os.getpid() or self._tree is None or self._closed:
            raise ExecutionError("Executor is not live in this process")

    @asynccontextmanager
    async def _cleanup(self) -> AsyncIterator[None]:
        """Order process registration and subtree removal across event loops."""
        with self._cleanup_lock:
            previous = self._cleanup_tail
            finished: Future[None] = Future()
            self._cleanup_tail = finished
        try:
            await asyncio.shield(asyncio.wrap_future(previous))
            yield
        finally:
            # A cancelled waiter must not let the next removal overtake the
            # operation that still owns the tree, even on another event loop.
            if previous.done():
                finished.set_result(None)
            else:
                previous.add_done_callback(lambda _: finished.set_result(None))

    async def close(self) -> None:
        """Reject new work, finish owned cleanup, and release the live handle."""
        if self._released:
            return
        if self._pid != os.getpid():
            raise ExecutionError("Executor belongs to a different process")
        self._closed = True
        errors: list[BaseException] = []
        for scope in sorted(tuple(self._scopes), key=lambda s: len(s._path.parts), reverse=True):
            result, = await asyncio.gather(scope.close(), return_exceptions=True)
            if isinstance(result, BaseException):
                errors.append(result)
        async with self._cleanup():
            if not self._scopes and not self._processes:
                if self._guard is not None:
                    os.close(self._guard)
                    self._guard = None
                self._released = True
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Execution cleanup failed", errors)


class Scope:
    """A live node handle in its Executor's tree; it stores no child topology."""

    _owner: Executor
    _path: Path
    _attach_host: bool
    _control_pids: int | None
    _memory: int | None
    _cpu: float | None
    _pids: int | None
    _gpu_devices: tuple[int, ...] | None
    _read_paths: tuple[str, ...] | None
    _write_paths: tuple[str, ...] | None
    _optional_read_paths: tuple[str, ...] | None
    _mask_paths: tuple[str, ...]
    _sandbox: bool
    _entered: bool
    _closed: bool
    _released: bool
    _inode: int | None
    _prior_limits: dict[str, str]
    _borrow: Confinement | None
    _launcher_dir: Path | None
    _scratch: tempfile.TemporaryDirectory[str] | None
    _borrow_memory_kills: int | None
    _dev_nodes: tuple[str, ...]
    _name: str

    def __init__(self) -> None:
        raise TypeError("Create a Scope through Executor.scope or Scope.scope")

    @classmethod
    def _create(
        cls, owner: Executor, parent_path: Path, *, name: str,
        memory_bytes: int | None, cpu_cores: float | None, pids: int | None,
        gpu_devices: tuple[int, ...] | None,
        read_paths: Collection[str | Path] | None,
        write_paths: Collection[str | Path] | None,
        optional_read_paths: Collection[str | Path] | None = None,
        mask_paths: Collection[str | Path] = (),
        sandbox: bool | None, attach_host: bool = False,
        control_pids: int | None = None, launcher_dir: str | Path | None = None,
    ) -> Scope:
        self = object.__new__(cls)
        self._owner = owner
        self._path = parent_path if attach_host else parent_path / f"{next(owner._sequence)}-{safe_tag(name)}"
        self._attach_host = attach_host
        self._control_pids = control_pids
        self._memory = memory_bytes
        self._cpu = cpu_cores
        self._pids = pids
        self._gpu_devices = gpu_devices
        self._read_paths = _declared_paths(read_paths)
        self._write_paths = _declared_paths(write_paths)
        self._optional_read_paths = _declared_paths(optional_read_paths)
        self._mask_paths = _declared_paths(mask_paths) or ()
        self._sandbox = bool(sandbox)
        self._entered = False
        self._closed = False
        self._released = False
        self._inode = None
        self._borrow = None
        self._launcher_dir = None if launcher_dir is None else Path(launcher_dir)
        self._scratch = None
        self._borrow_memory_kills = None
        self._prior_limits = {}
        self._dev_nodes = ()
        self._name = name
        for label, value in (("memory_bytes", memory_bytes), ("pids", pids), ("control_pids", control_pids)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{label} must be a positive integer")
        if cpu_cores is not None and (isinstance(cpu_cores, bool) or not math.isfinite(cpu_cores) or cpu_cores <= 0):
            raise ValueError("cpu_cores must be finite and positive")
        if control_pids is not None and not attach_host:
            raise ValueError("control_pids applies only to the host-attached scope")
        if gpu_devices is not None and (len(set(gpu_devices)) != len(gpu_devices) or any(type(i) is not int or i < 0 for i in gpu_devices)):
            raise ValueError("gpu_devices must name distinct physical device indices")

        owner._scopes.add(self)
        return self

    @property
    def _local_limits_enforced(self) -> bool:
        return self._owner._external_root is None

    async def __aenter__(self) -> Scope:
        self._owner._require_open()
        if self._entered or self._closed:
            raise ExecutionError("A Scope can be entered only once")
        tree = self._owner._tree
        if tree is None:
            raise ExecutionError("Executor has no tree")
        try:
            if self._sandbox:
                self._read_paths = self._read_paths or ()
                self._write_paths = self._write_paths or ()
                self._optional_read_paths = self._optional_read_paths or ()
                await asyncio.to_thread(probe_unprivileged_userns)
            if self._gpu_devices:
                # GPU discovery is loaded only for a scope that names real cards.
                from .gpu_nodes import device_minors, enumerate_nvidia_cap_nodes
                from .bwrap import device_dev_nodes
                self._dev_nodes = tuple((*device_dev_nodes(await device_minors(self._gpu_devices, 30)), *enumerate_nvidia_cap_nodes()))
            self._owner._require_open()
            if self._closed:
                raise ExecutionError("The Scope was closed while it was entering")
            limits = Limits(memory_max_bytes=self._memory, oom_group=False, pids_max=self._pids,
                            cpu_max=None if self._cpu is None else f"{int(Decimal(str(self._cpu)) * 100000)} 100000")
            if self._attach_host:
                if self._local_limits_enforced:
                    self._prior_limits = {name: (self._path / name).read_text().strip() for name in ("memory.max", "memory.swap.max", "memory.oom.group", "cpu.max", "cpu.weight", "pids.max") if (self._path / name).exists()}
                    tree.apply(self._path, limits)
                    if self._control_pids is not None:
                        self._prior_limits["framework/pids.max"] = (self._path / "framework/pids.max").read_text().strip()
                        _write_verify(self._path, "framework/pids.max", str(self._control_pids))
            else:
                if self._path.exists():
                    raise ExecutionError(f"Scope node already exists: {self._path}")
                tree.inner(self._path.parent, self._path.name, limits)
            self._inode = self._path.stat().st_ino
            self._entered = True
        except BaseException as error:
            self._closed = True
            if not self._attach_host and self._path.exists():
                self._path.rmdir()
            await self.close()
            if isinstance(error, OSError):
                raise ExecutionError(f"The execution scope could not be entered: {self._name}") from error
            raise
        return self

    async def __aexit__(self, _exc_type: type[BaseException] | None, exc: BaseException | None, _traceback: object) -> None:
        try:
            await self.close()
        except BaseException as cleanup_error:
            if exc is not None and cleanup_error is not exc:
                raise exc from cleanup_error
            raise

    def _require_open(self) -> None:
        self._owner._require_open()
        if not self._entered or self._closed or not self._path.exists() or self._path.stat().st_ino != self._inode:
            raise ExecutionError("Scope no longer refers to a live node")

    def scope(
        self, *, name: str, memory_bytes: int | None = None,
        cpu_cores: float | None = None, pids: int | None = None,
        gpu_devices: tuple[int, ...] | None = None,
        read_paths: Collection[str | Path] | None = None,
        write_paths: Collection[str | Path] | None = None,
        optional_read_paths: Collection[str | Path] | None = None,
        mask_paths: Collection[str | Path] = (),
        sandbox: bool | None = None, launcher_dir: str | Path | None = None,
    ) -> Scope:
        """Declare a Scope in this tree; enter it before starting work."""
        self._require_open()
        if self._sandbox and sandbox is False:
            raise ValueError("A child cannot disable its parent's file isolation")
        reads = self._read_paths if read_paths is None else _declared_paths(read_paths)
        optional = self._optional_read_paths if optional_read_paths is None else _declared_paths(optional_read_paths)
        writes = self._write_paths if write_paths is None else _declared_paths(write_paths)
        self._check_paths(reads, writes, optional)
        devices = self._gpu_devices if gpu_devices is None else gpu_devices
        if self._gpu_devices is not None and not set(devices or ()).issubset(self._gpu_devices):
            raise ValueError("A child cannot add a GPU outside its parent's set")
        return Scope._create(self._owner, self._path, name=name, memory_bytes=memory_bytes,
                     cpu_cores=cpu_cores, pids=pids, gpu_devices=devices,
                     read_paths=reads, write_paths=writes, optional_read_paths=optional,
                     mask_paths=(*self._mask_paths, *mask_paths), sandbox=self._sandbox if sandbox is None else sandbox, launcher_dir=launcher_dir)

    def _check_paths(self, read_paths: Collection[str | Path] | None,
                     write_paths: Collection[str | Path] | None,
                     optional_read_paths: Collection[str | Path] | None = ()) -> None:
        for path in (*(read_paths or ()), *(optional_read_paths or ()), *(write_paths or ())):
            if any(is_within(path, mask) for mask in self._mask_paths):
                raise ValueError("A requested path is masked by this scope")
        for paths, allowed in (((*(read_paths or ()), *(optional_read_paths or ())), None if self._read_paths is None else (*(OS_READ_BASELINE if self._sandbox else ()), *self._read_paths, *(self._optional_read_paths or ()), *(self._write_paths or ()))), (write_paths, self._write_paths)):
            if allowed is not None and any(not any(is_within(p, root) for root in allowed) for p in paths or ()):
                raise ValueError("A requested path is outside the scope file grants")

    async def start(self, spec: ExecSpec) -> Process:
        """Start one process in a distinct leaf under this Scope."""
        self._require_open()
        from .process import Process  # Break the Scope/Process ownership cycle.
        async with self._owner._cleanup():
            self._require_open()
            return await Process._start(self, spec)

    async def run(self, spec: ExecSpec) -> ExecOutcome:
        """Use start and wait; descendant cleanup is best effort without a cgroup."""
        process = await self.start(spec)
        return await process.wait()

    async def close(self) -> None:
        """Reject new work and serialize cleanup with the same tree's other scopes."""
        if self._owner._pid != os.getpid():
            raise ExecutionError("Scope belongs to a different process")
        for scope in tuple(self._owner._scopes):
            if is_within(scope._path, self._path):
                scope._closed = True
        async with self._owner._cleanup():
            if self._released:
                return
            processes = [p for p in tuple(self._owner._processes) if is_within(p._path, self._path)]
        results = await asyncio.gather(*(p.terminate() for p in processes), return_exceptions=True)
        async with self._owner._cleanup():
            await self._finish_close(processes, results)

    async def _finish_close(self, processes: list[Process], results: list[ExecOutcome | BaseException]) -> None:
        if self._released:
            return
        self._closed = True
        if not self._entered:
            for name, value in self._prior_limits.items():
                _write_verify(self._path, name, value)
            self._released = True
            self._owner._scopes.discard(self)
            return
        descendants = [s for s in tuple(self._owner._scopes) if is_within(s._path, self._path)]
        for scope in descendants:
            scope._closed = True
        if any(not process._cleaned for process in processes):
            failures = [result for result in results if isinstance(result, BaseException)]
            raise BaseExceptionGroup("Scope still owns unfinished process cleanup", failures)
        paths = [p for p in self._path.iterdir() if p.is_dir() and p.name != "framework"] if self._attach_host else [self._path]
        for path in paths:
            kill_tree(path, enforce=self._local_limits_enforced)
            for _ in range(500):
                if self._local_limits_enforced:
                    populated = "populated 1" in (path / "cgroup.events").read_text()
                    if populated:
                        await asyncio.sleep(0.01)
                        continue
                for scope in descendants:
                    if scope._borrow is not None and is_within(scope._borrow.cgroup, path):
                        scope._borrow_memory_kills = scope._borrow.memory_kills()
                for node in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                    if node.is_dir():
                        node.rmdir()
                    elif not self._local_limits_enforced:
                        node.unlink()
                path.rmdir()
                break
            else:
                raise ExecutionError(f"Scope did not become empty after forced cleanup: {path}")
        for name, value in self._prior_limits.items():
            _write_verify(self._path, name, value)
        for scope in descendants:
            if scope._scratch is not None:
                scope._scratch.cleanup()
            scope._released = True
        self._owner._scopes.difference_update(descendants)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _confinement(self, path: Path, cwd: str | None = None) -> Confinement:
        if cwd is not None and not Path(cwd).is_dir():
            raise ExecutionError(f"The working directory does not exist: {cwd}")
        if self._sandbox and cwd is not None:
            self._check_paths((cwd,), ())
        return Confinement(cgroup=path, apply_bwrap=self._sandbox, cwd=cwd,
            system_read_paths=OS_READ_BASELINE, hard_ro=self._read_paths or (),
            ro_paths=self._optional_read_paths or (), mask_paths=self._mask_paths,
            rw_paths=self._write_paths or (), dev_nodes=self._dev_nodes,
            pinned_device_count=len(self._gpu_devices or ()))


    def _borrow_confinement(self) -> Confinement:
        self._require_open()
        if self._borrow is None:
            tree = self._owner._tree
            if tree is None:
                raise ExecutionError("Executor has no resource tree")
            root = tree.inner(self._path, "conversation", Limits(memory_max_bytes=None, oom_group=False))
            session = tree.leaf(root, "session", Limits(memory_max_bytes=None, oom_group=True))
            commands = tree.inner(root, "commands", Limits(memory_max_bytes=None, oom_group=False))
            if self._launcher_dir is None:
                self._scratch = tempfile.TemporaryDirectory(prefix="aibuildai-exec-launch-")
                self._launcher_dir = Path(self._scratch.name)
            self._borrow = replace(self._confinement(session), commands_cgroup=commands, step_dir=str(self._launcher_dir))
        return self._borrow

    @property
    def apply_bwrap(self) -> bool:
        return self._sandbox

    def memory_kills(self) -> int | None:
        if self._closed:
            return self._borrow_memory_kills
        return self._borrow_confinement().memory_kills()

    def delegates_commands(self) -> bool:
        return True

    def wrap_command(self, command: str, tool_use_id: str) -> str:
        return self._borrow_confinement().wrap_command(command, tool_use_id)

    def reap_command(self, tool_use_id: str) -> None:
        if self._closed:
            return
        self._borrow_confinement().reap_command(tool_use_id)

    async def terminate_command(self, tool_use_id: str) -> None:
        """Stop a command; descendant cleanup is best effort without a cgroup."""
        if self._owner._pid != os.getpid():
            raise ExecutionError("Scope belongs to a different process")
        async with self._owner._cleanup():
            if self._closed or self._owner._closed:
                return
            await self._borrow_confinement().terminate_command(tool_use_id,
                enforce=self._local_limits_enforced)

    def as_script(
        self, inner_argv: list[str], *, setenv: dict[str, str] | None = None,
        reopen_fds: Mapping[int, str] | None = None,
        environment_keys: Collection[str] | None = None, prelude: str = "",
        read_paths: Collection[str] = (), write_paths: Collection[str] = (),
    ) -> str:
        """Compile the native provider's launch with this Scope's physical grants."""
        self._require_open()
        self._check_paths(read_paths, write_paths)
        for source in (reopen_fds or {}).values():
            prefix = f"/proc/{os.getpid()}/fd/"
            if source.startswith(prefix) and source[len(prefix):].isdigit():
                os.fstat(int(source[len(prefix):]))
            else:
                self._check_paths((source,), ())
        return self._borrow_confinement().as_script(inner_argv, setenv=setenv,
            reopen_fds=reopen_fds, environment_keys=environment_keys, prelude=prelude,
            )
