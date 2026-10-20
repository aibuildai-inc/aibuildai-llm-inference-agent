"""The shared start, drain, wait and subtree cleanup path."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import tempfile
import os
import signal
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal

from .errors import ExecutionError
from .spec import ExecOutcome, ExecSpec, Termination
from .tree import Limits, kill_tree, oom_kill_count

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long

if TYPE_CHECKING:
    from .executor import Scope


class Process:
    """One owned process leaf. wait observes the same invocation each time."""

    _scope: Scope
    _spec: ExecSpec
    _path: Path
    _proc: asyncio.subprocess.Process | None
    _task: asyncio.Task[ExecOutcome] | None
    _termination: Termination | None
    _stop: asyncio.Event
    _started: float
    _total: int
    _truncated: bool
    _stdout: bytearray
    _stderr: bytearray
    _destinations: dict[str, BinaryIO]
    _files: list[BinaryIO]
    _oom_before: int | None
    _cleaned: bool
    _cleanup_task: asyncio.Task[None] | None
    _readers: list[asyncio.Task[None]]
    _reaped: asyncio.Task[int] | None

    def __init__(self) -> None:
        raise TypeError("Create a Process through Scope.start")

    @classmethod
    def _create(cls, scope: Scope, spec: ExecSpec, path: Path) -> Process:
        self = object.__new__(cls)
        self._scope = scope
        self._spec = spec
        self._path = path
        self._proc = None
        self._task = None
        self._termination = None
        self._stop = asyncio.Event()
        self._started = time.monotonic()
        self._total = 0
        self._truncated = False
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._files = []
        self._destinations = {}
        self._oom_before = None
        self._cleaned = False
        self._cleanup_task = None
        self._readers = []
        self._reaped = None
        return self

    @classmethod
    async def _start(cls, scope: Scope, spec: ExecSpec) -> Process:
        tree = scope._owner._tree
        if tree is None:
            raise ExecutionError("Executor has no resource tree")
        path = scope._path / f"process-{next(scope._owner._sequence)}"
        tree.leaf(scope._path, path.name, Limits(memory_max_bytes=None, oom_group=True))
        process = cls._create(scope, spec, path)
        try:
            confinement = scope._confinement(path, str(Path(spec.cwd).resolve()))
            process._oom_before = oom_kill_count(path)
            env = dict(os.environ if spec.env is None else spec.env)
            if scope._sandbox and spec.env is not None and "PWD" in spec.env and spec.env["PWD"] != confinement.cwd:
                raise ValueError("A sandbox sets PWD to its resolved working directory")
            if scope._gpu_devices is not None:
                visible = ",".join(str(i) for i in scope._gpu_devices)
                if spec.env is not None and "CUDA_VISIBLE_DEVICES" in spec.env and spec.env["CUDA_VISIBLE_DEVICES"] != visible:
                    raise ValueError("The explicit CUDA environment conflicts with the Scope's devices")
                env["CUDA_VISIBLE_DEVICES"] = visible
                if scope._gpu_devices:
                    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            executable = spec.argv[0]
            if "/" in executable:
                executable = str((Path(spec.cwd) / executable).absolute())
            else:
                executable = shutil.which(executable, path=env.get("PATH", os.defpath)) or ""
            if not executable or not os.access(executable, os.X_OK) or not Path(executable).is_file():
                raise ExecutionError(f"The executable is unavailable: {spec.argv[0]}")
            if scope._sandbox:
                scope._check_paths((executable,), ())
            environment_file = tempfile.TemporaryFile()
            process._files.append(environment_file)
            environment_file.write(json.dumps(env).encode("utf-8"))
            environment_file.seek(0)
            wrapped = confinement.wrap([executable, *spec.argv[1:]], environment_fd=environment_file.fileno())
            outputs: list[int | BinaryIO] = []
            for channel, destination in zip(("stdout", "stderr"), (spec.stdout_path, spec.stderr_path)):
                if destination is None:
                    outputs.append(asyncio.subprocess.PIPE)
                else:
                    stream = open(destination, "ab", buffering=0)
                    process._files.append(stream)
                    process._destinations[channel] = stream
                    outputs.append(stream if spec.on_output is None else asyncio.subprocess.PIPE)
            try:
                process._proc = await asyncio.create_subprocess_exec(
                    *wrapped, cwd=confinement.cwd, env={"PATH": os.defpath},
                    pass_fds=(environment_file.fileno(),),
                    stdin=asyncio.subprocess.PIPE, stdout=outputs[0], stderr=outputs[1],
                    start_new_session=True,
                )
            except OSError as exc:
                raise ExecutionError(f"The process could not start: {spec.argv[0]}") from exc
            environment_file.close()
            process._files.remove(environment_file)
            scope._owner._processes.add(process)
            process._task = asyncio.create_task(process._wait(), name=f"exec-{process._proc.pid}")
        except BaseException:
            for stream in process._files:
                stream.close()
            kill_tree(path, enforce=scope._local_limits_enforced)
            if not scope._local_limits_enforced:
                for file in path.iterdir():
                    file.unlink()
            path.rmdir()
            raise
        return process

    @property
    def pid(self) -> int:
        if self._proc is None:
            raise ExecutionError("The process has not started")
        return self._proc.pid

    @property
    def stdin(self) -> asyncio.StreamWriter:
        if self._proc is None or self._proc.stdin is None:
            raise ExecutionError("The process has no input stream")
        return self._proc.stdin

    @property
    def stdout(self) -> bytes:
        return bytes(self._stdout)

    @property
    def stderr(self) -> bytes:
        return bytes(self._stderr)

    async def _drain(self, reader: asyncio.StreamReader | None, channel: Literal["stdout", "stderr"], tail: bytearray) -> None:
        if reader is None:
            return
        limit = (self._spec.capture_limit_bytes + (channel == "stdout")) // 2
        while chunk := await reader.read(65536):
            if channel in self._destinations:
                self._destinations[channel].write(chunk)
            else:
                self._total += len(chunk)
                tail.extend(chunk)
                if len(tail) > limit:
                    if limit:
                        del tail[:-limit]
                    else:
                        tail.clear()
                    self._truncated = True
            if self._spec.on_output is not None:
                self._spec.on_output(channel, chunk)
            if self._total > self._spec.capture_limit_bytes and self._termination is None:
                self._termination = Termination.OUTPUT_LIMIT
                self._stop.set()

    async def wait(self) -> ExecOutcome:
        """Wait for this invocation; cancellation cleans up and is re-raised."""
        if self._task is None:
            raise ExecutionError("The process has not started")
        try:
            return await asyncio.shield(self._task)
        except asyncio.CancelledError as cancelled:
            try:
                await asyncio.shield(self.terminate())
            except BaseException as cleanup_error:
                raise cancelled from cleanup_error
            raise
        finally:
            if self._cleaned:
                self._scope._owner._processes.discard(self)

    async def terminate(self) -> ExecOutcome:
        """Request termination and wait for the same completed outcome."""
        if self._task is None:
            raise ExecutionError("The process has not started")
        if self._task.done() and not self._cleaned:
            await self._cleanup()
        if not self._task.done() and self._termination is None:
            self._termination = Termination.TERMINATED
            self._stop.set()
        try:
            return await asyncio.shield(self._task)
        finally:
            if self._cleaned:
                self._scope._owner._processes.discard(self)

    async def _exited(self) -> None:
        """Observe direct-child exit independently of inherited pipe handles."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        try:
            # The supported frozen Python omits os.pidfd_open (#3176).
            descriptor = _LIBC.syscall(ctypes.c_long(434), ctypes.c_int(proc.pid), ctypes.c_uint(0))
            if descriptor < 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
        except ProcessLookupError:
            return
        except OSError as exc:
            raise ExecutionError("The host cannot provide a process-exit descriptor") from exc
        loop = asyncio.get_running_loop()
        exited: asyncio.Future[None] = loop.create_future()
        def ready() -> None:
            loop.remove_reader(descriptor)
            if not exited.done():
                exited.set_result(None)
        loop.add_reader(descriptor, ready)
        try:
            await exited
        finally:
            loop.remove_reader(descriptor)
            os.close(descriptor)

    async def _wait(self) -> ExecOutcome:
        proc = self._proc
        if proc is None:
            raise ExecutionError("The process has not started")
        readers = [asyncio.create_task(self._drain(proc.stdout, "stdout", self._stdout)),
                   asyncio.create_task(self._drain(proc.stderr, "stderr", self._stderr))]
        exited = asyncio.create_task(self._exited())
        reaped = asyncio.create_task(proc.wait())
        self._reaped = reaped
        self._readers = readers
        stopped = asyncio.create_task(self._stop.wait())
        cleanup_start = self._started
        memory_kills: int | None = None
        try:
            # Reader failures (including caller callback errors) must end the
            # process before the original error is returned to its caller.
            completion = asyncio.gather(reaped, *readers)
            completion.add_done_callback(lambda task: None if task.cancelled() else task.exception())
            remaining = None if self._spec.timeout_s is None else max(0.0, self._spec.timeout_s - (time.monotonic() - self._started))
            done, _ = await asyncio.wait((completion, exited, stopped), timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            cleanup_start = time.monotonic()
            if completion in done:
                await completion
            elif exited in done:
                await exited
                kill_tree(self._path, enforce=self._scope._local_limits_enforced)
                await asyncio.wait_for(asyncio.shield(completion), 5.0)
            else:
                if stopped not in done:
                    self._termination = Termination.TIMEOUT
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(asyncio.shield(completion), self._spec.stop_wait_seconds)
                except TimeoutError:
                    kill_tree(self._path, enforce=self._scope._local_limits_enforced)
                    await asyncio.wait_for(asyncio.shield(completion), 5.0)
            after = oom_kill_count(self._path)
            if after is not None and self._oom_before is not None:
                memory_kills = after - self._oom_before
            if self._termination is None:
                self._termination = Termination.MEMORY_LIMIT if memory_kills else Termination.EXITED
        finally:
            failure = sys.exception()
            stopped.cancel()
            exited.cancel()
            await asyncio.gather(stopped, exited, return_exceptions=True)
            try:
                await self._cleanup()
            except BaseException as cleanup_error:
                if failure is not None and cleanup_error is not failure:
                    raise failure from cleanup_error
                raise
        return ExecOutcome(termination=self._termination, return_code=proc.returncode,
            stdout=None if self._spec.stdout_path is not None else bytes(self._stdout),
            stderr=None if self._spec.stderr_path is not None else bytes(self._stderr),
            duration_s=cleanup_start - self._started, cleanup_s=time.monotonic() - cleanup_start,
            memory_kills=memory_kills, stdout_path=self._spec.stdout_path,
            stderr_path=self._spec.stderr_path, output_truncated=self._truncated)


    async def _cleanup(self) -> None:
        if self._cleaned:
            return
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._finish_cleanup())
        await asyncio.shield(self._cleanup_task)

    async def _finish_cleanup(self) -> None:
        if self._reaped is None:
            raise ExecutionError("The process reaper was not started")
        kill_tree(self._path, enforce=self._scope._local_limits_enforced)
        try:
            await asyncio.wait_for(asyncio.shield(self._reaped), 5.0)
            if self._readers:
                _, pending = await asyncio.wait(self._readers, timeout=5.0)
                if pending:
                    raise ExecutionError(f"Process output is still open after cleanup: {self._path}")
            for _ in range(500):
                if not self._scope._local_limits_enforced or "populated 1" not in (self._path / "cgroup.events").read_text():
                    break
                await asyncio.sleep(0.01)
            else:
                raise ExecutionError(f"Process subtree is still populated: {self._path}")
            if not self._scope._local_limits_enforced:
                for file in self._path.iterdir():
                    file.unlink()
            self._path.rmdir()
            self._cleaned = True
        except TimeoutError as exc:
            raise ExecutionError(f"Process cleanup did not finish: {self._path}") from exc
        finally:
            if self._cleaned:
                for stream in self._files:
                    stream.close()
