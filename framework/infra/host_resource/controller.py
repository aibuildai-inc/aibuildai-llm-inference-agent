"""Map product declarations to public execution scopes and GPU choices."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from aibuildai_exec import Executor, Scope
from infra.host_resource.cgroup import kubernetes_start_mode
from infra.host_resource.disk_space import free_bytes
from infra.host_resource.gpu_alloc import GpuSelector
from infra.host_resource.plan import GIB, FRAMEWORK_PIDS, RUN_PIDS_MAX

if TYPE_CHECKING:
    from config import ResourcesConfig


class RunResources:
    """Interpret one Run's policy; the public Executor owns its physical tree."""

    def __init__(self, resources: ResourcesConfig) -> None:
        self._resources = resources
        self._gpu = GpuSelector(
            gpu_busy_util_threshold=resources.gpu.busy_util_threshold,
            min_free_mib=resources.gpu.min_free_mib,
            util_samples=resources.gpu.util_samples,
            visible_devices=resources.cuda_visible_devices,
        )
        self._executor: Executor | None = None
        self._owners: dict[str, Scope] = {}
        self._stack = AsyncExitStack()

    async def acquire(self, *, playground_root: Path) -> None:
        """Enter the one physical owner and apply the product's Run limits."""
        run = self._resources.run
        self._executor = Executor(external_root=(playground_root / f".scopes-{os.getpid()}" if kubernetes_start_mode() else None))
        try:
            await self._stack.enter_async_context(self._executor)
            scope = self._executor.scope(name="run", attach_host=True,
                memory_bytes=None if run.memory_max_gb is None else run.memory_max_gb * GIB,
                cpu_cores=run.cpu_max_cores, pids=RUN_PIDS_MAX, control_pids=FRAMEWORK_PIDS)
            self._owners[""] = await self._stack.enter_async_context(scope)
        except BaseException:
            await self._stack.aclose()
            raise

    async def claim_level(self, path: str) -> Scope:
        """Map the existing identity path to unlimited structure in the same tree."""
        parent = self._owners[""]
        prefix = []
        for part in path.split("/"):
            prefix.append(part)
            key = "/".join(prefix)
            if key not in self._owners:
                self._owners[key] = await self._stack.enter_async_context(parent.scope(name=part))
            parent = self._owners[key]
        return parent

    async def action_scope(self, path: str, *, memory_max_bytes: int | None,
                           cpu_max_millicores: int | None, gpu_devices: tuple[int, ...]) -> Scope:
        """Place one Action beside its identity's other Actions and children."""
        identity, name = path.rsplit("/", 1)
        parent = await self.claim_level(identity)
        return parent.scope(name=name, memory_bytes=memory_max_bytes,
            cpu_cores=None if cpu_max_millicores is None else cpu_max_millicores / 1000,
            gpu_devices=gpu_devices)

    def sidecar_scope(self, name: str, *, pids: int | None = None) -> Scope:
        """Declare a host-owned sidecar directly under the Run's aggregate."""
        return self._owners[""].scope(name=name, pids=pids)

    def cpu_limits_available(self) -> bool:
        if self._executor is None:
            raise RuntimeError("Run resources have not been acquired")
        return self._executor.cpu_limits_available

    async def release(self) -> None:
        await self._stack.aclose()
        self._executor = None
        self._owners.clear()

    async def select_gpus(
        self,
        count: int,
        remaining_s: Callable[[], float],
        *,
        candidates: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """Choose ``count`` physical cards for one Action about to be recorded.

        Nothing is reserved: the answer is a placement, and a later Action may
        legally be placed on the same card. ``candidates`` defaults to the
        whole run universe; a caller narrows it only as its own scheduling
        policy, and a narrowed pool is still checked against the run's own
        cards, because choosing among fewer must never reach more. An empty
        pool offers nothing and cannot answer a positive count, so it fails
        instead of quietly becoming CPU work."""
        what = f"a {count}-card selection"
        # The host's own answer is read BEFORE the caller's pool: a caller
        # that offers no candidate is making a scheduling choice, not
        # reporting that the machine has no cards, and the two must not end
        # in the same silent CPU placement.
        universe = self._run_cards(what)
        if universe is None:
            return ()
        pool = universe if candidates is None else self.placed_gpus(
            tuple(candidates), what=what
        )
        return await self._gpu.choose(count, pool, remaining_s)

    def _run_cards(self, what: str) -> tuple[int, ...] | None:
        """This run's cards, or None where the host has none to give.

        The one place that answers "can anything be placed at all". A host
        with no reachable card and no configured device list places the work
        un-pinned, exactly as it ran before cards were selected at all, and
        both doors into placement read that from here rather than each
        keeping its own copy of the rule."""
        universe = self._gpu.universe
        if not universe and self._resources.cuda_visible_devices is None:
            logging.info("no GPU visible; placing %s un-pinned", what)
            return None
        return universe

    def check_within_run(
        self,
        *,
        cpu_max_cores: float | None,
        memory_max_gb: float | None,
        what: str,
    ) -> None:
        """Refuse one Action that alone declares more than the whole Run may use.

        The run root is the one aggregate authority, so several Actions may
        declare more than it in total and the kernel throttles them; that is
        deliberate and is not refused here. A SINGLE Action above the run
        ceiling is different: no placement could ever satisfy it, so the call
        that starts it is told now instead of leaving an unplaceable level to
        be discovered later."""
        run = self._resources.run
        if (
            run.cpu_max_cores is not None
            and cpu_max_cores is not None
            and cpu_max_cores > run.cpu_max_cores
        ):
            raise ValueError(
                f"{what} declares {cpu_max_cores} CPU cores, which is more "
                f"than the whole run may use ({run.cpu_max_cores})"
            )
        if (
            run.memory_max_gb is not None
            and memory_max_gb is not None
            and memory_max_gb > run.memory_max_gb
        ):
            raise ValueError(
                f"{what} declares {memory_max_gb} GB of memory, which is more "
                f"than the whole run may use ({run.memory_max_gb})"
            )

    def placed_gpus(self, indices: tuple[int, ...], *, what: str) -> tuple[int, ...]:
        """Check exact cards against the same Run boundary the selector picks from.

        Naming cards skips the placement heuristic, which is the point of
        naming them. It does not skip the operator's device list: a card this
        run may not use is refused here, before any Action records it. On a
        host with no reachable card and no configured list the answer is the
        one the counted path already gives -- un-pinned, on the CPU."""
        universe = self._run_cards(what)
        if universe is None:
            return ()
        outside = [index for index in indices if index not in universe]
        if outside:
            raise ValueError(
                f"{what} names GPU(s) {outside} this run may not use; its "
                f"cards are {list(universe)}"
            )
        return indices

    def gpu_universe(self) -> tuple[int, ...]:
        """Every physical card this run may place work on."""
        return self._gpu.universe

    def host_visible_gpu_count(self) -> int:
        """Return how many cards Clock-bounded startup measured."""
        return self._gpu.host_visible_gpu_count()

    async def probe_host_visible_gpu_count(
        self, remaining_s: Callable[[], float]
    ) -> int:
        """Measure the visible cards within the live Run Budget."""
        return await self._gpu.probe_host_visible_gpu_count(remaining_s)

    def disk_free_bytes(self, path: Path) -> int:
        """Return free bytes on the filesystem that holds this path."""
        return free_bytes(path)
