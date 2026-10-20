"""Typed, bounded GPU facts from a leaf NVML process."""
from __future__ import annotations

import asyncio
import sys
from typing import NamedTuple, cast

import subprocess

from .errors import ExecutionError

class NvmlQueryError(ExecutionError):
    """A bounded NVML process could not answer."""


class GpuStats(NamedTuple):
    """Available memory and utilization for one physical GPU."""
    index: int
    free_mib: int
    util_pct: int | None


class GpuTelemetry(NamedTuple):
    """The canonical product snapshot of one physical GPU."""
    index: int
    name: str
    temperature_c: int | None
    power_w: float | None
    power_cap_w: float | None
    memory_used_mb: int
    memory_total_mb: int
    utilization_pct: int | None
    memory_bw_pct: int | None


def _decode(query: str, returncode: int, stdout: str, stderr: str) -> object:
    if returncode:
        raise NvmlQueryError(
            f"NVML {query} query failed (exit {returncode}): {stderr.strip()}"
        )
    from pydantic import TypeAdapter, ValidationError

    result_types = {
        "minors": dict[int, int],
        "stats": list[GpuStats],
        "telemetry": list[GpuTelemetry],
        "topology": list[tuple[int, int, int]],
    }
    try:
        result = TypeAdapter(result_types[query]).validate_json(stdout, strict=True)
    except ValidationError as error:
        raise NvmlQueryError(f"NVML {query} returned invalid JSON: {error}") from error
    if query != "topology":
        return result
    topology = {frozenset((left, right)): rank for left, right, rank in result}
    if any(len(pair) != 2 or rank not in range(6) for pair, rank in topology.items()):
        raise NvmlQueryError("NVML topology returned an invalid pair or rank")
    if len(topology) != len(result):
        raise NvmlQueryError("NVML topology returned a duplicate pair")
    return topology or None


async def _query_for_run(query: str, timeout_s: float) -> object:
    if timeout_s <= 0:
        raise TimeoutError
    argv = ([sys.executable, "__nvml_query", query] if getattr(sys, "frozen", False)
            else [sys.executable, "-m", "aibuildai_exec._impl.nvml", query])
    task = asyncio.create_task(asyncio.to_thread(
        subprocess.run, argv, capture_output=True, text=True, timeout=timeout_s,
    ))
    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except subprocess.TimeoutExpired:
            pass
        raise
    except subprocess.TimeoutExpired as error:
        raise TimeoutError from error
    return _decode(query, result.returncode, result.stdout, result.stderr)


def run_gpu_query(query: str) -> int:
    """Run the isolated worker; frozen hosts dispatch __nvml_query here."""
    from .nvml_worker import main  # Optional NVML is loaded only in the worker.
    return main(query)



async def query_gpu_stats(timeout_s: float) -> list[GpuStats]:
    """Read bounded GPU memory and utilization facts."""
    return cast(list[GpuStats], await _query_for_run("stats", timeout_s))


async def query_topology(timeout_s: float) -> dict[frozenset[int], int] | None:
    """Read GPU connectivity ranks without choosing devices."""
    return cast(dict[frozenset[int], int] | None, await _query_for_run("topology", timeout_s))



def query_gpu_telemetry(timeout_s: float) -> list[GpuTelemetry]:
    """Read GPU display facts from a bounded worker."""
    return cast(list[GpuTelemetry], asyncio.run(_query_for_run("telemetry", timeout_s)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: nvml.py QUERY")
    raise SystemExit(run_gpu_query(sys.argv[1]))
