"""REFERENCE scorer -- the one the setup agent wrote for the run that produced
the published head. It is NOT loaded by a re-run.

A run writes its OWN private/score.py: the setup agent implements the contract in
task/README.md from scratch, so a fresh search on a new machine will have a
scorer of its own making. That is by design, and it is also the single largest
source of drift between two runs of "the same" search.

Keep this file for two purposes:
  1. diff a re-run's private/score.py against it, to see what the new agent
     decided differently;
  2. drop it in deliberately (copy to <run>/private/score.py after SETUP) if you
     want the scoring held fixed while something else varies.

Original docstring follows.

Score a speculative-decoding draft-head submission using the task's own harness.

Thin wrapper: run harness/check_draft.py as the cheap gate (minutes), and only if it
passes, run harness/benchmark.py (the authoritative measurement, ~40 min) and return its
'speedup'. Both scripts print one JSON object and their exit code is NOT the verdict --
only their JSON's "passed" / "gate_passed" fields are. See harness/benchmark.py and
harness/check_draft.py docstrings, and README.md's "Measurement notes".

Scoring path contract (see public/README.md): `output_dir` may be either the leaf
directory that directly holds head.pt + head_config.json, or its parent with a `full/`
subdirectory holding them. Both are resolved; the leaf is preferred when both exist.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TASK = Path(os.environ.get("SPECDEC_TASK")
            or Path(__file__).resolve().parent / "task")
PROGRAM_PYTHON = (
    "/program-environment/env/bin/python"
)

# speed_pair.min_free_vram_gb in models.json: one bf16 14B target + draft + KV budget.
MIN_FREE_VRAM_GB = 60
VRAM_WAIT_TIMEOUT_S = 1800
VRAM_WAIT_POLL_S = 20
CHECK_TIMEOUT_S = 1200        # check_draft.py: documented "minutes"
BENCH_TIMEOUT_S = 5400        # benchmark.py: documented "~40 min", generous margin

_DEFAULT_NUM_DRAFT_TOKENS = int(
    json.loads((TASK / "models.json").read_text())["sglang"]["speculative_num_draft_tokens"]
)


def _alpha(tau: float, num_draft_tokens: int) -> float:
    """accepted / evaluated draft tokens, derived from tau (tokens emitted per target
    forward, the harness's own measured statistic) and the tree size actually used.

    This harness reports tau, not accepted/evaluated draft-token counts directly, but
    under greedy decoding each cycle's full fixed-size tree is always evaluated and
    exactly one token (the target's own sample) is emitted beyond whatever draft tokens
    were accepted -- see skills/specdec-methods -- so accepted = tau - 1 and evaluated =
    num_draft_tokens per cycle. This is a real derivation from measured values, not a
    fabricated one; it must stay finite and JSON-serialisable, so it is clamped at 0.
    """
    if num_draft_tokens <= 0:
        return 0.0
    return max(0.0, (tau - 1.0) / num_draft_tokens)


def _resolve_submission(output_dir: str) -> Path:
    out = Path(output_dir)
    nested = out / "full"
    sub = nested if (nested / "head.pt").is_file() else out
    if not (sub / "head.pt").is_file():
        raise FileNotFoundError(
            f"head.pt not found in {sub} (also checked {nested}); "
            f"output_dir={output_dir} must contain head.pt either directly or under full/"
        )
    return sub


def _select_device_env() -> dict:
    """Build the subprocess env for exactly one GPU.

    If the caller already pinned a card via CUDA_VISIBLE_DEVICES, leave it alone -- it
    is already the one device the child should see, and re-deriving an index from
    nvidia-smi or re-writing the variable would silently point at a different physical
    card (see skills/gpu-shared-host). Only when unpinned (nothing set) do we pick a
    device ourselves, among the devices actually visible to this process, by free
    memory -- never by a physical/nvidia-smi index.
    """
    env = os.environ.copy()
    if env.get("CUDA_VISIBLE_DEVICES", "").strip():
        return env
    import torch

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no CUDA device visible to score.py; cannot benchmark on GPU")
    idx = max(range(n), key=lambda i: torch.cuda.mem_get_info(i)[0])
    env["CUDA_VISIBLE_DEVICES"] = str(idx)
    return env


def _wait_for_free_vram(env: dict, need_gb: float) -> None:
    """Another tenant can fill this card after we picked it; wait rather than die."""
    import torch

    want = need_gb * (1024 ** 3)
    # Respect whatever device selection is already encoded in env (pinned or chosen
    # above): with CUDA_VISIBLE_DEVICES set, device 0 in *our* process is the same card
    # the subprocess will see.
    restore = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", restore or "")
    try:
        t0 = time.monotonic()
        while True:
            free, _total = torch.cuda.mem_get_info(0)
            if free >= want:
                return
            if time.monotonic() - t0 > VRAM_WAIT_TIMEOUT_S:
                raise RuntimeError(
                    f"only {free / 1e9:.1f} GB free on the selected GPU after waiting "
                    f"{VRAM_WAIT_TIMEOUT_S}s, need {need_gb:.1f} GB"
                )
            time.sleep(VRAM_WAIT_POLL_S)
    finally:
        if restore is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = restore


def _run_json(script: str, args: list[str], env: dict, timeout_s: int, tmp: Path) -> dict:
    out_json = tmp / f"{script}.json"
    cmd = [PROGRAM_PYTHON, str(TASK / "harness" / script), *args, "--json", str(out_json)]
    try:
        proc = subprocess.run(
            cmd, cwd=str(TASK), env=env, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        tail = ((exc.stdout or "") + (exc.stderr or ""))[-4000:]
        raise RuntimeError(f"{script} timed out after {timeout_s}s:\n{tail}") from exc
    if not out_json.is_file():
        tail = ((proc.stdout or "") + (proc.stderr or ""))[-4000:]
        raise RuntimeError(
            f"{script} wrote no JSON (exit {proc.returncode}); its own exit code is not "
            f"the verdict, but it must still print a result -- stdout/stderr tail:\n{tail}"
        )
    return json.loads(out_json.read_text())


def score(output_dir: str) -> dict:
    submission = _resolve_submission(output_dir)
    env = _select_device_env()
    _wait_for_free_vram(env, MIN_FREE_VRAM_GB)

    with tempfile.TemporaryDirectory(prefix="specdec_score_") as tmp_s:
        tmp = Path(tmp_s)

        gate = _run_json("check_draft.py", ["--submission", str(submission)], env,
                          CHECK_TIMEOUT_S, tmp)
        if not gate.get("passed", False):
            # check_draft.py never writes an _overrides.json for the draft it packages
            # (see harness/check_draft.py), so this short run always used models.json's
            # default tree size regardless of the submission's own sglang_overrides.
            tau0 = float(gate.get("tau", 0.0))
            return {
                "score": 0.0,
                "components": {
                    "gate_passed": 0.0,
                    "method_tokens_per_second": 0.0,
                    "baseline_tokens_per_second": 0.0,
                    "alpha": _alpha(tau0, _DEFAULT_NUM_DRAFT_TOKENS),
                    "tau": tau0,
                },
            }

        _wait_for_free_vram(env, MIN_FREE_VRAM_GB)
        bench = _run_json("benchmark.py", ["--submission", str(submission)], env,
                           BENCH_TIMEOUT_S, tmp)

    gate_passed = bool(bench.get("gate_passed", False))
    speedup = float(bench.get("speedup", 0.0)) if gate_passed else 0.0
    tau = float(bench.get("tau", 0.0))
    # The actual tree size used for this run, honouring the submission's own
    # sglang_overrides if it set one (see harness/benchmark.py's engine_args_draft).
    num_draft_tokens = int(
        bench.get("engine_args_draft", {}).get(
            "speculative_num_draft_tokens", _DEFAULT_NUM_DRAFT_TOKENS
        )
    )
    return {
        "score": speedup,
        "components": {
            "gate_passed": 1.0 if gate_passed else 0.0,
            "method_tokens_per_second": float(bench.get("method_tokens_per_second", 0.0)),
            "baseline_tokens_per_second": float(bench.get("baseline_tokens_per_second", 0.0)),
            "alpha": _alpha(tau, num_draft_tokens) if gate_passed else 0.0,
            "tau": tau,
        },
    }


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1]), indent=2))
