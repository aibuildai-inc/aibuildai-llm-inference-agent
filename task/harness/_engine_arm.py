"""Run ONE SGLang arm in its own process and write its tok/s to a JSON file.

Separate process per arm, because two 14B engines cannot be resident at once. The
harness runs this twice -- without a draft, then with -- and takes the ratio.

Protocol matches the eager task's benchmark.py so numbers stay interpretable: the frozen
prompt list in file order, the same chat template, greedy, max_new_tokens from
models.json, ONE request at a time, one warmup pass, then `repeats` sweeps;
tok/s = total_tokens / total_seconds, median sweep reported.

  python _engine_arm.py --models M --prompts P --out OUT.json [--draft DIR] [--repeats N]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft", default=None, help="omit for the no-speculation arm")
    ap.add_argument("--repeats", type=int, default=3,
                    help="sweeps per arm. 3 is the floor for a real median: "
                         "statistics.median of TWO values is their mean, so "
                         "--repeats 2 rejects no outlier and one stalled sweep "
                         "moves the reported number by half the stall")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cfg = json.loads(Path(a.models).read_text())
    pair, sg = cfg["speed_pair"], cfg["sglang"]
    max_new = cfg["max_new_tokens"]
    limit = a.limit or cfg["benchmark_prompt_limit"]
    prompts = json.loads(Path(a.prompts).read_text())["prompts"][:limit]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pair["target"])
    ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True),
               return_tensors=None)["input_ids"] for p in prompts]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _toolchain import ensure as _ensure_toolchain
    _ensure_toolchain()          # ninja + C++ on PATH for flashinfer JIT
    import sglang as sgl

    # Pinned identically for both arms. Only `speculative_*` differs.
    common = dict(
        model_path=pair["target"],
        dtype=pair["dtype"],
        mem_fraction_static=sg["mem_fraction_static"],
        max_total_tokens=sg["max_total_tokens"],
        disable_cuda_graph=sg["disable_cuda_graph"],
        skip_tokenizer_init=True,
        log_level="info",          # keep WARNINGs: a silent fallback must be visible
    )
    spec = {}
    if a.draft:
        overrides = json.loads(Path(a.draft).joinpath("_overrides.json").read_text()) \
            if Path(a.draft, "_overrides.json").is_file() else {}
        spec = dict(
            speculative_algorithm=sg["speculative_algorithm"],
            speculative_draft_model_path=a.draft,
            speculative_num_steps=int(overrides.get("speculative_num_steps",
                                                    sg["speculative_num_steps"])),
            speculative_eagle_topk=int(overrides.get("speculative_eagle_topk",
                                                     sg["speculative_eagle_topk"])),
            speculative_num_draft_tokens=int(overrides.get("speculative_num_draft_tokens",
                                                           sg["speculative_num_draft_tokens"])),
        )
    engine = sgl.Engine(**common, **spec)
    sampling = {"temperature": 0.0, "max_new_tokens": max_new}

    def one(pid):
        t0 = time.perf_counter()
        r = engine.generate(input_ids=pid, sampling_params=sampling)
        dt = time.perf_counter() - t0
        o = r[0] if isinstance(r, list) else r
        mi = o.get("meta_info", {}) or {}
        n = mi.get("completion_tokens")
        if n is None:
            n = len(o.get("output_ids") or o.get("token_ids") or [])
        # spec_verify_ct is the number of target forwards SGLang actually ran for this
        # request. Read per request from meta_info rather than scraped from the log:
        # SGLang only logs aggregate spec metrics on a ~10s timer, so a short run emits
        # none and a log-based check cannot tell "disabled" from "too short to log".
        # tokens / verify_ct is tau, the counting statistic.
        return int(n), dt, list(o.get("output_ids") or o.get("token_ids") or []), \
            int(mi.get("spec_verify_ct") or 0)

    one(ids[0][:32])                       # warmup: also triggers graph capture / JIT

    sweeps, streams = [], None
    for s in range(a.repeats):
        tokens = seconds = 0.0
        verify = 0
        out_ids = []
        for pid in ids:
            n, dt, toks, vct = one(pid)
            tokens += n
            seconds += dt
            verify += vct
            out_ids.append(toks)
        sweeps.append({"tokens": tokens, "seconds": seconds,
                       "tokens_per_second": tokens / seconds if seconds else 0.0,
                       "spec_verify_ct": verify,
                       "tau": (tokens / verify) if verify else 0.0})
        if s == 0:
            streams = out_ids            # kept so the gate can compare token streams

    tps = [s["tokens_per_second"] for s in sweeps]
    Path(a.out).write_text(json.dumps({
        "arm": "draft" if a.draft else "plain",
        "tokens_per_second": statistics.median(tps),
        "tokens_per_second_sweeps": tps,
        "sweeps": sweeps,
        "spec_verify_ct": sum(sw["spec_verify_ct"] for sw in sweeps),
        "tau": statistics.median([sw["tau"] for sw in sweeps]),
        "engine_args": {**{k: v for k, v in common.items() if k != "model_path"}, **spec},
        "output_ids": streams,
    }, indent=2) + "\n")
    engine.shutdown()


if __name__ == "__main__":
    main()
