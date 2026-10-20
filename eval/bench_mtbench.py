"""MT-Bench speedup + acceptance length for a SGLang EAGLE draft head.

Follows the convention the speculative-decoding literature uses, so numbers can sit
beside published ones: all 80 MT-Bench questions, BOTH turns with turn 1 in context,
max_new_tokens 1024, batch 1, one arm per process.

  python bench_mtbench.py --models M.json --questions q.jsonl --out OUT.json
                          [--draft DIR] [--temperature 0] [--limit N]

Omit --draft for the no-speculation arm. Reports overall and per-category.
Prints one JSON object and writes it to --out.
"""
from __future__ import annotations
import argparse, json, statistics, sys, time
from pathlib import Path
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft", default=None, help="omit for the no-speculation arm")
    ap.add_argument("--backend", default="sglang", choices=["sglang", "hf"],
                    help="hf = plain HuggingFace generate(), the weak reference baseline")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (debug)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--harness", default=None,
                    help="task harness dir, for _toolchain (JIT cache + compiler on PATH)")
    a = ap.parse_args()

    cfg = json.loads(Path(a.models).read_text())
    pair, sg = cfg["speed_pair"], cfg["sglang"]
    qs = [json.loads(l) for l in Path(a.questions).read_text().splitlines() if l.strip()]
    if a.limit:
        qs = qs[: a.limit]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pair["target"])

    # The task harness's toolchain setup: ninja + C++ on PATH and a writable JIT cache.
    # Without this, flashinfer compiles into a default location or fails outright, and a
    # cold compile landing inside the timed window is a 2x measurement error.
    if a.harness:
        sys.path.insert(0, str(Path(a.harness).resolve()))
        from _toolchain import ensure as _ensure
        _ensure()
    if a.backend == "hf":
        # Plain HuggingFace generate(): KV-cached greedy, eager mode, no CUDA graphs, no
        # paged attention. Deliberately the weak reference, measured on the SAME protocol
        # so it can be a denominator without mixing workloads.
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            pair["target"], dtype=getattr(torch, pair["dtype"]), device_map="cuda:0")
        model.eval()
        eos = getattr(model.generation_config, "eos_token_id", None)
        stop = [eos] if isinstance(eos, int) else list(eos or [])
        extra = tok.convert_tokens_to_ids(["<|im_end|>"])
        stop += [i for i in extra if isinstance(i, int) and i >= 0 and i not in stop]

        def gen(ids):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    torch.tensor([ids], device="cuda:0"),
                    max_new_tokens=a.max_new_tokens,
                    do_sample=(a.temperature > 0),
                    **({"temperature": a.temperature} if a.temperature > 0 else {}),
                    eos_token_id=stop or None,
                    pad_token_id=(stop[0] if stop else None))
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            new_ids = out[0][len(ids):].tolist()
            return len(new_ids), dt, new_ids, 0

        # Warm up before timing anything: the first generate() pays CUDA context setup and
        # kernel autotuning. Unwarmed, that lands inside the measurement.
        gen(tok(tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                        add_generation_prompt=True))["input_ids"])
        _run(a, qs, tok, gen, {"backend": "hf", "note": "HuggingFace generate(), eager"})
        return

    import sglang as sgl

    common = dict(
        model_path=pair["target"], dtype=pair["dtype"],
        mem_fraction_static=sg["mem_fraction_static"],
        max_total_tokens=max(sg["max_total_tokens"], 8192),  # 1024-token answers x 2 turns
        disable_cuda_graph=sg["disable_cuda_graph"],
        skip_tokenizer_init=True, log_level="info",
    )
    spec = {}
    if a.draft:
        ov = {}
        f = Path(a.draft) / "_overrides.json"
        if f.is_file():
            ov = json.loads(f.read_text())
        spec = dict(
            speculative_algorithm=sg["speculative_algorithm"],
            speculative_draft_model_path=a.draft,
            speculative_num_steps=int(ov.get("speculative_num_steps", sg["speculative_num_steps"])),
            speculative_eagle_topk=int(ov.get("speculative_eagle_topk", sg["speculative_eagle_topk"])),
            speculative_num_draft_tokens=int(ov.get("speculative_num_draft_tokens", sg["speculative_num_draft_tokens"])),
        )
    engine = sgl.Engine(**common, **spec)
    sampling = {"temperature": a.temperature, "max_new_tokens": a.max_new_tokens}

    def gen(ids):
        t0 = time.perf_counter()
        r = engine.generate(input_ids=ids, sampling_params=sampling)
        dt = time.perf_counter() - t0
        o = r[0] if isinstance(r, list) else r
        mi = o.get("meta_info", {}) or {}
        out = list(o.get("output_ids") or o.get("token_ids") or [])
        n = mi.get("completion_tokens") or len(out)
        return int(n), dt, out, int(mi.get("spec_verify_ct") or 0)

    # warm up the engine and its decode kernels
    gen(tok(tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                    add_generation_prompt=True))["input_ids"])

    _run(a, qs, tok, gen, {**{k: v for k, v in common.items() if k != "model_path"}, **spec})
    engine.shutdown()


def _run(a, qs, tok, gen, engine_args) -> None:
    """The protocol itself: both turns, turn 1 in context, per-category aggregation.
    Shared by both backends so the comparison cannot drift."""
    import statistics
    from collections import defaultdict
    from pathlib import Path
    import json
    sweeps = []
    for _ in range(a.repeats):
        per_cat = defaultdict(lambda: {"tokens": 0.0, "seconds": 0.0, "verify": 0})
        tot = {"tokens": 0.0, "seconds": 0.0, "verify": 0}
        records = []
        for q in qs:
            msgs = []
            for turn_i, turn in enumerate(q["turns"], 1):
                msgs.append({"role": "user", "content": turn})
                ids = tok(tok.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True))["input_ids"]
                n, dt, out, vct = gen(ids)
                msgs.append({"role": "assistant",
                             "content": tok.decode(out, skip_special_tokens=True)})
                for d in (per_cat[q["category"]], tot):
                    d["tokens"] += n; d["seconds"] += dt; d["verify"] += vct
                records.append({"question_id": q["question_id"], "category": q["category"],
                                "turn": turn_i, "tokens": n, "seconds": dt,
                                "spec_verify_ct": vct})
        sweeps.append({
            "tokens_per_second": tot["tokens"] / tot["seconds"],
            "tokens": tot["tokens"], "seconds": tot["seconds"], "verify": tot["verify"],
            "tau": (tot["tokens"] / tot["verify"]) if tot["verify"] else 0.0,
            "per_category": {c: {"tokens_per_second": v["tokens"] / v["seconds"],
                                 "tau": (v["tokens"] / v["verify"]) if v["verify"] else 0.0}
                             for c, v in sorted(per_cat.items())},
            "records": records,
        })

    tps = [s["tokens_per_second"] for s in sweeps]
    result = {
        "arm": ("hf-eager" if a.backend == "hf" else ("draft" if a.draft else "plain")),
        "benchmark": "mt-bench",
        "questions": len(qs), "turns_per_question": 2,
        "temperature": a.temperature, "max_new_tokens": a.max_new_tokens,
        # best-of-sweeps: throughput noise is one-sided, a stall only ever slows a sweep
        "tokens_per_second": max(tps),
        "tokens_per_second_median": statistics.median(tps),
        "tokens_per_second_sweeps": tps,
        "tau": sweeps[tps.index(max(tps))]["tau"],
        "per_category": sweeps[tps.index(max(tps))]["per_category"],
        "engine_args": engine_args,
        "sweeps": sweeps,
        "protocol": ("MT-Bench 80 questions x 2 turns, turn 1 in context, batch 1, "
                     "max_new_tokens 1024; one arm per process; best of sweeps"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    brief = {k: result[k] for k in ("arm", "tokens_per_second", "tau", "temperature",
                                   "tokens_per_second_sweeps")}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
