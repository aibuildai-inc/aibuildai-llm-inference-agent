"""Quality parity: does the draft head change task scores?

Our accept rule is exact, so this is CONFIRMATORY -- it tests whether the measured
~0.86%/token bf16 divergence has any effect on task accuracy. Run both arms and compare.

  python bench_quality.py --bench gsm8k    --models M --harness H --out O [--draft D]
  python bench_quality.py --bench humaneval --models M --harness H --out O [--draft D]

SECURITY: --bench humaneval EXECUTES MODEL-GENERATED CODE. Each candidate runs in a
separate process, in a throwaway temp dir, under RLIMIT_CPU/AS/NPROC and a wall-clock
timeout. That is meaningful isolation, not a container: do not run it on a host where
arbitrary local code execution would be unacceptable.
"""
from __future__ import annotations
import argparse, json, re, resource, subprocess, sys, tempfile, time
from pathlib import Path


def extract_final_number(text: str) -> str | None:
    """GSM8K scoring: the last number in the generation, matching the #### convention."""
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return m[-1].rstrip(".") if m else None


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def run_humaneval_case(prompt: str, completion: str, test: str, entry: str) -> bool:
    """Execute one candidate in an isolated subprocess. True iff its tests pass."""
    prog = f"{prompt}{completion}\n\n{test}\n\ncheck({entry})\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "cand.py"
        f.write_text(prog)
        try:
            r = subprocess.run([sys.executable, str(f)], cwd=d, timeout=15,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               preexec_fn=_limits)
            return r.returncode == 0
        except Exception:
            return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["gsm8k", "humaneval"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--harness", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args()

    cfg = json.loads(Path(a.models).read_text())
    pair, sg = cfg["speed_pair"], cfg["sglang"]
    DATA = Path(__file__).resolve().parent   # data ships beside this script

    if a.bench == "gsm8k":
        rows = [json.loads(l) for l in (DATA / "gsm8k_test.jsonl").read_text().splitlines() if l.strip()]
        if a.limit: rows = rows[: a.limit]
        prompts = [r["question"] + "\n\nSolve step by step, then give the final numeric "
                   "answer on its own last line." for r in rows]
        golds = [gsm8k_gold(r["answer"]) for r in rows]
    else:
        rows = [json.loads(l) for l in (DATA / "humaneval.jsonl").read_text().splitlines() if l.strip()]
        if a.limit: rows = rows[: a.limit]
        prompts = ["Complete this Python function. Output ONLY the function body "
                   "continuation, no prose, no markdown fences.\n\n" + r["prompt"] for r in rows]
        golds = [None] * len(rows)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pair["target"])
    if a.harness:
        sys.path.insert(0, str(Path(a.harness).resolve()))
        from _toolchain import ensure as _ensure
        _ensure()
    import sglang as sgl

    common = dict(model_path=pair["target"], dtype=pair["dtype"],
                  mem_fraction_static=sg["mem_fraction_static"],
                  max_total_tokens=max(sg["max_total_tokens"], 8192),
                  disable_cuda_graph=sg["disable_cuda_graph"],
                  skip_tokenizer_init=True, log_level="info")
    spec = {}
    if a.draft:
        ov = {}
        f = Path(a.draft) / "_overrides.json"
        if f.is_file(): ov = json.loads(f.read_text())
        spec = dict(speculative_algorithm=sg["speculative_algorithm"],
                    speculative_draft_model_path=a.draft,
                    speculative_num_steps=int(ov.get("speculative_num_steps", sg["speculative_num_steps"])),
                    speculative_eagle_topk=int(ov.get("speculative_eagle_topk", sg["speculative_eagle_topk"])),
                    speculative_num_draft_tokens=int(ov.get("speculative_num_draft_tokens", sg["speculative_num_draft_tokens"])))
    engine = sgl.Engine(**common, **spec)
    sampling = {"temperature": 0.0, "max_new_tokens": a.max_new_tokens}

    def gen(p):
        ids = tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True))["input_ids"]
        r = engine.generate(input_ids=ids, sampling_params=sampling)
        o = r[0] if isinstance(r, list) else r
        out = list(o.get("output_ids") or o.get("token_ids") or [])
        mi = o.get("meta_info", {}) or {}
        return tok.decode(out, skip_special_tokens=True), len(out), int(mi.get("spec_verify_ct") or 0)

    gen("hi")  # warm up

    correct = 0; toks = 0; verify = 0; details = []
    t0 = time.perf_counter()
    for i, p in enumerate(prompts):
        text, n, vct = gen(p)
        toks += n; verify += vct
        if a.bench == "gsm8k":
            got = extract_final_number(text)
            ok = (got is not None and golds[i] is not None
                  and abs(float(got) - float(golds[i])) < 1e-6)
        else:
            body = text
            if "```" in body:  # strip fences if the model added them anyway
                parts = body.split("```")
                body = max(parts, key=len)
                body = re.sub(r"^(python|py)\n", "", body)
            ok = run_humaneval_case(rows[i]["prompt"], body, rows[i]["test"], rows[i]["entry_point"])
        correct += int(ok)
        details.append({"i": i, "ok": bool(ok)})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(prompts)}  running score {correct/(i+1):.4f}", flush=True)
    secs = time.perf_counter() - t0

    n = len(prompts)
    score = correct / n
    se = (score * (1 - score) / n) ** 0.5
    result = {
        "bench": a.bench, "arm": "draft" if a.draft else "plain",
        "n": n, "correct": correct, "score": score,
        "binomial_se": se, "score_ci95": [score - 1.96 * se, score + 1.96 * se],
        "tokens": toks, "seconds": secs, "tokens_per_second": toks / secs,
        "tau": (toks / verify) if verify else 0.0,
        "max_new_tokens": a.max_new_tokens, "temperature": 0.0,
        "engine_args": {**{k: v for k, v in common.items() if k != "model_path"}, **spec},
        "details": details,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("bench","arm","n","correct","score","binomial_se","tau")}, indent=2))
    engine.shutdown()


if __name__ == "__main__":
    main()
