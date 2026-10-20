"""NOT the gate for this task -- kept only to debug a head in isolation.

Under this task SGLang owns verification, so exactness is not the submission's property
and `harness/benchmark.py` does not call this. It remains because an fp32 eager check is
still the only way to inspect a head's own math on this host: SGLang cannot execute fp32
at all (no Float dispatch in its attention or fused rmsnorm).

Decide whether a submission is lossless, in fp32, at temperature 0.

The reference is this harness's own uncached argmax loop over the gate pair's
target weights. A submission passes only when its greedy output is
token-identical to that reference on every prompt.

Why fp32: in bf16 a KV-cached step and a batched verify forward round near-tied
logits differently, so a bf16 mismatch proves nothing. At a divergence this
prints the reference's top-2 logit gap; a gap at or below ~2 ulp is rounding,
not a broken accept rule.

Prints one JSON object on stdout and ALWAYS exits 0 when the gate ran to
completion. Read "passed" from the JSON; do not read the exit code as a verdict.

  python gate_eager_reference.py --submission DIR [--device cuda:0] [--json OUT]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = HERE.parent


def reference_greedy(model, input_ids, max_new_tokens, stop_ids, device):
    """Uncached argmax loop in fp32: the oracle. Returns (tokens, top2_gaps)."""
    ids = list(input_ids)
    produced: list[int] = []
    gaps: list[float] = []
    for _ in range(max_new_tokens):
        tensor = torch.tensor([ids], device=device)
        logits = model(tensor).logits[0, -1].float()
        top2 = torch.topk(logits, 2)
        gaps.append(float(top2.values[0] - top2.values[1]))
        nxt = int(top2.indices[0])
        produced.append(nxt)
        ids.append(nxt)
        if nxt in stop_ids:
            break
    return produced, gaps


def run(args: argparse.Namespace) -> dict:
    config = common.read_json(args.models)
    pair = config["gate_pair"]
    max_new = args.max_new_tokens or min(64, config["max_new_tokens"])
    limit = args.limit or config["gate_prompt_limit"]
    prompts = common.load_prompts(args.prompts, limit)

    tokenizer = common.load_tokenizer(pair["target"])

    # PHASE 1 - build the reference, then free it. At full precision one model can
    # fill most of a card, so the reference and the submission are never resident
    # together: collect the oracle's answers first, release, then load the method.
    target = common.load_model(pair["target"], pair["dtype"], args.device)
    stop_ids = common.stop_token_ids(target, tokenizer)
    encoded = [common.encode_prompt(tokenizer, p) for p in prompts]
    references, gaps = [], []
    with torch.no_grad():
        for ids in encoded:
            produced, gap = reference_greedy(target, ids, max_new, stop_ids, args.device)
            references.append(produced)
            gaps.append(gap)
    del target
    gc.collect()
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()

    # PHASE 2 - the submission, compared against what the oracle already said.
    method_cls = common.load_submission(args.submission)
    cfg = common.method_config(pair, args.device, args.submission, stop_ids)
    method = method_cls(cfg)

    cases = []
    passed = True
    with torch.no_grad():
        for index, ids in enumerate(encoded):
            produced = method.generate(ids, max_new, 0.0)["token_ids"]
            reference = references[index]
            # The spec path may overshoot max_new_tokens; compare the overlap.
            span = min(len(reference), len(produced))
            first_bad = None
            for position in range(span):
                if reference[position] != produced[position]:
                    first_bad = position
                    break
            ok = first_bad is None and span > 0
            passed = passed and ok
            case = {"prompt_index": index, "compared_tokens": span, "identical": ok}
            if first_bad is not None:
                case["first_divergence"] = first_bad
                case["reference_top2_gap_at_divergence"] = gaps[index][first_bad]
                case["reference_token"] = reference[first_bad]
                case["submission_token"] = produced[first_bad]
            cases.append(case)

    return {
        "passed": passed,
        "prompts": len(prompts),
        "identical_prompts": sum(1 for c in cases if c["identical"]),
        "cases": cases,
        "target": pair["target"],
        "draft": pair.get("draft"),
        "dtype": pair["dtype"],
        "device": args.device,
        "max_new_tokens": max_new,
        "protocol": "two phases: reference collected and freed before the submission loads",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", required=True, help="directory holding submission.py")
    parser.add_argument("--models", default=str(TASK / "models.json"))
    parser.add_argument("--prompts", default=str(TASK / "prompts.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(result, indent=2)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
