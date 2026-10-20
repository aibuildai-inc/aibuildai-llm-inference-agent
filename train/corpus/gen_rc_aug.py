"""Step 9 increment: generate ADDITIONAL disjoint reasoning + code slices to
UN-STARVE the two scored domains that the corpus under-represents.

Rationale (step 9): the scored 30-prompt benchmark is EXACTLY equal thirds
chat(0-9) / reasoning(10-19) / code(20-29). The step-5..7 corpus is chat 14000 /
reasoning 3500 / code 3500 -- 67% chat but only 33% chat scored, so reasoning and
code are each scored at ~2x their training share. The lineage repeatedly scaled
CHAT because dev shows chat weakest (accept 0.281), but chat SATURATED (step 7
+1.7%). The scored metric is the equal-weight average of per-domain tau, so the
marginal training token should go where marginal return is highest -- the starved,
responsive reasoning/code domains (dev accept 0.449 / 0.396, dev top1 0.87 / 0.80),
not the saturated chat domain. This script adds fresh reasoning+code supervision.

Reuses prepare.py's build_prompt / generate_continuations / collect so the feature
format is byte-compatible with data/train.pt / train_v2.pt. Writes data/rc_aug.pt;
merge_corpus_v3.py concatenates it onto train_v2.pt -> train_v3.pt.

Disjointness: we EXCLUDE every dev reasoning/code prompt prefix (protects eval.py's
held-out split exactly), and draw from a HIGH shuffle offset (seed 0, same as
prepare) past everything the original 3500-count slices could have used, so train
overlap is negligible. Reasoning (MetaMathQA, ~395k rows) has effectively unlimited
disjoint prompts; code (CodeAlpaca-20k) has ~16k rows left after the original 3500.

    python gen_rc_aug.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# train/, one level up: prepare.py and data/ live there, not in corpus/.
HERE = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(HERE))
from prepare import recipe, build_prompt, generate_continuations, collect  # noqa: E402

# new disjoint slices to add (chat held at 14000; raise the two starved scored domains)
AUG = [
    {"dataset": "meta-math/MetaMathQA", "name": None, "split": "train",
     "instruction_field": "query", "context_field": None, "response_field": "response",
     "domain": "reasoning", "count": 5000, "skip": 6000},
    {"dataset": "sahil2801/CodeAlpaca-20k", "name": None, "split": "train",
     "instruction_field": "instruction", "context_field": "input", "response_field": "output",
     "domain": "code", "count": 5000, "skip": 6000},
]
AUG_SEED = 0   # same permutation basis as prepare; SKIP offset makes the slice disjoint


def dev_prefix_set():
    """Token-id-prefix set of the dev reasoning/code prompts -> hard exclusion."""
    dev = torch.load(HERE / "data" / "dev.pt", weights_only=False)
    s = set()
    for ids, rs, dom in zip(dev["ids"], dev["resp_start"], dev["domain"]):
        if dom in ("reasoning", "code"):
            s.add(tuple(int(x) for x in ids[:int(rs)].tolist()))
    print(f"excluding {len(s)} dev reasoning/code prompt prefixes", flush=True)
    return s


def gather_rc_prompts(cfg, tok, excl):
    from datasets import load_dataset
    data = cfg["data"]
    prompt_max = int(data["prompt_max"])
    out = []
    for spec in AUG:
        rng = random.Random(AUG_SEED)
        ds = load_dataset(spec["dataset"], spec["name"], split=spec["split"])
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        idx = idx[int(spec["skip"]):]          # skip past everything the orig slice used
        kept = 0
        for i in idx:
            if kept >= int(spec["count"]):
                break
            row = ds[i]
            instr = str(row.get(spec["instruction_field"]) or "")
            ctx = str(row.get(spec["context_field"]) or "") if spec.get("context_field") else ""
            resp = str(row.get(spec["response_field"]) or "")
            if not instr.strip() or not resp.strip():
                continue
            pids = build_prompt(tok, instr, ctx, prompt_max)
            if pids is None:
                continue
            if tuple(pids) in excl:            # never a dev prompt
                continue
            out.append((pids, spec["domain"]))
            kept += 1
        print(f"{spec['domain']:9s} {spec['dataset']:28s} kept {kept} (want {spec['count']})", flush=True)
    return out


def main() -> None:
    cfg = recipe()
    models = json.loads((HERE / cfg["models_json"]).resolve().read_text())
    target = models["speed_pair"]["target"]
    tok = AutoTokenizer.from_pretrained(target)
    print(f"loading target {target} (bf16) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(target, torch_dtype=torch.bfloat16)
    model.to("cuda").eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    gen_batch = int(cfg["data"]["gen_batch"])
    gen_max_new = int(cfg["data"]["gen_max_new"])
    max_len = int(cfg["data"]["max_len"])
    tok.padding_side = "left"

    excl = dev_prefix_set()
    prompts = gather_rc_prompts(cfg, tok, excl)
    print(f"== generating {len(prompts)} reasoning+code continuations (greedy) ==", flush=True)
    examples = generate_continuations(model, tok, prompts, gen_batch, gen_max_new, max_len, pad_id)
    print(f"generated {len(examples)} usable sequences; collecting features ...", flush=True)
    bs = int(cfg["data"]["collect_batch"])
    aug = collect(model, examples, bs, pad_id)

    out = HERE / cfg["data"]["prepared"] / "rc_aug.pt"
    torch.save(aug, out)
    from collections import Counter
    print(f"rc_aug: {len(aug['ids'])} seqs / {aug['total_tokens']} tokens "
          f"{Counter(aug['domain'])} -> {out}", flush=True)


if __name__ == "__main__":
    main()
