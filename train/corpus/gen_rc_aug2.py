"""Step 10 increment: generate a SECOND fresh disjoint reasoning + code slice to
push the corpus composition further toward the scored equal thirds -- PURELY
ADDITIVELY (chat held at 14000 so no domain regresses).

Rationale (step 10): step 9 proved the composition lever still pays -- raising
reasoning+code from 3500 to 8500 each gained +2.8% harness tau (3.701 -> 3.804) at
fixed geometry+tree. The scored 30-prompt benchmark is exactly equal thirds chat /
reasoning / code, yet train_v3 is still chat 14000 / reasoning 8500 / code 8500
(45/27/27). The marginal training token still belongs in the starved, equally
scored, still-responsive reasoning/code domains. This script adds +3000 reasoning
+ +3000 code -> train_v4 = chat 14000 / reasoning 11500 / code 11500 (~38/31/31),
nearer true equal thirds, chat untouched.

Reuses prepare.py's build_prompt / generate_continuations / collect so the feature
format is byte-compatible with data/train.pt / train_v2.pt / train_v3.pt. Writes
data/rc_aug2.pt; merge_corpus_v4.py concatenates it onto train_v3.pt -> train_v4.pt.

Disjointness: EXCLUDE every dev reasoning/code prompt prefix (protects eval.py's
held-out split), and draw from a HIGHER shuffle offset (seed 0, same as prepare)
than BOTH the original 3500-count slices and the step-9 rc_aug slice (which used
skip=6000, count 5000, i.e. indices up to ~11000). skip=12000 here sits past all of
them. Reasoning (MetaMathQA, ~395k rows) has effectively unlimited disjoint prompts;
code (CodeAlpaca-20k, ~20k rows) has ~5k rows left after index 12000, enough for
count 3000 after filtering.

    python gen_rc_aug2.py
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

# new disjoint slices: push the two starved scored domains further, chat untouched
AUG = [
    {"dataset": "meta-math/MetaMathQA", "name": None, "split": "train",
     "instruction_field": "query", "context_field": None, "response_field": "response",
     "domain": "reasoning", "count": 3000, "skip": 12000},
    {"dataset": "sahil2801/CodeAlpaca-20k", "name": None, "split": "train",
     "instruction_field": "instruction", "context_field": "input", "response_field": "output",
     "domain": "code", "count": 3000, "skip": 12000},
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
        idx = idx[int(spec["skip"]):]          # skip past everything the orig + rc_aug slices used
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

    out = HERE / cfg["data"]["prepared"] / "rc_aug2.pt"
    torch.save(aug, out)
    from collections import Counter
    print(f"rc_aug2: {len(aug['ids'])} seqs / {aug['total_tokens']} tokens "
          f"{Counter(aug['domain'])} -> {out}", flush=True)


if __name__ == "__main__":
    main()
