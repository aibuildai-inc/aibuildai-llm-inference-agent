"""Step 7 increment: generate an ADDITIONAL chat slice from a SECOND, disjoint
chat/instruction dataset (yahma/alpaca-cleaned) and collect its target features,
WITHOUT regenerating the existing reasoning/code/dolly-chat corpus.

Rationale: the scored 30-prompt benchmark is equal thirds chat/reasoning/code, and
chat is the weakest domain (dev accept 0.293 vs 0.5+). Chat training data is pinned
at dolly's 8000-row single-dataset ceiling. alpaca-cleaned supplies disjoint
explanatory instructions (60% have empty input -> pure "Explain/Describe/Name..."
prompts that match the benchmark chat style). We filter to those empty-input rows.

Reuses prepare.py's tested build_prompt / generate_continuations / collect so the
feature format is byte-compatible with the existing data/train.pt. Writes the new
slice to data/chat_aug.pt; merge_corpus.py concatenates it onto train.pt -> train_v2.pt.

    python gen_chat_aug.py
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

# --- the new chat slice -----------------------------------------------------
AUG_DATASET = "yahma/alpaca-cleaned"
AUG_SPLIT = "train"
AUG_COUNT = 6000          # new chat prompts to add (8000 dolly -> 14000 chat total)
AUG_DOMAIN = "chat"
AUG_SEED = 7              # distinct from prepare's seed 0; disjoint dataset anyway


def gather_alpaca_chat_prompts(cfg, tok):
    """Empty-input (pure-instruction) alpaca rows, templated like prepare.gather_prompts."""
    from datasets import load_dataset
    data = cfg["data"]
    prompt_max = int(data["prompt_max"])
    rng = random.Random(AUG_SEED)
    ds = load_dataset(AUG_DATASET, None, split=AUG_SPLIT)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    kept = []
    for i in idx:
        if len(kept) >= AUG_COUNT:
            break
        row = ds[i]
        instr = str(row.get("instruction") or "")
        ctx = str(row.get("input") or "")
        resp = str(row.get("output") or "")
        if ctx.strip():
            continue                      # keep only pure-instruction (chat-like) rows
        if not instr.strip() or not resp.strip():
            continue
        pids = build_prompt(tok, instr, "", prompt_max)
        if pids is None:
            continue
        kept.append((pids, AUG_DOMAIN))
    print(f"alpaca-cleaned: kept {len(kept)} empty-input chat prompts "
          f"(requested {AUG_COUNT})", flush=True)
    return kept


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

    prompts = gather_alpaca_chat_prompts(cfg, tok)
    print("== generating alpaca chat continuations (greedy) ==", flush=True)
    examples = generate_continuations(model, tok, prompts, gen_batch, gen_max_new, max_len, pad_id)
    print(f"generated {len(examples)} usable chat sequences; collecting features ...", flush=True)
    bs = int(cfg["data"]["collect_batch"])
    aug = collect(model, examples, bs, pad_id)

    out = HERE / cfg["data"]["prepared"] / "chat_aug.pt"
    torch.save(aug, out)
    print(f"chat_aug: {len(aug['ids'])} seqs / {aug['total_tokens']} tokens -> {out}", flush=True)


if __name__ == "__main__":
    main()
