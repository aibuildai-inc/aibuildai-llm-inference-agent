"""Collect EAGLE training data: teacher-force the TARGET over a 4-domain
instruction mix and store its post-final-norm hidden states (the vectors the
target's lm_head consumes) alongside the token ids.

    python prepare.py

Writes <data.prepared>/train.pt and dev.pt, each a dict of parallel lists:
  ids[k]        int32  [L]      the templated conversation's token ids
  feats[k]      fp16   [L, H]   target post-norm hidden state at each position
  resp_start[k] int               first index of the assistant response
  domain[k]     str
The head predicts feats[j+1] from (feats[j], emb(ids[j+1])); see train.py.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent


def recipe() -> dict:
    return yaml.safe_load((HERE / "recipe.yaml").read_text(encoding="utf-8"))


def build_text(tok, instruction: str, context: str, response: str, max_len: int):
    """Return (full_ids, resp_start) or None if there is no usable response room."""
    user = instruction.strip()
    if context and context.strip():
        user = user + "\n\n" + context.strip()
    msgs = [{"role": "user", "content": user}]
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + response.strip() + tok.eos_token
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"][:max_len]
    resp_start = len(prompt_ids)
    if resp_start >= len(full_ids) - 2:       # need at least a couple response tokens
        return None
    return full_ids, resp_start


def build_prompt(tok, instruction: str, context: str, prompt_max: int):
    """Return prompt_ids (templated, add_generation_prompt) truncated to prompt_max, or None."""
    user = instruction.strip()
    if context and context.strip():
        user = user + "\n\n" + context.strip()
    msgs = [{"role": "user", "content": user}]
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > prompt_max:
        return None   # drop over-long prompts rather than truncate the chat template
    if len(prompt_ids) < 4:
        return None
    return prompt_ids


def gather_prompts(cfg, tok):
    """Self-distillation: gather PROMPTS only (gold response discarded). Returns
    (train, dev) lists of (prompt_ids, domain)."""
    data = cfg["data"]
    prompt_max = int(data["prompt_max"])
    rng = random.Random(int(data["seed"]))
    train, dev = [], []
    per_dev = int(data["eval_sequences"]) // len(data["mix"])
    for spec in data["mix"]:
        ds = load_dataset(spec["dataset"], spec["name"], split=spec["split"])
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        want = int(spec["count"]) + per_dev
        kept = []
        for i in idx:
            if len(kept) >= want:
                break
            row = ds[i]
            instr = str(row.get(spec["instruction_field"]) or "")
            ctx = str(row.get(spec["context_field"]) or "") if spec.get("context_field") else ""
            resp = str(row.get(spec["response_field"]) or "")
            if not instr.strip() or not resp.strip():
                continue          # keep the same prompt-usability filter as gold mode
            pids = build_prompt(tok, instr, ctx, prompt_max)
            if pids is None:
                continue
            kept.append((pids, spec["domain"]))
        train.extend(kept[: int(spec["count"])])
        dev.extend(kept[int(spec["count"]):])
        print(f"{spec['domain']:9s} {spec['dataset']:34s} prompts {len(kept)} "
              f"(train {min(len(kept), int(spec['count']))}, dev {max(0, len(kept) - int(spec['count']))})")
    rng.shuffle(train)
    return train, dev


@torch.no_grad()
def generate_continuations(model, tok, prompts, gen_batch, gen_max_new, max_len, pad_id):
    """Greedy-generate the target's own continuation for each prompt, then return
    (full_ids, resp_start, domain) with full_ids = prompt + generation (<= max_len).

    SGLang drafts over exactly this distribution: the target's own argmax trajectory.
    """
    device = next(model.parameters()).device
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tok.eos_token_id
    eos_set = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
    # length-bucket so each batch pads to a similar prompt length
    order = sorted(range(len(prompts)), key=lambda k: len(prompts[k][0]))
    out = []
    done = 0
    for b in range(0, len(order), gen_batch):
        chunk = [prompts[k] for k in order[b : b + gen_batch]]
        lens = [len(p) for p, _ in chunk]
        m = max(lens)
        # LEFT-pad for generation
        ids = torch.full((len(chunk), m), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), m), dtype=torch.long)
        for j, (pids, _) in enumerate(chunk):
            ids[j, m - len(pids):] = torch.tensor(pids, dtype=torch.long)
            attn[j, m - len(pids):] = 1
        ids, attn = ids.to(device), attn.to(device)
        this_new = min(gen_max_new, max_len - m)
        if this_new < 2:
            this_new = 2
        gen = model.generate(input_ids=ids, attention_mask=attn,
                             do_sample=False, num_beams=1,
                             max_new_tokens=this_new, pad_token_id=pad_id)
        new_tokens = gen[:, m:]                      # generated portion only
        for j, (pids, dom) in enumerate(chunk):
            g = new_tokens[j].tolist()
            cut = []
            for t in g:
                cut.append(t)
                if t in eos_set:
                    break                            # keep the eos, stop there
            full_ids = (pids + cut)[:max_len]
            if len(full_ids) < len(pids) + 2:        # need a couple response tokens
                continue
            out.append((full_ids, len(pids), dom))
        done += len(chunk)
        if (b // gen_batch) % 10 == 0:
            tot = sum(len(f) for f, _, _ in out)
            print(f"  generated {done}/{len(order)} prompts, {len(out)} usable, "
                  f"{tot} tokens so far", flush=True)
    return out


def gather_examples(cfg, tok):
    data = cfg["data"]
    max_len = int(data["max_len"])
    rng = random.Random(int(data["seed"]))
    train, dev = [], []
    per_dev = int(data["eval_sequences"]) // len(data["mix"])
    for spec in data["mix"]:
        ds = load_dataset(spec["dataset"], spec["name"], split=spec["split"])
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        want = int(spec["count"]) + per_dev
        kept = []
        for i in idx:
            if len(kept) >= want:
                break
            row = ds[i]
            instr = str(row.get(spec["instruction_field"]) or "")
            ctx = str(row.get(spec["context_field"]) or "") if spec.get("context_field") else ""
            resp = str(row.get(spec["response_field"]) or "")
            if not instr.strip() or not resp.strip():
                continue
            built = build_text(tok, instr, ctx, resp, max_len)
            if built is None:
                continue
            full_ids, resp_start = built
            kept.append((full_ids, resp_start, spec["domain"]))
        # disjoint split: first `count` train, last `per_dev` dev
        train.extend(kept[: int(spec["count"])])
        dev.extend(kept[int(spec["count"]):])
        print(f"{spec['domain']:9s} {spec['dataset']:34s} kept {len(kept)} "
              f"(train {min(len(kept), int(spec['count']))}, dev {max(0, len(kept) - int(spec['count']))})")
    rng.shuffle(train)
    return train, dev


@torch.no_grad()
def collect(model, examples, batch_size, pad_id):
    """Teacher-force in batches, return parallel lists (ids, feats fp16 CPU, resp_start, domain)."""
    device = next(model.parameters()).device
    out_ids, out_feats, out_rs, out_dom = [], [], [], []
    examples = sorted(examples, key=lambda e: len(e[0]))   # length-bucket to cut padding
    total_tokens = 0
    for b in range(0, len(examples), batch_size):
        chunk = examples[b : b + batch_size]
        lens = [len(e[0]) for e in chunk]
        m = max(lens)
        ids = torch.full((len(chunk), m), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), m), dtype=torch.long)
        for j, (fids, _, _) in enumerate(chunk):
            ids[j, : len(fids)] = torch.tensor(fids, dtype=torch.long)
            attn[j, : len(fids)] = 1
        ids, attn = ids.to(device), attn.to(device)
        hs = model.model(input_ids=ids, attention_mask=attn).last_hidden_state  # post-norm
        for j, (fids, rs, dom) in enumerate(chunk):
            L = len(fids)
            out_ids.append(torch.tensor(fids, dtype=torch.int32))
            out_feats.append(hs[j, :L].to(torch.float16).cpu().contiguous())
            out_rs.append(int(rs))
            out_dom.append(dom)
            total_tokens += L
        if (b // batch_size) % 25 == 0:
            print(f"  collected {b + len(chunk)}/{len(examples)} seqs, {total_tokens} tokens")
    return {"ids": out_ids, "feats": out_feats, "resp_start": out_rs, "domain": out_dom,
            "total_tokens": total_tokens}


def main() -> None:
    cfg = recipe()
    models = json.loads((HERE / cfg["models_json"]).resolve().read_text())
    target = models["speed_pair"]["target"]
    tok = AutoTokenizer.from_pretrained(target)
    print(f"loading target {target} (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(target, torch_dtype=torch.bfloat16)
    model.to("cuda").eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    if bool(cfg["data"].get("self_distill", False)):
        gen_batch = int(cfg["data"]["gen_batch"])
        gen_max_new = int(cfg["data"]["gen_max_new"])
        max_len = int(cfg["data"]["max_len"])
        # for batched generation we need left padding
        tok.padding_side = "left"
        train_pr, dev_pr = gather_prompts(cfg, tok)
        print(f"gathered {len(train_pr)} train, {len(dev_pr)} dev prompts; "
              f"generating target continuations (greedy, max_new {gen_max_new}) ...", flush=True)
        print("== generating TRAIN continuations ==", flush=True)
        train_ex = generate_continuations(model, tok, train_pr, gen_batch, gen_max_new, max_len, pad_id)
        print("== generating DEV continuations ==", flush=True)
        dev_ex = generate_continuations(model, tok, dev_pr, gen_batch, gen_max_new, max_len, pad_id)
    else:
        train_ex, dev_ex = gather_examples(cfg, tok)
    print(f"gathered {len(train_ex)} train, {len(dev_ex)} dev sequences; collecting features ...")
    bs = int(cfg["data"]["collect_batch"])
    train = collect(model, train_ex, bs, pad_id)
    dev = collect(model, dev_ex, bs, pad_id)

    out_dir = HERE / cfg["data"]["prepared"]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train, out_dir / "train.pt")
    torch.save(dev, out_dir / "dev.pt")
    print(f"train: {len(train['ids'])} seqs / {train['total_tokens']} tokens -> {out_dir/'train.pt'}")
    print(f"dev:   {len(dev['ids'])} seqs / {dev['total_tokens']} tokens -> {out_dir/'dev.pt'}")


if __name__ == "__main__":
    # This script takes no arguments and starts a multi-hour GPU job as soon as
    # it is called. Without this guard, `prepare.py --help` silently began
    # generating 21,000 teacher continuations.
    if len(sys.argv) > 1:
        sys.exit(__doc__)
    main()
