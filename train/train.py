"""Train the EAGLE draft head by multi-step (unroll) feature regression.

    python train.py

EAGLE relation (see head_model.py / qwen2_eagle.py): the head predicts the next
post-norm feature f_{j+1} from the current feature f_j and the embedding of the
realized next token x_{j+1}, at RoPE position j+1. Candidate logits come from the
target's FROZEN lm_head applied to the predicted feature (no final norm).

Single-step loss (STEP<=5):
  loss = SmoothL1(pred_feature, f_{j+1}) + ce_weight * CE(lm_head(pred), x_{j+2})

STEP 6 -- MULTI-STEP UNROLL. At decode SGLang drafts `speculative_num_steps`
tokens deep, feeding the head's OWN predicted feature (not the target's true
feature) at draft depth >= 1. Single-step training only ever shows the head the
TRUE feature as input -> an exposure-bias gap at exactly the depths the tree uses.
Unroll closes it: run the head K times per batch. Iteration t predicts f_{k+1+t}
from (feature, emb, pos) where
  - iteration 0: feature = true f_k
  - iteration t>0: feature = the head's OWN prediction from iteration t-1
    (detached; the head learns to be robust to its own imperfect features),
  - emb = embedding of the TRUE token x_{k+1+t} (greedy acceptance means the
    accepted token chain equals the target's), position = k+1+t (RoPE is
    relative, so a uniform +t shift leaves the head's internal attention intact).
The per-iteration losses are combined with decaying weights gamma^t (normalised
to sum 1, so the overall loss scale -- and thus the tuned lr -- is unchanged):
flat unroll weights were a prior-run null, so step 0 stays dominant while deeper
steps add the deep-acceptance supervision tau>2 comes from.

unroll_steps=1 reproduces the exact single-step objective of steps<=5.
Writes a plain state dict to <train.output_dir>/head.pt plus head_config.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoConfig, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent

import os
import pathlib
import sys
sys.path.insert(0, str(HERE))
from head_model import EagleHead  # noqa: E402

try:
    import aibuildai
except Exception:  # noqa: BLE001
    aibuildai = None


def recipe() -> dict:
    return yaml.safe_load((HERE / "recipe.yaml").read_text(encoding="utf-8"))


def geometry(target: str) -> dict:
    sys.path.insert(
        0,
        os.environ.get("SPECDEC_HARNESS")
        or str(pathlib.Path(__file__).resolve().parent.parent / "task" / "harness"),
    )
    from target_geometry import geometry as geo
    return geo(AutoConfig.from_pretrained(target))


def make_batches(data, batch_size, shuffle_rng=None):
    n = len(data["ids"])
    order = list(range(n))
    if shuffle_rng is not None:
        shuffle_rng.shuffle(order)
    for b in range(0, n, batch_size):
        yield [order[b + k] for k in range(min(batch_size, n - b))]


def build_batch(data, idxs, hidden, device, K):
    """Assemble one padded training batch supporting K unroll iterations.

    Slot k of a length-L sequence feeds f_k at iteration 0; at iteration t it
    predicts f_{k+1+t}. We precompute, for each t in 0..K-1, the shifted token
    embeddings (emb of x_{k+1+t}), positions (k+1+t), regression targets
    (f_{k+1+t}), CE targets (x_{k+2+t}) and the loss masks. The iteration-0
    feature input is the true feature; iteration-t (t>0) feature input is the
    head's own prediction from t-1 (supplied in the training loop).
    """
    seqs = idxs
    slot_lens = [len(data["ids"][i]) - 1 for i in seqs]      # L-1 input slots
    S = max(slot_lens)
    B = len(seqs)
    maxL = S + 1
    # padded full per-sequence feats [B, maxL+K, H] and ids [B, maxL+K+1]
    feats_full = torch.zeros(B, maxL + K, hidden, dtype=torch.float16)
    ids_full = torch.zeros(B, maxL + K + 1, dtype=torch.long)
    lens = torch.tensor(slot_lens, dtype=torch.long)         # = L-1 per seq
    resp = torch.zeros(B, dtype=torch.long)
    for j, i in enumerate(seqs):
        ids = data["ids"][i].long()
        feats = data["feats"][i]                             # [L,H] fp16
        L = ids.shape[0]
        feats_full[j, :L] = feats
        ids_full[j, :L] = ids
        resp[j] = data["resp_start"][i]

    # iteration-0 feature input: f_0..f_{L-2}
    feat_in0 = feats_full[:, :S].contiguous()                # [B,S,H]

    k_idx = torch.arange(S)                                  # slot index
    emb_list, pos_list, tgt_list, ce_list, rmask_list = [], [], [], [], []
    for t in range(K):
        # target f_{k+1+t}; valid when k+1+t <= L-1  <=>  k <= lens_j - 1 - t
        tgt_t = feats_full[:, 1 + t:1 + t + S].contiguous()          # [B,S,H]
        emb_t = ids_full[:, 1 + t:1 + t + S].contiguous()            # [B,S] token ids
        ce_t = ids_full[:, 2 + t:2 + t + S].clone()                  # [B,S] token ids
        pos_t = (k_idx + 1 + t).unsqueeze(0).expand(B, S).contiguous()
        # regression valid: target in-range AND predicted pos (k+1+t) on response
        reg_ok = (k_idx.unsqueeze(0) <= (lens.unsqueeze(1) - 1 - t)) & \
                 ((k_idx.unsqueeze(0) + 1 + t) >= resp.unsqueeze(1))
        # CE valid: its target x_{k+2+t} in-range (k <= lens_j - 2 - t) AND reg_ok
        ce_ok = reg_ok & (k_idx.unsqueeze(0) <= (lens.unsqueeze(1) - 2 - t))
        ce_t[~ce_ok] = -100
        emb_list.append(emb_t.to(device))
        pos_list.append(pos_t.to(device))
        tgt_list.append(tgt_t.to(device))
        ce_list.append(ce_t.to(device))
        rmask_list.append(reg_ok.to(device))

    # additive attention bias [B,1,S,S]: causal AND key slot valid (< L-1)
    key_valid = (k_idx.unsqueeze(0) < lens.unsqueeze(1))              # [B,S]
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    bias = torch.zeros(B, 1, S, S, dtype=torch.float32)
    bias.masked_fill_(causal[None, None], -1e9)
    bias.masked_fill_(~key_valid[:, None, None, :], -1e9)
    return (feat_in0.to(device), emb_list, pos_list, tgt_list, ce_list,
            rmask_list, bias.to(device))


def main() -> None:
    cfg = recipe()
    tcfg = cfg["train"]
    models = json.loads((HERE / cfg["models_json"]).resolve().read_text())
    target = models["speed_pair"]["target"]
    geo = geometry(target)
    hidden = geo["hidden"]
    device = "cuda"

    K = int(tcfg.get("unroll_steps", 1))
    gamma = float(tcfg.get("unroll_decay", 0.8))
    unroll_detach = bool(tcfg.get("unroll_detach", True))
    # decaying, normalised step weights (sum=1 so loss scale == single-step)
    raw_w = [gamma ** t for t in range(K)]
    sw = [w / sum(raw_w) for w in raw_w]
    print(f"unroll: K={K} decay={gamma} detach={unroll_detach} weights={[round(x,4) for x in sw]}")

    print(f"loading target {target} for embed_tokens + lm_head (frozen) ...")
    tgt = AutoModelForCausalLM.from_pretrained(target, torch_dtype=torch.bfloat16)
    tgt.to(device).eval()
    for p in tgt.parameters():
        p.requires_grad_(False)
    embed = tgt.get_input_embeddings()
    lm_head = tgt.get_output_embeddings()

    head = EagleHead(
        hidden=hidden, inter=int(cfg["head"]["inter"]),
        n_heads=geo["n_heads"], kv_heads=geo["kv_heads"], head_dim=geo["head_dim"],
        n_layers=int(cfg["head"]["n_layers"]), theta=geo["rope_theta"],
        eps=geo["rms_norm_eps"], fc_bias=bool(cfg["head"]["fc_bias"]),
    ).to(device)
    head.train()
    nparams = sum(p.numel() for p in head.parameters())
    print(f"head: {nparams/1e6:.1f}M params, {cfg['head']['n_layers']} layer(s), "
          f"inter {cfg['head']['inter']}, kv {geo['kv_heads']}, head_dim {geo['head_dim']}")

    train_file = cfg["data"].get("train_file", "train.pt")
    data = torch.load(HERE / cfg["data"]["prepared"] / train_file, weights_only=False)
    print(f"train data ({train_file}): {len(data['ids'])} seqs / {data['total_tokens']} tokens")

    bs = int(tcfg["batch_size"])
    accum = int(tcfg["grad_accum"])
    epochs = int(tcfg["epochs"])
    steps_per_epoch = math.ceil(len(data["ids"]) / bs)
    total_micro = steps_per_epoch * epochs
    total_opt = math.ceil(total_micro / accum)
    warmup = max(1, int(tcfg["warmup_frac"] * total_opt))
    beta = float(tcfg["smoothl1_beta"])
    ce_weight = float(tcfg["ce_weight"])

    opt = torch.optim.AdamW(head.parameters(), lr=float(tcfg["lr"]),
                            weight_decay=float(tcfg["weight_decay"]), betas=(0.9, 0.95))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_opt - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    out = HERE / tcfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    def save_head(tag: str = "") -> None:
        sd = {k: v.to(torch.float16).cpu() for k, v in head.state_dict().items()}
        torch.save(sd, out / "head.pt")
        head_cfg = {
            "n_layers": int(cfg["head"]["n_layers"]),
            "inter": int(cfg["head"]["inter"]),
            "hidden": hidden, "n_heads": geo["n_heads"], "kv_heads": geo["kv_heads"],
            "head_dim": geo["head_dim"], "rope_theta": geo["rope_theta"],
            "sglang_overrides": cfg.get("sglang_overrides", {}),
        }
        (out / "head_config.json").write_text(json.dumps(head_cfg, indent=2) + "\n")
        if tag:
            print(f"[checkpoint {tag}] saved head -> {out/'head.pt'}", flush=True)

    ckpt_every = int(tcfg.get("ckpt_every_opt", 2000))

    if aibuildai is not None:
        aibuildai.init()

    import random
    rng = random.Random(0)
    gstep = 0
    opt_step = 0
    opt.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        run_reg = [0.0] * K
        run_ce = run_acc = seen = 0.0
        for idxs in make_batches(data, bs, rng):
            feat_in0, emb_list, pos_list, tgt_list, ce_list, rmask_list, bias = \
                build_batch(data, idxs, hidden, device, K)

            loss = 0.0
            cur_feat = feat_in0.float()                      # iteration-0 feature input
            step_reg = [0.0] * K
            ce_acc_val = acc_val = 0.0
            for t in range(K):
                with torch.no_grad():
                    tok_emb = embed(emb_list[t])             # bf16
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = head(cur_feat, tok_emb.float(), pos_list[t], bias)  # [B,S,H]
                    rmask = rmask_list[t]
                    pr = pred[rmask]
                    ce_sel = ce_list[t][rmask]
                    valid = ce_sel != -100
                    logits = lm_head(pr[valid]) if valid.any() else None
                # losses in fp32 for stability
                reg = F.smooth_l1_loss(pr.float(), tgt_list[t][rmask].float(), beta=beta)
                if logits is not None:
                    ce = F.cross_entropy(logits.float(), ce_sel[valid])
                    if t == 0:
                        acc_val = (logits.argmax(-1) == ce_sel[valid]).float().mean().item()
                        ce_acc_val = ce.item()
                else:
                    ce = torch.zeros((), device=device)
                loss = loss + sw[t] * (reg + ce_weight * ce)
                step_reg[t] = reg.item()
                # feed the head's OWN prediction as next-step feature; detached by
                # default (robustness-to-own-features view). unroll_detach=False keeps
                # the graph so gradients flow through the full recursion (end-to-end).
                if t + 1 < K:
                    cur_feat = pred.detach().float() if unroll_detach else pred.float()

            (loss / accum).backward()
            gstep += 1
            if gstep % accum == 0:
                for g in opt.param_groups:
                    g["lr"] = float(tcfg["lr"]) * lr_at(opt_step)
                torch.nn.utils.clip_grad_norm_(head.parameters(), float(tcfg["max_grad_norm"]))
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % ckpt_every == 0:
                    save_head(f"opt{opt_step}")

            for t in range(K):
                run_reg[t] += step_reg[t]
            run_ce += ce_acc_val; run_acc += acc_val; seen += 1
            if int(gstep) % int(tcfg["log_every"]) == 0:
                m = {"epoch": epoch, "micro_step": gstep,
                     "reg0": round(run_reg[0] / seen, 5),
                     "reg_deep": [round(run_reg[t] / seen, 5) for t in range(K)],
                     "ce_loss": round(run_ce / seen, 5),
                     "top1_acc": round(run_acc / seen, 5),
                     "lr": round(opt.param_groups[0]["lr"], 7)}
                print(json.dumps(m))
                if aibuildai is not None:
                    log = {"train_reg_loss": run_reg[0] / seen,
                           "train_ce_loss": run_ce / seen,
                           "train_top1_acc": run_acc / seen,
                           "lr": opt.param_groups[0]["lr"]}
                    for t in range(K):
                        log[f"train_reg_step{t}"] = run_reg[t] / seen
                    aibuildai.log(log)
                run_reg = [0.0] * K
                run_ce = run_acc = seen = 0.0
        print(f"== epoch {epoch} done ==")

    if aibuildai is not None:
        aibuildai.finish()

    save_head("final")
    print(f"saved head -> {out/'head.pt'} ({nparams/1e6:.1f}M params)")


if __name__ == "__main__":
    main()
