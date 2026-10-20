"""Development evaluation of the trained head on the held-out dev features.

    python eval.py [head_dir]

Advisory only -- never the task score (that is SGLang tok/s, measured by the
harness). Two metrics, both per-domain so later steps can rebalance composition:

  top1   single-step next-token accuracy: argmax lm_head(head(f_j, emb(x_{j+1})))
         == x_{j+2}, on response positions.
  accept greedy chain-acceptance length: starting from a true feature/token, run
         the head autoregressively (feeding its own predicted feature and its own
         argmax token) and count consecutive tokens that match the target's, up to
         K. This is the offline analogue of tau and the better ranking signal.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from head_model import EagleHead  # noqa: E402

CHAIN_K = 6           # max chain length probed (>= deepest tree depth of interest)
MAX_STARTS = 4000     # cap chain starts for speed


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


@torch.no_grad()
def main() -> None:
    cfg = recipe()
    models = json.loads((HERE / cfg["models_json"]).resolve().read_text())
    target = models["speed_pair"]["target"]
    geo = geometry(target)
    hidden = geo["hidden"]
    device = "cuda"

    head_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / cfg["train"]["output_dir"]
    hc = json.loads((head_dir / "head_config.json").read_text())
    head = EagleHead(hidden=hidden, inter=int(hc["inter"]), n_heads=geo["n_heads"],
                     kv_heads=geo["kv_heads"], head_dim=geo["head_dim"],
                     n_layers=int(hc["n_layers"]), theta=geo["rope_theta"],
                     eps=geo["rms_norm_eps"], fc_bias=("fc.bias" in torch.load(
                         head_dir / "head.pt", map_location="cpu", weights_only=True)))
    sd = torch.load(head_dir / "head.pt", map_location="cpu", weights_only=True)
    head.load_state_dict(sd, strict=True)
    head.to(device).eval().to(torch.bfloat16)

    tgt = AutoModelForCausalLM.from_pretrained(target, torch_dtype=torch.bfloat16)
    tgt.to(device).eval()
    embed = tgt.get_input_embeddings()
    lm_head = tgt.get_output_embeddings()

    dev = torch.load(HERE / cfg["data"]["prepared"] / "dev.pt", weights_only=False)

    top1_hit = defaultdict(int); top1_n = defaultdict(int)
    accept_sum = defaultdict(float); accept_n = defaultdict(int)
    starts_used = 0

    for i in range(len(dev["ids"])):
        ids = dev["ids"][i].long().to(device)
        feats = dev["feats"][i].to(device).to(torch.bfloat16)      # [L,H]
        L = ids.shape[0]
        rs = dev["resp_start"][i]
        dom = dev["domain"][i]
        if L < rs + 3:
            continue

        # single-step top1 over all response slots, in one parallel head forward
        slots = L - 1
        feat_in = feats[:slots].unsqueeze(0)                       # f_0..f_{L-2}
        emb_in = embed(ids[1:L]).unsqueeze(0).to(torch.bfloat16)   # x_1..x_{L-1}
        pos = torch.arange(1, L, device=device).unsqueeze(0)
        pred = head(feat_in, emb_in, pos)                          # [1,slots,H]
        pred_pos = torch.arange(1, L, device=device)
        ce_slots = slots - 1
        if ce_slots >= 1:
            resp = pred_pos[:ce_slots] >= rs
            if resp.any():
                logits = lm_head(pred[0, :ce_slots][resp].to(torch.bfloat16))
                tgt_tok = ids[2:L][resp]
                top1_hit[dom] += int((logits.argmax(-1) == tgt_tok).sum().item())
                top1_n[dom] += int(resp.sum().item())

        # greedy chain acceptance from a few response start positions
        for j in range(rs, L - 1):
            if starts_used >= MAX_STARTS:
                break
            cur_feat = feats[j].view(1, 1, hidden)                 # true f_j
            next_tok = ids[j + 1].view(1)                          # true x_{j+1}
            accepted = 0
            p = j + 1
            for _ in range(CHAIN_K):
                if p + 1 >= L:
                    break
                emb_t = embed(next_tok).view(1, 1, hidden).to(torch.bfloat16)
                posj = torch.tensor([[p]], device=device)
                g = head(cur_feat, emb_t, posj)                    # predict f_{p}
                tok = lm_head(g[0, 0].to(torch.bfloat16)).argmax(-1)
                if int(tok.item()) == int(ids[p + 1].item()):
                    accepted += 1
                    cur_feat = g                                   # feed own prediction
                    next_tok = tok.view(1)
                    p += 1
                else:
                    break
            accept_sum[dom] += accepted
            accept_n[dom] += 1
            starts_used += 1

    print("per-domain dev metrics (single-step top1, greedy chain-accept length):")
    all_hit = all_n = 0
    acc_sum_all = acc_n_all = 0.0
    for dom in sorted(top1_n):
        t1 = top1_hit[dom] / max(1, top1_n[dom])
        ac = accept_sum[dom] / max(1, accept_n[dom])
        all_hit += top1_hit[dom]; all_n += top1_n[dom]
        acc_sum_all += accept_sum[dom]; acc_n_all += accept_n[dom]
        print(f"  {dom:9s} top1 {t1:.4f} ({top1_n[dom]} toks)   "
              f"accept {ac:.3f} ({accept_n[dom]} chains)")
    print(f"  OVERALL   top1 {all_hit/max(1,all_n):.4f}   "
          f"accept {acc_sum_all/max(1,acc_n_all):.3f}")


if __name__ == "__main__":
    main()
