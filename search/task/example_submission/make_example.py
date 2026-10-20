"""Write a minimal, UNTRAINED submission in the required format.

Kept as a script rather than a checked-in checkpoint because the head is ~250M
parameters (~0.5 GB) and this folder is copied into every run.

The identity initialisation means the head proposes roughly the target's current
feature unchanged, so acceptance is near zero and the score lands just under 1.0.
That is the useful floor: it proves the plumbing end to end before any training.

  python make_example.py --out DIR [--n-layers 1] [--inter 8192]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))
from eagle_head import EagleHead                                      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--inter", type=int, default=8192)
    ap.add_argument("--models", default=str(Path(__file__).resolve().parent.parent / "models.json"))
    a = ap.parse_args()

    cfg = json.loads(Path(a.models).read_text())
    from transformers import AutoConfig
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
    from target_geometry import geometry
    geo = geometry(AutoConfig.from_pretrained(cfg["speed_pair"]["target"]))

    head = EagleHead(hidden=geo["hidden"], inter=a.inter,
                     n_heads=geo["n_heads"], kv_heads=geo["kv_heads"],
                     n_layers=a.n_layers, theta=geo["rope_theta"],
                     eps=geo["rms_norm_eps"])

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out / "head.pt")
    (out / "head_config.json").write_text(json.dumps({
        "n_layers": a.n_layers,
        "inter": a.inter,
        "hidden": geo["hidden"],
        "n_heads": geo["n_heads"],
        "kv_heads": geo["kv_heads"],
        "head_dim": geo["head_dim"],
        "note": "untrained identity init; expect a score just under 1.0",
        "sglang_overrides": {},
    }, indent=2) + "\n")
    n = sum(v.numel() for v in head.state_dict().values())
    print(f"wrote {out}/head.pt  ({n/1e6:.1f}M params, {a.n_layers} layer(s), inter={a.inter})")


if __name__ == "__main__":
    main()
