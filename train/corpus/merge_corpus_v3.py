"""Step 9: concatenate the fresh reasoning+code increment (data/rc_aug.pt) onto the
step-7 corpus (data/train_v2.pt) -> data/train_v3.pt, shuffled.

Keeps train_v2.pt untouched (the step-7/8 head's reproducer). train.py loads
data.train_file; the step-9 recipe sets it to train_v3.pt.

    python merge_corpus_v3.py
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"   # train/data/, one level up from corpus/


def main() -> None:
    base = torch.load(DATA / "train_v2.pt", weights_only=False)
    aug = torch.load(DATA / "rc_aug.pt", weights_only=False)
    print(f"base (train_v2): {len(base['ids'])} seqs / {base['total_tokens']} tokens "
          f"{Counter(base['domain'])}")
    print(f"aug  (rc_aug):   {len(aug['ids'])} seqs / {aug['total_tokens']} tokens "
          f"{Counter(aug['domain'])}")

    keys = ["ids", "feats", "resp_start", "domain"]
    merged = {k: list(base[k]) + list(aug[k]) for k in keys}
    merged["total_tokens"] = int(base["total_tokens"]) + int(aug["total_tokens"])

    n = len(merged["ids"])
    order = list(range(n))
    random.Random(0).shuffle(order)
    for k in keys:
        merged[k] = [merged[k][i] for i in order]

    out = DATA / "train_v3.pt"
    torch.save(merged, out)
    print(f"train_v3: {len(merged['ids'])} seqs / {merged['total_tokens']} tokens "
          f"{Counter(merged['domain'])} -> {out}")


if __name__ == "__main__":
    main()
