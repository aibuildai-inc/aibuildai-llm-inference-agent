"""Concatenate the step-7 alpaca chat increment (data/chat_aug.pt) onto the
existing step-5/6 corpus (data/train.pt) -> data/train_v2.pt, shuffled.

Keeps train.pt untouched (it is the step-6 head's reproducer). train.py loads
data.train_file (default train.pt); step-7 recipe sets it to train_v2.pt.

    python merge_corpus.py
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"   # train/data/, one level up from corpus/


def main() -> None:
    base = torch.load(DATA / "train.pt", weights_only=False)
    aug = torch.load(DATA / "chat_aug.pt", weights_only=False)
    print(f"base: {len(base['ids'])} seqs / {base['total_tokens']} tokens "
          f"{Counter(base['domain'])}")
    print(f"aug:  {len(aug['ids'])} seqs / {aug['total_tokens']} tokens "
          f"{Counter(aug['domain'])}")

    keys = ["ids", "feats", "resp_start", "domain"]
    merged = {k: list(base[k]) + list(aug[k]) for k in keys}
    merged["total_tokens"] = int(base["total_tokens"]) + int(aug["total_tokens"])

    # shuffle so the new chat slice is interleaved, not a tail block
    n = len(merged["ids"])
    order = list(range(n))
    random.Random(0).shuffle(order)
    for k in keys:
        merged[k] = [merged[k][i] for i in order]

    out = DATA / "train_v2.pt"
    torch.save(merged, out)
    print(f"train_v2: {len(merged['ids'])} seqs / {merged['total_tokens']} tokens "
          f"{Counter(merged['domain'])} -> {out}")


if __name__ == "__main__":
    main()
