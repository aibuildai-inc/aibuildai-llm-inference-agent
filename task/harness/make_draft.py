"""Package a submission's head into an SGLang EAGLE draft directory.

The submission writes `head.pt` (state dict, SGLang key names -- see
reference_head.py) and `head_config.json`. This adds the target's embedding and
frozen lm_head, writes config.json, and emits a directory SGLang can load with
`speculative_draft_model_path`.

Refuses anything that is not the required architecture, naming what is wrong. A head
that cannot be loaded is a zero, so failing loudly here is better than a silent
fallback that quietly scores 1.0.

  python make_draft.py --submission DIR --models models.json --out DRAFT_DIR
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REQUIRED_PER_LAYER = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
]
BANNED = {
    "norm.weight": "a final RMSNorm (SGLang's EAGLE model returns the residual sum, no final norm)",
    "mlp_norm.weight": "an extra residual MLP block (no SGLang equivalent)",
    "mlp_gate.weight": "an extra residual MLP block (no SGLang equivalent)",
    "blocks.0.norm.weight": "an attention-free block stack (no SGLang equivalent)",
}


def load_head(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def validate(sd: dict) -> int:
    """Return the layer count, or exit explaining exactly what is wrong."""
    for bad, why in BANNED.items():
        if bad in sd:
            raise SystemExit(
                f"REJECTED: head contains `{bad}`, which is {why}.\n"
                "This task requires the architecture in reference_head.py; see its "
                "docstring for the state-dict keys SGLang can load.")
    if "fc.weight" not in sd:
        raise SystemExit("REJECTED: no `fc.weight`. The head must fuse "
                         "cat([token_embedding, target_feature]) -> hidden.")
    idx = sorted({int(m.group(1)) for k in sd
                  if (m := re.match(r"layers\.(\d+)\.", k))})
    if not idx:
        raise SystemExit(
            "REJECTED: no `layers.{i}.*` keys. The head must be one or more standard "
            "pre-norm decoder blocks (reference_head.py). Attention-free stacks and "
            "custom block names cannot be hosted by SGLang.")
    if idx != list(range(len(idx))):
        raise SystemExit(f"REJECTED: layer indices are not contiguous from 0: {idx}")
    missing = [f"layers.{i}.{suffix}" for i in idx for suffix in REQUIRED_PER_LAYER
               if f"layers.{i}.{suffix}" not in sd]
    if missing:
        raise SystemExit("REJECTED: missing required keys:\n  " + "\n  ".join(missing))
    return len(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sub = Path(a.submission)
    cfg = json.loads(Path(a.models).read_text())
    pair = cfg["speed_pair"]
    dtype = getattr(torch, pair["dtype"])

    head_pt = sub / "head.pt"
    if not head_pt.is_file():
        raise SystemExit(f"REJECTED: {head_pt} not found. The submission must write "
                         "head.pt plus head_config.json.")
    sd = load_head(head_pt)
    n_layers = validate(sd)

    # The TARGET's own config is authoritative for head geometry, never head_config.json.
    from transformers import AutoConfig
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from target_geometry import geometry
    geo = geometry(AutoConfig.from_pretrained(pair["target"]))
    hidden, n_heads, head_dim = geo["hidden"], geo["n_heads"], geo["head_dim"]

    if tuple(sd["fc.weight"].shape) != (hidden, 2 * hidden):
        raise SystemExit(f"REJECTED: fc.weight is {tuple(sd['fc.weight'].shape)}, "
                         f"expected {(hidden, 2 * hidden)}")
    kv_heads = sd["layers.0.self_attn.k_proj.weight"].shape[0] // head_dim
    inter = sd["layers.0.mlp.gate_proj.weight"].shape[0]

    out = {k: v for k, v in sd.items()}
    out.setdefault("fc.bias", torch.zeros(hidden, dtype=sd["fc.weight"].dtype))
    # The target's attention carries qkv bias; zeros are identical for a head without it.
    for i in range(n_layers):
        for proj in ("q", "k", "v"):
            w = out[f"layers.{i}.self_attn.{proj}_proj.weight"]
            out.setdefault(f"layers.{i}.self_attn.{proj}_proj.bias",
                           torch.zeros(w.shape[0], dtype=w.dtype))

    # The head rides on the target's embedding and FROZEN lm_head: copy both in.
    # Resolved file by file, never snapshot_download: a cache holding only weights and
    # config is "incomplete" to snapshot_download and it refuses offline, which is the
    # normal state here.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _toolchain import ensure as _ensure_toolchain
    _ensure_toolchain()          # ninja + C++ on PATH for flashinfer JIT
    from safetensors.torch import load_file, save_file
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> Path:
        return Path(hf_hub_download(pair["target"], name))

    index = json.loads(fetch("model.safetensors.index.json").read_text())["weight_map"]
    for want, key in (("embed_tokens.weight", "model.embed_tokens.weight"),
                      ("lm_head.weight", "lm_head.weight")):
        shard = index.get(key)
        if shard is None:
            raise SystemExit(f"{key} missing from the target index; cannot build the draft")
        out[want] = load_file(str(fetch(shard)))[key]

    out = {k: v.to(dtype).contiguous() for k, v in out.items()}
    dest = Path(a.out); dest.mkdir(parents=True, exist_ok=True)
    draft_cfg = {
        "architectures": [cfg["sglang"]["draft_architecture"]],
        "model_type": cfg["sglang"]["draft_model_type"],
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": kv_heads,
        "vocab_size": geo["vocab_size"],
        "max_position_embeddings": geo["max_position_embeddings"],
        "rms_norm_eps": geo["rms_norm_eps"],
        "rope_theta": geo["rope_theta"],
        "tie_word_embeddings": False,
        "torch_dtype": pair["dtype"],
        "attention_bias": True,
    }
    (dest / "config.json").write_text(json.dumps(draft_cfg, indent=2) + "\n")
    save_file(out, str(dest / "model.safetensors"), metadata={"format": "pt"})
    print(json.dumps({"draft_dir": str(dest), "tensors": len(out), "n_layers": n_layers,
                      "intermediate_size": inter, "num_key_value_heads": kv_heads}, indent=2))


if __name__ == "__main__":
    main()
