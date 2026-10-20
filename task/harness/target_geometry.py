"""Read the TARGET's geometry from its own config, across transformers versions.

transformers 5.x moved `rope_theta` into a `rope_parameters` dict, so reading it
directly raises AttributeError. Several places need these numbers, and getting them
wrong is silent -- a wrong rope_theta trains a head that works offline and proposes
garbage in SGLang -- so the lookup lives in one place with explicit fallbacks.
"""
from __future__ import annotations

from typing import Any


def rope_theta(cfg: Any) -> float:
    for attr in ("rope_theta",):
        v = getattr(cfg, attr, None)
        if v is not None:
            return float(v)
    for attr in ("rope_parameters", "rope_scaling"):
        d = getattr(cfg, attr, None)
        if isinstance(d, dict) and d.get("rope_theta") is not None:
            return float(d["rope_theta"])
    raise SystemExit(
        "cannot determine rope_theta from the target config: looked at .rope_theta, "
        ".rope_parameters and .rope_scaling. Add the right field here rather than "
        "guessing a default -- a wrong value trains a head that fails only in SGLang.")


def geometry(cfg: Any) -> dict:
    hidden = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", None) or hidden // n_heads)
    return {
        "hidden": hidden,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "kv_heads": int(getattr(cfg, "num_key_value_heads", n_heads)),
        "inter": int(cfg.intermediate_size),
        "vocab_size": int(cfg.vocab_size),
        "max_position_embeddings": int(cfg.max_position_embeddings),
        "rms_norm_eps": float(cfg.rms_norm_eps),
        "rope_theta": rope_theta(cfg),
    }
