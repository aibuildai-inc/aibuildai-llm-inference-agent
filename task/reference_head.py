"""The draft-head architecture this task requires, and a working implementation of it.

WHY THIS FILE IS NORMATIVE
--------------------------
The submission does not run inference. SGLang does. SGLang hosts an EAGLE draft as
the EAGLE class models.json names (sglang/srt/models/<model_type>_eagle.py),
which is a fixed module tree, so a head whose shape differs CANNOT be loaded no matter how good it is. Heads
from earlier eager-harness runs demonstrate the failure modes:

  * an extra SwiGLU residual MLP plus a final RMSNorm -> no SGLang equivalent, unmappable
  * an attention-free stack of SwiGLU blocks        -> no SGLang equivalent, unmappable

Both scored well against an eager HuggingFace baseline and neither can be deployed.
So this task fixes the architecture and lets the submission optimise everything else:
the training data, the loss, the schedule, the layer count, the MLP width, and the
speculative tree parameters.

THE ARCHITECTURE
----------------
    fused = fc(cat([token_embedding, target_feature], dim=-1))     # EMBEDDING FIRST
    h = fused
    for layer in layers:                       # standard pre-norm decoder block
        h = h + attn(input_layernorm(h))       # qkv carry bias, RoPE, GQA allowed
        h = h + mlp(post_attention_layernorm(h))
    return h                                   # NO final norm

`target_feature` is the target's post-final-norm hidden state (the vector its lm_head
consumes). Candidate logits come from the target's own FROZEN lm_head, which SGLang
supplies -- the submission never trains an output projection.

Note what is NOT here: no final RMSNorm, no extra residual MLP, and the concat order is
embedding-first. Those three are exactly where previous heads diverged from SGLang.

STATE DICT KEYS -- use these names, they are SGLang's own
---------------------------------------------------------
    fc.weight                                        (hidden, 2*hidden)
    fc.bias                                          (hidden,)        optional
    layers.{i}.input_layernorm.weight                (hidden,)
    layers.{i}.self_attn.q_proj.{weight,bias}        (hidden, hidden) / (hidden,)
    layers.{i}.self_attn.k_proj.{weight,bias}        (kv*hd, hidden)  / (kv*hd,)
    layers.{i}.self_attn.v_proj.{weight,bias}        (kv*hd, hidden)  / (kv*hd,)
    layers.{i}.self_attn.o_proj.weight               (hidden, hidden)
    layers.{i}.post_attention_layernorm.weight       (hidden,)
    layers.{i}.mlp.{gate,up,down}_proj.weight        (inter, hidden) x2, (hidden, inter)

`harness/make_draft.py` packages exactly these into an SGLang draft directory; it adds
the target's embedding and lm_head and writes config.json. Anything else fails the gate.

ONE PATCH THIS TASK DEPENDS ON
------------------------------
Upstream SGLang deletes layer 0's `input_layernorm` for parity with the reference EAGLE
implementation. This architecture USES it. The run's environment carries the one-line
patch that keeps it; `harness/check_draft.py` asserts the patch is present, because
without it the loaded head computes a different function (measured: relative error 1.84
versus 7.7e-17 with the patch).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dt)


def _rope(positions: torch.Tensor, head_dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    f = positions.float()[..., None] * inv[None, None, :]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rot_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class Attention(nn.Module):
    """Target-style self-attention: qkv bias, RoPE, optional GQA."""

    def __init__(self, hidden: int, n_heads: int, kv_heads: int, theta: float) -> None:
        super().__init__()
        self.n_heads, self.kv_heads = n_heads, kv_heads
        self.head_dim = hidden // n_heads
        self.theta = theta
        self.q_proj = nn.Linear(hidden, n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = _rope(positions, self.head_dim, self.theta, x.device, x.dtype)
        cos, sin = cos[:, None], sin[:, None]
        q = q * cos + _rot_half(q) * sin
        k = k * cos + _rot_half(k) * sin
        if self.kv_heads != self.n_heads:
            rep = self.n_heads // self.kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        mask = torch.full((S, S), torch.finfo(x.dtype).min, device=x.device,
                          dtype=x.dtype).triu(1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(o.transpose(1, 2).reshape(B, S, -1))


class MLP(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, hidden: int, inter: int, n_heads: int, kv_heads: int,
                 theta: float, eps: float) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = Attention(hidden, n_heads, kv_heads, theta)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = MLP(hidden, inter)

    def forward(self, h: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h = h + self.self_attn(self.input_layernorm(h), positions)
        return h + self.mlp(self.post_attention_layernorm(h))


class EagleHead(nn.Module):
    """The required architecture. Train this; the harness packages it for SGLang."""

    def __init__(self, hidden: int, inter: int, n_heads: int, kv_heads: int,
                 n_layers: int = 1, theta: float = 1_000_000.0, eps: float = 1e-6,
                 fc_bias: bool = False) -> None:
        super().__init__()
        self.hidden = hidden
        self.fc = nn.Linear(2 * hidden, hidden, bias=fc_bias)
        self.layers = nn.ModuleList(
            [Layer(hidden, inter, n_heads, kv_heads, theta, eps) for _ in range(n_layers)])
        self._init_identity()

    def _init_identity(self) -> None:
        """Start as (almost) the identity on the target feature.

        fc sees cat([embedding, feature]); zeroing the embedding half and setting the
        feature half to I means layer 0 initially reads the target's own feature, which
        is the near-identity warm start the residual blocks are designed around.
        """
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.weight[:, self.hidden:].copy_(torch.eye(self.hidden))
            if self.fc.bias is not None:
                self.fc.bias.zero_()
            for layer in self.layers:
                layer.self_attn.o_proj.weight.zero_()
                layer.mlp.down_proj.weight.zero_()

    def forward(self, feature: torch.Tensor, token_emb: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """feature,[B,S,H] target post-final-norm states; token_emb,[B,S,H]; positions,[B,S]."""
        h = self.fc(torch.cat([token_emb, feature], dim=-1))   # EMBEDDING FIRST
        for layer in self.layers:
            h = layer(h, positions)
        return h                                               # NO final norm
