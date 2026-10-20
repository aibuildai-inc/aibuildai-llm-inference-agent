"""The EAGLE draft-head architecture SGLang hosts, written from the contract.

This is my own implementation of the module whose state-dict keys
`harness/make_draft.py` validates and whose shape SGLang loads. The contract
(reference/eagle_head.py docstring, and sglang/srt/models/qwen2_eagle.py):

    fused = fc(cat([token_embedding, target_feature], dim=-1))   # EMBEDDING FIRST
    h = fused
    for layer in layers:                       # standard pre-norm decoder block
        h = h + attn(input_layernorm(h))       # qkv bias, RoPE, GQA
        h = h + mlp(post_attention_layernorm(h))
    return h                                    # NO final norm

State-dict keys (SGLang's own names):
    fc.weight                                   (hidden, 2*hidden)
    layers.{i}.input_layernorm.weight
    layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
    layers.{i}.self_attn.o_proj.weight
    layers.{i}.post_attention_layernorm.weight
    layers.{i}.mlp.{gate,up,down}_proj.weight
"""
from __future__ import annotations

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
    def __init__(self, hidden: int, n_heads: int, kv_heads: int, head_dim: int,
                 theta: float) -> None:
        super().__init__()
        self.n_heads, self.kv_heads, self.head_dim = n_heads, kv_heads, head_dim
        self.theta = theta
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
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
        if attn_bias is None:
            # reference behaviour: plain causal mask (matches SGLang inference)
            attn_bias = torch.full((S, S), torch.finfo(q.dtype).min, device=x.device,
                                   dtype=q.dtype).triu(1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.to(q.dtype))
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
    def __init__(self, hidden, inter, n_heads, kv_heads, head_dim, theta, eps):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = Attention(hidden, n_heads, kv_heads, head_dim, theta)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = MLP(hidden, inter)

    def forward(self, h: torch.Tensor, positions: torch.Tensor,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        h = h + self.self_attn(self.input_layernorm(h), positions, attn_bias)
        return h + self.mlp(self.post_attention_layernorm(h))


class EagleHead(nn.Module):
    def __init__(self, hidden: int, inter: int, n_heads: int, kv_heads: int,
                 head_dim: int | None = None, n_layers: int = 1,
                 theta: float = 1_000_000.0, eps: float = 1e-6,
                 fc_bias: bool = False) -> None:
        super().__init__()
        self.hidden = hidden
        head_dim = head_dim or hidden // n_heads
        self.fc = nn.Linear(2 * hidden, hidden, bias=fc_bias)
        self.layers = nn.ModuleList(
            [Layer(hidden, inter, n_heads, kv_heads, head_dim, theta, eps)
             for _ in range(n_layers)])
        self._init_identity()

    def _init_identity(self) -> None:
        """Start near the identity on the target feature: fc reads the feature half
        unchanged (embedding half zeroed), residual branches zeroed."""
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.weight[:, self.hidden:].copy_(torch.eye(self.hidden))
            if self.fc.bias is not None:
                self.fc.bias.zero_()
            for layer in self.layers:
                layer.self_attn.o_proj.weight.zero_()
                layer.mlp.down_proj.weight.zero_()

    def forward(self, feature: torch.Tensor, token_emb: torch.Tensor,
                positions: torch.Tensor,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc(torch.cat([token_emb, feature], dim=-1))   # EMBEDDING FIRST
        for layer in self.layers:
            h = layer(h, positions, attn_bias)
        return h                                               # NO final norm
