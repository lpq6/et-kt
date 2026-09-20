from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from a2g_incumbent import A2GMambaKT as IncumbentA2G


class IdentityInitializedSDPAOutputGate(nn.MultiheadAttention):
    @classmethod
    def from_incumbent(cls, original, enabled):
        if not original.batch_first or not original._qkv_same_embed_dim:
            raise ValueError("only the incumbent equal-width batch-first attention is supported")
        if original.bias_k is not None or original.bias_v is not None or original.add_zero_attn:
            raise ValueError("unexpected extra attention tokens")
        with torch.random.fork_rng(devices=[]):
            module = cls(
                original.embed_dim, original.num_heads, dropout=original.dropout,
                bias=original.in_proj_bias is not None, batch_first=True,
            )
        module.load_state_dict(original.state_dict(), strict=True)
        module.gate_weight = nn.Parameter(torch.zeros(original.embed_dim, original.embed_dim))
        module.use_sdpa_output_gate = bool(enabled)
        return module

    def forward(
        self, query, key, value, key_padding_mask=None, need_weights=True,
        attn_mask=None, average_attn_weights=True, is_causal=False,
    ):
        if not self.use_sdpa_output_gate:
            return super().forward(
                query, key, value, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                average_attn_weights=average_attn_weights, is_causal=is_causal,
            )
        if query is not key or query is not value or query.ndim != 3:
            raise ValueError("the candidate accepts only batched self-attention")
        if need_weights or key_padding_mask is not None or is_causal:
            raise ValueError("use the incumbent explicit causal mask without returned weights")
        batch_size, length, width = query.shape
        if attn_mask is None or attn_mask.shape != (length, length) or attn_mask.dtype != torch.bool:
            raise ValueError("the incumbent boolean causal mask is required")
        sequence = query.transpose(0, 1)
        queries, keys, values = functional._in_projection_packed(
            sequence, sequence, sequence, self.in_proj_weight, self.in_proj_bias
        )
        queries = queries.view(length, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        keys = keys.view(length, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        values = values.view(length, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        queries = queries.view(batch_size, self.num_heads, length, self.head_dim)
        keys = keys.view(batch_size, self.num_heads, length, self.head_dim)
        values = values.view(batch_size, self.num_heads, length, self.head_dim)
        additive_mask = query.new_zeros(length, length).masked_fill(attn_mask, -torch.inf)
        attended = functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=additive_mask.view(1, 1, length, length),
            dropout_p=self.dropout if self.training else 0.0,
        )
        gate = 2.0 * torch.sigmoid(functional.linear(query, self.gate_weight))
        gate = gate.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)
        attended = attended * gate
        concatenated = attended.permute(2, 0, 1, 3).contiguous().view(length * batch_size, width)
        projected = functional.linear(concatenated, self.out_proj.weight, self.out_proj.bias)
        return projected.view(length, batch_size, width).transpose(0, 1), None


class A2GMambaKT(IncumbentA2G):
    CANDIDATE_ID = "incumbent_a2g_identity_initialized_sdpa_output_gate_20260909_v11"

    def __init__(self, *args, use_sdpa_output_gate=1, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.blocks:
            block["attn"] = IdentityInitializedSDPAOutputGate.from_incumbent(
                block["attn"], bool(int(use_sdpa_output_gate))
            )


__all__ = ["A2GMambaKT", "IdentityInitializedSDPAOutputGate"]
