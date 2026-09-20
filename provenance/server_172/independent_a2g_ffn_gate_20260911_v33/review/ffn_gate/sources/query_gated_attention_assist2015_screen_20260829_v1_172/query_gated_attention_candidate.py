"""Identity-start query-gated attention for the frozen A2G screen.

The frozen MultiheadAttention remains the complete primary path.  Each
projected output channel group receives a query-dependent scalar gate:

    gate[b,t,h] = 2 * sigmoid(<x[b,t,h,:], w[h,:]>)

All gate weights start at zero, so the installed candidate is bit-exact to
the control before optimization.  No attention mask, Q/K/V parameter,
dropout call, state update, prediction head, input field, or loss is changed.
"""

from __future__ import annotations

import types

import torch
from torch import nn


def _query_gated_forward(
    self,
    query,
    key,
    value,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False,
):
    output, weights = self._a2g_control_forward(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        attn_mask=attn_mask,
        average_attn_weights=average_attn_weights,
        is_causal=is_causal,
    )
    if not self.batch_first or query.dim() != 3 or output.dim() != 3:
        raise RuntimeError("A2G query gate requires batch-first 3-D attention")
    if query.shape != output.shape or query.size(-1) != self.embed_dim:
        raise RuntimeError("unexpected frozen self-attention shape")
    batch, length, model_dim = output.shape
    head_dim = model_dim // self.num_heads
    query_groups = query.reshape(batch, length, self.num_heads, head_dim)
    gate_logits = torch.einsum(
        "blhd,hd->blh", query_groups, self.query_gate_weight
    )
    gates = 2.0 * torch.sigmoid(gate_logits)
    gated = output.reshape(batch, length, self.num_heads, head_dim)
    gated = gated * gates.unsqueeze(-1)
    return gated.reshape(batch, length, model_dim), weights


def install_query_gated_attention(model) -> None:
    """Install zero-start query gates without advancing constructor RNG."""

    for index, block in enumerate(model.blocks):
        attention = block["attn"]
        if not isinstance(attention, nn.MultiheadAttention):
            raise TypeError(f"blocks.{index}.attn is not MultiheadAttention")
        if hasattr(attention, "query_gate_weight"):
            raise RuntimeError(f"blocks.{index}.attn was already gated")
        if not attention.batch_first or attention.embed_dim % attention.num_heads:
            raise RuntimeError("unexpected frozen attention dimensions")
        head_dim = attention.embed_dim // attention.num_heads
        attention.register_parameter(
            "query_gate_weight",
            nn.Parameter(
                torch.zeros(
                    attention.num_heads,
                    head_dim,
                    device=attention.in_proj_weight.device,
                    dtype=attention.in_proj_weight.dtype,
                )
            ),
        )
        attention._a2g_control_forward = attention.forward
        attention.forward = types.MethodType(_query_gated_forward, attention)


__all__ = ["install_query_gated_attention"]
