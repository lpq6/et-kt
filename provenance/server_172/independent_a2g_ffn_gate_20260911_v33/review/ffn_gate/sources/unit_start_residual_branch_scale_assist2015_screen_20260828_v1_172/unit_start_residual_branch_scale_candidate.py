"""Unit-start learnable scales for frozen A2G residual branches.

The seven scalar parameters start at exactly 1.0, so the initial function is
the frozen Full control. Training may independently adjust four attention and
three FFN residual contributions without changing Q/K/V, dropout, postnorm,
SSM, statistics, the prediction head, or the objective.
"""

from __future__ import annotations

import torch
from torch import nn

from work.a2g_mambakt_rwce_item_dropout_candidate import (
    A2GMambaKT as _FrozenA2G,
)


class UnitStartResidualBranchScaleA2GMambaKT(_FrozenA2G):
    """Frozen Full plus seven unit-initialized residual branch scalars."""

    candidate_id = "unit_start_residual_branch_scale_v1"

    def __init__(self, *args, use_residual_branch_scales=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_residual_branch_scales = bool(int(use_residual_branch_scales))
        attention_count = len(self.blocks)
        ffn_count = sum(int("ffn" in block) for block in self.blocks)
        self.attention_residual_scales = nn.Parameter(torch.ones(attention_count))
        self.ffn_residual_scales = nn.Parameter(torch.ones(ffn_count))

    def _attend(self, x):
        if not self.use_residual_branch_scales:
            return super()._attend(x)
        length = x.size(1)
        future = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        ffn_index = 0
        for block_index, block in enumerate(self.blocks):
            attended, _ = block["attn"](
                x,
                x,
                x,
                attn_mask=future,
                need_weights=False,
            )
            attention_scale = self.attention_residual_scales[block_index]
            x = block["norm"](x + attention_scale * block["drop"](attended))
            if "ffn" in block:
                ffn_scale = self.ffn_residual_scales[ffn_index]
                x = block["ffn_norm"](x + ffn_scale * block["ffn"](x))
                ffn_index += 1
        return x


__all__ = ["UnitStartResidualBranchScaleA2GMambaKT"]
