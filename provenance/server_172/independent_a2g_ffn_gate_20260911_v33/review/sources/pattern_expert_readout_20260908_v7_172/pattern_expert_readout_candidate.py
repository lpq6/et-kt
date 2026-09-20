"""Pattern-conditioned expert readout on top of the v6 causal memory."""

from __future__ import annotations

import torch
from torch import nn

from a2g_causal_query_state_candidate import (
    A2GMambaKT as DualContextRFA,
)


class PatternExpertReadout(nn.Module):
    """Allocate readout capacity across heterogeneous causal response regimes."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        d_ff: int,
        dropout: float,
        num_experts: int = 2,
    ):
        super().__init__()
        if int(num_experts) < 2:
            raise ValueError("num_experts must be at least 2")
        self.num_experts = int(num_experts)
        self.norm = nn.LayerNorm(input_dim)
        self.router = nn.Linear(input_dim, self.num_experts)
        expert_hidden = max(64, int(d_ff) // self.num_experts)
        expert_bottleneck = max(64, int(d_model) // 4)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, expert_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, expert_bottleneck),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(expert_bottleneck, 1),
                )
                for _ in range(self.num_experts)
            ]
        )
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, features):
        normalized = self.norm(features)
        route_weights = torch.softmax(self.router(normalized), dim=-1)
        expert_logits = torch.cat(
            [expert(features) for expert in self.experts],
            dim=-1,
        )
        return (route_weights * expert_logits).sum(
            dim=-1,
            keepdim=True,
        )


class A2GMambaKT(DualContextRFA):
    """v6 dual-context RFA with a pattern-conditioned expert readout."""

    CANDIDATE_ID = "pattern_expert_readout_20260908_v7"

    def __init__(
        self,
        n_question,
        n_pid,
        d_model=256,
        d_ff=512,
        dropout=0.2,
        num_pattern_experts=2,
        **kwargs,
    ):
        super().__init__(
            n_question,
            n_pid,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            **kwargs,
        )
        self.model_name = "pattern_expert_readout"
        self.num_pattern_experts = int(num_pattern_experts)
        self.pred = PatternExpertReadout(
            input_dim=3 * int(d_model) + 2,
            d_model=int(d_model),
            d_ff=int(d_ff),
            dropout=float(dropout),
            num_experts=self.num_pattern_experts,
        )


__all__ = ["A2GMambaKT", "PatternExpertReadout"]
