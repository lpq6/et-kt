"""Token-conditioned expert capacity adapter for the retained A2G FFN.

The adapter consumes only the causal token representation supplied to the
first retained FFN. Its zero-initialized residual scale makes construction an
exact control function while allowing a two-expert mixture to specialize
during training. It adds no stochastic operation, auxiliary loss, response
feature, statistic, or prediction-level routing path.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class TokenCapacityMoEFFN(nn.Module):
    """Dense two-expert token mixer with an identity-start residual gate."""

    def __init__(self, d_model: int, expert_width: int):
        super().__init__()
        self.router = nn.Linear(d_model, 2, bias=False)
        self.expert_down = nn.ModuleList(
            [nn.Linear(d_model, expert_width, bias=False) for _ in range(2)]
        )
        self.expert_up = nn.ModuleList(
            [nn.Linear(expert_width, d_model, bias=False) for _ in range(2)]
        )
        self.gamma = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.router.weight)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.router(token), dim=-1)
        expert_outputs = torch.stack(
            [
                up(F.gelu(down(token)))
                for down, up in zip(self.expert_down, self.expert_up)
            ],
            dim=-2,
        )
        mixture = (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        return torch.tanh(self.gamma) * mixture

__all__ = ["TokenCapacityMoEFFN"]
