from __future__ import annotations

import torch
from torch import nn

from a2g_incumbent import A2GMambaKT as IncumbentA2G


class ResidualLinearSplineGELU(nn.Module):
    """Retain GELU exactly at initialization and learn local activation corrections."""

    def __init__(self, width, original_activation, enabled=True):
        super().__init__()
        self.base = original_activation
        self.enabled = bool(enabled)
        self.coefficients = nn.Parameter(torch.zeros(int(width), 17))
        self.register_buffer("knots", torch.linspace(-4.0, 4.0, 17), persistent=False)

    def forward(self, values):
        original = self.base(values)
        if not self.enabled:
            return original
        basis = (1.0 - (values.unsqueeze(-1) - self.knots).abs() / 0.5).clamp_min(0.0)
        return original + (basis * self.coefficients).sum(dim=-1)


class A2GMambaKT(IncumbentA2G):
    CANDIDATE_ID = "incumbent_a2g_residual_spline_activation_20260908_v9"

    def __init__(self, *args, use_readout_spline=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred[1] = ResidualLinearSplineGELU(
            self.pred[3].in_features, self.pred[1], bool(use_readout_spline)
        )


__all__ = ["A2GMambaKT", "ResidualLinearSplineGELU"]
