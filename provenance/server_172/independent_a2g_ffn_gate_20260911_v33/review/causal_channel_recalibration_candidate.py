"""Zero-initialized causal channel recalibration after the frozen attention stack."""

from __future__ import annotations

import torch
from torch import nn

try:
    from pykt.models.a2g_mambakt_final import A2GMambaKT as _FrozenFull
except ImportError:
    from a2g_mambakt_final import A2GMambaKT as _FrozenFull


class CausalChannelRecalibrationA2GMambaKT(_FrozenFull):
    """A causal SE-like channel gain with an identity start."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_channel_scale = nn.Parameter(torch.zeros(self.d_model))

    def _causal_channel_recalibrate(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        count = torch.arange(1, length + 1, device=x.device, dtype=x.dtype).view(1, length, 1)
        cumulative = x.cumsum(dim=1) / count
        rms = cumulative.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        context = torch.tanh(cumulative / rms)
        scale = self.causal_channel_scale.view(1, 1, -1)
        return x + x * scale * context

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        attended = super()._attend(x)
        return self._causal_channel_recalibrate(attended)


__all__ = ["CausalChannelRecalibrationA2GMambaKT"]
