"""Extracted from the reviewed history_pace_candidate.py; see source_map.json."""

import torch
from torch import nn


def historical_pace_feature(timestamps, hist_c):
    """At target t, use timestamp[t-1] - timestamp[t-2], never timestamp[t]."""
    if hist_c.ndim != 2 or hist_c.size(1) == 0:
        raise ValueError("history concepts must have a nonempty batch/length shape")
    (batch, length) = hist_c.shape
    zero = torch.zeros(batch, length, 1, dtype=torch.float32, device=hist_c.device)
    if timestamps is None or timestamps.numel() == 0:
        return zero
    if timestamps.shape != (batch, length - 1) or timestamps.dtype != torch.int64:
        raise ValueError("unshifted timestamps must be int64 with one fewer column")
    if timestamps.device != hist_c.device:
        raise ValueError("timestamps and history must share a device")
    if length < 3:
        return zero
    (earlier, later) = (timestamps[:, :-1], timestamps[:, 1:])
    valid = (
        hist_c[:, 1:-1].gt(0) & hist_c[:, 2:].gt(0) & earlier.ge(0) & later.ge(earlier)
    )
    gap = torch.where(valid, later - earlier, torch.zeros_like(later))
    feature = torch.log1p(gap.to(torch.float32) / 1000.0).unsqueeze(-1)
    return torch.cat([zero[:, :2], feature], dim=1)


class HistoricalPaceModulation(nn.Module):
    def __init__(self, width):
        super().__init__()
        if width < 1:
            raise ValueError("history width must be positive")
        self.scale = nn.Parameter(torch.zeros(width))
        self.shift = nn.Parameter(torch.zeros(width))

    def forward(self, history, feature):
        if history.shape[:2] != feature.shape[:2] or feature.size(-1) != 1:
            raise ValueError("history and temporal features must align")
        feature = feature.to(dtype=history.dtype)
        return history * (1.0 + feature * self.scale) + feature * self.shift
