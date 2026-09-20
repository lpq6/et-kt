"""Bounded, depthwise causal cognitive smoothing candidate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F


try:
    from work.a2g_mambakt_rwce_item_dropout_candidate import A2GMambaKT as _FrozenA2G
except ModuleNotFoundError as exc:
    if exc.name not in {"work", "work.a2g_mambakt_rwce_item_dropout_candidate"}:
        raise
    module_name = "_bounded_smoothing_local_control_20260817"
    source = Path(__file__).resolve().parents[1] / "pykt" / "models" / "a2g_mambakt_final.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local A2G control from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _FrozenA2G = module.A2GMambaKT


class BoundedDepthwiseCognitiveSmooth(nn.Module):
    """Causal per-channel trend filter with bounded residual mixing."""

    def __init__(self, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.kernel_size = 5
        self.kernel_logits = nn.Parameter(torch.zeros(hidden_size, 1, 5))
        self.mix_logit = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.out_dropout = nn.Dropout(float(dropout_rate))
        with torch.no_grad():
            coeff = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]).log()
            self.kernel_logits.copy_(coeff.view(1, 1, 5).expand_as(self.kernel_logits))

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        channel_first = token.transpose(1, 2)
        weights = torch.softmax(self.kernel_logits, dim=-1)
        padded = F.pad(channel_first, (self.kernel_size - 1, 0))
        trend = F.conv1d(padded, weights, groups=token.size(-1))
        trend = trend.transpose(1, 2)
        mix = torch.sigmoid(self.mix_logit)
        return token + self.out_dropout(mix * (trend - token))


class BoundedSmoothingA2GMambaKT(_FrozenA2G):
    """Add only bounded depthwise causal smoothing before the frozen stack."""

    candidate_id = "bounded_depthwise_cognitive_smoothing_v1"

    def __init__(self, *args, bounded_smooth_kernel=5, **kwargs):
        super().__init__(*args, **kwargs)
        if int(bounded_smooth_kernel) != 5:
            raise ValueError("bounded_smooth_kernel is locked to 5")
        self.bounded_smooth_kernel = 5
        hidden_size = int(self.item_emb.embedding_dim)
        dropout_rate = float(getattr(self.input[3], "p", 0.2))
        self.bounded_smooth = BoundedDepthwiseCognitiveSmooth(hidden_size, dropout_rate)
        original_input_forward = self.input.forward
        def smoothed_input_forward(module, features):
            del module
            return self.bounded_smooth(original_input_forward(features))
        import types
        self.input.forward = types.MethodType(smoothed_input_forward, self.input)
        self.model_name = "a2g_mambakt"


__all__ = ["BoundedSmoothingA2GMambaKT", "BoundedDepthwiseCognitiveSmooth"]
