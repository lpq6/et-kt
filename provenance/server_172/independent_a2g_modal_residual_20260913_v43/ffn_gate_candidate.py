"""V32 with input-dependent gates on the retained FFN hidden neurons."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from concept_graph_candidate import A2GMambaKT as _V32


class InputGatedFeedForward(nn.Sequential):
    def __init__(self, original, enabled=True):
        if (
            not isinstance(original, nn.Sequential) or isinstance(original, InputGatedFeedForward)
            or len(original) != 5
            or not isinstance(original[0], nn.Linear)
            or not isinstance(original[1], nn.GELU)
            or not isinstance(original[2], nn.Dropout)
            or not isinstance(original[3], nn.Linear)
            or not isinstance(original[4], nn.Dropout)
        ):
            raise ValueError("expected the original Linear/GELU/Dropout/Linear/Dropout FFN")
        first, last = original[0], original[3]
        if (
            first.bias is None or last.bias is None
            or first.out_features != last.in_features
            or first.in_features != last.out_features
        ):
            raise ValueError("the original biased FFN geometry changed")
        super().__init__(*original.children())
        self.gate_weight = nn.Parameter(torch.zeros_like(first.weight))
        self.gate_bias = nn.Parameter(torch.zeros_like(first.bias))
        self.enabled = bool(enabled)

    def forward(self, inputs):
        if not self.enabled:
            return super().forward(inputs)
        hidden = self[1](self[0](inputs))
        gate = 2.0 * torch.sigmoid(F.linear(inputs, self.gate_weight, self.gate_bias))
        hidden = hidden * gate
        for index in range(2, len(self)):
            hidden = self[index](hidden)
        return hidden


class A2GMambaKT(_V32):
    CANDIDATE_ID = "a2g_v32_input_gated_ffn_20260911_v33"

    def __init__(self, *args, use_ffn_gate=1, **kwargs):
        super().__init__(*args, **kwargs)
        if not any("ffn" in block for block in self.blocks):
            raise ValueError("at least one retained FFN is required")
        for block in self.blocks:
            if "ffn" in block:
                block["ffn"] = InputGatedFeedForward(block["ffn"])
        self.use_ffn_gate = use_ffn_gate

    @property
    def use_ffn_gate(self):
        return self._use_ffn_gate

    @use_ffn_gate.setter
    def use_ffn_gate(self, value):
        self._use_ffn_gate = bool(int(value))
        for block in self.blocks:
            if "ffn" in block:
                block["ffn"].enabled = self._use_ffn_gate


__all__ = ["A2GMambaKT", "InputGatedFeedForward"]
