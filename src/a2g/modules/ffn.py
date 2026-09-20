"""Extracted from the reviewed ffn_gate_candidate.py; see source_map.json."""

import torch
from torch import nn
import torch.nn.functional as F


class InputGatedFeedForward(nn.Sequential):
    def __init__(self, original, enabled=True):
        if (
            not isinstance(original, nn.Sequential)
            or isinstance(original, InputGatedFeedForward)
            or len(original) != 5
            or (not isinstance(original[0], nn.Linear))
            or (not isinstance(original[1], nn.GELU))
            or (not isinstance(original[2], nn.Dropout))
            or (not isinstance(original[3], nn.Linear))
            or (not isinstance(original[4], nn.Dropout))
        ):
            raise ValueError(
                "expected the original Linear/GELU/Dropout/Linear/Dropout FFN"
            )
        (first, last) = (original[0], original[3])
        if (
            first.bias is None
            or last.bias is None
            or first.out_features != last.in_features
            or (first.in_features != last.out_features)
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
