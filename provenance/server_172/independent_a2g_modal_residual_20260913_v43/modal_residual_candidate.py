"""V39 finite input memory with a zero-initialized modal residual readout."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from input_memory_candidate import A2GMambaKT as _V39
from input_memory_candidate import InputMemorySSM


class MultiModeResidualInputMemorySSM(InputMemorySSM):
    @classmethod
    def from_existing(cls, source, mode_count=8, enabled=True):
        if type(source) is not InputMemorySSM:
            raise TypeError("the modal residual requires the retained V39 input-memory SSM")
        if type(mode_count) is not int or not 2 <= mode_count <= 8:
            raise ValueError("supported synthetic/production mode counts are 2 through 8")
        width = source.in_proj.in_features
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.in_proj, module.norm, module.drop = source.in_proj, source.norm, source.drop
        module.input_decay_scale = source.input_decay_scale
        module.input_memory = source.input_memory
        module.use_input_memory = source.use_input_memory
        module.mode_count = mode_count
        module.use_multimode_residual = bool(enabled)
        module.mode_input_weight = nn.Parameter(
            source.in_proj.weight.new_zeros(mode_count, width)
        )
        module.mode_read_weight = nn.Parameter(
            source.in_proj.weight.new_zeros(mode_count, width)
        )
        powers = torch.arange(
            mode_count,
            device=source.in_proj.weight.device,
            dtype=source.in_proj.weight.dtype,
        )
        module.register_buffer(
            "rate_multipliers",
            torch.exp2(-powers),
            persistent=False,
        )
        return module

    def forward(self, x, scope_weight=None):
        if not self.use_multimode_residual:
            return super().forward(x, scope_weight=scope_weight)
        if (
            x.ndim != 3 or min(x.shape) < 1
            or x.size(-1) != self.in_proj.in_features
            or not x.is_floating_point()
            or x.device != self.in_proj.weight.device
            or x.dtype != self.in_proj.weight.dtype
        ):
            raise ValueError(
                "SSM inputs must share the original projection's nonempty "
                "B/L/D geometry, dtype and device"
            )
        if scope_weight is not None and (
            scope_weight.shape not in (x.shape, (*x.shape[:2], 1))
            or scope_weight.device != x.device
            or scope_weight.dtype != x.dtype
        ):
            raise ValueError(
                "scope weights must share the input B/L/D or B/L/1 layout, "
                "dtype and device"
            )

        projected = torch.stack(
            [self.in_proj(x[:, index]) for index in range(x.size(1))],
            dim=1,
        )
        corrected = (
            projected + self.input_memory(projected)
            if self.use_input_memory
            else projected
        )
        hidden = x.new_zeros(x.size(0), x.size(-1))
        modal_hidden = x.new_zeros(
            x.size(0), x.size(-1), self.mode_count
        )
        reference_hidden = x.new_zeros(
            x.size(0), x.size(-1), self.mode_count
        )
        outputs = []
        transition_rate = x.new_tensor(math.log(2.0))
        for index in range(x.size(1)):
            candidate, gate, delta = corrected[:, index].chunk(3, dim=-1)
            candidate = torch.tanh(candidate)
            gate = torch.sigmoid(gate)
            step = self.input_decay_scale * F.softplus(delta)
            retention = torch.exp(
                -transition_rate * step
            ).clamp(1e-4, 1.0 - 1e-4)
            update = 1.0 - retention
            hidden = (1.0 - update) * hidden + update * candidate

            modal_retention = retention.unsqueeze(-1).pow(
                self.rate_multipliers
            )
            writes = 2.0 * torch.sigmoid(
                F.linear(x[:, index], self.mode_input_weight)
            )
            reads = torch.softmax(
                F.linear(x[:, index], self.mode_read_weight),
                dim=-1,
            )
            modal_drive = candidate.unsqueeze(-1) * writes.unsqueeze(1)
            modal_hidden = (
                modal_retention * modal_hidden
                + (1.0 - modal_retention) * modal_drive
            )
            reference_hidden = (
                modal_retention * reference_hidden
                + (1.0 - modal_retention) * candidate.unsqueeze(-1)
            )
            reference_reads = x[:, index].new_full(
                (x.size(0), self.mode_count),
                1.0 / self.mode_count,
            )
            modal_readout = (
                modal_hidden * reads.unsqueeze(1)
            ).sum(dim=-1)
            reference_readout = (
                reference_hidden * reference_reads.unsqueeze(1)
            ).sum(dim=-1)
            if scope_weight is not None:
                gate = gate * scope_weight[:, index]
            original_output = (
                gate * hidden + (1.0 - gate) * x[:, index]
            )
            modal_output = (
                gate * modal_readout + (1.0 - gate) * x[:, index]
            )
            reference_output = (
                gate * reference_readout + (1.0 - gate) * x[:, index]
            )
            outputs.append(original_output + modal_output - reference_output)
        return self.drop(self.norm(torch.stack(outputs, dim=1)))


class A2GMambaKT(_V39):
    CANDIDATE_ID = "a2g_v33_modal_residual_20260913_v43"

    def __init__(self, *args, use_multimode_residual=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssm = MultiModeResidualInputMemorySSM.from_existing(
            self.ssm,
            mode_count=self.blocks[0]["attn"].num_heads,
            enabled=bool(int(use_multimode_residual)),
        )

    @property
    def use_multimode_residual(self):
        return self.ssm.use_multimode_residual

    @use_multimode_residual.setter
    def use_multimode_residual(self, value):
        self.ssm.use_multimode_residual = bool(int(value))


__all__ = [
    "A2GMambaKT",
    "MultiModeResidualInputMemorySSM",
]
