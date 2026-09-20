"""V39 candidate: finite causal SSM input memory (not an AUC claim)."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from .core import SelectiveSSMBlock


class CausalInputMemory(nn.Module):
    KERNEL_SIZE = 4

    def __init__(self, channels, *, device=None, dtype=None):
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("input-memory channels must be a positive integer")
        self.channels = channels
        self.weight = nn.Parameter(
            torch.zeros(channels, 1, self.KERNEL_SIZE, device=device, dtype=dtype)
        )

    def forward(self, projected):
        if (
            projected.ndim != 3
            or min(projected.shape) < 1
            or projected.size(-1) != self.channels
            or (not projected.is_floating_point())
            or (projected.device != self.weight.device)
            or (projected.dtype != self.weight.dtype)
        ):
            raise ValueError(
                "projected inputs must share the memory's nonempty B/L/C geometry, dtype and device"
            )
        history = F.pad(projected.transpose(1, 2), (self.KERNEL_SIZE, 0))[..., :-1]
        return F.conv1d(history, self.weight, groups=self.channels).transpose(1, 2)


class InputMemorySSM(SelectiveSSMBlock):
    @classmethod
    def from_existing(cls, source, enabled=True):
        if type(source) is not SelectiveSSMBlock:
            raise TypeError("expected the retained V33 selective SSM")
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        (module.in_proj, module.norm, module.drop) = (
            source.in_proj,
            source.norm,
            source.drop,
        )
        module.input_decay_scale = source.input_decay_scale
        module.use_umk_ssm = source.use_umk_ssm
        module.umk_lambda0 = source.umk_lambda0
        if source.use_umk_ssm:
            module.umk_alpha = source.umk_alpha
        module.input_memory = CausalInputMemory(
            source.in_proj.out_features,
            device=source.in_proj.weight.device,
            dtype=source.in_proj.weight.dtype,
        )
        module.use_input_memory = bool(enabled)
        return module

    def forward(self, x, scope_weight=None, concept_prior=None):
        if not self.use_input_memory:
            return super().forward(x, scope_weight=scope_weight, concept_prior=concept_prior)
        if (
            x.ndim != 3
            or min(x.shape) < 1
            or x.size(-1) != self.in_proj.in_features
            or (not x.is_floating_point())
            or (x.device != self.in_proj.weight.device)
            or (x.dtype != self.in_proj.weight.dtype)
        ):
            raise ValueError(
                "SSM inputs must share the original projection's nonempty B/L/D geometry, dtype and device"
            )
        if scope_weight is not None and (
            scope_weight.shape not in (x.shape, (*x.shape[:2], 1))
            or scope_weight.device != x.device
            or scope_weight.dtype != x.dtype
        ):
            raise ValueError(
                "scope weights must share the input B/L/D or B/L/1 layout, dtype and device"
            )
        umk_exponent = self._umk_exponent(x, concept_prior)
        projected = torch.stack(
            [self.in_proj(x[:, index]) for index in range(x.size(1))], dim=1
        )
        corrected = projected + self.input_memory(projected)
        hidden = x.new_zeros(x.size(0), x.size(-1))
        outputs = []
        transition_rate = x.new_tensor(math.log(2.0))
        for index in range(x.size(1)):
            (candidate, gate, delta) = corrected[:, index].chunk(3, dim=-1)
            candidate = torch.tanh(candidate)
            gate = torch.sigmoid(gate)
            step = self.input_decay_scale * F.softplus(delta)
            retention = torch.exp(-transition_rate * step).clamp(0.0001, 1.0 - 0.0001)
            if umk_exponent is not None:
                retention = retention.pow(umk_exponent[:, index])
            update = 1.0 - retention
            hidden = (1.0 - update) * hidden + update * candidate
            if scope_weight is not None:
                gate = gate * scope_weight[:, index]
            outputs.append(gate * hidden + (1.0 - gate) * x[:, index])
        return self.drop(self.norm(torch.stack(outputs, dim=1)))
