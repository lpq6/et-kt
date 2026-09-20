"""Parameter-neutral two-timescale Selective-SSM for the frozen A2G model.

The original projection, gate, normalization, and dropout are retained.  The
single recurrent state is replaced by tied fast/slow states whose transition
rates are fixed one octave around the control rate (2.0 and 0.5).  Their
arithmetic mean is read through the original gate.  No new trainable state is
introduced and no response at or after the prediction position is accessed.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F


FAST_RATE = 2.0
SLOW_RATE = 0.5


def _two_timescale_forward(self, x, scope_weight=None):
    fast = torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)
    slow = torch.zeros_like(fast)
    outputs = []
    transition_rate = x.new_tensor(math.log(2.0))
    for index in range(x.size(1)):
        candidate, gate, delta = self.in_proj(x[:, index]).chunk(3, dim=-1)
        candidate = torch.tanh(candidate)
        gate = torch.sigmoid(gate)
        step = self.input_decay_scale * F.softplus(delta)
        fast_retention = torch.exp(
            -transition_rate * FAST_RATE * step
        ).clamp(1e-4, 1.0 - 1e-4)
        slow_retention = torch.exp(
            -transition_rate * SLOW_RATE * step
        ).clamp(1e-4, 1.0 - 1e-4)
        fast = fast_retention * fast + (1.0 - fast_retention) * candidate
        slow = slow_retention * slow + (1.0 - slow_retention) * candidate
        state = 0.5 * (fast + slow)
        if scope_weight is not None:
            gate = gate * scope_weight[:, index]
        outputs.append(gate * state + (1.0 - gate) * x[:, index])
    return self.drop(self.norm(torch.stack(outputs, dim=1)))


def install_two_timescale_selective_ssm(model) -> None:
    """Install the fixed tied-state recurrence without changing parameters."""

    required = {"in_proj", "norm", "drop", "input_decay_scale"}
    missing = sorted(name for name in required if not hasattr(model.ssm, name))
    if missing:
        raise TypeError(f"frozen Selective-SSM contract missing: {missing}")
    model.ssm.forward = types.MethodType(_two_timescale_forward, model.ssm)
    model.use_two_timescale_selective_ssm = True


__all__ = [
    "FAST_RATE",
    "SLOW_RATE",
    "install_two_timescale_selective_ssm",
]
