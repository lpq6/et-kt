"""Robust smoothing plus zero-initialized evidence-conditioned attention gate."""

from __future__ import annotations

import types

import torch
from torch import nn


class RobustCognitiveSmooth(nn.Module):
    """Causal RobustKT smoothing block used by the previous screen."""

    def __init__(self, hidden_size: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        if kernel_size != 5:
            raise ValueError("the preregistered smoothing kernel is fixed at 5")
        self.kernel_size = int(kernel_size)
        self.causal_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            self.kernel_size,
            padding=self.kernel_size - 1,
        )
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.out_dropout = nn.Dropout(float(dropout_rate))
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        channel_first = token.transpose(1, 2)
        trend = self.causal_conv(channel_first)
        trend = trend[:, :, : token.size(1)].transpose(1, 2)
        random = token - trend
        cognitive = trend + self.sqrt_beta.square() * random
        return self.norm(token + self.out_dropout(cognitive))


class EvidenceAttentionGate(nn.Module):
    """Per-channel residual gate, zero initialized for exact baseline identity."""

    def __init__(self, input_dim: int, hidden_size: int, gate_max: float):
        super().__init__()
        if input_dim != 6:
            raise ValueError(f"expected six learned-stat channels, got {input_dim}")
        if float(gate_max) != 0.25:
            raise ValueError("the preregistered attention gate maximum is fixed at 0.25")
        self.gate_max = float(gate_max)
        self.linear = nn.Linear(input_dim, hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, learned_stats: torch.Tensor) -> torch.Tensor:
        return self.gate_max * torch.tanh(self.linear(learned_stats))


def install_candidate(model: nn.Module, overrides: dict) -> None:
    """Install only the preregistered smoothing and attention-gate changes."""

    if overrides.get("robust_smooth_kernel") != 5:
        raise ValueError("robust_smooth_kernel must be the locked value 5")
    if overrides.get("attention_gate_max") != 0.25:
        raise ValueError("attention_gate_max must be the locked value 0.25")

    # Module initialization is deliberately RNG-neutral so the control and
    # candidate consume identical dropout streams after construction.
    rng_state = torch.random.get_rng_state()
    try:
        hidden_size = int(model.item_emb.embedding_dim)
        dropout_rate = float(getattr(model.input[3], "p", 0.2))
        model.robust_smooth = RobustCognitiveSmooth(
            hidden_size, 5, dropout_rate
        )
        original_input_forward = model.input.forward

        def smoothed_input_forward(module, features):
            del module
            return model.robust_smooth(original_input_forward(features))

        model.input.forward = types.MethodType(
            smoothed_input_forward, model.input
        )

        stat_dim = int(model.learned_stat_indices.numel())
        model.evidence_attention_gate = EvidenceAttentionGate(
            stat_dim, hidden_size, 0.25
        )
        model.attention_gate_enabled = True

        original_replace = model._replace_learned_exposure

        def capture_learned_stats(module, learned_stats, *args, **kwargs):
            result = original_replace(learned_stats, *args, **kwargs)
            model._gate_learned_stats = result
            return result

        model._replace_learned_exposure = types.MethodType(
            capture_learned_stats, model
        )

        original_ssm_forward = model.ssm.forward

        def capture_state(module, token, scope_weight=None):
            result = original_ssm_forward(token, scope_weight=scope_weight)
            model._gate_state = result
            return result

        model.ssm.forward = types.MethodType(capture_state, model.ssm)

        original_attend = model._attend

        def gated_attend(module, token):
            attended = original_attend(token)
            if not model.attention_gate_enabled:
                return attended
            gate = model.evidence_attention_gate(model._gate_learned_stats)
            return attended + gate * (model._gate_state - attended)

        model._attend = types.MethodType(gated_attend, model)
    finally:
        torch.random.set_rng_state(rng_state)


__all__ = [
    "EvidenceAttentionGate",
    "RobustCognitiveSmooth",
    "install_candidate",
]
