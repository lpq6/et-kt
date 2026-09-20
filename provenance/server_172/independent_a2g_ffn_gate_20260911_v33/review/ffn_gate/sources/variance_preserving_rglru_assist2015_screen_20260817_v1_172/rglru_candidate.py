"""Single-point variance-preserving RG-LRU-style SSM candidate."""

from __future__ import annotations

import importlib.util
import math
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
    module_name = "_rglru_local_control_20260817"
    source = Path(__file__).resolve().parents[1] / "pykt" / "models" / "a2g_mambakt_final.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local A2G control from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _FrozenA2G = module.A2GMambaKT


def _frozen_args(kwargs):
    expected = {
        "d_model": 256,
        "d_ff": 512,
        "n_blocks": 4,
        "num_attn_heads": 8,
        "dropout": 0.2,
        "seq_len": 200,
        "normalization_topology": "full_postnorm",
        "use_split_boundary": 1,
        "use_recency_weighted_concept_evidence": 1,
        "use_evidence_equivalent_residual": 1,
        "item_residual_dropout": 0.0,
        "item_residual_dropout_schedule": "constant",
    }
    errors = []
    for key, expected_value in expected.items():
        actual = kwargs.get(key, expected_value)
        equal = abs(float(actual) - expected_value) <= 1e-12 if isinstance(expected_value, float) else actual == expected_value
        if not equal:
            errors.append(f"{key}={actual!r}, expected {expected_value!r}")
    if errors:
        raise ValueError("Exact Full architecture/contract mismatch: " + "; ".join(errors))


class _VariancePreservingRGLRU(nn.Module):
    """Capacity-matched recurrence with RG-LRU variance-preserving input."""

    def __init__(self, d_model, dropout, input_decay_scale):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 3)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.input_decay_scale = float(input_decay_scale)
        self.use_rglru = True

    def forward(self, x, scope_weight=None):
        h = torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)
        outputs = []
        transition_rate = x.new_tensor(math.log(2.0))
        for index in range(x.size(1)):
            candidate, gate, delta = self.in_proj(x[:, index]).chunk(3, dim=-1)
            candidate = torch.tanh(candidate)
            gate = torch.sigmoid(gate)
            step = self.input_decay_scale * F.softplus(delta)
            retention = torch.exp(-transition_rate * step).clamp(1e-4, 1.0 - 1e-4)
            if self.use_rglru:
                input_multiplier = torch.sqrt((1.0 - retention.square()).clamp_min(1e-6))
                h = retention * h + input_multiplier * candidate
            else:
                update = 1.0 - retention
                h = retention * h + update * candidate
            read_gate = gate if scope_weight is None else gate * scope_weight[:, index]
            outputs.append(read_gate * h + (1.0 - read_gate) * x[:, index])
        return self.drop(self.norm(torch.stack(outputs, dim=1)))


class RGLRUA2GMambaKT(_FrozenA2G):
    """Replace only the SSM state update with a variance-preserving branch."""

    candidate_id = "variance_preserving_rglru_recurrence_v1"

    def __init__(self, *args, use_rglru=1, **kwargs):
        _frozen_args(kwargs)
        super().__init__(*args, **kwargs)
        old_state = self.ssm.state_dict()
        replacement = _VariancePreservingRGLRU(
            self.d_model,
            float(self.blocks[0]["attn"].dropout),
            float(self.ssm.input_decay_scale),
        )
        replacement.load_state_dict(old_state, strict=True)
        replacement.use_rglru = bool(int(use_rglru))
        self.ssm = replacement
        self.use_rglru = bool(int(use_rglru))
        self.model_name = "a2g_mambakt"


__all__ = ["RGLRUA2GMambaKT"]
