"""Five isolated A2G module candidates for validation-only screening.

The public ``A2GMambaKT`` class is never modified here.  Each candidate keeps
the frozen control path available through a zero/disabled flag so CPU smoke
tests can prove that the ablation wrapper did not change the control.

Candidates in this file:

* ``CLCRA_A2GMambaKT``: cross-layer causal raw-logit residual attention.
* ``RGLRUA2GMambaKT``: variance-preserving RG-LRU-style recurrence.
* ``HistoryAdaptiveDepthA2GMambaKT``: soft token-wise depth routing.
* ``CausalDiffusionStateA2GMambaKT``: causal noisy-state score auxiliary and
  one-step state correction, with a fixed four-level noise schedule.

CEPA-v1 remains in its separately audited source file and is imported by the
runner rather than duplicated here.
"""

from __future__ import annotations

import math
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F


BASE_WRAPPER_SHA256 = "509189b84aeb383d712945ab90b5992e0d76359f292df3f9ddfca726993584b9"


try:
    from work.a2g_mambakt_rwce_item_dropout_candidate import (
        A2GMambaKT as _FrozenA2G,
    )
    CONTROL_SOURCE_MODE = "authoritative_remote_wrapper_chain"
except ModuleNotFoundError as exc:
    if exc.name not in {"work", "work.a2g_mambakt_rwce_item_dropout_candidate"}:
        raise
    _module_name = "_a2g_five_module_local_control_20260805"
    _source = Path(__file__).resolve().parents[1] / "pykt" / "models" / "a2g_mambakt_final.py"
    _spec = importlib.util.spec_from_file_location(_module_name, _source)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Cannot load local A2G control from {_source}")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_module_name] = _module
    _spec.loader.exec_module(_module)
    _FrozenA2G = _module.A2GMambaKT

    CONTROL_SOURCE_MODE = "local_exact_control_snapshot"


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
        if isinstance(expected_value, float):
            equal = abs(float(actual) - expected_value) <= 1e-12
        else:
            equal = actual == expected_value
        if not equal:
            errors.append(f"{key}={actual!r}, expected {expected_value!r}")
    if errors:
        raise ValueError("Exact Full architecture/contract mismatch: " + "; ".join(errors))


class _PackedQKV:
    @staticmethod
    def project(x, attention):
        d_model = int(attention.embed_dim)
        if attention.in_proj_weight is None:
            raise TypeError("A2G attention must use packed in_proj_weight")
        qkv = F.linear(x, attention.in_proj_weight, attention.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        heads = int(attention.num_heads)
        head_dim = d_model // heads
        shape = (x.size(0), x.size(1), heads, head_dim)
        q = q.reshape(shape).transpose(1, 2)
        k = k.reshape(shape).transpose(1, 2)
        v = v.reshape(shape).transpose(1, 2)
        return q, k, v


class CLCRA_A2GMambaKT(_FrozenA2G):
    """RealFormer-style causal raw-logit residual attention.

    ``use_clcra=0`` delegates to the exact control.  The enabled path uses
    PyTorch SDPA with the previous layer's raw QK score as an additive mask,
    preserving the fused dropout implementation while changing only the
    attention-logit responsibility.
    """

    candidate_id = "cross_layer_causal_residual_attention_v1"

    def __init__(self, *args, use_clcra=1, **kwargs):
        _frozen_args(kwargs)
        super().__init__(*args, **kwargs)
        self.use_clcra = bool(int(use_clcra))
        self.model_name = "a2g_mambakt"

    def _attend(self, x):
        if not self.use_clcra:
            return super()._attend(x)
        length = x.size(1)
        future = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        causal_bias = x.new_zeros(length, length).masked_fill(future, float("-inf"))
        running_raw = None
        for block in self.blocks:
            attention = block["attn"]
            # Keep the frozen Full post-norm topology.  Only the additive
            # cross-layer raw-logit path is changed in this candidate.
            attention_input = x
            q, k, v = _PackedQKV.project(attention_input, attention)
            current_raw = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            additive = causal_bias
            if running_raw is not None:
                additive = causal_bias.unsqueeze(0).unsqueeze(0) + running_raw
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=additive,
                dropout_p=float(attention.dropout) if self.training else 0.0,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).reshape_as(attention_input)
            attended = F.linear(attended, attention.out_proj.weight, attention.out_proj.bias)
            x = block["norm"](x + block["drop"](attended))
            if "ffn" in block:
                x = block["ffn_norm"](x + block["ffn"](x))
            running_raw = current_raw if running_raw is None else running_raw + current_raw
        return x


class _VariancePreservingRGLRU(nn.Module):
    """Three-way projection with an exact control and RG-LRU-style branch."""

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
                # RG-LRU's variance-preserving input multiplier.  The decay
                # and gate parameters retain the frozen A2G parameterization.
                input_multiplier = torch.sqrt((1.0 - retention.square()).clamp_min(1e-6))
                h = retention * h + input_multiplier * candidate
            else:
                update = 1.0 - retention
                h = (1.0 - update) * h + update * candidate
            read_gate = gate if scope_weight is None else gate * scope_weight[:, index]
            outputs.append(read_gate * h + (1.0 - read_gate) * x[:, index])
        return self.drop(self.norm(torch.stack(outputs, dim=1)))


class RGLRUA2GMambaKT(_FrozenA2G):
    """Capacity-matched variance-preserving RG-LRU-style state replacement."""

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


class HistoryAdaptiveDepthA2GMambaKT(_FrozenA2G):
    """Soft token-wise depth routing over the four existing attention blocks."""

    candidate_id = "history_adaptive_depth_v1"

    def __init__(self, *args, use_history_adaptive_depth=1, **kwargs):
        _frozen_args(kwargs)
        super().__init__(*args, **kwargs)
        self.use_history_adaptive_depth = bool(int(use_history_adaptive_depth))
        self.depth_router = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1),
        )
        nn.init.zeros_(self.depth_router[1].weight)
        nn.init.zeros_(self.depth_router[1].bias)
        # Start close to the frozen final-block representation, while keeping
        # every depth differentiable for the first validation screen.
        self.depth_prior = nn.Parameter(torch.full((len(self.blocks) - 1,), -4.0))
        self.model_name = "a2g_mambakt"

    def _attend(self, x):
        if not self.use_history_adaptive_depth:
            return super()._attend(x)
        length = x.size(1)
        future = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        states = []
        halt_logits = []
        for block in self.blocks:
            attended, _ = block["attn"](
                x,
                x,
                x,
                attn_mask=future,
                need_weights=False,
            )
            # Match Full's post-norm attention and FFN residual ordering.
            x = block["norm"](x + block["drop"](attended))
            if "ffn" in block:
                x = block["ffn_norm"](x + block["ffn"](x))
            states.append(x)
            halt_logits.append(self.depth_router(x).squeeze(-1))
        remaining = x.new_ones(x.size(0), x.size(1))
        weighted = x.new_zeros(x.shape)
        for index, state in enumerate(states[:-1]):
            halt = torch.sigmoid(halt_logits[index] + self.depth_prior[index])
            weight = remaining * halt
            weighted = weighted + weight.unsqueeze(-1) * state
            remaining = remaining * (1.0 - halt)
        weighted = weighted + remaining.unsqueeze(-1) * states[-1]
        return weighted


class _CausalDiffusionTransition(nn.Module):
    """One-step causal score correction with a fixed four-level noise schedule."""

    def __init__(self, d_model, dropout, input_decay_scale, aux_weight, sigma_infer):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 3)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.input_decay_scale = float(input_decay_scale)
        self.aux_weight = float(aux_weight)
        self.sigma_infer = float(sigma_infer)
        self.score_norm = nn.LayerNorm(d_model)
        self.score_proj = nn.Linear(d_model, d_model)
        self.register_buffer(
            "sigma_schedule",
            torch.tensor([0.025, 0.05, 0.1, 0.2], dtype=torch.float32),
            persistent=False,
        )
        self.last_aux_loss = torch.tensor(0.0)

    def _score(self, noisy, token):
        context = noisy + 0.5 * token
        return self.score_proj(torch.tanh(self.score_norm(context)))

    def forward(self, x, scope_weight=None):
        h = torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)
        outputs = []
        losses = []
        transition_rate = x.new_tensor(math.log(2.0))
        for index in range(x.size(1)):
            token = x[:, index]
            candidate, gate, delta = self.in_proj(token).chunk(3, dim=-1)
            candidate = torch.tanh(candidate)
            gate = torch.sigmoid(gate)
            step = self.input_decay_scale * F.softplus(delta)
            retention = torch.exp(-transition_rate * step).clamp(1e-4, 1.0 - 1e-4)
            update = 1.0 - retention
            h = (1.0 - update) * h + update * candidate
            if not getattr(self, "use_diffusion", True):
                read_gate = gate if scope_weight is None else gate * scope_weight[:, index]
                outputs.append(read_gate * h + (1.0 - read_gate) * token)
                continue
            if self.training:
                level = torch.randint(
                    0,
                    int(self.sigma_schedule.numel()),
                    (x.size(0), 1),
                    device=x.device,
                )
                sigma = self.sigma_schedule[level].to(dtype=x.dtype)
                noise = torch.randn_like(h)
                noisy = h + sigma * noise
                predicted_noise = self._score(noisy, token)
                corrected = noisy - sigma * predicted_noise
                losses.append(F.mse_loss(predicted_noise, noise, reduction="none").mean(dim=-1))
            else:
                sigma = x.new_full((x.size(0), 1), self.sigma_infer)
                noisy = h
                predicted_noise = self._score(noisy, token)
                corrected = h - sigma * predicted_noise
            read_gate = gate if scope_weight is None else gate * scope_weight[:, index]
            outputs.append(read_gate * corrected + (1.0 - read_gate) * token)
        if losses and self.training:
            self.last_aux_loss = torch.stack(losses, dim=1).mean() * self.aux_weight
        else:
            self.last_aux_loss = x.new_zeros(())
        return self.drop(self.norm(torch.stack(outputs, dim=1)))


class CausalDiffusionStateA2GMambaKT(_FrozenA2G):
    """Causal diffusion-inspired state transition with a fixed one-step path."""

    candidate_id = "causal_diffusion_state_transition_v1"

    def __init__(
        self,
        *args,
        use_causal_diffusion=1,
        diffusion_aux_weight=0.05,
        diffusion_sigma_infer=0.05,
        **kwargs,
    ):
        _frozen_args(kwargs)
        super().__init__(*args, **kwargs)
        old_state = self.ssm.state_dict()
        replacement = _CausalDiffusionTransition(
            self.d_model,
            float(self.blocks[0]["attn"].dropout),
            float(self.ssm.input_decay_scale),
            aux_weight=float(diffusion_aux_weight),
            sigma_infer=float(diffusion_sigma_infer),
        )
        base_state = {
            key: value
            for key, value in old_state.items()
            if key in {"in_proj.weight", "in_proj.bias", "norm.weight", "norm.bias"}
        }
        replacement.load_state_dict(base_state, strict=False)
        replacement.use_diffusion = bool(int(use_causal_diffusion))
        self.ssm = replacement
        self.use_causal_diffusion = bool(int(use_causal_diffusion))
        self.diffusion_aux_weight = float(diffusion_aux_weight)
        self.diffusion_sigma_infer = float(diffusion_sigma_infer)
        self.model_name = "a2g_mambakt"

    def forward(self, dcur, train=False, qtest=False):
        if self.use_causal_diffusion:
            result = super().forward(dcur, train=train, qtest=qtest)
            if not train:
                return result
            prediction, _, route = result
            return prediction, self.ssm.last_aux_loss, route
        return super().forward(dcur, train=train, qtest=qtest)


__all__ = [
    "CLCRA_A2GMambaKT",
    "RGLRUA2GMambaKT",
    "HistoryAdaptiveDepthA2GMambaKT",
    "CausalDiffusionStateA2GMambaKT",
]
