"""Server-equivalent SSM and folded normalization."""

import math
import torch
from torch import nn
import torch.nn.functional as F

from .umk import concept_rate, validate_lambda0


class SelectiveSSMBlock(nn.Module):
    def __init__(
        self, d_model, dropout=0.1, input_decay_scale=0.5,
        use_umk_ssm=0, umk_lambda0=0.3,
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 3)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.input_decay_scale = float(input_decay_scale)
        self.use_umk_ssm = bool(use_umk_ssm)
        self.umk_lambda0 = validate_lambda0(umk_lambda0)
        if self.use_umk_ssm:
            self.umk_alpha = nn.Parameter(self.in_proj.weight.new_tensor(0.3))

    def _umk_exponent(self, x, concept_prior):
        if not self.use_umk_ssm:
            return None
        rate = concept_rate(concept_prior, self.umk_lambda0)
        if rate.shape != x.shape[:2] or rate.device != x.device or rate.dtype != x.dtype:
            raise ValueError("UMK SSM prior must share input [B,T], dtype and device")
        # decay_eff = decay ** (1 + alpha * lambda_c), alpha starts at 0.3.
        return (1.0 + self.umk_alpha * rate).unsqueeze(-1)

    def forward(self, x, scope_weight=None, concept_prior=None):
        umk_exponent = self._umk_exponent(x, concept_prior)
        h = torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)
        outputs = []
        transition_rate = x.new_tensor(math.log(2.0))
        for index in range(x.size(1)):
            (candidate, gate, delta) = self.in_proj(x[:, index]).chunk(3, dim=-1)
            candidate = torch.tanh(candidate)
            gate = torch.sigmoid(gate)
            step = self.input_decay_scale * F.softplus(delta)
            retention = torch.exp(-transition_rate * step).clamp(0.0001, 1.0 - 0.0001)
            if umk_exponent is not None:
                retention = retention.pow(umk_exponent[:, index])
            update = 1.0 - retention
            h = (1.0 - update) * h + update * candidate
            if scope_weight is not None:
                gate = gate * scope_weight[:, index]
            outputs.append(gate * h + (1.0 - gate) * x[:, index])
        return self.drop(self.norm(torch.stack(outputs, dim=1)))


class FoldedZeroNormLinear(nn.Module):
    """Reference-width LayerNorm and linear projection with zero channels folded."""

    def __init__(self, active_indices, reference_dim, out_features, eps=1e-05):
        super().__init__()
        active_indices = torch.as_tensor(active_indices, dtype=torch.long)
        self.register_buffer("active_indices", active_indices, persistent=False)
        self.reference_dim = int(reference_dim)
        self.eps = float(eps)
        active_dim = int(active_indices.numel())
        self.norm_weight = nn.Parameter(torch.ones(active_dim))
        self.norm_bias = nn.Parameter(torch.zeros(active_dim))
        self.active_weight = nn.Parameter(torch.empty(out_features, active_dim))
        self.zero_norm_weight = nn.Parameter(torch.empty(out_features))
        self.linear_bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        full_weight = self.active_weight.new_empty(
            self.active_weight.size(0), self.reference_dim
        )
        nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.active_weight.copy_(full_weight.index_select(1, self.active_indices))
            active = torch.zeros(
                self.reference_dim, dtype=torch.bool, device=full_weight.device
            )
            active[self.active_indices] = True
            self.zero_norm_weight.copy_(full_weight[:, ~active].sum(dim=1))
        bound = 1.0 / math.sqrt(self.reference_dim)
        nn.init.uniform_(self.linear_bias, -bound, bound)

    def forward(self, x):
        mean = x.sum(dim=-1, keepdim=True) / self.reference_dim
        variance = (
            x.square().sum(dim=-1, keepdim=True) / self.reference_dim - mean.square()
        )
        inv_std = torch.rsqrt(variance.clamp_min(0.0) + self.eps)
        normalized = (x - mean) * inv_std
        normalized = normalized * self.norm_weight + self.norm_bias
        zero_normalized = (-mean * inv_std).squeeze(-1)
        return (
            F.linear(normalized, self.active_weight, self.linear_bias)
            + zero_normalized.unsqueeze(-1) * self.zero_norm_weight
        )


class FoldedSelectedNormLinear(nn.Module):
    """LayerNorm-linear pair with selected zero-valued channels folded out."""

    def __init__(self, active_indices, reference_dim, out_features, eps=1e-05):
        super().__init__()
        active_indices = torch.as_tensor(active_indices, dtype=torch.long)
        self.register_buffer("active_indices", active_indices, persistent=False)
        self.reference_dim = int(reference_dim)
        self.eps = float(eps)
        active_dim = int(active_indices.numel())
        self.norm_weight = nn.Parameter(torch.empty(active_dim))
        self.norm_bias = nn.Parameter(torch.empty(active_dim))
        self.active_weight = nn.Parameter(torch.empty(out_features, active_dim))
        self.zero_norm_weight = nn.Parameter(torch.empty(out_features))
        self.linear_bias = nn.Parameter(torch.empty(out_features))

    @staticmethod
    def _partition(reference_dim, active_indices, device):
        active = torch.zeros(reference_dim, dtype=torch.bool, device=device)
        active[active_indices] = True
        return (active, ~active)

    @classmethod
    def from_dense(cls, norm, linear, active_indices):
        reference_dim = int(norm.normalized_shape[0])
        active_indices = torch.as_tensor(
            active_indices, dtype=torch.long, device=linear.weight.device
        )
        module = cls(
            active_indices.cpu(), reference_dim, linear.out_features, eps=norm.eps
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        (_, removed) = cls._partition(
            reference_dim, active_indices, linear.weight.device
        )
        with torch.no_grad():
            module.norm_weight.copy_(norm.weight.index_select(0, active_indices))
            module.norm_bias.copy_(norm.bias.index_select(0, active_indices))
            module.active_weight.copy_(linear.weight.index_select(1, active_indices))
            module.zero_norm_weight.copy_(
                (linear.weight[:, removed] * norm.weight[removed]).sum(dim=1)
            )
            module.linear_bias.copy_(
                linear.bias
                + (linear.weight[:, removed] * norm.bias[removed]).sum(dim=1)
            )
        return module

    @classmethod
    def from_folded(cls, folded, retained_local_indices):
        retained_local_indices = torch.as_tensor(
            retained_local_indices, dtype=torch.long, device=folded.active_weight.device
        )
        local_active = torch.zeros(
            folded.active_weight.size(1),
            dtype=torch.bool,
            device=folded.active_weight.device,
        )
        local_active[retained_local_indices] = True
        local_removed = ~local_active
        reference_indices = folded.active_indices.to(
            folded.active_weight.device
        ).index_select(0, retained_local_indices)
        module = cls(
            reference_indices.cpu(),
            folded.reference_dim,
            folded.active_weight.size(0),
            eps=folded.eps,
        ).to(device=folded.active_weight.device, dtype=folded.active_weight.dtype)
        with torch.no_grad():
            module.norm_weight.copy_(
                folded.norm_weight.index_select(0, retained_local_indices)
            )
            module.norm_bias.copy_(
                folded.norm_bias.index_select(0, retained_local_indices)
            )
            module.active_weight.copy_(
                folded.active_weight.index_select(1, retained_local_indices)
            )
            module.zero_norm_weight.copy_(
                folded.zero_norm_weight
                + (
                    folded.active_weight[:, local_removed]
                    * folded.norm_weight[local_removed]
                ).sum(dim=1)
            )
            module.linear_bias.copy_(
                folded.linear_bias
                + (
                    folded.active_weight[:, local_removed]
                    * folded.norm_bias[local_removed]
                ).sum(dim=1)
            )
        return module

    def forward(self, x):
        mean = x.sum(dim=-1, keepdim=True) / self.reference_dim
        variance = x.square().sum(dim=-1, keepdim=True) / self.reference_dim
        variance = variance - mean.square()
        inv_std = torch.rsqrt(variance.clamp_min(0.0) + self.eps)
        normalized = (x - mean) * inv_std
        normalized = normalized * self.norm_weight + self.norm_bias
        zero_normalized = (-mean * inv_std).squeeze(-1)
        return (
            F.linear(normalized, self.active_weight, self.linear_bias)
            + zero_normalized.unsqueeze(-1) * self.zero_norm_weight
        )


class OneEpochConceptNoveltyDropout(nn.Module):
    """Protect each concept's first causal token during the first epoch."""

    def __init__(self, dropout):
        super().__init__()
        if not isinstance(dropout, nn.Dropout):
            raise TypeError(f"Expected nn.Dropout, got {type(dropout).__name__}")
        self.p = float(dropout.p)
        self.warmup_active = True
        self._protection_mask = None

    def set_protection_mask(self, mask):
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise TypeError("Protection mask must be a rank-2 boolean tensor")
        self._protection_mask = mask

    def clear_protection_mask(self):
        self._protection_mask = None

    def complete_warmup(self):
        self.warmup_active = False
        self.clear_protection_mask()

    def forward(self, inputs):
        dropped = F.dropout(inputs, p=self.p, training=self.training, inplace=False)
        if not self.training or not self.warmup_active:
            return dropped
        if self._protection_mask is None:
            raise RuntimeError("Concept-novelty mask was not set before input dropout")
        if tuple(self._protection_mask.shape) != tuple(inputs.shape[:2]):
            raise ValueError("Concept-novelty mask does not match the input sequence")
        return torch.where(self._protection_mask.unsqueeze(-1), inputs, dropped)
