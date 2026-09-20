from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from work.a2g_mambakt_joint_weak_stat_prune_candidate import (
    A2GMambaKT as _FunctionalA2GMambaKT,
    OneEpochConceptNoveltyDropout,
    SelectiveSSMBlock,
)


LEARNED_STAT_INDICES = (0, 1, 2, 3, 4, 6)
REMOVED_STAT_INDICES = (5, 7)


class FoldedSelectedNormLinear(nn.Module):
    """LayerNorm-linear pair with selected zero-valued channels folded out."""

    def __init__(self, active_indices, reference_dim, out_features, eps=1e-5):
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
        return active, ~active

    @classmethod
    def from_dense(cls, norm, linear, active_indices):
        reference_dim = int(norm.normalized_shape[0])
        active_indices = torch.as_tensor(
            active_indices,
            dtype=torch.long,
            device=linear.weight.device,
        )
        module = cls(
            active_indices.cpu(),
            reference_dim,
            linear.out_features,
            eps=norm.eps,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        _, removed = cls._partition(reference_dim, active_indices, linear.weight.device)
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
            retained_local_indices,
            dtype=torch.long,
            device=folded.active_weight.device,
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


class A2GMambaKT(_FunctionalA2GMambaKT):
    """Physical form with two weak learned-stat channels folded out."""

    def __init__(self, *args, **kwargs):
        kwargs["use_joint_weak_stat_prune"] = 1
        super().__init__(*args, **kwargs)
        rng_state = torch.random.get_rng_state().clone()

        input_reference_dim = self.d_model * 4 + 10
        input_removed = {
            self.d_model * 4 + 2 + index for index in REMOVED_STAT_INDICES
        }
        input_active = [
            index
            for index in range(input_reference_dim)
            if index not in input_removed
        ]
        old_input = self.input
        folded_input = FoldedSelectedNormLinear.from_dense(
            old_input[0], old_input[1], input_active
        )
        self.input = nn.Sequential(
            folded_input,
            nn.Identity(),
            old_input[2],
            old_input[3],
        )

        old_pred_input = self.pred[0]
        pred_retained_local = list(range(self.d_model))
        pred_retained_local.extend(
            self.d_model + index for index in LEARNED_STAT_INDICES
        )
        self.pred[0] = FoldedSelectedNormLinear.from_folded(
            old_pred_input, pred_retained_local
        )
        self.register_buffer(
            "learned_stat_indices",
            torch.tensor(LEARNED_STAT_INDICES, dtype=torch.long),
            persistent=False,
        )
        torch.random.set_rng_state(rng_state)

    def _learned_stats(self, stats):
        return stats.index_select(-1, self.learned_stat_indices.to(stats.device))


def fold_functional_state_dict(state, d_model):
    """Convert a trained functional-deletion checkpoint to the physical form."""
    converted = OrderedDict((key, value.clone()) for key, value in state.items())

    input_reference_dim = d_model * 4 + 10
    input_removed = [d_model * 4 + 2 + index for index in REMOVED_STAT_INDICES]
    input_active = [
        index for index in range(input_reference_dim) if index not in input_removed
    ]
    input_active_tensor = torch.tensor(
        input_active,
        dtype=torch.long,
        device=state["input.1.weight"].device,
    )
    input_removed_tensor = torch.tensor(
        input_removed,
        dtype=torch.long,
        device=state["input.1.weight"].device,
    )
    input_norm_weight = state["input.0.weight"]
    input_norm_bias = state["input.0.bias"]
    input_linear_weight = state["input.1.weight"]
    input_linear_bias = state["input.1.bias"]
    for key in (
        "input.0.weight",
        "input.0.bias",
        "input.1.weight",
        "input.1.bias",
    ):
        converted.pop(key)
    converted["input.0.norm_weight"] = input_norm_weight.index_select(
        0, input_active_tensor
    ).clone()
    converted["input.0.norm_bias"] = input_norm_bias.index_select(
        0, input_active_tensor
    ).clone()
    converted["input.0.active_weight"] = input_linear_weight.index_select(
        1, input_active_tensor
    ).clone()
    converted["input.0.zero_norm_weight"] = (
        input_linear_weight.index_select(1, input_removed_tensor)
        * input_norm_weight.index_select(0, input_removed_tensor)
    ).sum(dim=1)
    converted["input.0.linear_bias"] = input_linear_bias + (
        input_linear_weight.index_select(1, input_removed_tensor)
        * input_norm_bias.index_select(0, input_removed_tensor)
    ).sum(dim=1)

    pred_retained_local = list(range(d_model))
    pred_retained_local.extend(d_model + index for index in LEARNED_STAT_INDICES)
    pred_removed_local = [d_model + index for index in REMOVED_STAT_INDICES]
    pred_retained_tensor = torch.tensor(
        pred_retained_local,
        dtype=torch.long,
        device=state["pred.0.active_weight"].device,
    )
    pred_removed_tensor = torch.tensor(
        pred_removed_local,
        dtype=torch.long,
        device=state["pred.0.active_weight"].device,
    )
    pred_norm_weight = state["pred.0.norm_weight"]
    pred_norm_bias = state["pred.0.norm_bias"]
    pred_active_weight = state["pred.0.active_weight"]
    converted["pred.0.norm_weight"] = pred_norm_weight.index_select(
        0, pred_retained_tensor
    ).clone()
    converted["pred.0.norm_bias"] = pred_norm_bias.index_select(
        0, pred_retained_tensor
    ).clone()
    converted["pred.0.active_weight"] = pred_active_weight.index_select(
        1, pred_retained_tensor
    ).clone()
    converted["pred.0.zero_norm_weight"] = state["pred.0.zero_norm_weight"] + (
        pred_active_weight.index_select(1, pred_removed_tensor)
        * pred_norm_weight.index_select(0, pred_removed_tensor)
    ).sum(dim=1)
    converted["pred.0.linear_bias"] = state["pred.0.linear_bias"] + (
        pred_active_weight.index_select(1, pred_removed_tensor)
        * pred_norm_bias.index_select(0, pred_removed_tensor)
    ).sum(dim=1)
    return converted
