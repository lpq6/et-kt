"""Unified A2G with a causal concept-keyed gated delta-state residual.

This candidate changes exactly one mechanism relative to the frozen unified
simplified control.  RWCE, split-boundary routing, and item dropout remain
disabled by the parent class; a small learner-local associative state is added
before causal attention.  The state predicts the value currently associated
with a concept key and applies a response-conditioned delta-rule correction.

The output LayerScale is initialized to zero.  With the mechanism enabled at
construction, all retained control parameters and the caller RNG stream remain
paired, while the initial forward function is exactly the unified control.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from work.a2g_mambakt_unified_simplified_candidate_20260730 import (
    A2GMambaKT as _UnifiedSimplifiedA2G,
)


class ConceptKeyedGatedDeltaState(nn.Module):
    """Maintain a compact per-learner key-value state with gated delta writes."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        key_dim: int = 8,
        value_dim: int = 8,
        initial_retention: float = 0.98,
        initial_write_rate: float = 0.10,
    ):
        super().__init__()
        if min(d_model, num_heads, key_dim, value_dim) <= 0:
            raise ValueError("gated-delta dimensions must be positive")
        if not 0.0 < initial_retention < 1.0:
            raise ValueError("initial_retention must lie in (0, 1)")
        if not 0.0 < initial_write_rate < 1.0:
            raise ValueError("initial_write_rate must lie in (0, 1)")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)

        key_width = self.num_heads * self.key_dim
        value_width = self.num_heads * self.value_dim
        self.key_proj = nn.Linear(self.d_model, key_width, bias=False)
        self.value_proj = nn.Linear(self.d_model, value_width, bias=False)
        self.retention_proj = nn.Linear(self.d_model, self.num_heads)
        self.write_proj = nn.Linear(self.d_model, self.num_heads)
        self.output_proj = nn.Linear(value_width, self.d_model, bias=False)
        self.raw_output_scale = nn.Parameter(torch.zeros(self.d_model))

        with torch.no_grad():
            self.retention_proj.weight.zero_()
            self.retention_proj.bias.fill_(
                math.log(initial_retention / (1.0 - initial_retention))
            )
            self.write_proj.weight.zero_()
            self.write_proj.bias.fill_(
                math.log(initial_write_rate / (1.0 - initial_write_rate))
            )

    def effective_output_scale(self):
        return torch.tanh(self.raw_output_scale)

    def dynamic_state_elements(self, batch_size: int) -> int:
        return (
            int(batch_size)
            * self.num_heads
            * self.value_dim
            * self.key_dim
        )

    @staticmethod
    def _read(memory: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return (memory * key.unsqueeze(-2)).sum(dim=-1)

    @staticmethod
    def _outer(error: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        # Avoid the experimental torch native bmm_outer_product dispatch used
        # by the remote nightly build.  key_dim is frozen to 8, so this small
        # explicit stack preserves the exact delta-rule algebra at low cost.
        return torch.stack(
            [error * key[..., index].unsqueeze(-1) for index in range(key.size(-1))],
            dim=-1,
        )

    def forward(
        self,
        target_concept: torch.Tensor,
        history_concept: torch.Tensor,
        history_interaction: torch.Tensor,
        history_concept_ids: torch.Tensor,
        history_responses: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = target_concept.shape
        memory = target_concept.new_zeros(
            batch_size,
            self.num_heads,
            self.value_dim,
            self.key_dim,
        )
        query = self.key_proj(target_concept).view(
            batch_size, sequence_length, self.num_heads, self.key_dim
        )
        write_key = self.key_proj(history_concept).view(
            batch_size, sequence_length, self.num_heads, self.key_dim
        )
        value = torch.tanh(self.value_proj(history_interaction)).view(
            batch_size, sequence_length, self.num_heads, self.value_dim
        )
        query = F.normalize(query, dim=-1, eps=1e-6)
        write_key = F.normalize(write_key, dim=-1, eps=1e-6)
        retention = torch.sigmoid(self.retention_proj(history_interaction))
        write_rate = torch.sigmoid(self.write_proj(history_interaction))
        valid_write = history_concept_ids.gt(0) & history_responses.ge(0)
        valid_write = valid_write & history_responses.le(1)

        reads = []
        for index in range(sequence_length):
            valid = valid_write[:, index].unsqueeze(-1)
            retain = torch.where(
                valid,
                retention[:, index],
                torch.ones_like(retention[:, index]),
            )
            rate = torch.where(
                valid,
                write_rate[:, index],
                torch.zeros_like(write_rate[:, index]),
            )
            key = write_key[:, index]
            predicted_value = self._read(memory, key)
            error = value[:, index] - predicted_value
            correction = self._outer(error, key)
            memory = retain.unsqueeze(-1).unsqueeze(-1) * memory
            memory = memory + rate.unsqueeze(-1).unsqueeze(-1) * correction
            reads.append(self._read(memory, query[:, index]))

        read = torch.stack(reads, dim=1).reshape(
            batch_size, sequence_length, self.num_heads * self.value_dim
        )
        residual = self.output_proj(read)
        scale = self.effective_output_scale().to(dtype=residual.dtype)
        return residual * scale.view(1, 1, -1)


class A2GMambaKT(_UnifiedSimplifiedA2G):
    """Frozen unified control plus one causal gated-delta state residual."""

    def __init__(
        self,
        *args,
        use_gated_delta_state=1,
        gated_delta_heads: int = 8,
        gated_delta_key_dim: int = 8,
        gated_delta_value_dim: int = 8,
        gated_delta_init_seed: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_gated_delta_state = bool(int(use_gated_delta_state))
        self.gated_delta_init_seed = int(gated_delta_init_seed)
        if not self.use_gated_delta_state:
            return
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.gated_delta_init_seed)
            self.gated_delta_state = ConceptKeyedGatedDeltaState(
                self.d_model,
                num_heads=int(gated_delta_heads),
                key_dim=int(gated_delta_key_dim),
                value_dim=int(gated_delta_value_dim),
            )

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_gated_delta_state:
            return super().forward(dcur, train=train, qtest=qtest)

        dropout = self.input[3]
        target_q, target_c, hist_q, hist_c, hist_r = self._sequences(dcur)
        if self.training and dropout.warmup_active:
            dropout.set_protection_mask(self._concept_novelty_mask(target_c))
        try:
            concept_prior = self.concept_prior(target_c.clamp_min(0))
            has_item_sequence = (
                self.n_pid > 0
                and target_q.numel() > 0
                and target_q.size(1) == target_c.size(1)
            )
            if has_item_sequence:
                item_prior = self.item_prior(target_q.clamp_min(0))
            else:
                item_prior = concept_prior.new_zeros(concept_prior.shape)
            stats = self._stats(target_c, hist_c, hist_r, item_prior, concept_prior)
            learned_stats = self._learned_stats(stats)
            learned_stats = self._replace_learned_exposure(
                learned_stats,
                target_c,
                hist_c,
                hist_r,
            )

            target_concept = self.concept_emb(target_c.clamp_min(0))
            if has_item_sequence:
                target_item = self.item_emb(target_q.clamp_min(0))
                target_item = target_item * self._item_residual_reliability(
                    target_q
                ).unsqueeze(-1)
                target_item = self._apply_item_residual_dropout(target_item, target_q)
            else:
                target_item = torch.zeros_like(target_concept)
            target = target_item + target_concept

            history_concept = self.hist_concept_emb(hist_c.clamp_min(0))
            has_history_items = (
                self.n_pid > 0
                and hist_q.numel() > 0
                and hist_q.size(1) == hist_c.size(1)
            )
            if has_history_items:
                history_item = self.hist_item_emb(hist_q.clamp_min(0))
                history_item = history_item * self._item_residual_reliability(
                    hist_q
                ).unsqueeze(-1)
                history_item = self._apply_item_residual_dropout(history_item, hist_q)
            else:
                history_item = torch.zeros_like(history_concept)
            history = history_item + history_concept
            history = history + self.resp_emb(hist_r.clamp(0, 2))

            token = self.input(
                torch.cat(
                    [
                        target,
                        history,
                        target_item,
                        target_concept,
                        item_prior,
                        concept_prior,
                        learned_stats,
                    ],
                    dim=-1,
                )
            )
            scope_weight = self._boundary_weights(token, target_c)
            state = self.ssm(token, scope_weight=scope_weight)
            delta_state = self.gated_delta_state(
                target_concept,
                history_concept,
                history,
                hist_c,
                hist_r,
            )
            sequence = self._attend(token + state + delta_state)
            fused = torch.cat([sequence, learned_stats], dim=-1)
            prior_logit = (item_prior + concept_prior).squeeze(-1)
            logits = (
                self.pred(fused).squeeze(-1)
                + self.prior_scale * prior_logit
                + self.stat_scale * self._direct_stat_logit(stats)
            )
            prediction = torch.sigmoid(logits)
            if qtest and not train:
                return prediction, fused
            if not train:
                return prediction
            route = sequence.new_zeros(4)
            return prediction, sequence.new_tensor(0.0), route
        finally:
            dropout.clear_protection_mask()


__all__ = ["A2GMambaKT", "ConceptKeyedGatedDeltaState"]
