"""Within-interaction attentive field-set adapter for frozen A2G-MambaKT.

The adapter transfers the interaction-set representation responsibility from
KTST to the five fields already available at one prediction step: target item,
target concept, historical item, historical concept, and historical response.
It never mixes positions. A content-adaptive, permutation-invariant attentive
pool is projected through an exactly zero-initialized residual immediately
before the frozen selective SSM. No metadata, temporal attention, dropout,
objective, or RNG draw is added to the forward path.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from work.a2g_mambakt_rwce_item_dropout_candidate import (
    A2GMambaKT as _FrozenA2G,
)


class InteractionSetEncoderA2GMambaKT(_FrozenA2G):
    """Frozen Full control plus a zero-start within-step field-set residual."""

    candidate_id = "interaction_set_encoder_v1"
    SET_SCALE = 0.10

    def __init__(self, *args, use_interaction_set_encoder=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_interaction_set_encoder = bool(int(use_interaction_set_encoder))
        rng_state = torch.random.get_rng_state().clone()
        self.interaction_set_projection = nn.Linear(
            self.d_model,
            self.d_model,
            bias=False,
        )
        nn.init.zeros_(self.interaction_set_projection.weight)
        torch.random.set_rng_state(rng_state)
        self.model_name = "a2g_mambakt"

    @staticmethod
    def _attentive_set_pool(fields: torch.Tensor) -> torch.Tensor:
        """Pool [batch, time, field, channel] without temporal mixing."""
        normalized = F.normalize(fields, p=2.0, dim=-1, eps=1e-6)
        query = normalized.mean(dim=-2, keepdim=True)
        scores = (normalized * query).sum(dim=-1) / math.sqrt(fields.size(-1))
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * fields).sum(dim=-2)

    def _interaction_set_residual(self, *fields: torch.Tensor) -> torch.Tensor:
        reference = fields[0]
        if not self.use_interaction_set_encoder:
            return reference.new_zeros(reference.shape)
        pooled = self._attentive_set_pool(torch.stack(fields, dim=-2))
        return self.SET_SCALE * self.interaction_set_projection(pooled)

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_interaction_set_encoder:
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
            stats = self._stats(
                target_c,
                hist_c,
                hist_r,
                item_prior,
                concept_prior,
            )
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
                target_item = self._apply_item_residual_dropout(
                    target_item, target_q
                )
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
                history_item = self._apply_item_residual_dropout(
                    history_item, hist_q
                )
            else:
                history_item = torch.zeros_like(history_concept)
            history_response = self.resp_emb(hist_r.clamp(0, 2))
            history = history_item + history_concept + history_response

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
            token = token + self._interaction_set_residual(
                target_item,
                target_concept,
                history_item,
                history_concept,
                history_response,
            )
            scope_weight = self._boundary_weights(token, target_c)
            state = self.ssm(token, scope_weight=scope_weight)
            sequence = self._attend(token + state)
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


__all__ = ["InteractionSetEncoderA2GMambaKT"]
