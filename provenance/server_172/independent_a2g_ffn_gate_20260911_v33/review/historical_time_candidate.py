"""Causal elapsed-time adapter for the Assist2017 exploratory screen.

Assist2017 exposes timestamps in its leakage-safe sequence files. The adapter
uses only the elapsed time from the previous interaction to the current
question, which is observable before the response being predicted. The gate is
initialized to zero so the frozen A2G path is unchanged at construction time;
no extra random draws are used for the new parameters.
"""

from __future__ import annotations

import torch
from torch import nn

from work.a2g_mambakt_rwce_item_dropout_candidate import (
    A2GMambaKT as _FrozenA2GMambaKT,
)


class A2GMambaKT(_FrozenA2GMambaKT):
    """Frozen A2G plus a causal elapsed-time token adapter."""

    candidate_id = "causal_elapsed_time_adapter_assist2017_v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Deterministic fixed projection avoids shifting the frozen RNG stream.
        self.elapsed_time_projection = nn.Linear(1, self.d_model, bias=True)
        nn.init.constant_(self.elapsed_time_projection.weight, 1.0 / self.d_model)
        nn.init.zeros_(self.elapsed_time_projection.bias)
        self.elapsed_time_gate = nn.Parameter(torch.zeros(self.d_model))

    @staticmethod
    def _elapsed_time_feature(dcur, target_c):
        """Return log-scaled current-minus-previous timestamp in seconds."""
        if "tseqs" not in dcur or "shft_tseqs" not in dcur:
            raise ValueError("Assist2017 timestamp tensors are required")
        current = dcur["tseqs"].to(device=target_c.device, dtype=torch.float32)
        shifted = dcur["shft_tseqs"].to(device=target_c.device, dtype=torch.float32)
        if current.ndim != 2 or shifted.shape != current.shape:
            raise ValueError("timestamp tensor shape mismatch")
        target_time = torch.cat([current[:, :1], shifted], dim=1)
        previous_time = torch.cat(
            [torch.zeros_like(current[:, :1]), current], dim=1
        )
        delta_ms = (target_time - previous_time).clamp_min(0.0)
        delta_ms[:, 0] = 0.0
        # Milliseconds to seconds, then a bounded log transform.
        feature = torch.log1p(delta_ms / 1000.0).clamp_max(12.0)
        return feature.unsqueeze(-1)

    def _time_adapter(self, dcur, target_c):
        feature = self._elapsed_time_feature(dcur, target_c)
        return self.elapsed_time_projection(feature) * self.elapsed_time_gate

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_recency_weighted_concept_evidence:
            raise RuntimeError("The exploratory contract requires RWCE")
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
                target_c, hist_c, hist_r, item_prior, concept_prior
            )
            learned_stats = self._learned_stats(stats)
            learned_stats = self._replace_learned_exposure(
                learned_stats, target_c, hist_c, hist_r
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
            token = token + self._time_adapter(dcur, target_c)
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


__all__ = ["A2GMambaKT"]
