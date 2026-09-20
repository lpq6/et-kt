"""V19 with history-only elapsed-time conditioning of the history embedding."""

from __future__ import annotations

import torch
from torch import nn

from attempt_stage_candidate import A2GMambaKT as _V19


def historical_pace_feature(timestamps, hist_c):
    """At target t, use timestamp[t-1] - timestamp[t-2], never timestamp[t]."""
    if hist_c.ndim != 2 or hist_c.size(1) == 0:
        raise ValueError("history concepts must have a nonempty batch/length shape")
    batch, length = hist_c.shape
    zero = torch.zeros(batch, length, 1, dtype=torch.float32, device=hist_c.device)
    if timestamps is None or timestamps.numel() == 0:
        return zero
    if timestamps.shape != (batch, length - 1) or timestamps.dtype != torch.int64:
        raise ValueError("unshifted timestamps must be int64 with one fewer column")
    if timestamps.device != hist_c.device:
        raise ValueError("timestamps and history must share a device")
    if length < 3:
        return zero
    earlier, later = timestamps[:, :-1], timestamps[:, 1:]
    valid = (
        hist_c[:, 1:-1].gt(0) & hist_c[:, 2:].gt(0)
        & earlier.ge(0) & later.ge(earlier)
    )
    # Subtract epoch milliseconds as integers before converting the difference.
    gap = torch.where(valid, later - earlier, torch.zeros_like(later))
    feature = torch.log1p(gap.to(torch.float32) / 1000.0).unsqueeze(-1)
    return torch.cat([zero[:, :2], feature], dim=1)


class HistoricalPaceModulation(nn.Module):
    def __init__(self, width):
        super().__init__()
        if width < 1:
            raise ValueError("history width must be positive")
        self.scale = nn.Parameter(torch.zeros(width))
        self.shift = nn.Parameter(torch.zeros(width))

    def forward(self, history, feature):
        if history.shape[:2] != feature.shape[:2] or feature.size(-1) != 1:
            raise ValueError("history and temporal features must align")
        feature = feature.to(dtype=history.dtype)
        return history * (1.0 + feature * self.scale) + feature * self.shift


class A2GMambaKT(_V19):
    CANDIDATE_ID = "a2g_v19_historical_pace_input_20260910_v27"

    def __init__(self, *args, use_history_pace=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_history_pace = bool(int(use_history_pace))
        self.history_pace = HistoricalPaceModulation(self.d_model)

    def _modulate_history(self, history, dcur, hist_c):
        feature = historical_pace_feature(dcur.get("tseqs"), hist_c)
        return self.history_pace(history, feature)

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_history_pace:
            return super().forward(dcur, train=train, qtest=qtest)
        dropout = self.input[3]
        target_q, target_c, hist_q, hist_c, hist_r = self._sequences(dcur)
        if self.training and dropout.warmup_active:
            dropout.set_protection_mask(self._concept_novelty_mask(target_c))
        try:
            concept_prior = self.concept_prior(target_c.clamp_min(0))
            has_items = (
                self.n_pid > 0
                and target_q.numel() > 0
                and target_q.size(1) == target_c.size(1)
            )
            item_prior = (
                self.item_prior(target_q.clamp_min(0))
                if has_items
                else torch.zeros_like(concept_prior)
            )
            stats = self._stats(target_c, hist_c, hist_r, item_prior, concept_prior)
            learned_stats = self._replace_learned_exposure(
                self._learned_stats(stats), target_c, hist_c, hist_r
            )
            target_concept = self.concept_emb(target_c.clamp_min(0))
            history_concept = self.hist_concept_emb(hist_c.clamp_min(0))
            if has_items:
                target_item = self.item_emb(target_q.clamp_min(0))
                history_item = self.hist_item_emb(hist_q.clamp_min(0))
                target_reliability = self._item_residual_reliability(target_q)
                history_reliability = self._item_residual_reliability(hist_q)
                if self.use_evidence_branch:
                    target_item = target_item * target_reliability.unsqueeze(-1)
                    history_item = history_item * history_reliability.unsqueeze(-1)
                target_item = self._apply_item_residual_dropout(target_item, target_q)
                history_item = self._apply_item_residual_dropout(history_item, hist_q)
            else:
                target_item = torch.zeros_like(target_concept)
                history_item = torch.zeros_like(history_concept)

            direct_logit = (
                self.prior_scale * (item_prior + concept_prior).squeeze(-1)
                + self.stat_scale * self._direct_stat_logit(stats)
            )
            if not self.use_evidence_branch:
                item_prior = torch.zeros_like(item_prior)
                concept_prior = torch.zeros_like(concept_prior)
                learned_stats = torch.zeros_like(learned_stats)
                direct_logit = torch.zeros_like(direct_logit)

            target = target_item + target_concept
            history = history_item + history_concept + self.resp_emb(hist_r.clamp(0, 2))
            history = self._modulate_history(history, dcur, hist_c)
            raw_input = torch.cat(
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
            token = self._factorized_token(raw_input, self.input(raw_input))
            full_state = self.ssm(token, scope_weight=self._boundary_weights(token, target_c))
            state = full_state if self.use_ssm_branch else torch.zeros_like(full_state)
            attention_input = token + state
            full_attention = self._attend(attention_input)
            sequence = full_attention if self.use_attention_branch else attention_input
            gate = torch.sigmoid(self.memory_readout_gate(token))
            sequence = sequence + gate * state
            fused = torch.cat([sequence, learned_stats], dim=-1)
            if self.use_item_attempt_stage:
                direct_logit = direct_logit + self.attempt_readout(
                    sequence, target, target_q, hist_q, hist_r,
                )
            prediction = torch.sigmoid(self.pred(fused).squeeze(-1) + direct_logit)
            if qtest and not train:
                return prediction, fused
            if not train:
                return prediction
            return prediction, fused.new_tensor(0.0), fused.new_zeros(4)
        finally:
            dropout.clear_protection_mask()


__all__ = ["A2GMambaKT", "HistoricalPaceModulation", "historical_pace_feature"]
