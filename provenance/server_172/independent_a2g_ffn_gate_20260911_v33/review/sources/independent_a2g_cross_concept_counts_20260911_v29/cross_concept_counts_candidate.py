"""V28 plus a directed, off-diagonal readout of past concept outcome counts."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from prequential_newton_candidate import A2GMambaKT as _V28


def prefix_outcome_counts(hist_c, hist_r, num_ids):
    """History column t holds event t-1, so its inclusive prefix is causal."""
    if hist_c.ndim != 2 or min(hist_c.shape) < 1 or hist_r.shape != hist_c.shape:
        raise ValueError("history concepts and responses need matching nonempty shapes")
    if hist_c.dtype != torch.int64 or hist_c.device != hist_r.device:
        raise ValueError("concepts must be int64 on the responses' device")
    if not isinstance(num_ids, int) or num_ids < 3:
        raise ValueError("capacity must include padding, unknown and a known concept")
    known = hist_c.ge(2) & hist_c.lt(num_ids)
    valid = known & (hist_r.eq(0) | hist_r.eq(1))
    index = (hist_c - 2).clamp(0, num_ids - 3)
    one_hot = F.one_hot(index, num_classes=num_ids - 2)
    success = one_hot * (valid & hist_r.eq(1)).unsqueeze(-1)
    failure = one_hot * (valid & hist_r.eq(0)).unsqueeze(-1)
    return success.cumsum(dim=1), failure.cumsum(dim=1)


class CrossConceptCountReadout(nn.Module):
    def __init__(self, num_ids):
        super().__init__()
        if not isinstance(num_ids, int) or num_ids < 3:
            raise ValueError("capacity must include at least one known concept")
        self.num_ids = num_ids
        self.known_concepts = num_ids - 2
        shape = (self.known_concepts, self.known_concepts - 1)
        self.success_weight = nn.Parameter(torch.zeros(shape))
        self.failure_weight = nn.Parameter(torch.zeros(shape))

    def forward(self, target_c, hist_c, hist_r):
        if target_c.shape != hist_c.shape or target_c.dtype != torch.int64:
            raise ValueError("target concepts must align with int64 history concepts")
        if target_c.device != self.success_weight.device or target_c.device != hist_c.device:
            raise ValueError("concepts and weights must share a device")
        success, failure = prefix_outcome_counts(hist_c, hist_r, self.num_ids)
        target_index = (target_c - 2).clamp(0, self.known_concepts - 1)
        # Each compact row omits its own concept without dormant diagonal weights.
        source_index = torch.arange(self.known_concepts - 1, device=target_c.device)
        source_index = source_index + (source_index >= target_index.unsqueeze(-1))
        success = success.gather(-1, source_index).to(self.success_weight.dtype)
        failure = failure.gather(-1, source_index).to(self.failure_weight.dtype)
        value = (
            F.embedding(target_index, self.success_weight) * torch.log1p(success)
            + F.embedding(target_index, self.failure_weight) * torch.log1p(failure)
        ).sum(-1)
        known_target = target_c.ge(2) & target_c.lt(self.num_ids)
        return torch.where(known_target, value, torch.zeros_like(value))


class A2GMambaKT(_V28):
    CANDIDATE_ID = "a2g_v28_cross_concept_count_readout_20260911_v29"

    def __init__(self, *args, use_cross_concept_counts=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cross_concept_counts = bool(int(use_cross_concept_counts))
        self.cross_concept_counts = CrossConceptCountReadout(self.concept_emb.num_embeddings)

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_cross_concept_counts:
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
            if self.use_history_pace:
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
            base_logits = self.pred(fused).squeeze(-1) + direct_logit
            correction = (
                self.prequential_newton(base_logits, target, target_c, dcur["rseqs"])
                if self.use_prequential_newton else torch.zeros_like(base_logits)
            )
            count_responses = torch.cat(
                [torch.full_like(dcur["rseqs"][:, :1], 2), dcur["rseqs"]], dim=1,
            )
            cross_counts = self.cross_concept_counts(target_c, hist_c, count_responses)
            prediction = torch.sigmoid(base_logits + correction + cross_counts)
            if qtest and not train:
                return prediction, fused
            if not train:
                return prediction
            return prediction, fused.new_tensor(0.0), fused.new_zeros(4)
        finally:
            dropout.clear_protection_mask()


__all__ = ["A2GMambaKT", "CrossConceptCountReadout", "prefix_outcome_counts"]
