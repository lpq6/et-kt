"""V27 with a causal, target-conditioned Newton readout from past errors."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from history_pace_candidate import A2GMambaKT as _V27


def prefix_newton_scores(base_logits, features, responses, observed):
    """One ridge Newton step from events strictly earlier than each target."""
    if base_logits.ndim != 2 or min(base_logits.shape) < 1:
        raise ValueError("logits must have a nonempty batch/length shape")
    batch, length = base_logits.shape
    if features.ndim != 3 or features.shape[:2] != (batch, length) or features.size(-1) < 1:
        raise ValueError("features must align with the target sequence")
    if responses.shape != (batch, length - 1) or observed.shape != (batch, length):
        raise ValueError("past responses and event masks must align with targets")
    if observed.dtype != torch.bool:
        raise ValueError("observed event masks must be boolean")
    if not base_logits.is_floating_point() or features.dtype != base_logits.dtype:
        raise ValueError("logits and features must share a floating dtype")
    if any(value.device != base_logits.device for value in (features, responses, observed)):
        raise ValueError("all inputs must share a device")
    rank = features.size(-1)
    inverse = torch.eye(rank, dtype=features.dtype, device=features.device)
    inverse = inverse.unsqueeze(0).expand(batch, rank, rank)
    score = features.new_zeros(batch, rank)
    probability = torch.sigmoid(base_logits)
    outputs = [base_logits.new_zeros(batch)]

    for target in range(1, length):
        past = target - 1
        valid = observed[:, past] & (responses[:, past].eq(0) | responses[:, past].eq(1))
        feature = torch.where(valid.unsqueeze(-1), features[:, past], torch.zeros_like(score))
        p = probability[:, past]
        weight = torch.where(valid, p * (1.0 - p), torch.zeros_like(p))
        innovation = torch.where(valid, responses[:, past].to(p.dtype) - p, torch.zeros_like(p))
        # Sherman-Morrison updates (I + sum Fisher outer products)^(-1).
        vector = torch.bmm(inverse, feature.unsqueeze(-1)).squeeze(-1)
        denominator = 1.0 + weight * (feature * vector).sum(-1)
        inverse = inverse - (
            (weight / denominator).view(batch, 1, 1)
            * vector.unsqueeze(-1) * vector.unsqueeze(-2)
        )
        score = score + innovation.unsqueeze(-1) * feature
        offset = torch.bmm(inverse, score.unsqueeze(-1)).squeeze(-1)
        value = (features[:, target] * offset).sum(-1)
        outputs.append(torch.where(observed[:, target], value, torch.zeros_like(value)))
    return torch.stack(outputs, dim=1)


class PrequentialNewtonReadout(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError("projection width must equal one attention head")
        self.d_model = int(d_model)
        self.rank = int(d_model // num_heads)
        self.projection = nn.Linear(self.d_model, self.rank, bias=False)
        self.output_scale = nn.Parameter(torch.zeros(()))

    def features(self, target):
        if target.ndim != 3 or target.size(-1) != self.d_model:
            raise ValueError("target embedding width changed")
        normalized = F.layer_norm(target, (self.d_model,))
        coordinates = torch.tanh(self.projection(normalized)) / math.sqrt(self.rank)
        return torch.cat([torch.ones_like(coordinates[..., :1]), coordinates], dim=-1)

    def forward(self, base_logits, target, target_c, responses):
        score = prefix_newton_scores(
            base_logits, self.features(target), responses, target_c.gt(0),
        )
        return self.output_scale * score


class A2GMambaKT(_V27):
    CANDIDATE_ID = "a2g_v27_prequential_newton_readout_20260910_v28"

    def __init__(self, *args, use_prequential_newton=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_prequential_newton = bool(int(use_prequential_newton))
        with torch.random.fork_rng(devices=[]):
            self.prequential_newton = PrequentialNewtonReadout(
                self.d_model, self.blocks[0]["attn"].num_heads,
            )

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_prequential_newton:
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
            correction = self.prequential_newton(
                base_logits, target, target_c, dcur["rseqs"],
            )
            prediction = torch.sigmoid(base_logits + correction)
            if qtest and not train:
                return prediction, fused
            if not train:
                return prediction
            return prediction, fused.new_tensor(0.0), fused.new_zeros(4)
        finally:
            dropout.clear_protection_mask()


__all__ = ["A2GMambaKT", "PrequentialNewtonReadout", "prefix_newton_scores"]
