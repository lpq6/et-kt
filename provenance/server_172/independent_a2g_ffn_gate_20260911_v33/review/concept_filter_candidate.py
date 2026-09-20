"""V19 plus an item-conditioned, two-state concept forward filter."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from attempt_stage_candidate import A2GMambaKT as _V19


def matrix_prefix(matrices):
    """Inclusive products M[t] @ ... @ M[0], up to a positive scale."""
    if matrices.ndim < 3 or matrices.shape[-2:] != (2, 2) or not matrices.size(-3):
        raise ValueError("nonempty sequences of 2x2 matrices required")
    prefix = matrices
    offset = 1
    while offset < matrices.size(-3):
        products = prefix[..., offset:, :, :] @ prefix[..., :-offset, :, :]
        products = products / products.sum(dim=(-2, -1), keepdim=True)
        prefix = torch.cat((prefix[..., :offset, :, :], products), dim=-3)
        offset *= 2
    return prefix


class ConceptForwardFilter(nn.Module):
    def __init__(self, n_concept, n_item, seq_len=200):
        super().__init__()
        self.n_concept, self.n_item = int(n_concept), int(n_item)
        self.seq_len = int(seq_len)
        self.prior_logit = nn.Embedding(n_concept + 1, 1)
        self.transition_logit = nn.Embedding(n_concept + 1, 2)
        self.emission_center = nn.Embedding(n_item + 1, 1, padding_idx=0)
        self.emission_gap = nn.Embedding(n_concept + 1, 1)
        self.output_scale = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.prior_logit.weight)
        nn.init.constant_(self.transition_logit.weight, -math.log(19.0))
        nn.init.zeros_(self.emission_center.weight)
        nn.init.constant_(self.emission_gap.weight, math.log(2.0))

    def emissions(self, items, concepts):
        center = self.emission_center(items.clamp(0, self.n_item)).squeeze(-1)
        center = center * items.ge(2).to(center.dtype)
        gap = F.softplus(self.emission_gap(concepts.clamp(0, self.n_concept)).squeeze(-1))
        probability = torch.stack((torch.sigmoid(center - gap), torch.sigmoid(center + gap)), dim=-1)
        eps = torch.finfo(probability.dtype).eps
        return probability.clamp(eps, 1.0 - eps)

    def state(self, target_c, hist_q, hist_c, hist_r):
        if target_c.ndim != 2 or any(x.shape != target_c.shape for x in (hist_q, hist_c, hist_r)):
            raise ValueError("all sequences must have identical batch/length shapes")
        length = target_c.size(1)
        if not 1 <= length <= self.seq_len:
            raise ValueError("sequence length outside frozen bounds")
        probability = self.emissions(hist_q, hist_c)
        likelihood = torch.where(hist_r.eq(1).unsqueeze(-1), probability, 1.0 - probability)
        rates = torch.sigmoid(self.transition_logit(hist_c.clamp(0, self.n_concept)))
        eps = torch.finfo(rates.dtype).eps
        learn, forget = rates.clamp(eps, 1.0 - eps).unbind(-1)
        transition = torch.stack(
            (torch.stack((1.0 - learn, forget), -1), torch.stack((learn, 1.0 - forget), -1)),
            dim=-2,
        )
        event = transition * likelihood.unsqueeze(-2)
        concepts = torch.arange(self.n_concept + 1, device=target_c.device)
        valid = hist_c.ge(2) & (hist_r.eq(0) | hist_r.eq(1))
        matches = hist_c.unsqueeze(1).eq(concepts.view(1, -1, 1)) & valid.unsqueeze(1)
        identity = torch.eye(2, device=event.device, dtype=event.dtype)
        # A concept's track is an identity at all unrelated events. Positive
        # rescaling keeps the parallel forward recursion numerically stable.
        matrices = torch.where(matches[..., None, None], event.unsqueeze(1), identity)
        prefix = matrix_prefix(matrices)
        prior = torch.sigmoid(self.prior_logit.weight[:, 0])
        prior_pair = torch.stack((1.0 - prior, prior), -1)
        mass = (prefix @ prior_pair[None, :, None, :, None]).squeeze(-1)
        posterior = mass[..., 1] / mass.sum(-1)
        index = target_c.clamp(0, self.n_concept).unsqueeze(1)
        selected = posterior.gather(1, index).squeeze(1)
        observed = matches.cumsum(-1).gather(1, index).squeeze(1).gt(0)
        observed = observed & target_c.ge(2)
        return selected, observed

    def forward(self, target_q, target_c, hist_q, hist_c, hist_r):
        posterior, observed = self.state(target_c, hist_q, hist_c, hist_r)
        emission = self.emissions(target_q, target_c)
        prior = torch.sigmoid(self.prior_logit(target_c.clamp(0, self.n_concept)).squeeze(-1))
        novice, mastered = emission.unbind(-1)
        after = novice + (mastered - novice) * posterior
        before = novice + (mastered - novice) * prior
        eps = torch.finfo(after.dtype).eps
        evidence = torch.logit(after.clamp(eps, 1.0 - eps)) - torch.logit(before.clamp(eps, 1.0 - eps))
        return self.output_scale * evidence * observed.to(evidence.dtype)


class A2GMambaKT(_V19):
    CANDIDATE_ID = "a2g_v19_item_conditioned_concept_filter_20260910_v20"

    def __init__(self, *args, use_concept_filter=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_concept_filter = bool(int(use_concept_filter))
        with torch.random.fork_rng(devices=[]):
            self.concept_filter = ConceptForwardFilter(
                self.n_question, self.n_pid, int(kwargs.get("seq_len", 200)),
            )

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_concept_filter:
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
            direct_logit = direct_logit + self.concept_filter(
                target_q, target_c, hist_q, hist_c, hist_r,
            )
            prediction = torch.sigmoid(self.pred(fused).squeeze(-1) + direct_logit)
            if qtest and not train:
                return prediction, fused
            if not train:
                return prediction
            return prediction, fused.new_tensor(0.0), fused.new_zeros(4)
        finally:
            dropout.clear_protection_mask()


__all__ = ["A2GMambaKT", "ConceptForwardFilter", "matrix_prefix"]
