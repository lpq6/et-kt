"""V12 with a zero-start, item-attempt-conditioned logit readout."""

from __future__ import annotations

import torch
from torch import nn

from factorized_input_candidate import A2GMambaKT as _V12


def item_attempt_indices(target_q, hist_q, hist_r):
    """Return four exact counts using shifted history through column t."""
    if target_q.ndim != 2 or hist_q.shape != target_q.shape or hist_r.shape != target_q.shape:
        raise ValueError("item and response sequences must have identical batch/length shapes")
    length = target_q.size(1)
    if length == 0:
        raise ValueError("at least one target is required")
    positions = torch.arange(length, device=target_q.device)
    valid_history = hist_q.ge(2) & (hist_r.eq(0) | hist_r.eq(1))
    available = positions.view(1, 1, -1) <= positions.view(1, -1, 1)
    same = (
        target_q.unsqueeze(-1).eq(hist_q.unsqueeze(1))
        & target_q.ge(2).unsqueeze(-1)
        & valid_history.unsqueeze(1)
        & available
    )
    attempts = same.sum(-1)
    successes = (same & hist_r.eq(1).unsqueeze(1)).sum(-1)
    adjacent = target_q.eq(hist_q) & target_q.ge(2) & valid_history
    failed = adjacent & hist_r.eq(0)

    def run_length(continues):
        breaks = torch.where(continues, torch.zeros_like(positions), positions)
        last_break = breaks.cummax(dim=1).values
        return torch.where(continues, positions - last_break, torch.zeros_like(last_break))

    return attempts, successes, run_length(adjacent), run_length(failed)


class ItemAttemptReadout(nn.Module):
    def __init__(self, d_model, num_heads, seq_len):
        super().__init__()
        if d_model % num_heads or seq_len < 2:
            raise ValueError("readout rank must match one attention head, with length>=2")
        rank = d_model // num_heads
        self.seq_len = int(seq_len)
        self.rank = rank
        self.attempt_embedding = nn.Embedding(seq_len, rank, padding_idx=0)
        self.success_embedding = nn.Embedding(seq_len, rank, padding_idx=0)
        self.run_embedding = nn.Embedding(seq_len, rank, padding_idx=0)
        self.failure_embedding = nn.Embedding(seq_len, rank, padding_idx=0)
        self.target_norm = nn.LayerNorm(d_model)
        self.sequence_norm = nn.LayerNorm(d_model)
        self.target_projection = nn.Linear(d_model, rank, bias=False)
        self.sequence_projection = nn.Linear(d_model, rank, bias=False)
        self.output = nn.Linear(3 * rank, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, sequence, target, target_q, hist_q, hist_r):
        if target_q.size(1) > self.seq_len:
            raise ValueError("sequence exceeds the frozen ordinal table width")
        indices = item_attempt_indices(target_q, hist_q, hist_r)
        # Four unit-scale embeddings share one state vector; no rates or bins
        # are fitted from the descriptive profile.
        state = (
            self.attempt_embedding(indices[0])
            + self.success_embedding(indices[1])
            + self.run_embedding(indices[2])
            + self.failure_embedding(indices[3])
        ) / 2.0
        query = torch.tanh(self.target_projection(self.target_norm(target)))
        learner = torch.tanh(self.sequence_projection(self.sequence_norm(sequence)))
        delta = self.output(torch.cat([state, state * query, state * learner], dim=-1)).squeeze(-1)
        return delta * indices[0].gt(0).to(delta.dtype)


class A2GMambaKT(_V12):
    CANDIDATE_ID = "a2g_v12_item_attempt_stage_readout_20260910_v19"

    def __init__(self, *args, use_item_attempt_stage=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_item_attempt_stage = bool(int(use_item_attempt_stage))
        with torch.random.fork_rng(devices=[]):
            self.attempt_readout = ItemAttemptReadout(
                self.d_model, self.blocks[0]["attn"].num_heads, int(kwargs.get("seq_len", 200)),
            )

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_item_attempt_stage:
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


__all__ = ["A2GMambaKT", "ItemAttemptReadout", "item_attempt_indices"]
