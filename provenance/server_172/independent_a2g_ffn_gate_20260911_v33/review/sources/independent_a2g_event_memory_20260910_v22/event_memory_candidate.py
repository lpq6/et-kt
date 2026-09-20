"""V19 with event-aligned keys and values in its existing attention stack."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from attempt_stage_candidate import A2GMambaKT as _V19


def event_attention_mask(hist_c, hist_r, num_heads):
    if hist_c.ndim != 2 or hist_r.shape != hist_c.shape:
        raise ValueError("history concepts and responses must share batch/length")
    batch, length = hist_c.shape
    if length < 1 or num_heads < 1:
        raise ValueError("positive sequence length and head count required")
    valid = hist_c.gt(0) & (hist_r.eq(0) | hist_r.eq(1))
    causal = torch.ones(length, length, dtype=torch.bool, device=hist_c.device).tril()
    allowed = causal.unsqueeze(0) & valid.unsqueeze(1)
    has_history = allowed.any(-1)
    # Give empty rows one dummy key for a finite softmax, then zero their read.
    allowed[:, :, 0] |= ~has_history
    blocked = (~allowed).unsqueeze(1).expand(-1, num_heads, -1, -1)
    return blocked.reshape(batch * num_heads, length, length), has_history


class A2GMambaKT(_V19):
    CANDIDATE_ID = "a2g_v19_event_aligned_attention_memory_20260910_v22"

    def __init__(self, *args, use_event_memory=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_event_memory = bool(int(use_event_memory))
        if self.use_event_memory and self.normalization_topology != "full_postnorm":
            raise ValueError("event memory requires the frozen full_postnorm topology")

    def _attend_event_memory(self, x, history_semantic, history, hist_c, hist_r):
        if history_semantic.shape != x.shape or history.shape != x.shape:
            raise ValueError("queries, history semantics and events must share shape")
        if hist_c.shape != x.shape[:2] or x.size(-1) != self.d_model:
            raise ValueError("event memory dimensions differ from the query")
        keys = F.layer_norm(history_semantic, (self.d_model,))
        values = F.layer_norm(history, (self.d_model,))
        mask, has_history = event_attention_mask(
            hist_c, hist_r, self.blocks[0]["attn"].num_heads,
        )
        for block in self.blocks:
            attended, _ = block["attn"](
                x, keys, values, attn_mask=mask, need_weights=False,
            )
            attended = attended * has_history.unsqueeze(-1).to(attended.dtype)
            x = block["norm"](x + block["drop"](attended))
            if "ffn" in block:
                x = block["ffn_norm"](x + block["ffn"](x))
        return x

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_event_memory:
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
            full_attention = self._attend_event_memory(
                attention_input, history_item + history_concept, history, hist_c, hist_r,
            )
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


__all__ = ["A2GMambaKT", "event_attention_mask"]
