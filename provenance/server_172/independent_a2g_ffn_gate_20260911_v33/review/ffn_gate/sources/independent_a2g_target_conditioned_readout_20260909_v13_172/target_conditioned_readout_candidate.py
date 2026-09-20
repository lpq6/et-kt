from __future__ import annotations

import torch
from torch import nn

from a2g_incumbent import A2GMambaKT as _IncumbentA2GMambaKT


class A2GMambaKT(_IncumbentA2GMambaKT):
    CANDIDATE_ID = "incumbent_a2g_target_conditioned_readout_20260909_v13"
    LEARNED_STAT_DIM = 6
    INTERACTION_RANK = 32

    def __init__(
        self,
        *args,
        use_target_conditioned_readout=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_target_conditioned_readout = bool(
            int(use_target_conditioned_readout)
        )
        interaction_width = self.INTERACTION_RANK * 4 + self.LEARNED_STAT_DIM
        self.target_readout_norm = nn.LayerNorm(interaction_width)
        self.target_readout_query = nn.Linear(
            self.d_model,
            self.INTERACTION_RANK,
            bias=False,
        )
        self.target_readout_state = nn.Linear(
            self.d_model,
            self.INTERACTION_RANK,
            bias=False,
        )
        self.target_readout_attention = nn.Linear(
            self.d_model,
            self.INTERACTION_RANK,
            bias=False,
        )
        self.target_readout_projection = nn.Linear(
            interaction_width,
            self.d_model + self.LEARNED_STAT_DIM,
        )
        nn.init.zeros_(self.target_readout_projection.weight)
        nn.init.zeros_(self.target_readout_projection.bias)

    def _conditioned_readout_delta(
        self,
        target,
        state,
        attention,
        learned_stats,
    ):
        if not self.use_target_conditioned_readout:
            return torch.zeros(
                *target.shape[:-1],
                self.d_model + self.LEARNED_STAT_DIM,
                device=target.device,
                dtype=target.dtype,
            )
        query = self.target_readout_query(target)
        memory_state = self.target_readout_state(state)
        memory_attention = self.target_readout_attention(attention)
        interactions = torch.cat(
            [
                query * memory_state,
                query * memory_attention,
                memory_state * memory_attention,
                query * memory_state * memory_attention,
                learned_stats,
            ],
            dim=-1,
        )
        return self.target_readout_projection(
            self.target_readout_norm(interactions)
        )

    def forward(self, dcur, train=False, qtest=False):
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
            stats = self._stats(
                target_c,
                hist_c,
                hist_r,
                item_prior,
                concept_prior,
            )
            learned_stats = self._replace_learned_exposure(
                self._learned_stats(stats),
                target_c,
                hist_c,
                hist_r,
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
                target_item = self._apply_item_residual_dropout(
                    target_item,
                    target_q,
                )
                history_item = self._apply_item_residual_dropout(
                    history_item,
                    hist_q,
                )
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
            history = (
                history_item
                + history_concept
                + self.resp_emb(hist_r.clamp(0, 2))
            )
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
            full_state = self.ssm(
                token,
                scope_weight=self._boundary_weights(token, target_c),
            )
            state = (
                full_state
                if self.use_ssm_branch
                else torch.zeros_like(full_state)
            )
            attention_input = token + state
            full_attention = self._attend(attention_input)
            attention = (
                full_attention
                if self.use_attention_branch
                else torch.zeros_like(full_attention)
            )
            sequence = (
                full_attention
                if self.use_attention_branch
                else attention_input
            )
            gate = torch.sigmoid(self.memory_readout_gate(token))
            sequence = sequence + gate * state
            base_fused = torch.cat([sequence, learned_stats], dim=-1)
            readout_delta = self._conditioned_readout_delta(
                target,
                state,
                attention,
                learned_stats,
            )
            fused = base_fused + readout_delta
            prediction = torch.sigmoid(
                self.pred(fused).squeeze(-1) + direct_logit
            )
            if qtest and not train:
                return prediction, fused
            if not train:
                return prediction
            return prediction, fused.new_tensor(0.0), fused.new_zeros(4)
        finally:
            dropout.clear_protection_mask()


__all__ = ["A2GMambaKT"]
