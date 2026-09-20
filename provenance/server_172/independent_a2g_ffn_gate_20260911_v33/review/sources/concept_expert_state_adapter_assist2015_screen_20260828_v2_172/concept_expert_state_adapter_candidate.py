"""Concept-conditioned low-rank expert adapter for the frozen A2G state.

This is a single fixed three-expert state-readout module inspired by the
concept-level multi-expert acquisition network in Q-MCKT (arXiv:2403.07322).
The adapter reads only the current causal SSM state and the known target
concept embedding. Its residual output is zero at initialization, so the
frozen control forward path and dropout stream remain unchanged while the
added expert parameters receive gradients.
"""

from __future__ import annotations

import torch
from torch import nn

from work.a2g_mambakt_rwce_item_dropout_candidate import (
    A2GMambaKT as _FrozenA2G,
)


class _LowRankExpert(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.orthogonal_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.up(torch.tanh(self.down(state)))


class ConceptExpertStateAdapterA2GMambaKT(_FrozenA2G):
    """Frozen Full control plus a zero-start concept-conditioned state adapter."""

    candidate_id = "concept_expert_state_adapter_v1"
    NUM_EXPERTS = 3
    RANK = 8
    RESIDUAL_SCALE = 0.10

    def __init__(
        self,
        *args,
        use_concept_expert_state_adapter=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_concept_expert_state_adapter = bool(
            int(use_concept_expert_state_adapter)
        )
        rng_state = torch.random.get_rng_state().clone()
        self.concept_expert_state_adapter = nn.ModuleList(
            [_LowRankExpert(self.d_model, self.RANK) for _ in range(self.NUM_EXPERTS)]
        )
        self.concept_expert_gate = nn.Linear(
            self.d_model,
            self.NUM_EXPERTS,
            bias=True,
        )
        nn.init.zeros_(self.concept_expert_gate.weight)
        nn.init.zeros_(self.concept_expert_gate.bias)
        # Added modules must not consume construction RNG used by frozen modules.
        torch.random.set_rng_state(rng_state)
        self.model_name = "a2g_mambakt"

    def _adapt_state(
        self,
        state: torch.Tensor,
        target_concept: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_concept_expert_state_adapter:
            return state
        gate = torch.softmax(self.concept_expert_gate(target_concept), dim=-1)
        experts = torch.stack(
            [expert(state) for expert in self.concept_expert_state_adapter],
            dim=-2,
        )
        residual = (gate.unsqueeze(-1) * experts).sum(dim=-2)
        return state + self.RESIDUAL_SCALE * residual

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_concept_expert_state_adapter:
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
            state = self._adapt_state(state, target_concept)
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


__all__ = ["ConceptExpertStateAdapterA2GMambaKT"]
