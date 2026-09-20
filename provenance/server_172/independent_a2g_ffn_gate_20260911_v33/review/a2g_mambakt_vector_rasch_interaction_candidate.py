import torch
from torch import nn

from work.a2g_mambakt_cascaded_postnorm_candidate import (
    A2GMambaKT as _PostNormA2GMambaKT,
)


class A2GMambaKT(_PostNormA2GMambaKT):
    """Condition target semantics on a train-fold-seeded item difficulty vector."""

    def __init__(
        self,
        *args,
        use_vector_rasch_interaction=1,
        vector_rasch_prior_scale=0.5,
        vector_rasch_prior_clip=0.25,
        **kwargs,
    ):
        self.use_vector_rasch_interaction = bool(
            int(use_vector_rasch_interaction)
        )
        self.vector_rasch_prior_scale = float(vector_rasch_prior_scale)
        self.vector_rasch_prior_clip = float(vector_rasch_prior_clip)
        super().__init__(*args, **kwargs)

        if not self.use_vector_rasch_interaction:
            return

        rng_state = torch.random.get_rng_state().clone()
        self.vector_difficulty = nn.Embedding(
            self.n_pid + 1,
            self.d_model,
            padding_idx=0,
        )
        self.concept_variation = nn.Embedding(
            self.n_question + 1,
            self.d_model,
            padding_idx=0,
        )
        with torch.no_grad():
            self.vector_difficulty.weight.zero_()
            self.concept_variation.weight[0].zero_()
        torch.random.set_rng_state(rng_state)

    @torch.no_grad()
    def set_prior_logits(self, item_logits, concept_logits):
        super().set_prior_logits(item_logits, concept_logits)
        if not self.use_vector_rasch_interaction:
            return

        item_logits = item_logits.to(
            device=self.vector_difficulty.weight.device,
            dtype=self.vector_difficulty.weight.dtype,
        ).view(-1, 1)
        seeded = self.vector_rasch_prior_scale * item_logits
        if self.vector_rasch_prior_clip > 0.0:
            seeded = seeded.clamp(
                -self.vector_rasch_prior_clip,
                self.vector_rasch_prior_clip,
            )
        count = min(self.vector_difficulty.weight.size(0), seeded.size(0))
        self.vector_difficulty.weight.zero_()
        self.vector_difficulty.weight[:count].copy_(
            seeded[:count].expand(-1, self.d_model)
        )
        self.vector_difficulty.weight[0].zero_()

    def _vector_rasch_target(self, target_q, target_c):
        if not self.use_vector_rasch_interaction:
            return self.item_emb(target_q.clamp_min(0)).new_zeros(
                *target_q.shape,
                self.d_model,
            )
        difficulty = self.vector_difficulty(
            target_q.clamp(0, self.n_pid)
        )
        variation = self.concept_variation(
            target_c.clamp(0, self.n_question)
        )
        return difficulty * variation

    def vector_rasch_diagnostics(self, dcur):
        if not self.use_vector_rasch_interaction:
            raise RuntimeError("Vector Rasch interaction is disabled")
        target_q, target_c, _, _, _ = self._sequences(dcur)
        difficulty = self.vector_difficulty(
            target_q.clamp(0, self.n_pid)
        )
        variation = self.concept_variation(
            target_c.clamp(0, self.n_question)
        )
        return {
            "difficulty": difficulty,
            "variation": variation,
            "target_delta": difficulty * variation,
            "prior_scale": difficulty.new_tensor(
                self.vector_rasch_prior_scale
            ),
            "prior_clip": difficulty.new_tensor(
                self.vector_rasch_prior_clip
            ),
        }

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_vector_rasch_interaction:
            return super().forward(dcur, train=train, qtest=qtest)

        dropout = self.input[3]
        target_q, target_c, hist_q, hist_c, hist_r = self._sequences(dcur)
        if self.training and dropout.warmup_active:
            dropout.set_protection_mask(self._concept_novelty_mask(target_c))
        try:
            item_prior = self.item_prior(target_q.clamp_min(0))
            concept_prior = self.concept_prior(target_c.clamp_min(0))
            stats = self._stats(
                target_c,
                hist_c,
                hist_r,
                item_prior,
                concept_prior,
            )
            learned_stats = self._learned_stats(stats)

            target_item = self.item_emb(target_q.clamp_min(0))
            target_concept = self.concept_emb(target_c.clamp_min(0))
            target = target_item + target_concept
            target = target + self._vector_rasch_target(target_q, target_c)
            history = self.hist_item_emb(hist_q.clamp_min(0))
            history = history + self.hist_concept_emb(hist_c.clamp_min(0))
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
