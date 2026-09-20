import torch

from work.a2g_mambakt_recency_weighted_concept_evidence_candidate import (
    A2GMambaKT as _RwceA2GMambaKT,
)


class A2GMambaKT(_RwceA2GMambaKT):
    """RWCE core with train-time item residual dropout."""

    def __init__(
        self,
        *args,
        item_residual_dropout=0.0,
        item_residual_dropout_schedule="constant",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.item_residual_dropout = max(0.0, min(1.0, float(item_residual_dropout)))
        self.item_residual_dropout_schedule = str(item_residual_dropout_schedule)

    def _scheduled_dropout(self):
        if self.item_residual_dropout_schedule == "constant":
            return self.item_residual_dropout
        if self.item_residual_dropout_schedule == "warmup":
            return self.item_residual_dropout if self.training else 0.0
        raise ValueError(
            "Unknown item_residual_dropout_schedule: "
            f"{self.item_residual_dropout_schedule!r}"
        )

    def _apply_item_residual_dropout(self, residual, item_ids):
        rate = self._scheduled_dropout()
        if not self.training or rate <= 0.0:
            return residual
        keep = torch.rand_like(item_ids.float()) >= rate
        if not keep.any():
            keep = torch.ones_like(keep, dtype=torch.bool)
        return residual * keep.unsqueeze(-1).to(residual.dtype)

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_recency_weighted_concept_evidence:
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
                target_item = self._apply_item_residual_dropout(target_item, target_q)
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
                history_item = self._apply_item_residual_dropout(history_item, hist_q)
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
