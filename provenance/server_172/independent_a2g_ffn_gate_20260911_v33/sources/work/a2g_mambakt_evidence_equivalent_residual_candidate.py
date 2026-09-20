import torch

from work.a2g_mambakt_joint_weak_stat_prune_physical import (
    A2GMambaKT as _PhysicalPrunedA2GMambaKT,
)


class A2GMambaKT(_PhysicalPrunedA2GMambaKT):
    """Shrink sparse item residuals and recover exact identity at mature support."""

    IDENTITY_EVIDENCE_MULTIPLIER = 2.0

    def __init__(
        self,
        *args,
        use_evidence_equivalent_residual=1,
        support_smoothing=24.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_evidence_equivalent_residual = bool(
            int(use_evidence_equivalent_residual)
        )
        self.support_smoothing = max(1e-6, float(support_smoothing))
        if self.use_evidence_equivalent_residual:
            self.register_buffer(
                "item_support_count",
                torch.zeros(self.n_pid + 1),
            )

    @property
    def identity_support(self):
        return self.IDENTITY_EVIDENCE_MULTIPLIER * self.support_smoothing

    @torch.no_grad()
    def set_prior_support(self, item_count, concept_count=None):
        del concept_count
        if not self.use_evidence_equivalent_residual:
            return
        item_count = item_count.float().view(-1)
        count = min(self.item_support_count.numel(), item_count.numel())
        self.item_support_count.zero_()
        self.item_support_count[:count].copy_(item_count[:count])

    def _item_residual_reliability(self, item_ids):
        if not self.use_evidence_equivalent_residual:
            return torch.ones_like(item_ids, dtype=self.item_emb.weight.dtype)
        item_ids = item_ids.clamp(
            min=0,
            max=self.item_support_count.numel() - 1,
        )
        support = self.item_support_count[item_ids]
        raw_reliability = support / (support + self.support_smoothing)
        identity_reliability = self.identity_support / (
            self.identity_support + self.support_smoothing
        )
        normalized = raw_reliability / identity_reliability
        return torch.where(
            support >= self.identity_support,
            torch.ones_like(normalized),
            normalized,
        )

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_evidence_equivalent_residual:
            return super().forward(dcur, train=train, qtest=qtest)

        dropout = self.input[3]
        target_q, target_c, hist_q, hist_c, hist_r = self._sequences(dcur)
        if self.training and dropout.warmup_active:
            dropout.set_protection_mask(self._concept_novelty_mask(target_c))
        try:
            item_prior = self.item_prior(target_q.clamp_min(0))
            concept_prior = self.concept_prior(target_c.clamp_min(0))
            stats = self._stats(target_c, hist_c, hist_r, item_prior, concept_prior)
            learned_stats = self._learned_stats(stats)

            target_item = self.item_emb(target_q.clamp_min(0))
            target_item = target_item * self._item_residual_reliability(
                target_q
            ).unsqueeze(-1)
            target_concept = self.concept_emb(target_c.clamp_min(0))
            target = target_item + target_concept

            history_item = self.hist_item_emb(hist_q.clamp_min(0))
            history_item = history_item * self._item_residual_reliability(
                hist_q
            ).unsqueeze(-1)
            history = history_item + self.hist_concept_emb(hist_c.clamp_min(0))
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
