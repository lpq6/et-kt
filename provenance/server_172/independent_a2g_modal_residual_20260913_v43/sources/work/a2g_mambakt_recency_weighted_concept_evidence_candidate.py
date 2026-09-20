import torch

from work.a2g_mambakt_evidence_postnorm_candidate import (
    A2GMambaKT as _EvidencePostNormA2GMambaKT,
)
from work.a2g_mambakt_joint_weak_stat_prune_physical import (
    LEARNED_STAT_INDICES,
)


EXPOSURE_STAT_INDEX = 2
EXPOSURE_LOCAL_INDEX = LEARNED_STAT_INDICES.index(EXPOSURE_STAT_INDEX)


class A2GMambaKT(_EvidencePostNormA2GMambaKT):
    """Replace the learned exposure copy with causal local outcome evidence."""

    def __init__(
        self,
        *args,
        use_recency_weighted_concept_evidence=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_recency_weighted_concept_evidence = bool(
            int(use_recency_weighted_concept_evidence)
        )

    @staticmethod
    def _recency_weighted_concept_evidence_components(
        target_c,
        hist_c,
        hist_r,
    ):
        sequence_length = target_c.size(1)
        positions = torch.arange(sequence_length, device=target_c.device)
        aligned_past = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=target_c.device,
                dtype=torch.bool,
            ),
            diagonal=0,
        )
        same = target_c.unsqueeze(2).eq(hist_c.unsqueeze(1))
        same = same & aligned_past.unsqueeze(0)
        same = same & target_c.gt(0).unsqueeze(2)
        same = same & hist_c.gt(0).unsqueeze(1)
        valid_response = hist_r.ge(0) & hist_r.le(1)
        same = same & valid_response.unsqueeze(1)

        distance = (
            positions.view(1, sequence_length, 1)
            - positions.view(1, 1, sequence_length)
            + 1
        ).clamp_min(1)
        weight = torch.reciprocal(distance.to(dtype=torch.float32))
        weight = weight * same.to(dtype=weight.dtype)
        signed_response = hist_r.clamp(0, 1).to(dtype=weight.dtype)
        signed_response = signed_response.mul(2.0).sub(1.0)
        weighted_mass = weight.sum(dim=-1)
        signed_mass = (weight * signed_response.unsqueeze(1)).sum(dim=-1)
        evidence = signed_mass / (1.0 + weighted_mass)
        exposure = same.sum(dim=-1)
        return evidence, weighted_mass, exposure

    @classmethod
    def _recency_weighted_concept_evidence(cls, target_c, hist_c, hist_r):
        evidence, _, _ = cls._recency_weighted_concept_evidence_components(
            target_c,
            hist_c,
            hist_r,
        )
        return evidence

    def _replace_learned_exposure(
        self,
        learned_stats,
        target_c,
        hist_c,
        hist_r,
    ):
        if not self.use_recency_weighted_concept_evidence:
            return learned_stats
        replaced = learned_stats.clone()
        evidence = self._recency_weighted_concept_evidence(
            target_c,
            hist_c,
            hist_r,
        )
        replaced[..., EXPOSURE_LOCAL_INDEX] = evidence.to(dtype=replaced.dtype)
        return replaced

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
                target_q.numel() > 0 and target_q.size(1) == target_c.size(1)
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
            else:
                target_item = torch.zeros_like(target_concept)
            target = target_item + target_concept

            history_concept = self.hist_concept_emb(hist_c.clamp_min(0))
            has_history_items = (
                hist_q.numel() > 0 and hist_q.size(1) == hist_c.size(1)
            )
            if has_history_items:
                history_item = self.hist_item_emb(hist_q.clamp_min(0))
                history_item = history_item * self._item_residual_reliability(
                    hist_q
                ).unsqueeze(-1)
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


__all__ = [
    "A2GMambaKT",
    "EXPOSURE_LOCAL_INDEX",
    "EXPOSURE_STAT_INDEX",
]
