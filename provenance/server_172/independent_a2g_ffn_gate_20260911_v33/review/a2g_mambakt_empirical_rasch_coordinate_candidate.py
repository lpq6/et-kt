import torch
from torch import nn
import torch.nn.functional as F

from work.a2g_mambakt_joint_weak_stat_prune_physical import (
    A2GMambaKT as _PhysicalA2GMambaKT,
)


class A2GMambaKT(_PhysicalA2GMambaKT):
    """Inject train-fold item priors into a minimal shared Rasch coordinate."""

    def __init__(
        self,
        *args,
        use_empirical_rasch_coordinate=1,
        backbone_prior_scale=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_empirical_rasch_coordinate = bool(
            int(use_empirical_rasch_coordinate)
        )
        self.backbone_prior_scale = float(backbone_prior_scale)
        if self.use_empirical_rasch_coordinate:
            self.prior_concept_coordinate = nn.Parameter(
                torch.zeros(self.n_question + 1, self.d_model)
            )
            self.prior_response_coordinate = nn.Parameter(
                torch.zeros(3, self.d_model)
            )

    def _rasch_coordinates(
        self,
        target_q,
        target_c,
        hist_q,
        hist_c,
        hist_r,
    ):
        if not self.use_empirical_rasch_coordinate:
            zero = self.item_emb(target_q.clamp_min(0)).new_zeros(
                *target_q.shape,
                self.d_model,
            )
            return zero, zero

        target_prior = self.item_prior(target_q.clamp_min(0))
        history_prior = self.item_prior(hist_q.clamp_min(0))
        target_direction = F.embedding(
            target_c.clamp(0, self.n_question),
            self.prior_concept_coordinate,
            padding_idx=0,
        )
        history_concept_direction = F.embedding(
            hist_c.clamp(0, self.n_question),
            self.prior_concept_coordinate,
            padding_idx=0,
        )
        history_response_direction = F.embedding(
            hist_r.clamp(0, 2),
            self.prior_response_coordinate,
            padding_idx=2,
        )
        target_coordinate = (
            self.backbone_prior_scale * target_prior * target_direction
        )
        history_coordinate = self.backbone_prior_scale * history_prior * (
            history_concept_direction + history_response_direction
        )
        return target_coordinate, history_coordinate

    def rasch_coordinate_diagnostics(self, dcur):
        if not self.use_empirical_rasch_coordinate:
            raise RuntimeError("Empirical Rasch coordinate is disabled")
        target_q, target_c, hist_q, hist_c, hist_r = self._sequences(dcur)
        target_coordinate, history_coordinate = self._rasch_coordinates(
            target_q,
            target_c,
            hist_q,
            hist_c,
            hist_r,
        )
        return {
            "backbone_prior_scale": target_coordinate.new_tensor(
                self.backbone_prior_scale
            ),
            "target_item_prior": self.item_prior(target_q.clamp_min(0)),
            "history_item_prior": self.item_prior(hist_q.clamp_min(0)),
            "target_coordinate": target_coordinate,
            "history_coordinate": history_coordinate,
            "concept_coordinate_rms": self.prior_concept_coordinate.square()
            .mean()
            .sqrt(),
            "response_coordinate_rms": self.prior_response_coordinate.square()
            .mean()
            .sqrt(),
        }

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_empirical_rasch_coordinate:
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
            history = self.hist_item_emb(hist_q.clamp_min(0))
            history = history + self.hist_concept_emb(hist_c.clamp_min(0))
            history = history + self.resp_emb(hist_r.clamp(0, 2))
            target_coordinate, history_coordinate = self._rasch_coordinates(
                target_q,
                target_c,
                hist_q,
                hist_c,
                hist_r,
            )
            target = target_item + target_concept + target_coordinate
            history = history + history_coordinate
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
