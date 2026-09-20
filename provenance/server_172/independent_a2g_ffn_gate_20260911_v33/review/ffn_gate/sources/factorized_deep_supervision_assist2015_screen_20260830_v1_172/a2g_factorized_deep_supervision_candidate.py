"""Factorized A2G candidate with isolated evidence, SSM, and attention heads.

The candidate keeps the frozen causal input/statistic construction, but gives
the three information paths disjoint prediction interfaces.  A small fixed
deep-supervision term in the runner trains each path to remain predictive.  The
late direct-statistic residual is intentionally removed; statistics remain in
the learned fusion input and the evidence branch.
"""

from __future__ import annotations

import torch
from torch import nn

# The frozen runner imports the current-best wrapper as a top-level module from
# its immutable `sources/` directory; keep the same import contract here.
from a2g_mambakt_final import A2GMambaKT as _CurrentBestA2GMambaKT


class A2GMambaKT(_CurrentBestA2GMambaKT):
    """Current A2G with three isolated, ablatable predictive branches."""

    CANDIDATE_ID = "factorized_deep_supervision_a2g_assist2015_v1"
    EVIDENCE_SCALE = 0.10
    SSM_SCALE = 0.10
    ATTENTION_SCALE = 0.10

    def __init__(
        self,
        *args,
        use_evidence_branch=1,
        use_ssm_branch=1,
        use_attention_branch=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_evidence_branch = bool(int(use_evidence_branch))
        self.use_ssm_branch = bool(int(use_ssm_branch))
        self.use_attention_branch = bool(int(use_attention_branch))

        evidence_dim = 8  # item prior, concept prior, and six learned statistics
        branch_hidden = max(32, self.d_model // 4)
        self.evidence_head = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, branch_hidden),
            nn.GELU(),
            nn.Linear(branch_hidden, 1),
        )
        self.ssm_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, branch_hidden),
            nn.GELU(),
            nn.Linear(branch_hidden, 1),
        )
        self.attention_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, branch_hidden),
            nn.GELU(),
            nn.Linear(branch_hidden, 1),
        )
        self.last_branch_logits = None

    def _branch_logits(self, item_prior, concept_prior, learned_stats, state, sequence):
        evidence_features = torch.cat(
            [item_prior, concept_prior, learned_stats], dim=-1
        )
        return {
            "evidence": self.evidence_head(evidence_features).squeeze(-1),
            "ssm": self.ssm_head(state).squeeze(-1),
            "attention": self.attention_head(sequence).squeeze(-1),
        }

    def forward(self, dcur, train=False, qtest=False):
        dropout = self.input[3]
        target_q, target_c, hist_q, hist_c, hist_r = self._sequences(dcur)
        if self.training and getattr(dropout, "warmup_active", False):
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
                target_c, hist_c, hist_r, item_prior, concept_prior
            )
            learned_stats = self._learned_stats(stats)
            learned_stats = self._replace_learned_exposure(
                learned_stats, target_c, hist_c, hist_r
            )

            target_concept = self.concept_emb(target_c.clamp_min(0))
            if has_item_sequence:
                target_item = self.item_emb(target_q.clamp_min(0))
                target_item = target_item * self._item_residual_reliability(
                    target_q
                ).unsqueeze(-1)
                if hasattr(self, "_apply_item_residual_dropout"):
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
                if hasattr(self, "_apply_item_residual_dropout"):
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
            full_state = self.ssm(token, scope_weight=scope_weight)
            state = full_state if self.use_ssm_branch else torch.zeros_like(full_state)

            full_sequence = self._attend(token + state)
            sequence = (
                full_sequence if self.use_attention_branch else token + state
            )

            fused = torch.cat([sequence, learned_stats], dim=-1)
            main_logit = self.pred(fused).squeeze(-1)
            branches = self._branch_logits(
                item_prior, concept_prior, learned_stats, state, sequence
            )
            self.last_branch_logits = branches
            logits = main_logit
            if self.use_evidence_branch:
                logits = logits + self.EVIDENCE_SCALE * branches["evidence"]
            if self.use_ssm_branch:
                logits = logits + self.SSM_SCALE * branches["ssm"]
            if self.use_attention_branch:
                logits = logits + self.ATTENTION_SCALE * branches["attention"]
            prediction = torch.sigmoid(logits)
            if qtest and not train:
                return prediction, fused
            if not train:
                return prediction
            route = sequence.new_zeros(4)
            return prediction, sequence.new_tensor(0.0), route
        finally:
            if hasattr(dropout, "clear_protection_mask"):
                dropout.clear_protection_mask()


__all__ = ["A2GMambaKT"]
