from __future__ import annotations

import torch
from torch import nn

from factorized_input_candidate import A2GMambaKT as _FactorizedA2GMambaKT


class PairwiseRoleRelation(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        relation_width = max(32, d_model // 2)
        self.target_private = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.history_private = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.evidence_private = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
        )
        self.target_history_target = nn.Linear(
            d_model, relation_width, bias=False
        )
        self.target_history_history = nn.Linear(
            d_model, relation_width, bias=False
        )
        self.target_evidence_target = nn.Linear(
            d_model, relation_width, bias=False
        )
        self.target_evidence_evidence = nn.Linear(
            d_model, relation_width, bias=False
        )
        self.history_evidence_history = nn.Linear(
            d_model, relation_width, bias=False
        )
        self.history_evidence_evidence = nn.Linear(
            d_model, relation_width, bias=False
        )
        self.target_history_out = nn.Sequential(
            nn.LayerNorm(relation_width),
            nn.Linear(relation_width, d_model),
            nn.GELU(),
        )
        self.target_evidence_out = nn.Sequential(
            nn.LayerNorm(relation_width),
            nn.Linear(relation_width, d_model),
            nn.GELU(),
        )
        self.history_evidence_out = nn.Sequential(
            nn.LayerNorm(relation_width),
            nn.Linear(relation_width, d_model),
            nn.GELU(),
        )
        self.target_update = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
        )
        self.history_update = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
        )
        self.evidence_update = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
        )

    @staticmethod
    def _interaction(left, right, left_projection, right_projection):
        return torch.tanh(left_projection(left)) * torch.tanh(
            right_projection(right)
        )

    def forward(self, target, history, evidence):
        target_private = self.target_private(target)
        history_private = self.history_private(history)
        evidence_private = self.evidence_private(evidence)
        shared = self.shared(
            torch.cat([target_private, history_private, evidence_private], dim=-1)
        )

        target_history = self.target_history_out(
            self._interaction(
                target_private,
                history_private,
                self.target_history_target,
                self.target_history_history,
            )
        )
        target_evidence = self.target_evidence_out(
            self._interaction(
                target_private,
                evidence_private,
                self.target_evidence_target,
                self.target_evidence_evidence,
            )
        )
        history_evidence = self.history_evidence_out(
            self._interaction(
                history_private,
                evidence_private,
                self.history_evidence_history,
                self.history_evidence_evidence,
            )
        )

        target_role = self.target_update(
            torch.cat(
                [target_private, shared, target_history, target_evidence],
                dim=-1,
            )
        )
        history_role = self.history_update(
            torch.cat(
                [history_private, shared, target_history, history_evidence],
                dim=-1,
            )
        )
        evidence_role = self.evidence_update(
            torch.cat(
                [evidence_private, shared, target_evidence, history_evidence],
                dim=-1,
            )
        )
        return self.output(
            torch.cat([target_role, history_role, evidence_role], dim=-1)
        )


class A2GMambaKT(_FactorizedA2GMambaKT):
    CANDIDATE_ID = "incumbent_a2g_role_relation_residual_20260909_v14"

    def __init__(self, *args, use_role_relation=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_role_relation = bool(int(use_role_relation))
        self.role_relation = PairwiseRoleRelation(self.d_model)
        self.role_relation_residual = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.role_relation_residual.weight)
        nn.init.zeros_(self.role_relation_residual.bias)

    def _factorized_token(self, raw_input, token):
        if not self.use_factorized_input and not self.use_role_relation:
            return token

        semantic_raw = raw_input[..., : self.semantic_width]
        evidence_raw = raw_input[..., self.semantic_width :]
        semantic = self.semantic_stream(semantic_raw)
        evidence = self.evidence_stream(evidence_raw)
        factorized = token
        if self.use_factorized_input:
            factorized = factorized + self.factorized_residual(
                torch.cat([semantic, evidence], dim=-1)
            )
        if not self.use_role_relation:
            return factorized

        target = raw_input[..., : self.d_model]
        history = raw_input[..., self.d_model : self.d_model * 2]
        relation = self.role_relation(target, history, evidence)
        return factorized + self.role_relation_residual(relation)


__all__ = ["A2GMambaKT", "PairwiseRoleRelation"]
