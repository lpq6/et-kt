"""Parent-preserving causal role-interaction correction candidate."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from a2g_zero_init_memory_readout_candidate import (
    A2GMambaKT as _ZeroInitReadoutA2GMambaKT,
)
from work.a2g_mambakt_promoted_no_duplicate_folded_bias_20260715 import (
    SelectiveSSMBlock,
)


class ZeroInitRoleLogitHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, value):
        return self.network(value).squeeze(-1)


class CausalEvidenceFeatures(nn.Module):
    EXTRA_DIM = 12

    @staticmethod
    def _mean(mask, values, fallback=0.5):
        denominator = mask.sum(dim=-1)
        numerator = (
            mask.to(values.dtype) * values.unsqueeze(1)
        ).sum(dim=-1)
        result = numerator / denominator.clamp_min(1).to(values.dtype)
        return torch.where(
            denominator.gt(0),
            result,
            result.new_full(result.shape, float(fallback)),
        )

    @staticmethod
    def _last(mask, values, fallback=0.5):
        sequence_length = values.size(1)
        positions = torch.arange(
            sequence_length,
            device=values.device,
            dtype=torch.long,
        ).view(1, 1, sequence_length)
        last_index = (
            mask.to(torch.long) * positions
        ).amax(dim=-1).clamp_max(sequence_length - 1)
        gathered = values.gather(1, last_index)
        return torch.where(
            mask.any(dim=-1),
            gathered,
            gathered.new_full(gathered.shape, float(fallback)),
        )

    @staticmethod
    def _weighted_mean(mask, values, distance, decay, fallback=0.5):
        weights = torch.pow(
            values.new_tensor(float(decay)),
            distance.to(values.dtype),
        )
        weighted_mask = mask.to(values.dtype) * weights
        denominator = weighted_mask.sum(dim=-1)
        numerator = (
            weighted_mask * values.unsqueeze(1)
        ).sum(dim=-1)
        result = numerator / denominator.clamp_min(1.0e-6)
        return torch.where(
            denominator.gt(0),
            result,
            result.new_full(result.shape, float(fallback)),
        )

    def forward(
        self,
        target_q,
        target_c,
        hist_q,
        hist_c,
        hist_r,
        stats,
    ):
        sequence_length = target_c.size(1)
        target_positions = torch.arange(
            sequence_length,
            device=target_c.device,
        ).view(1, sequence_length, 1)
        history_positions = torch.arange(
            sequence_length,
            device=target_c.device,
        ).view(1, 1, sequence_length)
        previous = history_positions.lt(target_positions)
        valid_history = hist_c.gt(0)
        same_concept = (
            target_c.unsqueeze(-1).eq(hist_c.unsqueeze(-2))
            & target_c.gt(0).unsqueeze(-1)
            & valid_history.unsqueeze(1)
            & previous
        )
        same_question = (
            target_q.unsqueeze(-1).eq(hist_q.unsqueeze(-2))
            & target_q.gt(0).unsqueeze(-1)
            & hist_q.gt(0).unsqueeze(1)
            & valid_history.unsqueeze(1)
            & previous
        )
        distance = (
            target_positions - history_positions
        ).clamp_min(1).to(torch.float32)
        concept_accuracy = self._mean(same_concept, hist_r)
        concept_last = self._last(same_concept, hist_r)
        concept_ewma_short = self._weighted_mean(
            same_concept,
            hist_r,
            distance,
            decay=0.90,
        )
        concept_ewma_long = self._weighted_mean(
            same_concept,
            hist_r,
            distance,
            decay=0.98,
        )
        question_accuracy = self._mean(same_question, hist_r)
        question_last = self._last(same_question, hist_r)
        recent_features = []
        for window in (4, 16, 64):
            recent = (
                previous
                & valid_history.unsqueeze(1)
                & history_positions.ge(target_positions - window)
            )
            recent_features.append(self._mean(recent, hist_r))
        global_history = previous & valid_history.unsqueeze(1)
        global_accuracy = self._mean(global_history, hist_r)
        global_last = self._last(global_history, hist_r)
        concept_count = torch.log1p(
            same_concept.sum(dim=-1).to(torch.float32)
        ) / math.log1p(max(2, sequence_length))
        question_count = torch.log1p(
            same_question.sum(dim=-1).to(torch.float32)
        ) / math.log1p(max(2, sequence_length))
        return torch.stack(
            [
                concept_accuracy,
                concept_last,
                concept_ewma_short,
                concept_ewma_long,
                question_accuracy,
                question_last,
                recent_features[0],
                recent_features[1],
                recent_features[2],
                global_accuracy,
                global_last,
                concept_count + question_count,
            ],
            dim=-1,
        )


class CausalDenseSparseRelationAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        minimum_temperature: float = 0.55,
    ):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.minimum_temperature = float(minimum_temperature)
        self.query_norm = nn.LayerNorm(d_model)
        self.key_norm = nn.LayerNorm(d_model)
        self.value_norm = nn.LayerNorm(d_model)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.relation_projection = nn.Linear(4, self.num_heads, bias=False)
        self.temperature_projection = nn.Linear(d_model, self.num_heads)
        self.sparsity_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)
        self.last_support_count = None

    @staticmethod
    def sparsemax(logits, dim=-1):
        shifted = logits - logits.max(dim=dim, keepdim=True).values
        sorted_logits, _ = torch.sort(shifted, dim=dim, descending=True)
        cumulative = sorted_logits.cumsum(dim=dim)
        width = logits.size(dim)
        view_shape = [1] * logits.ndim
        view_shape[dim] = width
        ranks = torch.arange(
            1,
            width + 1,
            device=logits.device,
            dtype=logits.dtype,
        ).view(view_shape)
        support = 1.0 + ranks * sorted_logits > cumulative
        support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
        support_index = support_size.to(torch.long) - 1
        threshold_sum = cumulative.gather(dim, support_index)
        threshold = (
            threshold_sum - 1.0
        ) / support_size.to(logits.dtype)
        return F.relu(shifted - threshold)

    @staticmethod
    def relation_features(target_q, target_c, hist_q, hist_c):
        sequence_length = target_c.size(1)
        target_positions = torch.arange(
            sequence_length,
            device=target_c.device,
        ).view(1, sequence_length, 1)
        history_positions = torch.arange(
            sequence_length,
            device=target_c.device,
        ).view(1, 1, sequence_length)
        distance = (
            target_positions - history_positions + 1
        ).to(torch.float32)
        causal = distance.ge(1)
        distance = distance.clamp_min(1)
        same_concept = (
            target_c.unsqueeze(-1).eq(hist_c.unsqueeze(-2))
            & target_c.gt(0).unsqueeze(-1)
            & hist_c.gt(0).unsqueeze(1)
        )
        same_item = (
            target_q.unsqueeze(-1).eq(hist_q.unsqueeze(-2))
            & target_q.gt(0).unsqueeze(-1)
            & hist_q.gt(0).unsqueeze(1)
        )
        recency = torch.reciprocal(distance) * causal.to(torch.float32)
        valid_history = hist_c.gt(0)
        valid_history = valid_history | history_positions.eq(0).squeeze(0)
        relation = torch.stack(
            [
                same_concept.to(torch.float32),
                same_item.to(torch.float32),
                recency.expand(target_c.size(0), -1, -1),
                valid_history.unsqueeze(1)
                .expand(-1, sequence_length, -1)
                .to(torch.float32),
            ],
            dim=-1,
        )
        return relation, causal, valid_history

    def forward(
        self,
        query,
        keys,
        values,
        target_q,
        target_c,
        hist_q,
        hist_c,
    ):
        batch_size, sequence_length, _ = query.shape
        relation, causal, valid_history = self.relation_features(
            target_q,
            target_c,
            hist_q,
            hist_c,
        )
        query_input = query
        query = self.query_norm(query)
        keys = self.key_norm(keys)
        values = self.value_norm(values)
        query = self.query_projection(query).view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        keys = self.key_projection(keys).view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        values = self.value_projection(values).view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        scores = torch.matmul(
            query,
            keys.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)
        relation_bias = self.relation_projection(relation).permute(
            0,
            3,
            1,
            2,
        )
        scores = scores + relation_bias
        query_flat = query.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.d_model,
        )
        temperature = self.minimum_temperature + F.softplus(
            self.temperature_projection(query_flat)
        )
        scores = scores / temperature.transpose(1, 2).unsqueeze(-1)
        blocked = (
            (~causal).unsqueeze(1)
            | ~valid_history.unsqueeze(1).unsqueeze(1)
        )
        masked_scores = scores.masked_fill(blocked, -1.0e4)
        dense_weights = F.softmax(masked_scores, dim=-1)
        dense_weights = dense_weights.masked_fill(blocked, 0.0)
        dense_weights = dense_weights / dense_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-12)
        sparse_weights = self.sparsemax(masked_scores, dim=-1)
        sparse_weights = sparse_weights.masked_fill(blocked, 0.0)
        sparse_weights = sparse_weights / sparse_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-12)
        sparse_gate = torch.sigmoid(
            self.sparsity_projection(query_flat)
        ).transpose(1, 2).unsqueeze(-1)
        weights = (
            sparse_gate * sparse_weights
            + (1.0 - sparse_gate) * dense_weights
        )
        self.last_support_count = weights.gt(1.0e-7).sum(
            dim=-1
        ).detach()
        weights = self.dropout(weights)
        weights = weights / weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-12)
        attended = torch.matmul(weights, values)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.d_model,
        )
        attended = self.output_projection(attended)
        return self.output_norm(
            query_input + self.dropout(attended)
        )


class A2GMambaKT(_ZeroInitReadoutA2GMambaKT):
    """A2G parent plus causal evidence, dual-memory, and relation roles."""

    CANDIDATE_ID = "single_trunk_cric_parent_preserving_20260908_v3"

    def __init__(
        self,
        *args,
        use_evidence_branch=1,
        use_ssm_branch=1,
        use_attention_branch=1,
        cric_residual_scale=0.25,
        cric_joint_scale=0.25,
        cric_minimum_temperature=0.55,
        **kwargs,
    ):
        super().__init__(
            *args,
            use_evidence_branch=use_evidence_branch,
            use_ssm_branch=use_ssm_branch,
            use_attention_branch=use_attention_branch,
            **kwargs,
        )
        self.use_evidence_branch = bool(int(use_evidence_branch))
        self.use_ssm_branch = bool(int(use_ssm_branch))
        self.use_attention_branch = bool(int(use_attention_branch))
        self.cric_residual_scale = float(cric_residual_scale)
        self.cric_joint_scale = float(cric_joint_scale)
        d_model = int(self.d_model)
        dropout = float(kwargs.get("dropout", 0.2))
        num_heads = int(kwargs.get("num_attn_heads", 8))
        input_decay_scale = float(kwargs.get("input_decay_scale", 0.5))
        hidden = max(64, d_model // 2)
        learned_stat_dim = int(self.learned_stat_indices.numel())
        evidence_dim = (
            2
            + 8
            + learned_stat_dim
            + CausalEvidenceFeatures.EXTRA_DIM
        )
        self.cric_evidence_features = CausalEvidenceFeatures()
        self.cric_evidence_encoder = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.GELU(),
        )
        self.cric_question_input = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.cric_knowledge_input = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.cric_question_ssm = SelectiveSSMBlock(
            d_model,
            dropout=dropout,
            input_decay_scale=input_decay_scale,
        )
        self.cric_knowledge_ssm = SelectiveSSMBlock(
            d_model,
            dropout=dropout,
            input_decay_scale=input_decay_scale,
        )
        self.cric_ssm_merge = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.cric_attention_query = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.cric_attention_history = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
        )
        self.cric_attention_keys = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.cric_attention_values = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.cric_attention = CausalDenseSparseRelationAttention(
            d_model,
            num_heads=num_heads,
            dropout=dropout,
            minimum_temperature=cric_minimum_temperature,
        )
        self.cric_evidence_head = ZeroInitRoleLogitHead(
            2 * d_model,
            hidden,
            dropout,
        )
        self.cric_ssm_head = ZeroInitRoleLogitHead(
            2 * d_model,
            hidden,
            dropout,
        )
        self.cric_attention_head = ZeroInitRoleLogitHead(
            2 * d_model,
            hidden,
            dropout,
        )
        self.cric_joint_head = ZeroInitRoleLogitHead(
            7 * d_model,
            hidden,
            dropout,
        )
        self.last_role_logits = None

    def _cric_path(self, dcur, base_fused):
        target_q, target_c, hist_q, hist_c, hist_r = self._sequences(
            dcur
        )
        concept_prior = self.concept_prior(target_c.clamp_min(0))
        has_items = (
            self.n_pid > 0
            and target_q.numel() > 0
            and target_q.size(1) == target_c.size(1)
        )
        if has_items:
            item_prior = self.item_prior(target_q.clamp_min(0))
            target_item = self.item_emb(target_q.clamp_min(0))
            history_item = self.hist_item_emb(hist_q.clamp_min(0))
        else:
            item_prior = torch.zeros_like(concept_prior)
            target_item = torch.zeros_like(
                self.concept_emb(target_c.clamp_min(0))
            )
            history_item = torch.zeros_like(target_item)
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
        extra_features = self.cric_evidence_features(
            target_q,
            target_c,
            hist_q,
            hist_c,
            hist_r,
            stats,
        )
        evidence_input = torch.cat(
            [
                item_prior,
                concept_prior,
                stats,
                learned_stats,
                extra_features,
            ],
            dim=-1,
        )
        evidence_full = self.cric_evidence_encoder(evidence_input)
        evidence_role = (
            evidence_full
            if self.use_evidence_branch
            else torch.zeros_like(evidence_full)
        )

        target_concept = self.concept_emb(target_c.clamp_min(0))
        history_concept = self.hist_concept_emb(hist_c.clamp_min(0))
        response = self.resp_emb(hist_r.clamp(0, 2))
        question_history = history_item + history_concept
        knowledge_history = history_concept + response
        question_input = self.cric_question_input(question_history)
        knowledge_input = self.cric_knowledge_input(knowledge_history)
        scope_weight = self._boundary_weights(
            question_input,
            target_c,
        )
        question_state = self.cric_question_ssm(
            question_input,
            scope_weight=scope_weight,
        )
        knowledge_state = self.cric_knowledge_ssm(
            knowledge_input,
            scope_weight=scope_weight,
        )
        merged_state_full = self.cric_ssm_merge(
            torch.cat([question_state, knowledge_state], dim=-1)
        )
        ssm_role = (
            merged_state_full
            if self.use_ssm_branch
            else torch.zeros_like(merged_state_full)
        )

        target_content = self.cric_attention_query(
            torch.cat([target_item, target_concept], dim=-1)
        )
        history_content = self.cric_attention_history(
            torch.cat(
                [history_item, history_concept, response],
                dim=-1,
            )
        )
        keys = self.cric_attention_keys(
            torch.cat([history_content, knowledge_input], dim=-1)
        )
        values = self.cric_attention_values(
            torch.cat([history_content, knowledge_input], dim=-1)
        )
        attention_full = self.cric_attention(
            target_content,
            keys,
            values,
            target_q,
            target_c,
            hist_q,
            hist_c,
        )
        attention_role = (
            attention_full
            if self.use_attention_branch
            else torch.zeros_like(attention_full)
        )

        base_context = base_fused[..., : self.d_model]
        evidence_logit_raw = self.cric_evidence_head(
            torch.cat([base_context, evidence_role], dim=-1)
        )
        ssm_logit_raw = self.cric_ssm_head(
            torch.cat([base_context, ssm_role], dim=-1)
        )
        attention_logit_raw = self.cric_attention_head(
            torch.cat([base_context, attention_role], dim=-1)
        )
        interaction = torch.cat(
            [
                base_context,
                evidence_role,
                ssm_role,
                attention_role,
                evidence_role * ssm_role,
                evidence_role * attention_role,
                ssm_role * attention_role,
            ],
            dim=-1,
        )
        joint_logit_raw = self.cric_joint_head(interaction)
        joint_active = (
            self.use_evidence_branch
            and self.use_ssm_branch
            and self.use_attention_branch
        )
        evidence_logit = (
            evidence_logit_raw
            if self.use_evidence_branch
            else torch.zeros_like(evidence_logit_raw)
        )
        ssm_logit = (
            ssm_logit_raw
            if self.use_ssm_branch
            else torch.zeros_like(ssm_logit_raw)
        )
        attention_logit = (
            attention_logit_raw
            if self.use_attention_branch
            else torch.zeros_like(attention_logit_raw)
        )
        joint_logit = (
            joint_logit_raw
            if joint_active
            else torch.zeros_like(joint_logit_raw)
        )
        correction = self.cric_residual_scale * (
            evidence_logit + ssm_logit + attention_logit
        )
        correction = correction + self.cric_joint_scale * joint_logit
        return correction, {
            "evidence": evidence_logit,
            "ssm": ssm_logit,
            "attention": attention_logit,
            "joint": joint_logit,
            "evidence_state": evidence_role,
            "ssm_state": ssm_role,
            "attention_state": attention_role,
            "extra_features": extra_features,
        }

    def forward(self, dcur, train=False, qtest=False):
        base_prediction, base_fused = super().forward(
            dcur,
            train=False,
            qtest=True,
        )
        correction, role_outputs = self._cric_path(dcur, base_fused)
        self.last_role_logits = role_outputs
        base_logit = torch.logit(
            base_prediction.clamp(1.0e-6, 1.0 - 1.0e-6)
        )
        prediction = torch.sigmoid(base_logit + correction)
        if qtest and not train:
            return prediction, base_fused
        if not train:
            return prediction
        route = base_fused.new_zeros(4)
        return prediction, base_fused.new_tensor(0.0), route


__all__ = [
    "A2GMambaKT",
    "CausalDenseSparseRelationAttention",
    "CausalEvidenceFeatures",
    "ZeroInitRoleLogitHead",
]
