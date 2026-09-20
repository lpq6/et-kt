"""Complete single-trunk A2G, consolidating the verified server V33 chain.

The forward is written once. Component flags preserve the original intervention
semantics, including nested effects; they do not assert positive ablation gains.
"""

from __future__ import annotations

import torch
from torch import nn

from ._initialization import (
    LegacyInitialization,
    LEARNED_STAT_INDICES,
    REMOVED_STAT_INDICES,
)
from .modules.core import FoldedSelectedNormLinear, OneEpochConceptNoveltyDropout
from .modules.evidence import EXPOSURE_LOCAL_INDEX, EvidenceMethods
from .modules.attempt import ItemAttemptReadout
from .modules.pace import HistoricalPaceModulation, historical_pace_feature
from .modules.newton import PrequentialNewtonReadout
from .modules.graph import DirectedConceptGraph
from .modules.ffn import InputGatedFeedForward
from .modules.umk import umk_attention_bias, validate_lambda0


class A2G(EvidenceMethods, LegacyInitialization):
    CANDIDATE_ID = "a2g_v33_local_equivalent"
    IDENTITY_EVIDENCE_MULTIPLIER = 2.0

    def __init__(
        self,
        n_question,
        n_pid,
        *,
        d_model=256,
        d_ff=512,
        n_blocks=4,
        num_attn_heads=8,
        dropout=0.2,
        seq_len=200,
        emb_type="qid",
        emb_path="",
        prior_scale=0.08,
        stat_scale=0.12,
        input_decay_scale=0.5,
        boundary_local_floor=0.5,
        use_split_boundary=1,
        use_shared_embeddings=0,
        normalization_topology="full_postnorm",
        support_smoothing=24.0,
        item_residual_dropout=0.4,
        item_residual_dropout_schedule="constant",
        use_evidence_equivalent_residual=1,
        use_recency_weighted_concept_evidence=1,
        use_evidence_branch=1,
        use_ssm_branch=1,
        use_attention_branch=1,
        use_factorized_input=1,
        use_item_attempt_stage=1,
        use_history_pace=1,
        use_prequential_newton=1,
        use_concept_graph=1,
        use_ffn_gate=1,
        graph_checkpoint_steps=0,
        use_umk_ssm=0,
        use_umk_attn=0,
        use_umk_rwce=0,
        umk_lambda0=0.3,
        use_transfer_residual=0,
        use_causal_transfer_gate=0,
        use_memory_readout=1,
    ):
        if n_blocks < 2 or d_model % num_attn_heads or seq_len < 2:
            raise ValueError(
                "A2G requires >=2 blocks, valid head geometry and seq_len>=2"
            )
        if normalization_topology not in {
            "control",
            "attention_postnorm",
            "full_postnorm",
        }:
            raise ValueError("unknown normalization topology")
        self.normalization_topology = normalization_topology
        super().__init__(
            n_question,
            n_pid,
            d_model=d_model,
            d_ff=d_ff,
            n_blocks=n_blocks,
            num_attn_heads=num_attn_heads,
            dropout=dropout,
            seq_len=seq_len,
            emb_type=emb_type,
            emb_path=emb_path,
            prior_scale=prior_scale,
            stat_scale=stat_scale,
            input_decay_scale=input_decay_scale,
            boundary_local_floor=boundary_local_floor,
            use_split_boundary=use_split_boundary,
            use_shared_embeddings=use_shared_embeddings,
            use_umk_ssm=use_umk_ssm,
            umk_lambda0=umk_lambda0,
        )
        self.seq_len = int(seq_len)
        self.use_joint_weak_stat_prune = True
        self.input[3] = OneEpochConceptNoveltyDropout(self.input[3])
        rng = torch.random.get_rng_state().clone()
        reference_dim = 4 * d_model + 10
        removed = {4 * d_model + 2 + index for index in REMOVED_STAT_INDICES}
        active = [index for index in range(reference_dim) if index not in removed]
        old = self.input
        self.input = nn.Sequential(
            FoldedSelectedNormLinear.from_dense(old[0], old[1], active),
            nn.Identity(),
            old[2],
            old[3],
        )
        pred_active = list(range(d_model)) + [d_model + i for i in LEARNED_STAT_INDICES]
        self.pred[0] = FoldedSelectedNormLinear.from_folded(self.pred[0], pred_active)
        self.register_buffer(
            "learned_stat_indices",
            torch.tensor(LEARNED_STAT_INDICES),
            persistent=False,
        )
        torch.random.set_rng_state(rng)
        self.use_evidence_equivalent_residual = bool(use_evidence_equivalent_residual)
        self.support_smoothing = max(1e-6, float(support_smoothing))
        if self.use_evidence_equivalent_residual:
            self.register_buffer("item_support_count", torch.zeros(self.n_pid + 1))
        self.use_recency_weighted_concept_evidence = bool(
            use_recency_weighted_concept_evidence
        )
        self.item_residual_dropout = max(0.0, min(1.0, float(item_residual_dropout)))
        self.item_residual_dropout_schedule = item_residual_dropout_schedule
        self.use_evidence_branch = bool(use_evidence_branch)
        self.use_ssm_branch = bool(use_ssm_branch)
        self.use_attention_branch = bool(use_attention_branch)
        self.use_memory_readout = bool(use_memory_readout)
        self.memory_readout_gate = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.memory_readout_gate.weight)
        nn.init.constant_(self.memory_readout_gate.bias, -2.0)

        self.use_factorized_input = bool(use_factorized_input)
        self.semantic_width = 4 * d_model
        self.evidence_width = self.input[0].active_weight.size(1) - self.semantic_width
        self.semantic_stream = nn.Sequential(
            nn.LayerNorm(self.semantic_width),
            nn.Linear(self.semantic_width, d_model),
            nn.GELU(),
        )
        self.evidence_stream = nn.Sequential(
            nn.LayerNorm(self.evidence_width),
            nn.Linear(self.evidence_width, d_model),
            nn.GELU(),
        )
        self.factorized_residual = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.factorized_residual.weight)
        nn.init.zeros_(self.factorized_residual.bias)

        self.use_item_attempt_stage = bool(use_item_attempt_stage)
        with torch.random.fork_rng(devices=[]):
            self.attempt_readout = ItemAttemptReadout(d_model, num_attn_heads, seq_len)
        self.use_history_pace = bool(use_history_pace)
        self.history_pace = HistoricalPaceModulation(d_model)
        self.use_prequential_newton = bool(use_prequential_newton)
        with torch.random.fork_rng(devices=[]):
            self.prequential_newton = PrequentialNewtonReadout(d_model, num_attn_heads)
        self.use_concept_graph = bool(use_concept_graph)
        with torch.random.fork_rng(devices=[]):
            self.concept_graph = DirectedConceptGraph(
                d_model,
                num_attn_heads,
                checkpoint_steps=graph_checkpoint_steps,
            )
        for block in self.blocks:
            if "ffn" in block:
                block["ffn"] = InputGatedFeedForward(block["ffn"])
        self.use_ffn_gate = use_ffn_gate
        self.use_umk_attn = bool(use_umk_attn)
        self.use_umk_rwce = bool(use_umk_rwce)
        self.umk_lambda0 = validate_lambda0(umk_lambda0)
        if self.use_umk_attn:
            self.umk_beta = nn.Parameter(self.concept_prior.weight.new_tensor(1.0))
        self.use_transfer_residual = bool(use_transfer_residual)
        if self.use_transfer_residual:
            self.transfer_gate = nn.Parameter(
                self.concept_prior.weight.new_tensor(0.1)
            )
        self.use_causal_transfer_gate = bool(use_causal_transfer_gate)
        if self.use_causal_transfer_gate:
            self.causal_transfer_gate = nn.Parameter(
                self.concept_prior.weight.new_tensor(0.35)
            )

    @property
    def use_umk_ssm(self):
        return self.ssm.use_umk_ssm

    @property
    def use_ffn_gate(self):
        return self._use_ffn_gate

    @use_ffn_gate.setter
    def use_ffn_gate(self, value):
        self._use_ffn_gate = bool(value)
        for block in self.blocks:
            if "ffn" in block:
                block["ffn"].enabled = self._use_ffn_gate

    def _learned_stats(self, stats):
        return stats.index_select(-1, self.learned_stat_indices)

    @staticmethod
    def _direct_stat_logit(stats):
        return (
            stats[..., 1]
            + stats[..., 2]
            + stats[..., 3]
            - stats[..., 0]
            - stats[..., 4]
            - stats[..., 5]
        )

    def _factorized_token(self, raw, token):
        if not self.use_factorized_input:
            return token
        semantic = self.semantic_stream(raw[..., : self.semantic_width])
        evidence = self.evidence_stream(raw[..., self.semantic_width :])
        return token + self.factorized_residual(torch.cat([semantic, evidence], dim=-1))

    def _attend(self, x, target_c=None, concept_prior=None):
        future = torch.ones(
            x.size(1), x.size(1), device=x.device, dtype=torch.bool
        ).triu(1)
        if self.use_umk_attn:
            if target_c is None or target_c.shape != x.shape[:2] or target_c.device != x.device:
                raise ValueError("UMK attention requires input-aligned target concepts")
            if concept_prior is None or concept_prior.dtype != x.dtype:
                raise ValueError("UMK attention prior must share input dtype")
            # score_ij += beta * log K_cj(Delta_ij); one precomputed mask for all
            # blocks/heads, retaining the existing scaled-dot-product operator.
            bias = umk_attention_bias(target_c, concept_prior, self.umk_beta, self.umk_lambda0)
            heads = self.blocks[0]["attn"].num_heads
            future = bias.unsqueeze(1).expand(-1, heads, -1, -1).reshape(
                x.size(0) * heads, x.size(1), x.size(1)
            )
        for block in self.blocks:
            if self.normalization_topology == "control":
                normalized = block["norm"](x)
                attended, _ = block["attn"](
                    normalized,
                    normalized,
                    normalized,
                    attn_mask=future,
                    need_weights=False,
                )
                x = x + block["drop"](attended)
            else:
                attended, _ = block["attn"](
                    x, x, x, attn_mask=future, need_weights=False
                )
                x = block["norm"](x + block["drop"](attended))
            if "ffn" in block:
                if self.normalization_topology == "full_postnorm":
                    x = block["ffn_norm"](x + block["ffn"](x))
                else:
                    x = x + block["ffn"](block["ffn_norm"](x))
        return x

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
            stats = self._stats(target_c, hist_c, hist_r, item_prior, concept_prior)
            learned = self._replace_learned_exposure(
                self._learned_stats(stats), target_c, hist_c, hist_r,
                concept_prior=concept_prior.squeeze(-1),
            )
            if self.use_transfer_residual:
                history_concept_prior = self.concept_prior(
                    hist_c.clamp_min(0)
                ).squeeze(-1)
                history_item_prior = (
                    self.item_prior(hist_q.clamp_min(0)).squeeze(-1)
                    if has_items
                    else torch.zeros_like(history_concept_prior)
                )
                transfer_residual = self._transfer_calibrated_concept_residual(
                    target_q,
                    target_c,
                    hist_q,
                    hist_c,
                    hist_r,
                    history_item_prior + history_concept_prior,
                )
                learned = learned.clone()
                learned[..., EXPOSURE_LOCAL_INDEX] = (
                    learned[..., EXPOSURE_LOCAL_INDEX]
                    + torch.tanh(self.transfer_gate) * transfer_residual
                )
            target_concept = self.concept_emb(target_c.clamp_min(0))
            history_concept = self.hist_concept_emb(hist_c.clamp_min(0))
            if has_items:
                target_item = self.item_emb(target_q.clamp_min(0))
                history_item = self.hist_item_emb(hist_q.clamp_min(0))
                if self.use_evidence_branch:
                    target_item = target_item * self._item_residual_reliability(
                        target_q
                    ).unsqueeze(-1)
                    history_item = history_item * self._item_residual_reliability(
                        hist_q
                    ).unsqueeze(-1)
                target_item = self._apply_item_residual_dropout(target_item, target_q)
                history_item = self._apply_item_residual_dropout(history_item, hist_q)
            else:
                target_item = torch.zeros_like(target_concept)
                history_item = torch.zeros_like(history_concept)
            direct_logit = self.prior_scale * (item_prior + concept_prior).squeeze(
                -1
            ) + self.stat_scale * self._direct_stat_logit(stats)
            if self.use_causal_transfer_gate:
                history_concept_prior = self.concept_prior(
                    hist_c.clamp_min(0)
                ).squeeze(-1)
                history_item_prior = (
                    self.item_prior(hist_q.clamp_min(0)).squeeze(-1)
                    if has_items
                    else torch.zeros_like(history_concept_prior)
                )
                causal_transfer = self._causal_transfer_gate_signal(
                    target_q,
                    target_c,
                    hist_q,
                    hist_c,
                    hist_r,
                    history_item_prior + history_concept_prior,
                )
                direct_logit = direct_logit + (
                    torch.tanh(self.causal_transfer_gate) * causal_transfer
                )
            if not self.use_evidence_branch:
                item_prior = torch.zeros_like(item_prior)
                concept_prior = torch.zeros_like(concept_prior)
                learned = torch.zeros_like(learned)
                direct_logit = torch.zeros_like(direct_logit)
            target = target_item + target_concept
            history = history_item + history_concept + self.resp_emb(hist_r.clamp(0, 2))
            if self.use_history_pace:
                history = self.history_pace(
                    history, historical_pace_feature(dcur.get("tseqs"), hist_c)
                )
            raw = torch.cat(
                [
                    target,
                    history,
                    target_item,
                    target_concept,
                    item_prior,
                    concept_prior,
                    learned,
                ],
                dim=-1,
            )
            token = self._factorized_token(raw, self.input(raw))
            if self.use_concept_graph:
                token = token + self.concept_graph(
                    history,
                    hist_c,
                    hist_r,
                    target_c,
                    self.hist_concept_emb.weight[: self.n_question],
                )
            full_state = self.ssm(
                token, scope_weight=self._boundary_weights(token, target_c),
                concept_prior=concept_prior.squeeze(-1),
            )
            state = full_state if self.use_ssm_branch else torch.zeros_like(full_state)
            attention_input = token + state
            full_attention = self._attend(
                attention_input, target_c=target_c, concept_prior=concept_prior.squeeze(-1)
            )
            sequence = full_attention if self.use_attention_branch else attention_input
            if self.use_memory_readout:
                sequence = sequence + torch.sigmoid(self.memory_readout_gate(token)) * state
            fused = torch.cat([sequence, learned], dim=-1)
            if self.use_item_attempt_stage:
                direct_logit = direct_logit + self.attempt_readout(
                    sequence, target, target_q, hist_q, hist_r
                )
            base_logits = self.pred(fused).squeeze(-1) + direct_logit
            correction = (
                self.prequential_newton(base_logits, target, target_c, dcur["rseqs"])
                if self.use_prequential_newton
                else torch.zeros_like(base_logits)
            )
            prediction = torch.sigmoid(base_logits + correction)
            if qtest and not train:
                return prediction, fused
            if train:
                return prediction, fused.new_tensor(0.0), fused.new_zeros(4)
            return prediction
        finally:
            dropout.clear_protection_mask()


A2GMambaKT = A2G
