"""V28 with learner-local, directed concept-graph recurrent features."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from prequential_newton_candidate import A2GMambaKT as _V28


def graph_state_rollout(events, hist_c, hist_r, target_c, edges, cell, state_projection):
    """Update with shifted event t-1 before reading the known target at t."""
    if events.ndim != 3 or min(events.shape) < 1 or not events.is_floating_point():
        raise ValueError("event features must have a nonempty floating B/L/R shape")
    batch, length, rank = events.shape
    if any(value.shape != (batch, length) for value in (hist_c, hist_r, target_c)):
        raise ValueError("concepts and responses must align with event features")
    if hist_c.dtype != torch.long or target_c.dtype != torch.long:
        raise ValueError("concept IDs must be int64")
    if edges.ndim != 2 or edges.size(0) != edges.size(1):
        raise ValueError("edge weights must be square")
    if edges.dtype != events.dtype or any(
        value.device != events.device for value in (hist_c, hist_r, target_c, edges)
    ):
        raise ValueError("edge dtype and all devices must agree")
    if not isinstance(cell, nn.GRUCell) or (cell.input_size, cell.hidden_size) != (rank, rank):
        raise ValueError("the recurrent engine must be a matching PyTorch GRUCell")
    nodes = edges.size(0)
    if nodes == 0:
        return torch.zeros_like(events)
    known_source = hist_c.ge(2) & hist_c.lt(nodes + 2)
    valid = known_source & (hist_r.eq(0) | hist_r.eq(1))
    known_target = target_c.ge(2) & target_c.lt(nodes + 2)
    source_index = (hist_c - 2).clamp(0, nodes - 1)
    target_index = (target_c - 2).clamp(0, nodes - 1)
    learners = torch.arange(batch, device=events.device)
    state = events.new_zeros(batch, nodes, rank)
    reads = []
    for position in range(length):
        source_state = state[learners, source_index[:, position]]
        message = torch.tanh(events[:, position] + state_projection(source_state))
        proposed = cell(
            message[:, None, :].expand(batch, nodes, rank).reshape(-1, rank),
            state.reshape(-1, rank),
        ).reshape(batch, nodes, rank)
        weights = edges.index_select(0, source_index[:, position]).unsqueeze(-1)
        updated = torch.lerp(state, proposed, weights)
        state = torch.where(valid[:, position, None, None], updated, state)
        value = state[learners, target_index[:, position]]
        reads.append(torch.where(known_target[:, position, None], value, torch.zeros_like(value)))
    return torch.stack(reads, dim=1)


class DirectedConceptGraph(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError("graph state width must equal one attention head")
        self.d_model = int(d_model)
        self.rank = int(d_model // num_heads)
        self.source_projection = nn.Linear(self.d_model, self.rank, bias=False)
        self.destination_projection = nn.Linear(self.d_model, self.rank, bias=False)
        self.event_projection = nn.Linear(self.d_model, self.rank)
        self.state_projection = nn.Linear(self.rank, self.rank, bias=False)
        self.cell = nn.GRUCell(self.rank, self.rank)
        self.output = nn.Linear(self.rank, self.d_model, bias=False)
        nn.init.zeros_(self.output.weight)

    def edge_weights(self, concept_table):
        if concept_table.ndim != 2 or concept_table.size(-1) != self.d_model:
            raise ValueError("concept table width changed")
        concepts = concept_table[2:]
        nodes = concepts.size(0)
        if nodes <= 1:
            return concept_table.new_ones(nodes, nodes)
        normalized = F.layer_norm(concepts, (self.d_model,))
        source = self.source_projection(normalized)
        destination = self.destination_projection(normalized)
        scores = source @ destination.T / math.sqrt(self.rank)
        diagonal = torch.eye(nodes, device=scores.device, dtype=torch.bool)
        neighbors = torch.softmax(scores.masked_fill(diagonal, -torch.inf), dim=-1)
        # An observed source updates fully; its other outgoing weights sum to one.
        return neighbors + diagonal.to(dtype=scores.dtype)

    def forward(self, history, hist_c, hist_r, target_c, concept_table):
        if history.ndim != 3 or history.size(-1) != self.d_model:
            raise ValueError("history embedding width changed")
        events = self.event_projection(F.layer_norm(history, (self.d_model,)))
        reads = graph_state_rollout(
            events, hist_c, hist_r, target_c, self.edge_weights(concept_table),
            self.cell, self.state_projection,
        )
        return self.output(reads)


class A2GMambaKT(_V28):
    CANDIDATE_ID = "a2g_v28_directed_concept_graph_20260911_v32"

    def __init__(self, *args, use_concept_graph=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_concept_graph = bool(int(use_concept_graph))
        with torch.random.fork_rng(devices=[]):
            self.concept_graph = DirectedConceptGraph(
                self.d_model, self.blocks[0]["attn"].num_heads,
            )

    def forward(self, dcur, train=False, qtest=False):
        if not self.use_concept_graph:
            return super().forward(dcur, train=train, qtest=qtest)
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
            learned_stats = self._replace_learned_exposure(
                self._learned_stats(stats), target_c, hist_c, hist_r
            )
            target_concept = self.concept_emb(target_c.clamp_min(0))
            history_concept = self.hist_concept_emb(hist_c.clamp_min(0))
            if has_items:
                target_item = self.item_emb(target_q.clamp_min(0))
                history_item = self.hist_item_emb(hist_q.clamp_min(0))
                target_reliability = self._item_residual_reliability(target_q)
                history_reliability = self._item_residual_reliability(hist_q)
                if self.use_evidence_branch:
                    target_item = target_item * target_reliability.unsqueeze(-1)
                    history_item = history_item * history_reliability.unsqueeze(-1)
                target_item = self._apply_item_residual_dropout(target_item, target_q)
                history_item = self._apply_item_residual_dropout(history_item, hist_q)
            else:
                target_item = torch.zeros_like(target_concept)
                history_item = torch.zeros_like(history_concept)
            direct_logit = (
                self.prior_scale * (item_prior + concept_prior).squeeze(-1)
                + self.stat_scale * self._direct_stat_logit(stats)
            )
            if not self.use_evidence_branch:
                item_prior = torch.zeros_like(item_prior)
                concept_prior = torch.zeros_like(concept_prior)
                learned_stats = torch.zeros_like(learned_stats)
                direct_logit = torch.zeros_like(direct_logit)
            target = target_item + target_concept
            history = history_item + history_concept + self.resp_emb(hist_r.clamp(0, 2))
            if self.use_history_pace:
                history = self._modulate_history(history, dcur, hist_c)
            raw_input = torch.cat(
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
            token = self._factorized_token(raw_input, self.input(raw_input))
            token = token + self.concept_graph(
                history, hist_c, hist_r, target_c,
                self.hist_concept_emb.weight[:self.n_question],
            )
            full_state = self.ssm(token, scope_weight=self._boundary_weights(token, target_c))
            state = full_state if self.use_ssm_branch else torch.zeros_like(full_state)
            attention_input = token + state
            full_attention = self._attend(attention_input)
            sequence = full_attention if self.use_attention_branch else attention_input
            gate = torch.sigmoid(self.memory_readout_gate(token))
            sequence = sequence + gate * state
            fused = torch.cat([sequence, learned_stats], dim=-1)
            if self.use_item_attempt_stage:
                direct_logit = direct_logit + self.attempt_readout(
                    sequence, target, target_q, hist_q, hist_r,
                )
            base_logits = self.pred(fused).squeeze(-1) + direct_logit
            correction = (
                self.prequential_newton(base_logits, target, target_c, dcur["rseqs"])
                if self.use_prequential_newton else torch.zeros_like(base_logits)
            )
            prediction = torch.sigmoid(base_logits + correction)
            if qtest and not train:
                return prediction, fused
            if not train:
                return prediction
            return prediction, fused.new_tensor(0.0), fused.new_zeros(4)
        finally:
            dropout.clear_protection_mask()


__all__ = ["A2GMambaKT", "DirectedConceptGraph", "graph_state_rollout"]
