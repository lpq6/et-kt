"""Extracted from the reviewed concept_graph_candidate.py; see source_map.json."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def checkpointed_graph_rollout(
    events,
    source_index,
    target_index,
    valid,
    known_target,
    edges,
    cell,
    state_projection,
    chunk_size,
):
    batch, length, rank = events.shape
    nodes = edges.size(0)
    learners = torch.arange(batch, device=events.device)
    state = events.new_zeros(batch, nodes, rank)
    reads = []

    def segment(state, events, source_index, target_index, valid, known_target, edges):
        values = []
        for position in range(events.size(1)):
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
            values.append(
                torch.where(
                    known_target[:, position, None], value, torch.zeros_like(value)
                )
            )
        return state, torch.stack(values, dim=1)

    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        state, values = checkpoint(
            segment,
            state,
            events[:, start:end],
            source_index[:, start:end],
            target_index[:, start:end],
            valid[:, start:end],
            known_target[:, start:end],
            edges,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        reads.append(values)
    return torch.cat(reads, dim=1)


def graph_state_rollout(
    events,
    hist_c,
    hist_r,
    target_c,
    edges,
    cell,
    state_projection,
    checkpoint_steps=0,
):
    """Update with shifted event t-1 before reading the known target at t."""
    if events.ndim != 3 or min(events.shape) < 1 or (not events.is_floating_point()):
        raise ValueError("event features must have a nonempty floating B/L/R shape")
    (batch, length, rank) = events.shape
    if any((value.shape != (batch, length) for value in (hist_c, hist_r, target_c))):
        raise ValueError("concepts and responses must align with event features")
    if hist_c.dtype != torch.long or target_c.dtype != torch.long:
        raise ValueError("concept IDs must be int64")
    if edges.ndim != 2 or edges.size(0) != edges.size(1):
        raise ValueError("edge weights must be square")
    if edges.dtype != events.dtype or any(
        (value.device != events.device for value in (hist_c, hist_r, target_c, edges))
    ):
        raise ValueError("edge dtype and all devices must agree")
    if not isinstance(cell, nn.GRUCell) or (cell.input_size, cell.hidden_size) != (
        rank,
        rank,
    ):
        raise ValueError("the recurrent engine must be a matching PyTorch GRUCell")
    nodes = edges.size(0)
    if nodes == 0:
        return torch.zeros_like(events)
    known_source = hist_c.ge(2) & hist_c.lt(nodes + 2)
    valid = known_source & (hist_r.eq(0) | hist_r.eq(1))
    known_target = target_c.ge(2) & target_c.lt(nodes + 2)
    source_index = (hist_c - 2).clamp(0, nodes - 1)
    target_index = (target_c - 2).clamp(0, nodes - 1)
    if checkpoint_steps and torch.is_grad_enabled():
        return checkpointed_graph_rollout(
            events,
            source_index,
            target_index,
            valid,
            known_target,
            edges,
            cell,
            state_projection,
            checkpoint_steps,
        )
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
        reads.append(
            torch.where(known_target[:, position, None], value, torch.zeros_like(value))
        )
    return torch.stack(reads, dim=1)


class DirectedConceptGraph(nn.Module):
    def __init__(self, d_model, num_heads, checkpoint_steps=0):
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError("graph state width must equal one attention head")
        self.d_model = int(d_model)
        self.rank = int(d_model // num_heads)
        if type(checkpoint_steps) is not int or checkpoint_steps < 0:
            raise ValueError("checkpoint_steps must be a nonnegative integer")
        self.checkpoint_steps = checkpoint_steps
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
        return neighbors + diagonal.to(dtype=scores.dtype)

    def forward(self, history, hist_c, hist_r, target_c, concept_table):
        if history.ndim != 3 or history.size(-1) != self.d_model:
            raise ValueError("history embedding width changed")
        events = self.event_projection(F.layer_norm(history, (self.d_model,)))
        reads = graph_state_rollout(
            events,
            hist_c,
            hist_r,
            target_c,
            self.edge_weights(concept_table),
            self.cell,
            self.state_projection,
            self.checkpoint_steps if self.training else 0,
        )
        return self.output(reads)
