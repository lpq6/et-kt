"""Extracted from the reviewed attempt_stage_candidate.py; see source_map.json."""

import torch
from torch import nn


def item_attempt_indices(target_q, hist_q, hist_r):
    """Return four exact counts using shifted history through column t."""
    if (
        target_q.ndim != 2
        or hist_q.shape != target_q.shape
        or hist_r.shape != target_q.shape
    ):
        raise ValueError(
            "item and response sequences must have identical batch/length shapes"
        )
    length = target_q.size(1)
    if length == 0:
        raise ValueError("at least one target is required")
    positions = torch.arange(length, device=target_q.device)
    valid_history = hist_q.ge(2) & (hist_r.eq(0) | hist_r.eq(1))
    available = positions.view(1, 1, -1) <= positions.view(1, -1, 1)
    same = (
        target_q.unsqueeze(-1).eq(hist_q.unsqueeze(1))
        & target_q.ge(2).unsqueeze(-1)
        & valid_history.unsqueeze(1)
        & available
    )
    attempts = same.sum(-1)
    successes = (same & hist_r.eq(1).unsqueeze(1)).sum(-1)
    adjacent = target_q.eq(hist_q) & target_q.ge(2) & valid_history
    failed = adjacent & hist_r.eq(0)

    def run_length(continues):
        breaks = torch.where(continues, torch.zeros_like(positions), positions)
        last_break = breaks.cummax(dim=1).values
        return torch.where(
            continues, positions - last_break, torch.zeros_like(last_break)
        )

    return (attempts, successes, run_length(adjacent), run_length(failed))


class ItemAttemptReadout(nn.Module):
    def __init__(self, d_model, num_heads, seq_len):
        super().__init__()
        if d_model % num_heads or seq_len < 2:
            raise ValueError(
                "readout rank must match one attention head, with length>=2"
            )
        rank = d_model // num_heads
        self.seq_len = int(seq_len)
        self.rank = rank
        self.attempt_embedding = nn.Embedding(seq_len, rank, padding_idx=0)
        self.success_embedding = nn.Embedding(seq_len, rank, padding_idx=0)
        self.run_embedding = nn.Embedding(seq_len, rank, padding_idx=0)
        self.failure_embedding = nn.Embedding(seq_len, rank, padding_idx=0)
        self.target_norm = nn.LayerNorm(d_model)
        self.sequence_norm = nn.LayerNorm(d_model)
        self.target_projection = nn.Linear(d_model, rank, bias=False)
        self.sequence_projection = nn.Linear(d_model, rank, bias=False)
        self.output = nn.Linear(3 * rank, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, sequence, target, target_q, hist_q, hist_r):
        if target_q.size(1) > self.seq_len:
            raise ValueError("sequence exceeds the frozen ordinal table width")
        indices = item_attempt_indices(target_q, hist_q, hist_r)
        state = (
            self.attempt_embedding(indices[0])
            + self.success_embedding(indices[1])
            + self.run_embedding(indices[2])
            + self.failure_embedding(indices[3])
        ) / 2.0
        query = torch.tanh(self.target_projection(self.target_norm(target)))
        learner = torch.tanh(self.sequence_projection(self.sequence_norm(sequence)))
        delta = self.output(
            torch.cat([state, state * query, state * learner], dim=-1)
        ).squeeze(-1)
        return delta * indices[0].gt(0).to(delta.dtype)
