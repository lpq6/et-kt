"""A causal pre/post-event decoder with training-fold empirical evidence."""

from __future__ import annotations

import math

import torch
from torch import nn


class EventCausalBlock(nn.Module):
    def __init__(self, width, hidden_width, heads, dropout):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("rotary attention requires an even head width")
        self.heads = int(heads)
        self.head_width = int(width // heads)
        self.attention_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, width),
            nn.Dropout(dropout),
        )
        self.register_buffer(
            "inverse_frequency",
            10000.0 ** (-torch.arange(0, self.head_width, 2).float() / self.head_width),
            persistent=False,
        )
        self.capture_attention = False
        self.last_attention = None

    def _heads(self, values):
        return values.view(
            values.size(0), values.size(1), self.heads, self.head_width
        ).transpose(1, 2)

    def _rotate(self, values, event_positions):
        angles = event_positions.to(values.dtype).unsqueeze(-1) * self.inverse_frequency
        cosine = angles.cos()[None, None]
        sine = angles.sin()[None, None]
        even, odd = values[..., 0::2], values[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)

    def forward(self, sequence, valid, event_positions):
        normalized = self.attention_norm(sequence)
        queries = self._rotate(self._heads(self.query(normalized)), event_positions)
        keys = self._rotate(self._heads(self.key(normalized)), event_positions)
        values = self._heads(self.value(normalized))
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_width)
        length = sequence.size(1)
        allowed = torch.ones(length, length, dtype=torch.bool, device=sequence.device).tril()
        allowed = allowed[None, None] & valid[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).masked_fill(~allowed, 0.0)
        if self.capture_attention:
            self.last_attention = weights.detach()
        else:
            self.last_attention = None
        attended = torch.matmul(self.attention_dropout(weights), values)
        attended = attended.transpose(1, 2).contiguous().flatten(-2)
        sequence = sequence + self.residual_dropout(self.output(attended))
        sequence = sequence + self.ffn(self.ffn_norm(sequence))
        return sequence.masked_fill(~valid.unsqueeze(-1), 0.0)


class A2GMambaKT(nn.Module):
    CANDIDATE_ID = "support_calibrated_paired_event_decoder_20260908_v8"

    def __init__(
        self,
        n_question,
        n_pid,
        d_model=256,
        d_ff=512,
        n_blocks=4,
        num_attn_heads=8,
        dropout=0.2,
        seq_len=200,
        prior_scale=0.08,
        item_residual_dropout=0.4,
        item_residual_dropout_schedule="constant",
        use_alternating_events=1,
        use_evidence=1,
    ):
        super().__init__()
        if n_question <= 0 or n_pid < 0 or n_blocks < 2 or seq_len < 1:
            raise ValueError("invalid vocabulary, depth, or sequence length")
        if item_residual_dropout_schedule != "constant":
            raise ValueError("only the frozen constant dropout schedule is supported")
        if not 0.0 <= item_residual_dropout <= 1.0:
            raise ValueError("invalid item dropout")
        self.model_name = "support_calibrated_paired_event_decoder"
        self.n_question, self.n_pid = int(n_question), int(n_pid)
        self.d_model, self.seq_len = int(d_model), int(seq_len)
        self.prior_scale = float(prior_scale)
        self.item_residual_dropout = float(item_residual_dropout)
        self.use_alternating_events = bool(use_alternating_events)
        self.use_evidence = bool(use_evidence)
        self.item_emb = nn.Embedding(self.n_pid + 1, d_model, padding_idx=0)
        self.concept_emb = nn.Embedding(self.n_question + 1, d_model, padding_idx=0)
        self.hist_item_emb = nn.Embedding(self.n_pid + 1, d_model, padding_idx=0)
        self.hist_concept_emb = nn.Embedding(self.n_question + 1, d_model, padding_idx=0)
        self.resp_emb = nn.Embedding(3, d_model, padding_idx=2)
        self.item_prior = nn.Embedding(self.n_pid + 1, 1, padding_idx=0)
        self.concept_prior = nn.Embedding(self.n_question + 1, 1, padding_idx=0)
        self.register_buffer("item_support_count", torch.zeros(self.n_pid + 1))
        self.register_buffer("concept_support_count", torch.zeros(self.n_question + 1))
        self.pre_event = nn.Sequential(
            nn.LayerNorm(2 * d_model + 2),
            nn.Linear(2 * d_model + 2, d_model),
            nn.GELU(),
        )
        self.post_event = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
        )
        self.phase_emb = nn.Embedding(2, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            EventCausalBlock(d_model, d_ff, num_attn_heads, dropout)
            for _ in range(n_blocks)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.pred = nn.Sequential(
            nn.LayerNorm(3 * d_model + 2),
            nn.Linear(3 * d_model + 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(128, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(max(128, d_model // 2), 1),
        )
        nn.init.zeros_(self.item_prior.weight)
        nn.init.zeros_(self.concept_prior.weight)
        nn.init.normal_(self.phase_emb.weight, std=0.02)

    @torch.no_grad()
    def set_prior_logits(self, item_logits, concept_logits):
        for embedding, values in (
            (self.item_prior, item_logits), (self.concept_prior, concept_logits)
        ):
            embedding.weight.zero_()
            width = min(embedding.num_embeddings, values.numel())
            embedding.weight[:width, 0].copy_(values[:width].to(embedding.weight))
            embedding.weight[0].zero_()

    @torch.no_grad()
    def set_prior_support(self, item_count, concept_count):
        for counts, values in (
            (self.item_support_count, item_count),
            (self.concept_support_count, concept_count),
        ):
            counts.zero_()
            width = min(counts.numel(), values.numel())
            counts[:width].copy_(values[:width].to(counts))
            counts[0] = 0.0

    def _item_residual(self, embedding, identifiers):
        residual = embedding(identifiers)
        if self.use_evidence:
            support = self.item_support_count[identifiers]
            reliability = (1.5 * support / (support + 24.0)).clamp_max(1.0)
            residual = residual * reliability.unsqueeze(-1)
        if self.training and self.item_residual_dropout > 0.0:
            keep = torch.rand_like(identifiers.float()) >= self.item_residual_dropout
            if not keep.any():
                keep = torch.ones_like(keep)
            residual = residual * keep.unsqueeze(-1)
        return residual

    def event_inputs(self, dcur):
        concepts = torch.cat(
            (dcur["cseqs"][:, :1], dcur["shft_cseqs"]), dim=1
        ).long()
        if concepts.size(1) > self.seq_len:
            raise ValueError("event count exceeds the frozen observation window")
        if self.n_pid > 0 and dcur["qseqs"].numel():
            questions = torch.cat(
                (dcur["qseqs"][:, :1], dcur["shft_qseqs"]), dim=1
            ).long()
        else:
            questions = torch.zeros_like(concepts)
        questions, concepts = questions.clamp_min(0), concepts.clamp_min(0)
        valid = concepts.gt(0) | questions.gt(0)
        responses = dcur["rseqs"].long()
        if responses.size(1) != concepts.size(1) - 1:
            raise ValueError("unshifted responses must precede the final query")
        post_valid = valid[:, :-1] & responses.ge(0) & responses.le(1)
        response_ids = responses.masked_fill(~post_valid, 2)
        target_concept = self.concept_emb(concepts)
        history_concept = self.hist_concept_emb(concepts[:, :-1])
        if self.n_pid:
            target_item = self._item_residual(self.item_emb, questions)
            history_item = self._item_residual(self.hist_item_emb, questions[:, :-1])
            item_prior = self.item_prior(questions)
        else:
            target_item = torch.zeros_like(target_concept)
            history_item = torch.zeros_like(history_concept)
            item_prior = torch.zeros_like(concepts, dtype=target_concept.dtype).unsqueeze(-1)
        concept_prior = self.concept_prior(concepts)
        if not self.use_evidence:
            item_prior, concept_prior = torch.zeros_like(item_prior), torch.zeros_like(concept_prior)
        pre = self.pre_event(
            torch.cat((target_item, target_concept, item_prior, concept_prior), dim=-1)
        ) + self.phase_emb.weight[0]
        post = self.post_event(
            torch.cat((history_item, history_concept, self.resp_emb(response_ids)), dim=-1)
        ) + self.phase_emb.weight[1]
        pre = pre.masked_fill(~valid.unsqueeze(-1), 0.0)
        post = post.masked_fill(~post_valid.unsqueeze(-1), 0.0)
        readout = (target_item, target_concept, item_prior, concept_prior)
        return pre, post, valid, post_valid, readout

    def forward(self, dcur, train=False, qtest=False):
        pre, post, valid, post_valid, readout = self.event_inputs(dcur)
        event_count = pre.size(1)
        if self.use_alternating_events:
            sequence = torch.cat(
                (torch.stack((pre[:, :-1], post), dim=2).flatten(1, 2), pre[:, -1:]),
                dim=1,
            )
            token_valid = torch.cat(
                (torch.stack((valid[:, :-1], post_valid), dim=2).flatten(1, 2), valid[:, -1:]),
                dim=1,
            )
            positions = torch.arange(sequence.size(1), device=pre.device) // 2
        else:
            previous_post = torch.cat((torch.zeros_like(pre[:, :1]), post), dim=1)
            sequence, token_valid = pre + previous_post, valid
            positions = torch.arange(event_count, device=pre.device)
        sequence = self.input_dropout(sequence)
        for block in self.blocks:
            sequence = block(sequence, token_valid, positions)
        state = sequence[:, ::2] if self.use_alternating_events else sequence
        state = self.final_norm(state)
        target_item, target_concept, item_prior, concept_prior = readout
        fused = torch.cat(
            (state, target_item, target_concept, item_prior, concept_prior), dim=-1
        )
        logits = self.pred(fused).squeeze(-1)
        logits = logits + self.prior_scale * (item_prior + concept_prior).squeeze(-1)
        prediction = logits.sigmoid()
        if qtest and not train:
            return prediction, fused
        if train:
            return prediction, prediction.new_zeros(()), prediction.new_zeros(4)
        return prediction


__all__ = ["A2GMambaKT", "EventCausalBlock"]
