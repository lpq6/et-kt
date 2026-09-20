import math

import torch
from torch import nn
import torch.nn.functional as F


class SelectiveSSMBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1, input_decay_scale=0.5):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 3)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.input_decay_scale = float(input_decay_scale)

    def forward(
        self,
        x,
        scope_weight=None,
    ):
        h = torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)
        outputs = []
        transition_rate = x.new_tensor(math.log(2.0))
        for index in range(x.size(1)):
            candidate, gate, delta = self.in_proj(x[:, index]).chunk(3, dim=-1)
            candidate = torch.tanh(candidate)
            gate = torch.sigmoid(gate)
            step = self.input_decay_scale * F.softplus(delta)
            retention = torch.exp(-transition_rate * step).clamp(1e-4, 1.0 - 1e-4)
            update = 1.0 - retention
            h = (1.0 - update) * h + update * candidate
            if scope_weight is not None:
                gate = gate * scope_weight[:, index]
            outputs.append(gate * h + (1.0 - gate) * x[:, index])
        return self.drop(self.norm(torch.stack(outputs, dim=1)))


class FoldedZeroNormLinear(nn.Module):
    """Reference-width LayerNorm and linear projection with zero channels folded."""

    def __init__(self, active_indices, reference_dim, out_features, eps=1e-5):
        super().__init__()
        active_indices = torch.as_tensor(active_indices, dtype=torch.long)
        self.register_buffer("active_indices", active_indices, persistent=False)
        self.reference_dim = int(reference_dim)
        self.eps = float(eps)
        active_dim = int(active_indices.numel())
        self.norm_weight = nn.Parameter(torch.ones(active_dim))
        self.norm_bias = nn.Parameter(torch.zeros(active_dim))
        self.active_weight = nn.Parameter(torch.empty(out_features, active_dim))
        self.zero_norm_weight = nn.Parameter(torch.empty(out_features))
        self.linear_bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        full_weight = self.active_weight.new_empty(self.active_weight.size(0), self.reference_dim)
        nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.active_weight.copy_(full_weight.index_select(1, self.active_indices))
            active = torch.zeros(self.reference_dim, dtype=torch.bool, device=full_weight.device)
            active[self.active_indices] = True
            self.zero_norm_weight.copy_(full_weight[:, ~active].sum(dim=1))
        bound = 1.0 / math.sqrt(self.reference_dim)
        nn.init.uniform_(self.linear_bias, -bound, bound)

    def forward(self, x):
        mean = x.sum(dim=-1, keepdim=True) / self.reference_dim
        variance = x.square().sum(dim=-1, keepdim=True) / self.reference_dim - mean.square()
        inv_std = torch.rsqrt(variance.clamp_min(0.0) + self.eps)
        normalized = (x - mean) * inv_std
        normalized = normalized * self.norm_weight + self.norm_bias
        zero_normalized = (-mean * inv_std).squeeze(-1)
        return (
            F.linear(normalized, self.active_weight, self.linear_bias)
            + zero_normalized.unsqueeze(-1) * self.zero_norm_weight
        )


class A2GMambaKT(nn.Module):
    """A2G core with the nonessential learned static decay removed."""

    def __init__(
        self,
        n_question,
        n_pid,
        d_model=160,
        d_ff=320,
        n_blocks=2,
        num_attn_heads=4,
        dropout=0.18,
        seq_len=200,
        emb_type="qid",
        emb_path="",
        prior_scale=0.08,
        stat_scale=0.12,
        input_decay_scale=0.5,
        boundary_local_floor=0.5,
        use_split_boundary=1,
        **kwargs,
    ):
        super().__init__()
        self.model_name = "a2g_mambakt"
        self.emb_type = emb_type
        self.n_question = int(n_question)
        self.n_pid = int(n_pid)
        self.d_model = int(d_model)
        self.prior_scale = float(prior_scale)
        self.stat_scale = float(stat_scale)
        self.boundary_local_floor = min(1.0, max(0.0, float(boundary_local_floor)))
        self.use_split_boundary = bool(int(use_split_boundary))

        self.item_emb = nn.Embedding(self.n_pid + 1, d_model, padding_idx=0)
        self.concept_emb = nn.Embedding(self.n_question + 1, d_model, padding_idx=0)
        self.hist_item_emb = nn.Embedding(self.n_pid + 1, d_model, padding_idx=0)
        self.hist_concept_emb = nn.Embedding(self.n_question + 1, d_model, padding_idx=0)
        self.resp_emb = nn.Embedding(3, d_model, padding_idx=2)

        # Preserve the validated seed stream without registering a position module.
        _position_initialization_compat = nn.Embedding(int(seq_len) + 2, d_model)
        del _position_initialization_compat

        self.item_prior = nn.Embedding(self.n_pid + 1, 1, padding_idx=0)
        self.concept_prior = nn.Embedding(self.n_question + 1, 1, padding_idx=0)
        self.input = nn.Sequential(
            nn.LayerNorm(d_model * 4 + 10),
            nn.Linear(d_model * 4 + 10, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ssm = SelectiveSSMBlock(d_model, dropout, input_decay_scale=input_decay_scale)
        blocks = []
        for block_index in range(int(n_blocks)):
            block = nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(d_model),
                    "attn": nn.MultiheadAttention(
                        d_model,
                        num_attn_heads,
                        dropout=dropout,
                        batch_first=True,
                    ),
                    "drop": nn.Dropout(dropout),
                    "ffn_norm": nn.LayerNorm(d_model),
                    "ffn": nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_ff, d_model),
                        nn.Dropout(dropout),
                    ),
                }
            )
            if block_index == int(n_blocks) - 1:
                block.pop("ffn_norm")
                block.pop("ffn")
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        reference_fused_dim = d_model * 4 + 8
        active_indices = list(range(d_model))
        active_indices.extend(range(d_model * 4, reference_fused_dim))
        pred_input = FoldedZeroNormLinear(active_indices, reference_fused_dim, d_ff)
        self.pred = nn.Sequential(
            pred_input,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(64, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(max(64, d_model // 2), 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.item_prior.weight)
        nn.init.zeros_(self.concept_prior.weight)

    @torch.no_grad()
    def set_prior_logits(self, item_logits, concept_logits):
        self.item_prior.weight.zero_()
        self.concept_prior.weight.zero_()
        item_count = min(self.item_prior.weight.size(0), item_logits.numel())
        concept_count = min(self.concept_prior.weight.size(0), concept_logits.numel())
        self.item_prior.weight[:item_count, 0].copy_(item_logits[:item_count])
        self.concept_prior.weight[:concept_count, 0].copy_(concept_logits[:concept_count])

    def _sequences(self, dcur):
        q = dcur["qseqs"].long()
        c = dcur["cseqs"].long()
        r = dcur["rseqs"].long()
        target_c = torch.cat([c[:, 0:1], dcur["shft_cseqs"].long()], dim=1)
        hist_c = torch.cat([torch.zeros_like(c[:, 0:1]), c], dim=1)
        hist_r = torch.cat([torch.full_like(r[:, 0:1], 2), r.clamp(0, 1)], dim=1)
        if self.n_pid <= 0 or q.numel() == 0 or q.size(1) == 0:
            target_q = torch.zeros_like(target_c)
            hist_q = torch.zeros_like(hist_c)
        else:
            target_q = torch.cat([q[:, 0:1], dcur["shft_qseqs"].long()], dim=1)
            hist_q = torch.cat([torch.zeros_like(q[:, 0:1]), q], dim=1)
        return target_q, target_c, hist_q, hist_c, hist_r

    def _stats(self, target_c, hist_c, hist_r, item_prior, concept_prior):
        sequence_length = target_c.size(1)
        previous = torch.tril(
            torch.ones(sequence_length, sequence_length, device=target_c.device),
            diagonal=-1,
        ).unsqueeze(0)
        same = target_c.unsqueeze(2).eq(hist_c.unsqueeze(1)).float() * previous
        same = same * hist_c.gt(0).float().unsqueeze(1)
        correct = same * hist_r.eq(1).float().unsqueeze(1)
        wrong = same * hist_r.eq(0).float().unsqueeze(1)
        exposure = same.sum(-1)
        success = correct.sum(-1)
        failure = wrong.sum(-1)
        accuracy = success / exposure.clamp_min(1.0)
        positions = torch.arange(sequence_length, device=target_c.device)
        last_seen = (positions.view(1, 1, sequence_length) * same).max(-1).values
        gap = (positions.view(1, sequence_length) - last_seen).clamp_min(0).float()
        gap = gap / max(1.0, float(sequence_length))
        prior = (item_prior + concept_prior).squeeze(-1)
        return torch.stack(
            [
                torch.sigmoid(-prior),
                accuracy,
                torch.log1p(exposure) / math.log1p(max(2, sequence_length)),
                success / (success + failure + 1.0),
                failure / (success + failure + 1.0),
                gap,
                item_prior.squeeze(-1).tanh(),
                concept_prior.squeeze(-1).tanh(),
            ],
            dim=-1,
        )

    def _attend(self, x):
        sequence_length = x.size(1)
        future = torch.triu(
            torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            normalized = block["norm"](x)
            attended, _ = block["attn"](
                normalized,
                normalized,
                normalized,
                attn_mask=future,
                need_weights=False,
            )
            x = x + block["drop"](attended)
            if "ffn" in block:
                x = x + block["ffn"](block["ffn_norm"](x))
        return x

    def _boundary_weights(self, token, target_c):
        if not self.use_split_boundary:
            return None
        same_run = target_c[:, 1:].eq(target_c[:, :-1]) & target_c[:, 1:].gt(0)
        same_run = torch.cat(
            [torch.zeros_like(target_c[:, :1], dtype=torch.bool), same_run],
            dim=1,
        )
        split = self.d_model // 2
        global_weight = token.new_ones(token.size(0), token.size(1), split)
        local_width = self.d_model - split
        local_weight = self.boundary_local_floor + (
            1.0 - self.boundary_local_floor
        ) * same_run.float()
        local_weight = local_weight.unsqueeze(-1).expand(-1, -1, local_width)
        return torch.cat([global_weight, local_weight], dim=-1)

    def forward(self, dcur, train=False, qtest=False):
        target_q, target_c, hist_q, hist_c, hist_r = self._sequences(dcur)
        item_prior = self.item_prior(target_q.clamp_min(0))
        concept_prior = self.concept_prior(target_c.clamp_min(0))
        stats = self._stats(target_c, hist_c, hist_r, item_prior, concept_prior)

        target_item = self.item_emb(target_q.clamp_min(0))
        target_concept = self.concept_emb(target_c.clamp_min(0))
        target = target_item + target_concept
        history = self.hist_item_emb(hist_q.clamp_min(0))
        history = history + self.hist_concept_emb(hist_c.clamp_min(0))
        history = history + self.resp_emb(hist_r.clamp(0, 2))
        token = self.input(
            torch.cat(
                [target, history, target_item, target_concept, item_prior, concept_prior, stats],
                dim=-1,
            )
        )
        scope_weight = self._boundary_weights(token, target_c)
        state = self.ssm(
            token,
            scope_weight=scope_weight,
        )
        sequence = self._attend(token + state)
        fused = torch.cat([sequence, stats], dim=-1)
        stat_logit = (
            stats[..., 1]
            + stats[..., 2]
            + stats[..., 3]
            - stats[..., 0]
            - stats[..., 4]
            - stats[..., 5]
        )
        prior_logit = (item_prior + concept_prior).squeeze(-1)
        logits = (
            self.pred(fused).squeeze(-1)
            + self.prior_scale * prior_logit
            + self.stat_scale * stat_logit
        )
        prediction = torch.sigmoid(logits)
        if qtest and not train:
            return prediction, fused
        if not train:
            return prediction
        route = sequence.new_zeros(4)
        return prediction, sequence.new_tensor(0.0), route
