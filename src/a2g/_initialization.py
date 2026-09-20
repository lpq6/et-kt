"""Legacy seed stream and folded initialization; no model forward."""

import math
import torch
from torch import nn
from .modules.core import (
    SelectiveSSMBlock,
    FoldedZeroNormLinear,
)

LEARNED_STAT_INDICES = (0, 1, 2, 3, 4, 6)
REMOVED_STAT_INDICES = (5, 7)


class LegacyInitialization(nn.Module):
    history_stat_diagonal = -1

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
        use_shared_embeddings=0,
        use_umk_ssm=0,
        umk_lambda0=0.1,
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
        if bool(int(use_shared_embeddings)):
            # v46: weight tying —— target/history 双流复用同一词表
            # （跨域借鉴 Transformer 权重共享：同一概念空间在不同时间位置用同一向量）
            self.hist_item_emb = self.item_emb
            self.hist_concept_emb = self.concept_emb
        else:
            self.hist_item_emb = nn.Embedding(self.n_pid + 1, d_model, padding_idx=0)
            self.hist_concept_emb = nn.Embedding(
                self.n_question + 1, d_model, padding_idx=0
            )
        self.resp_emb = nn.Embedding(3, d_model, padding_idx=2)
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
        self.ssm = SelectiveSSMBlock(
            d_model, dropout, input_decay_scale=input_decay_scale,
            use_umk_ssm=use_umk_ssm, umk_lambda0=umk_lambda0,
        )
        blocks = []
        for block_index in range(int(n_blocks)):
            block = nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(d_model),
                    "attn": nn.MultiheadAttention(
                        d_model, num_attn_heads, dropout=dropout, batch_first=True
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
        self.concept_prior.weight[:concept_count, 0].copy_(
            concept_logits[:concept_count]
        )

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
        return (target_q, target_c, hist_q, hist_c, hist_r)

    def _stats(self, target_c, hist_c, hist_r, item_prior, concept_prior):
        sequence_length = target_c.size(1)
        previous = torch.tril(
            torch.ones(sequence_length, sequence_length, device=target_c.device),
            diagonal=self.history_stat_diagonal,
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

    def _boundary_weights(self, token, target_c):
        if not self.use_split_boundary:
            return None
        same_run = target_c[:, 1:].eq(target_c[:, :-1]) & target_c[:, 1:].gt(0)
        same_run = torch.cat(
            [torch.zeros_like(target_c[:, :1], dtype=torch.bool), same_run], dim=1
        )
        split = self.d_model // 2
        global_weight = token.new_ones(token.size(0), token.size(1), split)
        local_width = self.d_model - split
        local_weight = (
            self.boundary_local_floor
            + (1.0 - self.boundary_local_floor) * same_run.float()
        )
        local_weight = local_weight.unsqueeze(-1).expand(-1, -1, local_width)
        return torch.cat([global_weight, local_weight], dim=-1)
