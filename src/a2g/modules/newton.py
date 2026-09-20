"""Extracted from the reviewed prequential_newton_candidate.py; see source_map.json."""

import math
import torch
from torch import nn
import torch.nn.functional as F


def prefix_newton_scores(base_logits, features, responses, observed):
    """One ridge Newton step from events strictly earlier than each target."""
    if base_logits.ndim != 2 or min(base_logits.shape) < 1:
        raise ValueError("logits must have a nonempty batch/length shape")
    (batch, length) = base_logits.shape
    if (
        features.ndim != 3
        or features.shape[:2] != (batch, length)
        or features.size(-1) < 1
    ):
        raise ValueError("features must align with the target sequence")
    if responses.shape != (batch, length - 1) or observed.shape != (batch, length):
        raise ValueError("past responses and event masks must align with targets")
    if observed.dtype != torch.bool:
        raise ValueError("observed event masks must be boolean")
    if not base_logits.is_floating_point() or features.dtype != base_logits.dtype:
        raise ValueError("logits and features must share a floating dtype")
    if any(
        (
            value.device != base_logits.device
            for value in (features, responses, observed)
        )
    ):
        raise ValueError("all inputs must share a device")
    rank = features.size(-1)
    inverse = torch.eye(rank, dtype=features.dtype, device=features.device)
    inverse = inverse.unsqueeze(0).expand(batch, rank, rank)
    score = features.new_zeros(batch, rank)
    probability = torch.sigmoid(base_logits)
    outputs = [base_logits.new_zeros(batch)]
    for target in range(1, length):
        past = target - 1
        valid = observed[:, past] & (
            responses[:, past].eq(0) | responses[:, past].eq(1)
        )
        feature = torch.where(
            valid.unsqueeze(-1), features[:, past], torch.zeros_like(score)
        )
        p = probability[:, past]
        weight = torch.where(valid, p * (1.0 - p), torch.zeros_like(p))
        innovation = torch.where(
            valid, responses[:, past].to(p.dtype) - p, torch.zeros_like(p)
        )
        vector = torch.bmm(inverse, feature.unsqueeze(-1)).squeeze(-1)
        denominator = 1.0 + weight * (feature * vector).sum(-1)
        inverse = inverse - (weight / denominator).view(batch, 1, 1) * vector.unsqueeze(
            -1
        ) * vector.unsqueeze(-2)
        score = score + innovation.unsqueeze(-1) * feature
        offset = torch.bmm(inverse, score.unsqueeze(-1)).squeeze(-1)
        value = (features[:, target] * offset).sum(-1)
        outputs.append(torch.where(observed[:, target], value, torch.zeros_like(value)))
    return torch.stack(outputs, dim=1)


class PrequentialNewtonReadout(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError("projection width must equal one attention head")
        self.d_model = int(d_model)
        self.rank = int(d_model // num_heads)
        self.projection = nn.Linear(self.d_model, self.rank, bias=False)
        self.output_scale = nn.Parameter(torch.zeros(()))

    def features(self, target):
        if target.ndim != 3 or target.size(-1) != self.d_model:
            raise ValueError("target embedding width changed")
        normalized = F.layer_norm(target, (self.d_model,))
        coordinates = torch.tanh(self.projection(normalized)) / math.sqrt(self.rank)
        return torch.cat([torch.ones_like(coordinates[..., :1]), coordinates], dim=-1)

    def forward(self, base_logits, target, target_c, responses):
        score = prefix_newton_scores(
            base_logits, self.features(target), responses, target_c.gt(0)
        )
        return self.output_scale * score
