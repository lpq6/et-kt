"""Shared v45 kernel: K_c(gap) = exp(-lambda0 * (1 - sigmoid(prior_c)) * gap)."""

import math

import torch


def validate_lambda0(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("umk_lambda0 must be finite and positive")
    return value


def concept_rate(concept_prior, lambda0=0.3):
    if concept_prior is None or concept_prior.ndim != 2 or not concept_prior.is_floating_point():
        raise ValueError("UMK requires floating concept_prior [B,T]")
    # Corrected direction: a larger prior (better mastery) gives slower decay.
    return validate_lambda0(lambda0) * (1.0 - torch.sigmoid(concept_prior))


def concept_last_seen_gaps(target_c):
    """Delta[i,j] = i - max{k < i: c[k] == c[j]}, with a separate seen mask."""
    if target_c.ndim != 2 or min(target_c.shape) < 1 or target_c.dtype != torch.long:
        raise ValueError("UMK concepts must be nonempty int64 [B,T]")
    length = target_c.size(1)
    positions = torch.arange(length, device=target_c.device)
    same = target_c.unsqueeze(2).eq(target_c.unsqueeze(1)) & target_c.gt(0).unsqueeze(2)
    # Prefix scan is O(B*T*T); shifting excludes the current target occurrence.
    observed = torch.where(same, positions.view(1, length, 1) + 1, 0)
    inclusive_last = observed.cummax(dim=1).values
    last = torch.cat([torch.zeros_like(inclusive_last[:, :1]), inclusive_last[:, :-1]], 1)
    return positions.view(1, length, 1) + 1 - last, last.gt(0)


def umk_attention_bias(target_c, concept_prior, beta, lambda0=0.3):
    rate = concept_rate(concept_prior, lambda0)
    if rate.shape != target_c.shape or rate.device != target_c.device:
        raise ValueError("UMK attention prior and concepts must align")
    gap, seen = concept_last_seen_gaps(target_c)
    length = target_c.size(1)
    causal = torch.ones(length, length, dtype=torch.bool, device=target_c.device).tril()
    allowed = seen & causal
    # log K is evaluated directly, avoiding exp underflow followed by log(0).
    bias = -beta * rate.unsqueeze(1) * gap.to(rate.dtype)
    bias = bias.masked_fill(~allowed, -torch.inf)
    # An empty row has no historical evidence. A neutral self token is causal
    # (its response is already shifted) and avoids all--inf softmax/NaN.
    empty = ~allowed.any(dim=-1)
    diagonal = torch.eye(length, dtype=torch.bool, device=target_c.device)
    return torch.where(empty.unsqueeze(-1) & diagonal, torch.zeros_like(bias), bias)
