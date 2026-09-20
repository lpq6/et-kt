"""Validation-only binary prediction metrics and strict admission checks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_metrics(labels, predictions):
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if not y.size or y.shape != p.shape:
        raise ValueError("labels and predictions must be nonempty and aligned")
    if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("AUC requires both binary classes")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("predictions must be finite probabilities")
    # The frozen server scorer clips before every metric, including AUC.
    p = np.clip(p, 1e-7, 1 - 1e-7)
    # [0, 1/15), ..., [14/15, 1], including p == 1.
    bin_id = np.minimum((p * 15).astype(np.int64), 14)
    ece = sum(
        abs(y[bin_id == i].mean() - p[bin_id == i].mean()) * (bin_id == i).mean()
        for i in range(15)
        if np.any(bin_id == i)
    )
    return {
        "auc": float(roc_auc_score(y, p)),
        "acc": float(((p >= 0.5) == y).mean()),
        "nll": float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
        "brier": float(np.square(p - y).mean()),
        "ece_15bin": float(ece),
        "interaction_count": int(y.size),
        "positive_count": int(y.sum()),
    }


def selection_key(metrics):
    """Higher is better; equal scores retain the earlier checkpoint."""
    return (
        metrics["auc"],
        -metrics["nll"],
        -metrics["brier"],
        -metrics["ece_15bin"],
    )


def exceeds_floor(auc, floor=0.8174):
    return bool(np.isfinite(auc) and auc > floor)
