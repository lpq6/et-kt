import torch
import torch.nn.functional as F
from torch import nn

from work.a2g_mambakt_promoted_no_duplicate_folded_bias_20260715 import (
    A2GMambaKT as _FormalA2GMambaKT,
    FoldedZeroNormLinear,
    SelectiveSSMBlock,
)


class OneEpochConceptNoveltyDropout(nn.Module):
    """Protect each concept's first causal token during the first epoch."""

    def __init__(self, dropout):
        super().__init__()
        if not isinstance(dropout, nn.Dropout):
            raise TypeError(f"Expected nn.Dropout, got {type(dropout).__name__}")
        self.p = float(dropout.p)
        self.warmup_active = True
        self._protection_mask = None

    def set_protection_mask(self, mask):
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise TypeError("Protection mask must be a rank-2 boolean tensor")
        self._protection_mask = mask

    def clear_protection_mask(self):
        self._protection_mask = None

    def complete_warmup(self):
        self.warmup_active = False
        self.clear_protection_mask()

    def forward(self, inputs):
        dropped = F.dropout(inputs, p=self.p, training=self.training, inplace=False)
        if not self.training or not self.warmup_active:
            return dropped
        if self._protection_mask is None:
            raise RuntimeError("Concept-novelty mask was not set before input dropout")
        if tuple(self._protection_mask.shape) != tuple(inputs.shape[:2]):
            raise ValueError("Concept-novelty mask does not match the input sequence")
        return torch.where(self._protection_mask.unsqueeze(-1), inputs, dropped)


class A2GMambaKT(_FormalA2GMambaKT):
    """Formal core with one-epoch protection for first concept occurrences."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.input) != 4 or not isinstance(self.input[3], nn.Dropout):
            raise TypeError("Expected input path: LayerNorm, Linear, GELU, Dropout")
        self.input[3] = OneEpochConceptNoveltyDropout(self.input[3])

    @staticmethod
    def _concept_novelty_mask(target_c):
        sequence_length = target_c.size(1)
        strict_past = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=target_c.device,
                dtype=torch.bool,
            ),
            diagonal=-1,
        )
        same_concept = target_c.unsqueeze(2).eq(target_c.unsqueeze(1))
        seen_before = (same_concept & strict_past.unsqueeze(0)).any(dim=-1)
        return target_c.gt(0) & ~seen_before

    def complete_concept_novelty_warmup(self):
        self.input[3].complete_warmup()

    def train(self, mode=True):
        if not mode and hasattr(self, "input"):
            dropout = self.input[3]
            if isinstance(dropout, OneEpochConceptNoveltyDropout):
                dropout.complete_warmup()
        return super().train(mode)

    def forward(self, dcur, train=False, qtest=False):
        dropout = self.input[3]
        if self.training and dropout.warmup_active:
            _, target_c, _, _, _ = self._sequences(dcur)
            dropout.set_protection_mask(self._concept_novelty_mask(target_c))
        try:
            return super().forward(dcur, train=train, qtest=qtest)
        finally:
            dropout.clear_protection_mask()
