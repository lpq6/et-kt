"""Causal evidence utilities; legacy lag semantics are preserved."""

import torch
from .core import OneEpochConceptNoveltyDropout
from .umk import concept_rate

EXPOSURE_LOCAL_INDEX = 2


class EvidenceMethods:
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

    @property
    def identity_support(self):
        return self.IDENTITY_EVIDENCE_MULTIPLIER * self.support_smoothing

    @torch.no_grad()
    def set_prior_support(self, item_count, concept_count=None):
        del concept_count
        if not self.use_evidence_equivalent_residual:
            return
        item_count = item_count.float().view(-1)
        count = min(self.item_support_count.numel(), item_count.numel())
        self.item_support_count.zero_()
        self.item_support_count[:count].copy_(item_count[:count])

    def _item_residual_reliability(self, item_ids):
        if not self.use_evidence_equivalent_residual:
            return torch.ones_like(item_ids, dtype=self.item_emb.weight.dtype)
        item_ids = item_ids.clamp(min=0, max=self.item_support_count.numel() - 1)
        support = self.item_support_count[item_ids]
        raw_reliability = support / (support + self.support_smoothing)
        identity_reliability = self.identity_support / (
            self.identity_support + self.support_smoothing
        )
        normalized = raw_reliability / identity_reliability
        return torch.where(
            support >= self.identity_support, torch.ones_like(normalized), normalized
        )

    @staticmethod
    def _recency_weighted_concept_evidence_components(
        target_c, hist_c, hist_r, concept_prior=None, use_umk_rwce=0, umk_lambda0=0.3,
    ):
        sequence_length = target_c.size(1)
        positions = torch.arange(sequence_length, device=target_c.device)
        aligned_past = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=target_c.device,
                dtype=torch.bool,
            ),
            diagonal=0,
        )
        same = target_c.unsqueeze(2).eq(hist_c.unsqueeze(1))
        same = same & aligned_past.unsqueeze(0)
        same = same & target_c.gt(0).unsqueeze(2)
        same = same & hist_c.gt(0).unsqueeze(1)
        valid_response = hist_r.ge(0) & hist_r.le(1)
        same = same & valid_response.unsqueeze(1)
        distance = (
            positions.view(1, sequence_length, 1)
            - positions.view(1, 1, sequence_length)
            + 1
        ).clamp_min(1)
        if use_umk_rwce:
            rate = concept_rate(concept_prior, umk_lambda0)
            if rate.shape != target_c.shape or rate.device != target_c.device:
                raise ValueError("UMK RWCE prior and concepts must align")
            # Shifted hist[:,j] is event j-1: K_c(distance-1) gives its
            # newest observation unit weight, using no current response.
            weight = torch.exp(-rate.unsqueeze(-1) * (distance - 1).to(rate.dtype))
        else:
            weight = torch.reciprocal(distance.to(dtype=torch.float32))
        weight = weight * same.to(dtype=weight.dtype)
        signed_response = hist_r.clamp(0, 1).to(dtype=weight.dtype)
        signed_response = signed_response.mul(2.0).sub(1.0)
        weighted_mass = weight.sum(dim=-1)
        signed_mass = (weight * signed_response.unsqueeze(1)).sum(dim=-1)
        evidence = signed_mass / (1.0 + weighted_mass)
        exposure = same.sum(dim=-1)
        return (evidence, weighted_mass, exposure)

    @classmethod
    def _recency_weighted_concept_evidence(
        cls, target_c, hist_c, hist_r, concept_prior=None, use_umk_rwce=0, umk_lambda0=0.3,
    ):
        (evidence, _, _) = cls._recency_weighted_concept_evidence_components(
            target_c, hist_c, hist_r, concept_prior, use_umk_rwce, umk_lambda0
        )
        return evidence

    @staticmethod
    def _transfer_calibrated_concept_residual(
        target_q, target_c, hist_q, hist_c, hist_r, hist_logit,
    ):
        """Aggregate aligned strict-past cross-item response residuals.

        ``hist_*[:, i]`` is the observed event immediately before target
        ``i``; therefore diagonal alignment is causal and must be included.
        A history event contributes only when it covers the same concept and a
        different known item. If item identifiers are unavailable for both
        sides, the pair falls back to concept-only transfer. Each past event is
        calibrated by its concept prior and downweighted by repeated use of its
        historical item, so repeated-item memorization cannot dominate the
        transfer signal.
        """
        if (
            target_q.ndim != 2
            or target_c.shape != target_q.shape
            or hist_q.shape != target_q.shape
            or hist_c.shape != target_q.shape
            or hist_r.shape != target_q.shape
            or hist_logit.shape != target_q.shape
            or hist_logit.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError("transfer residual inputs must be aligned [B,T] tensors")
        if target_q.device != target_c.device or hist_q.device != target_q.device:
            raise ValueError("transfer residual inputs must share a device")
        length = target_c.size(1)
        positions = torch.arange(length, device=target_c.device)
        aligned_past = torch.tril(
            torch.ones(
                length, length, device=target_c.device, dtype=torch.bool
            ),
            diagonal=0,
        )
        same_concept = target_c.unsqueeze(2).eq(hist_c.unsqueeze(1))
        target_item = target_q.unsqueeze(2)
        history_item = hist_q.unsqueeze(1)
        known_item_pair = target_item.gt(0) & history_item.gt(0)
        different_item = known_item_pair & target_item.ne(history_item)
        concept_only_pair = target_item.eq(0) & history_item.eq(0)
        valid = (
            target_c.gt(0).unsqueeze(2)
            & hist_c.gt(0).unsqueeze(1)
            & hist_r.ge(0).unsqueeze(1)
            & hist_r.le(1).unsqueeze(1)
        )
        transfer = aligned_past.unsqueeze(0) & same_concept & valid & (
            different_item | concept_only_pair
        )

        # A repeated historical item gets diminishing influence, independent
        # of the target position. This is a causal prefix count.
        same_history_item = hist_q.unsqueeze(2).eq(hist_q.unsqueeze(1))
        history_past = (
            torch.tril(
                torch.ones(
                    length, length, device=target_c.device, dtype=torch.bool
                ),
                diagonal=-1,
            ).unsqueeze(0)
            & same_history_item
            & hist_q.gt(0).unsqueeze(2)
        )
        repeat_count = history_past.sum(dim=-1).to(dtype=hist_logit.dtype)
        repeat_balance = torch.rsqrt(repeat_count + 1.0).unsqueeze(1)

        distance = (
            positions.view(1, length, 1)
            - positions.view(1, 1, length)
        ).clamp_min(1)
        weight = torch.reciprocal(distance.to(dtype=hist_logit.dtype))
        weight = weight * repeat_balance
        weight = weight * transfer.to(dtype=weight.dtype)
        residual = (
            hist_r.to(dtype=weight.dtype)
            - torch.sigmoid(hist_logit).to(dtype=weight.dtype)
        )
        mass = weight.sum(dim=-1)
        signed_mass = (weight * residual.unsqueeze(1)).sum(dim=-1)
        return signed_mass / (1.0 + mass)

    @staticmethod
    def _causal_transfer_gate_signal(
        target_q, target_c, hist_q, hist_c, hist_r, hist_logit,
    ):
        """Return a bounded, strict-past cross-item transfer signal.

        The signal separates two effects that are easy to conflate in a
        concept-level history: repeated attempts on the same item and
        transferable evidence from a different item.  A history response is
        first centered by its train-fold item/concept prior.  Only aligned
        strict-past events with the same concept and a different known item
        contribute.  The target-item novelty factor prevents this signal from
        overriding the dedicated same-item attempt path.  Sparse evidence is
        already shrunk by the ``1 + sum(weight)`` denominator, while
        contradictory evidence cancels through the signed residual sum; adding
        separate coverage/agreement gates would suppress the same signal twice.

        ``hist_*[:, i]`` is the event immediately before target ``i``.  Thus
        the diagonal of the pairwise mask is causal even though the pairwise
        indices are equal.
        """
        tensors = (target_q, target_c, hist_q, hist_c, hist_r, hist_logit)
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("causal transfer inputs must be rank-2 tensors")
        shape = target_c.shape
        if any(tensor.shape != shape for tensor in tensors):
            raise ValueError("causal transfer inputs must be aligned [B,T] tensors")
        if target_q.dtype != torch.long or target_c.dtype != torch.long:
            raise ValueError("causal transfer ids must be int64")
        if hist_q.dtype != torch.long or hist_c.dtype != torch.long:
            raise ValueError("causal transfer history ids must be int64")
        if hist_r.dtype != torch.long or hist_logit.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("causal transfer responses/logits have invalid dtype")
        if any(tensor.device != target_c.device for tensor in tensors):
            raise ValueError("causal transfer inputs must share a device")

        length = target_c.size(1)
        positions = torch.arange(length, device=target_c.device)
        aligned_past = torch.tril(
            torch.ones(length, length, device=target_c.device, dtype=torch.bool),
            diagonal=0,
        )
        valid_history = hist_r.ge(0) & hist_r.le(1)
        valid_target = target_c.gt(0)
        valid_history = valid_history & hist_c.gt(0)
        same_concept = target_c.unsqueeze(2).eq(hist_c.unsqueeze(1))
        known_pair = target_q.unsqueeze(2).gt(0) & hist_q.unsqueeze(1).gt(0)
        different_item = known_pair & target_q.unsqueeze(2).ne(hist_q.unsqueeze(1))
        concept_only = target_q.unsqueeze(2).eq(0) & hist_q.unsqueeze(1).eq(0)
        eligible = (
            aligned_past.unsqueeze(0)
            & valid_target.unsqueeze(2)
            & valid_history.unsqueeze(1)
            & same_concept
            & (different_item | concept_only)
        )

        # A repeated history item receives diminishing weight according to
        # occurrences strictly before that history event.
        same_history_item = hist_q.unsqueeze(2).eq(hist_q.unsqueeze(1))
        history_prefix = torch.tril(
            torch.ones(length, length, device=target_c.device, dtype=torch.bool),
            diagonal=-1,
        ).unsqueeze(0)
        repeat_count = (
            history_prefix
            & same_history_item
            & hist_q.gt(0).unsqueeze(2)
        ).sum(dim=-1)
        repeat_balance = torch.rsqrt(
            repeat_count.to(dtype=hist_logit.dtype) + 1.0
        ).unsqueeze(1)

        distance = (
            positions.view(1, length, 1)
            - positions.view(1, 1, length)
        ).clamp_min(1)
        weight = torch.reciprocal(distance.to(dtype=hist_logit.dtype))
        weight = weight * repeat_balance
        weight = weight * eligible.to(dtype=weight.dtype)
        residual = hist_r.to(dtype=weight.dtype) - torch.sigmoid(hist_logit)
        mass = weight.sum(dim=-1)
        signed_mass = (weight * residual.unsqueeze(1)).sum(dim=-1)
        raw = signed_mass / (1.0 + mass)

        # The target novelty term is deliberately based only on strict-past
        # same-item events, so the transfer path complements rather than
        # duplicates the same-item attempt readout.
        same_target_item = (
            aligned_past.unsqueeze(0)
            & target_q.unsqueeze(2).gt(0)
            & hist_q.unsqueeze(1).gt(0)
            & target_q.unsqueeze(2).eq(hist_q.unsqueeze(1))
            & valid_history.unsqueeze(1)
        )
        target_item_count = same_target_item.sum(dim=-1).to(
            dtype=hist_logit.dtype
        )
        novelty = torch.rsqrt(target_item_count + 1.0)
        # The evidence-mass denominator is the only confidence calibration:
        # it shrinks sparse prefixes, and signed residuals cancel conflicts.
        # Keeping this transport signal unsplit avoids a second heuristic
        # attenuation before the single learnable scalar gate.
        return raw * novelty

    def _replace_learned_exposure(
        self, learned_stats, target_c, hist_c, hist_r, concept_prior=None,
    ):
        if not self.use_recency_weighted_concept_evidence:
            return learned_stats
        replaced = learned_stats.clone()
        evidence = self._recency_weighted_concept_evidence(
            target_c, hist_c, hist_r, concept_prior,
            getattr(self, "use_umk_rwce", False), getattr(self, "umk_lambda0", 0.3),
        )
        replaced[..., EXPOSURE_LOCAL_INDEX] = evidence.to(dtype=replaced.dtype)
        return replaced

    def _scheduled_dropout(self):
        if self.item_residual_dropout_schedule == "constant":
            return self.item_residual_dropout
        if self.item_residual_dropout_schedule == "warmup":
            return self.item_residual_dropout if self.training else 0.0
        raise ValueError(
            f"Unknown item_residual_dropout_schedule: {self.item_residual_dropout_schedule!r}"
        )

    def _apply_item_residual_dropout(self, residual, item_ids):
        rate = self._scheduled_dropout()
        if not self.training or rate <= 0.0:
            return residual
        keep = torch.rand_like(item_ids.float()) >= rate
        if not keep.any():
            keep = torch.ones_like(keep, dtype=torch.bool)
        return residual * keep.unsqueeze(-1).to(residual.dtype)
