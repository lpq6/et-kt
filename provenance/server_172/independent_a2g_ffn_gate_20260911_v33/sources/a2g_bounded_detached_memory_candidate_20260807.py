"""Bounded detached recurrent-attention memory with segment-local features."""

import io
from dataclasses import dataclass

import torch


try:
    _FrozenA2G = A2GMambaKT
except NameError:
    from pykt.models.a2g_mambakt_final import A2GMambaKT as _FrozenA2G

try:
    _boundary_weights_segment = _boundary_weights_segment
    _selective_recurrence_segment = _selective_recurrence_segment
    _attention_stack_segment = _attention_stack_segment
except NameError:
    from a2g_detached_training_segment_core_20260807 import (
        _attention_stack_segment,
        _boundary_weights_segment,
        _selective_recurrence_segment,
    )


@dataclass
class A2GBoundedDetachedMemoryState:
    ssm_hidden: torch.Tensor
    attention_inputs: list[torch.Tensor | None]
    previous_target_c: torch.Tensor
    position: int
    memory_steps: int

    def state_dict(self):
        return {
            "schema_version": 1,
            "ssm_hidden": self.ssm_hidden.clone(),
            "attention_inputs": [
                None if value is None else value.clone()
                for value in self.attention_inputs
            ],
            "previous_target_c": self.previous_target_c.clone(),
            "position": int(self.position),
            "memory_steps": int(self.memory_steps),
        }

    @classmethod
    def from_state_dict(cls, payload):
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("unsupported bounded-memory state schema")
        return cls(
            ssm_hidden=payload["ssm_hidden"].clone(),
            attention_inputs=[
                None if value is None else value.clone()
                for value in payload["attention_inputs"]
            ],
            previous_target_c=payload["previous_target_c"].clone(),
            position=int(payload["position"]),
            memory_steps=int(payload["memory_steps"]),
        )


def _validate_bounded_state(model, state, batch_size, memory_steps):
    if not isinstance(state, A2GBoundedDetachedMemoryState):
        raise TypeError("state must be A2GBoundedDetachedMemoryState")
    if state.memory_steps != memory_steps:
        raise ValueError("memory_steps cannot change across segments")
    if state.position < 0:
        raise ValueError("state position must be non-negative")
    if state.ssm_hidden.shape != (batch_size, model.d_model):
        raise ValueError("SSM hidden-state shape mismatch")
    if state.previous_target_c.shape != (batch_size,):
        raise ValueError("previous_target_c shape mismatch")
    if len(state.attention_inputs) != len(model.blocks):
        raise ValueError("attention-state depth mismatch")
    expected_memory = min(state.position, memory_steps)
    for value in state.attention_inputs:
        if value is None:
            if expected_memory != 0:
                raise ValueError("nonzero position requires bounded attention memory")
            continue
        if value.shape != (batch_size, expected_memory, model.d_model):
            raise ValueError("bounded attention-memory shape mismatch")
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError("attention memory must be detached")
    if state.ssm_hidden.requires_grad or state.ssm_hidden.grad_fn is not None:
        raise ValueError("SSM hidden state must be detached")


def initialize_a2g_bounded_detached_memory_state(
    model, batch_size, memory_steps, device=None
):
    if isinstance(memory_steps, bool) or not isinstance(memory_steps, int):
        raise ValueError("memory_steps must be an integer")
    if memory_steps < 1:
        raise ValueError("memory_steps must be positive")
    if device is None:
        device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    return A2GBoundedDetachedMemoryState(
        ssm_hidden=torch.zeros(
            batch_size, model.d_model, device=device, dtype=dtype
        ),
        attention_inputs=[None for _ in model.blocks],
        previous_target_c=torch.zeros(
            batch_size, device=device, dtype=torch.long
        ),
        position=0,
        memory_steps=memory_steps,
    )


def a2g_bounded_detached_core_segment(
    model,
    token,
    target_c,
    *,
    state=None,
    memory_steps,
):
    if token.ndim != 3 or token.size(-1) != model.d_model:
        raise ValueError("token must have shape [B, L, D]")
    if token.size(1) < 1:
        raise ValueError("zero-length segments are unsupported")
    if target_c.shape != token.shape[:2]:
        raise ValueError("target_c must have shape [B, L]")
    if state is None:
        state = initialize_a2g_bounded_detached_memory_state(
            model, token.size(0), memory_steps, device=token.device
        )
    _validate_bounded_state(model, state, token.size(0), memory_steps)
    scope_weight = _boundary_weights_segment(model, token, target_c, state)
    recurrent, final_hidden = _selective_recurrence_segment(
        model, token, scope_weight, state.ssm_hidden
    )
    sequence, attention_inputs = _attention_stack_segment(
        model, token + recurrent, state.attention_inputs
    )
    attention_inputs = [
        value[:, -memory_steps:].detach() for value in attention_inputs
    ]
    next_state = A2GBoundedDetachedMemoryState(
        ssm_hidden=final_hidden.detach(),
        attention_inputs=attention_inputs,
        previous_target_c=target_c[:, -1].detach().clone(),
        position=state.position + token.size(1),
        memory_steps=memory_steps,
    )
    _validate_bounded_state(model, next_state, token.size(0), memory_steps)
    return sequence, next_state


class BoundedDetachedMemoryA2GMambaKT(_FrozenA2G):
    """Explicit segment API; the inherited ordinary forward remains unchanged."""

    def __init__(self, *args, cross_slice_memory_steps=200, **kwargs):
        self.cross_slice_memory_steps = int(cross_slice_memory_steps)
        if self.cross_slice_memory_steps < 1:
            raise ValueError("cross_slice_memory_steps must be positive")
        super().__init__(*args, **kwargs)

    def forward_segment(self, dcur, state=None, train=False, qtest=False):
        if not self.use_recency_weighted_concept_evidence:
            raise ValueError("segment prototype requires the frozen RWCE control")
        dropout = self.input[3]
        target_q, target_c, hist_q, hist_c, hist_r = self._sequences(dcur)
        if self.training and dropout.warmup_active:
            dropout.set_protection_mask(self._concept_novelty_mask(target_c))
        try:
            concept_prior = self.concept_prior(target_c.clamp_min(0))
            has_item_sequence = (
                self.n_pid > 0
                and target_q.numel() > 0
                and target_q.size(1) == target_c.size(1)
            )
            if has_item_sequence:
                item_prior = self.item_prior(target_q.clamp_min(0))
            else:
                item_prior = concept_prior.new_zeros(concept_prior.shape)
            stats = self._stats(
                target_c, hist_c, hist_r, item_prior, concept_prior
            )
            learned_stats = self._learned_stats(stats)
            learned_stats = self._replace_learned_exposure(
                learned_stats, target_c, hist_c, hist_r
            )

            target_concept = self.concept_emb(target_c.clamp_min(0))
            if has_item_sequence:
                target_item = self.item_emb(target_q.clamp_min(0))
                target_item = target_item * self._item_residual_reliability(
                    target_q
                ).unsqueeze(-1)
                target_item = self._apply_item_residual_dropout(
                    target_item, target_q
                )
            else:
                target_item = torch.zeros_like(target_concept)
            target = target_item + target_concept

            history_concept = self.hist_concept_emb(hist_c.clamp_min(0))
            has_history_items = (
                self.n_pid > 0
                and hist_q.numel() > 0
                and hist_q.size(1) == hist_c.size(1)
            )
            if has_history_items:
                history_item = self.hist_item_emb(hist_q.clamp_min(0))
                history_item = history_item * self._item_residual_reliability(
                    hist_q
                ).unsqueeze(-1)
                history_item = self._apply_item_residual_dropout(
                    history_item, hist_q
                )
            else:
                history_item = torch.zeros_like(history_concept)
            history = history_item + history_concept
            history = history + self.resp_emb(hist_r.clamp(0, 2))

            token = self.input(
                torch.cat(
                    [
                        target,
                        history,
                        target_item,
                        target_concept,
                        item_prior,
                        concept_prior,
                        learned_stats,
                    ],
                    dim=-1,
                )
            )
            sequence, state = a2g_bounded_detached_core_segment(
                self,
                token,
                target_c,
                state=state,
                memory_steps=self.cross_slice_memory_steps,
            )
            fused = torch.cat([sequence, learned_stats], dim=-1)
            prior_logit = (item_prior + concept_prior).squeeze(-1)
            logits = (
                self.pred(fused).squeeze(-1)
                + self.prior_scale * prior_logit
                + self.stat_scale * self._direct_stat_logit(stats)
            )
            prediction = torch.sigmoid(logits)
            if qtest and not train:
                return prediction, fused, state
            if not train:
                return prediction, state
            route = sequence.new_zeros(4)
            return prediction, sequence.new_tensor(0.0), route, state
        finally:
            dropout.clear_protection_mask()

    def cross_slice_contract(self):
        return {
            "candidate_id": "bounded_detached_recurrent_attention_memory_v1",
            "parameters_added": 0,
            "persistent_buffers_added": 0,
            "ordinary_forward_changed": False,
            "segment_api_explicit": True,
            "ssm_hidden_horizon": "learner_lifetime_detached",
            "attention_memory_steps": self.cross_slice_memory_steps,
            "statistics": "segment_local_frozen_control",
            "rwce": "segment_local_frozen_control",
            "novelty": "segment_local_frozen_control",
            "lineage": "Transformer-XL-style detached segment recurrence",
            "quality_claim_authorized": False,
        }


def serialize_a2g_bounded_detached_memory_state(state):
    buffer = io.BytesIO()
    torch.save(state.state_dict(), buffer)
    return buffer.getvalue()


def deserialize_a2g_bounded_detached_memory_state(serialized, *, map_location="cpu"):
    payload = torch.load(
        io.BytesIO(serialized), map_location=map_location, weights_only=True
    )
    return A2GBoundedDetachedMemoryState.from_state_dict(payload)


__all__ = [
    "A2GBoundedDetachedMemoryState",
    "BoundedDetachedMemoryA2GMambaKT",
    "a2g_bounded_detached_core_segment",
    "deserialize_a2g_bounded_detached_memory_state",
    "initialize_a2g_bounded_detached_memory_state",
    "serialize_a2g_bounded_detached_memory_state",
]
