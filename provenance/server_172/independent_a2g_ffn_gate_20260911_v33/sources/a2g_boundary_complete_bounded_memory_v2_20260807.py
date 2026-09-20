"""Boundary-complete bounded detached memory for sliced A2G sequences."""

from dataclasses import dataclass
import io

import torch


try:
    _V1Candidate = BoundedDetachedMemoryA2GMambaKT
except NameError:
    from a2g_bounded_detached_memory_candidate_20260807 import (
        BoundedDetachedMemoryA2GMambaKT as _V1Candidate,
    )

try:
    _attention_stack_segment = _attention_stack_segment
    _boundary_weights_segment = _boundary_weights_segment
    _selective_recurrence_segment = _selective_recurrence_segment
except NameError:
    from a2g_detached_training_segment_core_20260807 import (
        _attention_stack_segment,
        _boundary_weights_segment,
        _selective_recurrence_segment,
    )


@dataclass
class A2GBoundaryCompleteMemoryState:
    ssm_hidden: torch.Tensor
    attention_inputs: list[torch.Tensor | None]
    previous_target_q: torch.Tensor
    previous_target_c: torch.Tensor
    previous_response: torch.Tensor
    position: int
    memory_steps: int

    def state_dict(self):
        return {
            "schema_version": 2,
            "ssm_hidden": self.ssm_hidden.clone(),
            "attention_inputs": [
                None if value is None else value.clone()
                for value in self.attention_inputs
            ],
            "previous_target_q": self.previous_target_q.clone(),
            "previous_target_c": self.previous_target_c.clone(),
            "previous_response": self.previous_response.clone(),
            "position": int(self.position),
            "memory_steps": int(self.memory_steps),
        }

    @classmethod
    def from_state_dict(cls, payload):
        if int(payload.get("schema_version", -1)) != 2:
            raise ValueError("unsupported boundary-complete state schema")
        return cls(
            ssm_hidden=payload["ssm_hidden"].clone(),
            attention_inputs=[
                None if value is None else value.clone()
                for value in payload["attention_inputs"]
            ],
            previous_target_q=payload["previous_target_q"].clone(),
            previous_target_c=payload["previous_target_c"].clone(),
            previous_response=payload["previous_response"].clone(),
            position=int(payload["position"]),
            memory_steps=int(payload["memory_steps"]),
        )


def _validate_boundary_complete_state(model, state, batch_size, memory_steps):
    if not isinstance(state, A2GBoundaryCompleteMemoryState):
        raise TypeError("state must be A2GBoundaryCompleteMemoryState")
    if state.memory_steps != memory_steps:
        raise ValueError("memory_steps cannot change across segments")
    if state.position < 0:
        raise ValueError("state position must be non-negative")
    if state.ssm_hidden.shape != (batch_size, model.d_model):
        raise ValueError("SSM hidden-state shape mismatch")
    for name in (
        "previous_target_q",
        "previous_target_c",
        "previous_response",
    ):
        if getattr(state, name).shape != (batch_size,):
            raise ValueError(f"{name} shape mismatch")
    if state.position == 0:
        if bool(state.previous_target_q.ne(0).any()):
            raise ValueError("initial previous_target_q must be zero")
        if bool(state.previous_target_c.ne(0).any()):
            raise ValueError("initial previous_target_c must be zero")
        if bool(state.previous_response.ne(2).any()):
            raise ValueError("initial previous_response must be sentinel 2")
    elif bool(((state.previous_response < 0) | (state.previous_response > 1)).any()):
        raise ValueError("committed previous_response must be binary")
    if model.n_pid <= 0 and bool(state.previous_target_q.ne(0).any()):
        raise ValueError("concept-only state cannot retain an item ID")
    if state.ssm_hidden.requires_grad or state.ssm_hidden.grad_fn is not None:
        raise ValueError("SSM hidden state must be detached")
    if len(state.attention_inputs) != len(model.blocks):
        raise ValueError("attention-state depth mismatch")
    expected_memory = min(state.position, memory_steps)
    for value in state.attention_inputs:
        if value is None:
            if expected_memory != 0:
                raise ValueError("nonzero position requires attention memory")
            continue
        if value.shape != (batch_size, expected_memory, model.d_model):
            raise ValueError("bounded attention-memory shape mismatch")
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError("attention memory must be detached")


def initialize_a2g_boundary_complete_memory_state(
    model, batch_size, memory_steps, device=None
):
    if isinstance(memory_steps, bool) or not isinstance(memory_steps, int):
        raise ValueError("memory_steps must be an integer")
    if memory_steps < 1:
        raise ValueError("memory_steps must be positive")
    if device is None:
        device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    state = A2GBoundaryCompleteMemoryState(
        ssm_hidden=torch.zeros(
            batch_size, model.d_model, device=device, dtype=dtype
        ),
        attention_inputs=[None for _ in model.blocks],
        previous_target_q=torch.zeros(
            batch_size, device=device, dtype=torch.long
        ),
        previous_target_c=torch.zeros(
            batch_size, device=device, dtype=torch.long
        ),
        previous_response=torch.full(
            (batch_size,), 2, device=device, dtype=torch.long
        ),
        position=0,
        memory_steps=memory_steps,
    )
    _validate_boundary_complete_state(model, state, batch_size, memory_steps)
    return state


def _boundary_complete_core(model, token, target_c, state):
    _validate_boundary_complete_state(
        model, state, token.size(0), model.cross_slice_memory_steps
    )
    scope_weight = _boundary_weights_segment(model, token, target_c, state)
    recurrent, final_hidden = _selective_recurrence_segment(
        model, token, scope_weight, state.ssm_hidden
    )
    sequence, attention_inputs = _attention_stack_segment(
        model, token + recurrent, state.attention_inputs
    )
    attention_inputs = [
        value[:, -model.cross_slice_memory_steps :].detach()
        for value in attention_inputs
    ]
    return sequence, final_hidden, attention_inputs


class BoundaryCompleteBoundedMemoryA2GMambaKT(_V1Candidate):
    """V2 segment API that carries the last observed interaction boundary."""

    def forward_segment(self, dcur, state=None, train=False, qtest=False):
        if not self.use_recency_weighted_concept_evidence:
            raise ValueError("segment prototype requires the frozen RWCE control")
        dropout = self.input[3]
        target_q, target_c, hist_q, hist_c, hist_r = self._sequences(dcur)
        if state is None:
            state = initialize_a2g_boundary_complete_memory_state(
                self,
                target_c.size(0),
                self.cross_slice_memory_steps,
                device=target_c.device,
            )
        _validate_boundary_complete_state(
            self, state, target_c.size(0), self.cross_slice_memory_steps
        )
        if state.position > 0:
            hist_q = hist_q.clone()
            hist_c = hist_c.clone()
            hist_r = hist_r.clone()
            hist_q[:, 0] = state.previous_target_q
            hist_c[:, 0] = state.previous_target_c
            hist_r[:, 0] = state.previous_response
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
            sequence, final_hidden, attention_inputs = _boundary_complete_core(
                self, token, target_c, state
            )
            shifted_response = dcur.get("shft_rseqs")
            if (
                not torch.is_tensor(shifted_response)
                or shifted_response.ndim != 2
                or shifted_response.size(0) != target_c.size(0)
                or shifted_response.size(1) != target_c.size(1) - 1
            ):
                raise ValueError("shft_rseqs must bind every predicted response")
            next_state = A2GBoundaryCompleteMemoryState(
                ssm_hidden=final_hidden.detach(),
                attention_inputs=attention_inputs,
                previous_target_q=target_q[:, -1].detach().clone(),
                previous_target_c=target_c[:, -1].detach().clone(),
                previous_response=(
                    shifted_response[:, -1].long().clamp(0, 1).detach().clone()
                ),
                position=state.position + token.size(1),
                memory_steps=self.cross_slice_memory_steps,
            )
            _validate_boundary_complete_state(
                self,
                next_state,
                target_c.size(0),
                self.cross_slice_memory_steps,
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
                return prediction, fused, next_state
            if not train:
                return prediction, next_state
            route = sequence.new_zeros(4)
            return prediction, sequence.new_tensor(0.0), route, next_state
        finally:
            dropout.clear_protection_mask()

    def cross_slice_contract(self):
        return {
            "candidate_id": "boundary_complete_bounded_detached_memory_v2",
            "supersedes": "bounded_detached_recurrent_attention_memory_v1",
            "supersession_reason": "v1 omits the last observed response at each slice boundary",
            "parameters_added": 0,
            "persistent_buffers_added": 0,
            "ordinary_forward_changed": False,
            "segment_api_explicit": True,
            "ssm_hidden_horizon": "learner_lifetime_detached",
            "attention_memory_steps": self.cross_slice_memory_steps,
            "boundary_history": "previous target item, concept, and observed response",
            "statistics": "segment_local_frozen_control with boundary history at first position",
            "rwce": "segment_local_frozen control",
            "novelty": "segment_local_frozen control",
            "lineage": "Transformer-XL-style detached segment recurrence",
            "quality_claim_authorized": False,
        }


def serialize_a2g_boundary_complete_memory_state(state):
    buffer = io.BytesIO()
    torch.save(state.state_dict(), buffer)
    return buffer.getvalue()


def deserialize_a2g_boundary_complete_memory_state(
    serialized, *, map_location="cpu"
):
    payload = torch.load(
        io.BytesIO(serialized), map_location=map_location, weights_only=True
    )
    return A2GBoundaryCompleteMemoryState.from_state_dict(payload)


__all__ = [
    "A2GBoundaryCompleteMemoryState",
    "BoundaryCompleteBoundedMemoryA2GMambaKT",
    "deserialize_a2g_boundary_complete_memory_state",
    "initialize_a2g_boundary_complete_memory_state",
    "serialize_a2g_boundary_complete_memory_state",
]
