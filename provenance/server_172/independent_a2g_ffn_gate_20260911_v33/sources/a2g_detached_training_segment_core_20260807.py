"""Detached cross-segment training core for A2G recurrence and attention."""

import io
import math
from dataclasses import dataclass

import torch


@dataclass
class A2GDetachedTrainingCoreState:
    ssm_hidden: torch.Tensor
    attention_inputs: list[torch.Tensor | None]
    previous_target_c: torch.Tensor
    position: int

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
        }

    @classmethod
    def from_state_dict(cls, payload):
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("unsupported detached training-core state schema")
        return cls(
            ssm_hidden=payload["ssm_hidden"].clone(),
            attention_inputs=[
                None if value is None else value.clone()
                for value in payload["attention_inputs"]
            ],
            previous_target_c=payload["previous_target_c"].clone(),
            position=int(payload["position"]),
        )


def _validate_state(model, state, batch_size):
    if not isinstance(state, A2GDetachedTrainingCoreState):
        raise TypeError("state must be A2GDetachedTrainingCoreState")
    if state.ssm_hidden.shape != (batch_size, model.d_model):
        raise ValueError("SSM hidden-state shape mismatch")
    if state.previous_target_c.shape != (batch_size,):
        raise ValueError("previous_target_c shape mismatch")
    if len(state.attention_inputs) != len(model.blocks):
        raise ValueError("attention state depth mismatch")
    if state.position < 0:
        raise ValueError("state position must be non-negative")
    for value in state.attention_inputs:
        if value is None:
            if state.position != 0:
                raise ValueError("nonzero state position requires attention history")
            continue
        if value.shape != (batch_size, state.position, model.d_model):
            raise ValueError("attention-history shape mismatch")
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError("cross-segment attention history must be detached")
    if state.ssm_hidden.requires_grad or state.ssm_hidden.grad_fn is not None:
        raise ValueError("cross-segment SSM state must be detached")


def initialize_a2g_detached_training_core_state(model, batch_size, device=None):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError("batch_size must be an integer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device is None:
        device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    return A2GDetachedTrainingCoreState(
        ssm_hidden=torch.zeros(
            batch_size, model.d_model, device=device, dtype=dtype
        ),
        attention_inputs=[None for _ in model.blocks],
        previous_target_c=torch.zeros(
            batch_size, device=device, dtype=torch.long
        ),
        position=0,
    )


def _boundary_weights_segment(model, token, target_c, state):
    if not model.use_split_boundary:
        return None
    if target_c.shape != token.shape[:2]:
        raise ValueError("target_c must have shape [B, L]")
    if state.position == 0:
        first_same = torch.zeros_like(target_c[:, :1], dtype=torch.bool)
    else:
        first_same = target_c[:, :1].eq(
            state.previous_target_c.unsqueeze(1)
        ) & target_c[:, :1].gt(0)
    later_same = target_c[:, 1:].eq(target_c[:, :-1]) & target_c[:, 1:].gt(0)
    same_run = torch.cat([first_same, later_same], dim=1)
    split = model.d_model // 2
    global_weight = token.new_ones(token.size(0), token.size(1), split)
    local_width = model.d_model - split
    local_weight = model.boundary_local_floor + (
        1.0 - model.boundary_local_floor
    ) * same_run.to(dtype=token.dtype)
    local_weight = local_weight.unsqueeze(-1).expand(-1, -1, local_width)
    return torch.cat([global_weight, local_weight], dim=-1)


def _selective_recurrence_segment(model, token, scope_weight, initial_hidden):
    hidden = initial_hidden
    outputs = []
    transition_rate = token.new_tensor(math.log(2.0))
    for index in range(token.size(1)):
        candidate, gate, delta = model.ssm.in_proj(token[:, index]).chunk(
            3, dim=-1
        )
        candidate = torch.tanh(candidate)
        gate = torch.sigmoid(gate)
        step = model.ssm.input_decay_scale * torch.nn.functional.softplus(delta)
        retention = torch.exp(-transition_rate * step).clamp(1e-4, 1.0 - 1e-4)
        update = 1.0 - retention
        hidden = (1.0 - update) * hidden + update * candidate
        if scope_weight is not None:
            gate = gate * scope_weight[:, index]
        outputs.append(
            gate * hidden + (1.0 - gate) * token[:, index]
        )
    raw_output = torch.stack(outputs, dim=1)
    return model.ssm.drop(model.ssm.norm(raw_output)), hidden


def _segment_attention_mask(current_length, past_length, device):
    current_future = torch.triu(
        torch.ones(
            current_length,
            current_length,
            device=device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    if past_length == 0:
        return current_future
    past_allowed = torch.zeros(
        current_length, past_length, device=device, dtype=torch.bool
    )
    return torch.cat([past_allowed, current_future], dim=1)


def _attention_stack_segment(model, x, prior_inputs):
    topology = getattr(model, "normalization_topology", "control")
    new_inputs = []
    for index, block in enumerate(model.blocks):
        if topology == "control":
            attention_input = block["norm"](x)
        else:
            attention_input = x
        past = prior_inputs[index]
        past_length = 0 if past is None else past.size(1)
        key_value = (
            attention_input
            if past is None
            else torch.cat([past, attention_input], dim=1)
        )
        mask = _segment_attention_mask(
            attention_input.size(1), past_length, attention_input.device
        )
        attended, _ = block["attn"](
            attention_input,
            key_value,
            key_value,
            attn_mask=mask,
            need_weights=False,
        )
        if topology == "control":
            x = x + block["drop"](attended)
            if "ffn" in block:
                x = x + block["ffn"](block["ffn_norm"](x))
        else:
            x = block["norm"](x + block["drop"](attended))
            if "ffn" in block:
                if topology == "full_postnorm":
                    x = block["ffn_norm"](x + block["ffn"](x))
                else:
                    x = x + block["ffn"](block["ffn_norm"](x))
        new_inputs.append(key_value.detach())
    return x, new_inputs


def a2g_detached_training_core_segment(
    model,
    token,
    target_c,
    *,
    state=None,
    carry_enabled=True,
    detach_boundary=True,
):
    """Run recurrence and attention with an explicit detached segment boundary."""

    if token.ndim != 3 or token.size(-1) != model.d_model:
        raise ValueError("token must have shape [B, L, D]")
    if token.size(1) < 1:
        raise ValueError("zero-length segments are unsupported")
    if target_c.shape != token.shape[:2]:
        raise ValueError("target_c must have shape [B, L]")
    if not carry_enabled:
        if state is not None:
            raise ValueError("disabled carry cannot consume a state")
        scope_weight = model._boundary_weights(token, target_c)
        return model._attend(
            token + model.ssm(token, scope_weight=scope_weight)
        ), None
    if detach_boundary is not True:
        raise ValueError("only explicit detached segment boundaries are supported")
    if state is None:
        state = initialize_a2g_detached_training_core_state(
            model, token.size(0), device=token.device
        )
    _validate_state(model, state, token.size(0))
    scope_weight = _boundary_weights_segment(model, token, target_c, state)
    recurrent, final_hidden = _selective_recurrence_segment(
        model, token, scope_weight, state.ssm_hidden
    )
    sequence, attention_inputs = _attention_stack_segment(
        model, token + recurrent, state.attention_inputs
    )
    next_state = A2GDetachedTrainingCoreState(
        ssm_hidden=final_hidden.detach(),
        attention_inputs=attention_inputs,
        previous_target_c=target_c[:, -1].detach().clone(),
        position=state.position + token.size(1),
    )
    _validate_state(model, next_state, token.size(0))
    return sequence, next_state


def serialize_a2g_detached_training_core_state(state):
    buffer = io.BytesIO()
    torch.save(state.state_dict(), buffer)
    return buffer.getvalue()


def deserialize_a2g_detached_training_core_state(serialized, *, map_location="cpu"):
    payload = torch.load(
        io.BytesIO(serialized), map_location=map_location, weights_only=True
    )
    return A2GDetachedTrainingCoreState.from_state_dict(payload)


__all__ = [
    "A2GDetachedTrainingCoreState",
    "a2g_detached_training_core_segment",
    "deserialize_a2g_detached_training_core_state",
    "initialize_a2g_detached_training_core_state",
    "serialize_a2g_detached_training_core_state",
]
