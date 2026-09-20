"""UID-isolated evaluator mechanics for boundary-complete A2G memory v2."""

import copy
from dataclasses import dataclass
import io

import torch


try:
    _BoundaryState = A2GBoundaryCompleteMemoryState
    _initialize_boundary_state = initialize_a2g_boundary_complete_memory_state
except NameError:
    from a2g_boundary_complete_bounded_memory_v2_20260807 import (
        A2GBoundaryCompleteMemoryState as _BoundaryState,
        initialize_a2g_boundary_complete_memory_state as _initialize_boundary_state,
    )


class StatefulEvaluatorError(RuntimeError):
    """Base class for fail-closed evaluator errors."""


class StatefulEvaluatorScopeError(StatefulEvaluatorError):
    """Raised when a dataset-local UID crosses split or fold."""


class StatefulEvaluatorOrderError(StatefulEvaluatorError):
    """Raised for missing, duplicate, reordered, or completed segments."""


class StatefulEvaluatorTransactionError(StatefulEvaluatorError):
    """Raised for invalid batch transaction boundaries."""


class StatefulEvaluatorStateError(StatefulEvaluatorError):
    """Raised for malformed boundary-complete state."""


@dataclass(frozen=True, order=True)
class StatefulEvaluationKey:
    dataset_id: str
    split: str
    fold: int
    learner_uid: str

    def __post_init__(self):
        for name in ("dataset_id", "split", "learner_uid"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.fold, bool) or not isinstance(self.fold, int):
            raise ValueError("fold must be an integer")
        if self.fold < 0:
            raise ValueError("fold must be non-negative")

    @property
    def uid_claim(self):
        return self.dataset_id, self.learner_uid

    @property
    def scope(self):
        return self.split, self.fold

    def state_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "fold": self.fold,
            "learner_uid": self.learner_uid,
        }

    @classmethod
    def from_state_dict(cls, payload):
        return cls(
            dataset_id=payload["dataset_id"],
            split=payload["split"],
            fold=int(payload["fold"]),
            learner_uid=payload["learner_uid"],
        )


@dataclass
class _StoredEvaluationState:
    state: _BoundaryState
    next_segment_index: int


def _clone_state(state):
    if not isinstance(state, _BoundaryState):
        raise StatefulEvaluatorStateError(
            "state must be A2GBoundaryCompleteMemoryState"
        )
    return _BoundaryState.from_state_dict(state.state_dict())


def assemble_boundary_complete_states(states):
    states = tuple(states)
    if not states:
        raise StatefulEvaluatorStateError("cannot assemble an empty state batch")
    positions = {state.position for state in states}
    memory_steps = {state.memory_steps for state in states}
    depths = {len(state.attention_inputs) for state in states}
    if len(positions) != 1 or len(memory_steps) != 1 or len(depths) != 1:
        raise StatefulEvaluatorStateError(
            "one evaluator batch must share position, memory cap, and depth"
        )
    attention_inputs = []
    for layer in range(len(states[0].attention_inputs)):
        values = [state.attention_inputs[layer] for state in states]
        if all(value is None for value in values):
            attention_inputs.append(None)
        elif any(value is None for value in values):
            raise StatefulEvaluatorStateError(
                "one evaluator batch mixes empty and populated attention state"
            )
        else:
            attention_inputs.append(torch.cat(values, dim=0))
    return _BoundaryState(
        ssm_hidden=torch.cat([state.ssm_hidden for state in states], dim=0),
        attention_inputs=attention_inputs,
        previous_target_q=torch.cat(
            [state.previous_target_q for state in states], dim=0
        ),
        previous_target_c=torch.cat(
            [state.previous_target_c for state in states], dim=0
        ),
        previous_response=torch.cat(
            [state.previous_response for state in states], dim=0
        ),
        position=states[0].position,
        memory_steps=states[0].memory_steps,
    )


def split_boundary_complete_state(state):
    if not isinstance(state, _BoundaryState):
        raise StatefulEvaluatorStateError(
            "state must be A2GBoundaryCompleteMemoryState"
        )
    batch_size = state.ssm_hidden.size(0)
    fields = (
        state.previous_target_q,
        state.previous_target_c,
        state.previous_response,
    )
    if batch_size < 1 or any(value.shape != (batch_size,) for value in fields):
        raise StatefulEvaluatorStateError("batched boundary fields are malformed")
    results = []
    for index in range(batch_size):
        results.append(
            _BoundaryState(
                ssm_hidden=state.ssm_hidden[index : index + 1].detach().clone(),
                attention_inputs=[
                    None
                    if value is None
                    else value[index : index + 1].detach().clone()
                    for value in state.attention_inputs
                ],
                previous_target_q=state.previous_target_q[
                    index : index + 1
                ].detach().clone(),
                previous_target_c=state.previous_target_c[
                    index : index + 1
                ].detach().clone(),
                previous_response=state.previous_response[
                    index : index + 1
                ].detach().clone(),
                position=state.position,
                memory_steps=state.memory_steps,
            )
        )
    return tuple(results)


class UIDBoundaryCompleteEvaluationStore:
    """Atomic per-UID v2 states with terminal discard and no eviction."""

    SCHEMA_VERSION = 1

    def __init__(self, max_learners, memory_steps, sequence_width):
        for value, name in (
            (max_learners, "max_learners"),
            (memory_steps, "memory_steps"),
            (sequence_width, "sequence_width"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_learners = max_learners
        self.memory_steps = memory_steps
        self.sequence_width = sequence_width
        self._claims = {}
        self._entries = {}
        self._completed = set()
        self._inflight = None

    def __len__(self):
        return len(self._entries)

    @property
    def batch_in_flight(self):
        return self._inflight is not None

    @property
    def completed_keys(self):
        return tuple(sorted(self._completed))

    def _claim(self, key):
        prior = self._claims.get(key.uid_claim)
        if prior is not None and prior != key.scope:
            raise StatefulEvaluatorScopeError(
                "dataset-local UID cannot cross split or fold"
            )

    def begin_batch(self, requests, initializer):
        if self._inflight is not None:
            raise StatefulEvaluatorTransactionError(
                "an evaluation batch is already in flight"
            )
        requests = tuple(requests)
        if not requests:
            raise StatefulEvaluatorTransactionError("empty batch is forbidden")
        keys = [key for key, _ in requests]
        if len(keys) != len(set(keys)):
            raise StatefulEvaluatorTransactionError("batch repeats a learner")
        states = []
        staged_new = []
        for key, segment_index in requests:
            if not isinstance(key, StatefulEvaluationKey):
                raise TypeError("key must be StatefulEvaluationKey")
            if (
                isinstance(segment_index, bool)
                or not isinstance(segment_index, int)
                or segment_index < 0
            ):
                raise ValueError("segment_index must be non-negative")
            self._claim(key)
            if key in self._completed:
                raise StatefulEvaluatorOrderError(
                    "completed learner cannot re-enter evaluation"
                )
            entry = self._entries.get(key)
            if entry is None:
                if segment_index != 0:
                    raise StatefulEvaluatorOrderError(
                        "new learner must start at segment zero"
                    )
                if len(self._entries) + len(staged_new) >= self.max_learners:
                    raise StatefulEvaluatorStateError(
                        "capacity reached; implicit eviction is forbidden"
                    )
                state = initializer()
                if state.position != 0 or state.memory_steps != self.memory_steps:
                    raise StatefulEvaluatorStateError(
                        "initializer returned an incompatible state"
                    )
                staged_new.append((key, _StoredEvaluationState(_clone_state(state), 0)))
                states.append(_clone_state(state))
            else:
                if segment_index != entry.next_segment_index:
                    raise StatefulEvaluatorOrderError(
                        f"expected segment {entry.next_segment_index}, got {segment_index}"
                    )
                states.append(_clone_state(entry.state))
        for key, entry in staged_new:
            self._entries[key] = entry
            self._claims[key.uid_claim] = key.scope
        self._inflight = {
            "requests": requests,
            "new_keys": tuple(key for key, _ in staged_new),
        }
        return tuple(states)

    def commit_batch(self, resolutions):
        if self._inflight is None:
            raise StatefulEvaluatorTransactionError("no batch is in flight")
        resolutions = tuple(resolutions)
        requests = self._inflight["requests"]
        if tuple((key, segment) for key, segment, *_ in resolutions) != requests:
            raise StatefulEvaluatorTransactionError(
                "resolutions differ from checked-out requests"
            )
        validated = []
        for key, segment_index, steps_consumed, terminal, state in resolutions:
            entry = self._entries[key]
            if (
                isinstance(steps_consumed, bool)
                or not isinstance(steps_consumed, int)
                or steps_consumed < 2
                or steps_consumed > self.sequence_width
            ):
                raise StatefulEvaluatorStateError("invalid consumed sequence width")
            if not isinstance(terminal, bool):
                raise StatefulEvaluatorStateError("terminal must be boolean")
            if not isinstance(state, _BoundaryState):
                raise StatefulEvaluatorStateError("resolution state type changed")
            if state.memory_steps != self.memory_steps:
                raise StatefulEvaluatorStateError("memory cap changed")
            if state.position != entry.state.position + self.sequence_width:
                raise StatefulEvaluatorStateError(
                    "model state must advance by the fixed padded width"
                )
            validated.append(
                (key, segment_index, steps_consumed, terminal, _clone_state(state))
            )
        for key, segment_index, _steps, terminal, state in validated:
            if terminal:
                del self._entries[key]
                self._completed.add(key)
            else:
                self._entries[key] = _StoredEvaluationState(
                    state=state,
                    next_segment_index=segment_index + 1,
                )
        self._inflight = None

    def abort_batch(self):
        if self._inflight is None:
            raise StatefulEvaluatorTransactionError("no batch is in flight")
        for key in self._inflight["new_keys"]:
            del self._entries[key]
            self._claims.pop(key.uid_claim, None)
        self._inflight = None

    def peek_state(self, key):
        if self._inflight is not None:
            raise StatefulEvaluatorTransactionError(
                "cannot inspect state during an in-flight batch"
            )
        self._claim(key)
        if key not in self._entries:
            raise KeyError(key)
        return _clone_state(self._entries[key].state)

    def state_dict(self):
        if self._inflight is not None:
            raise StatefulEvaluatorTransactionError(
                "cannot serialize an in-flight evaluation batch"
            )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "max_learners": self.max_learners,
            "memory_steps": self.memory_steps,
            "sequence_width": self.sequence_width,
            "claims": [
                {
                    "dataset_id": dataset_id,
                    "learner_uid": learner_uid,
                    "split": scope[0],
                    "fold": scope[1],
                }
                for (dataset_id, learner_uid), scope in sorted(self._claims.items())
            ],
            "entries": [
                {
                    "key": key.state_dict(),
                    "next_segment_index": entry.next_segment_index,
                    "state": entry.state.state_dict(),
                }
                for key, entry in sorted(self._entries.items())
            ],
            "completed": [key.state_dict() for key in sorted(self._completed)],
        }

    @classmethod
    def from_state_dict(cls, payload):
        if int(payload.get("schema_version", -1)) != cls.SCHEMA_VERSION:
            raise StatefulEvaluatorStateError("unsupported evaluator-store schema")
        store = cls(
            int(payload["max_learners"]),
            int(payload["memory_steps"]),
            int(payload["sequence_width"]),
        )
        for record in payload.get("claims", []):
            claim = (record["dataset_id"], record["learner_uid"])
            if claim in store._claims:
                raise StatefulEvaluatorStateError("duplicate UID claim")
            store._claims[claim] = (record["split"], int(record["fold"]))
        for record in payload.get("entries", []):
            key = StatefulEvaluationKey.from_state_dict(record["key"])
            if key in store._entries:
                raise StatefulEvaluatorStateError("duplicate state entry")
            store._claim(key)
            state = _BoundaryState.from_state_dict(record["state"])
            store._entries[key] = _StoredEvaluationState(
                state=state,
                next_segment_index=int(record["next_segment_index"]),
            )
        store._completed = {
            StatefulEvaluationKey.from_state_dict(record)
            for record in payload.get("completed", [])
        }
        if store._completed.intersection(store._entries):
            raise StatefulEvaluatorStateError("completed and active keys overlap")
        if len(store._entries) > store.max_learners:
            raise StatefulEvaluatorStateError("serialized store exceeds capacity")
        return store


def _as_list(value, batch_size, transform):
    if torch.is_tensor(value):
        if value.ndim != 1 or value.numel() != batch_size:
            raise StatefulEvaluatorStateError("metadata tensor must have shape [B]")
        values = value.detach().cpu().tolist()
    else:
        values = list(value)
        if len(values) != batch_size:
            raise StatefulEvaluatorStateError("metadata list must contain B values")
    return [transform(item) for item in values]


def _prefix_mask(dcur, valid_lengths):
    required = {"cseqs", "shft_rseqs", "masks", "smasks"}
    if not required.issubset(dcur):
        raise StatefulEvaluatorStateError("evaluation batch lacks required tensors")
    batch_size, shifted_width = dcur["cseqs"].shape
    sequence_width = shifted_width + 1
    if sequence_width < 2:
        raise StatefulEvaluatorStateError("zero-prediction segment is forbidden")
    lengths = torch.tensor(
        valid_lengths, device=dcur["cseqs"].device, dtype=torch.long
    )
    if bool(((lengths < 2) | (lengths > sequence_width)).any()):
        raise StatefulEvaluatorStateError("valid length is outside segment width")
    positions = torch.arange(shifted_width, device=dcur["cseqs"].device)
    mask = positions.unsqueeze(0) < (lengths - 1).unsqueeze(1)
    if not torch.equal(dcur["masks"].bool(), mask):
        raise StatefulEvaluatorStateError("masks is not a contiguous valid prefix")
    if not torch.equal(dcur["smasks"].bool(), mask):
        raise StatefulEvaluatorStateError("smasks is not a contiguous valid prefix")
    if tuple(dcur["shft_rseqs"].shape) != (batch_size, shifted_width):
        raise StatefulEvaluatorStateError("shifted response shape changed")
    return sequence_width, mask


def evaluate_boundary_complete_uid_batch(model, dcur, state_meta, store):
    """Evaluate and atomically commit one depth-topological UID batch."""

    if not isinstance(store, UIDBoundaryCompleteEvaluationStore):
        raise TypeError("store must be UIDBoundaryCompleteEvaluationStore")
    if model.training:
        raise StatefulEvaluatorError("stateful evaluator requires model.eval()")
    batch_size = dcur["cseqs"].size(0)
    dataset_ids = _as_list(state_meta["dataset_id"], batch_size, str)
    splits = _as_list(state_meta["split"], batch_size, str)
    folds = _as_list(state_meta["fold"], batch_size, int)
    learner_uids = _as_list(state_meta["learner_uid"], batch_size, str)
    segment_indices = _as_list(state_meta["segment_index"], batch_size, int)
    valid_lengths = _as_list(state_meta["valid_length"], batch_size, int)
    dataset_indices = _as_list(state_meta["dataset_index"], batch_size, int)
    is_last = _as_list(state_meta["is_last"], batch_size, bool)
    if len(set(segment_indices)) != 1:
        raise StatefulEvaluatorOrderError("one batch must contain one depth")
    sequence_width, valid_mask = _prefix_mask(dcur, valid_lengths)
    if sequence_width != store.sequence_width:
        raise StatefulEvaluatorStateError("batch width differs from store contract")
    if any(
        length != sequence_width and not terminal
        for length, terminal in zip(valid_lengths, is_last)
    ):
        raise StatefulEvaluatorStateError("short nonterminal segment is forbidden")
    keys = tuple(
        StatefulEvaluationKey(dataset_id, split, fold, uid)
        for dataset_id, split, fold, uid in zip(
            dataset_ids, splits, folds, learner_uids
        )
    )
    requests = tuple(zip(keys, segment_indices))
    states = store.begin_batch(
        requests,
        lambda: _initialize_boundary_state(
            model,
            1,
            store.memory_steps,
            device=dcur["cseqs"].device,
        ),
    )
    try:
        batch_state = assemble_boundary_complete_states(states)
        with torch.inference_mode():
            prediction, next_state = model.forward_segment(dcur, state=batch_state)
        next_states = split_boundary_complete_state(next_state)
        resolutions = tuple(
            (
                key,
                segment_index,
                valid_length,
                terminal,
                state,
            )
            for key, segment_index, valid_length, terminal, state in zip(
                keys,
                segment_indices,
                valid_lengths,
                is_last,
                next_states,
            )
        )
        store.commit_batch(resolutions)
    except Exception:
        if store.batch_in_flight:
            store.abort_batch()
        raise
    selected_prediction = prediction[:, 1:][valid_mask].detach().cpu()
    selected_target = dcur["shft_rseqs"][valid_mask].detach().cpu()
    learner_keys = []
    provenance = []
    for row, (key, segment_index, dataset_index) in enumerate(
        zip(keys, segment_indices, dataset_indices)
    ):
        for prediction_index in torch.nonzero(
            valid_mask[row], as_tuple=False
        ).flatten().tolist():
            learner_keys.append(key)
            provenance.append(
                (
                    key.dataset_id,
                    key.split,
                    key.fold,
                    key.learner_uid,
                    segment_index,
                    dataset_index,
                    prediction_index,
                )
            )
    return {
        "prediction": selected_prediction,
        "target": selected_target,
        "learner_keys": tuple(learner_keys),
        "provenance": tuple(provenance),
        "dataset_indices": tuple(dataset_indices),
        "segment_index": segment_indices[0],
    }


class StatefulEvaluationAccumulator:
    """Resumable exact-once prediction pool with learner provenance."""

    SCHEMA_VERSION = 1

    def __init__(self):
        self._predictions = []
        self._targets = []
        self._learner_keys = []
        self._provenance = []
        self._seen = set()

    def append(self, record):
        prediction = record["prediction"].detach().cpu().view(-1)
        target = record["target"].detach().cpu().view(-1)
        learner_keys = tuple(record["learner_keys"])
        provenance = tuple(record["provenance"])
        count = prediction.numel()
        if target.numel() != count or len(learner_keys) != count or len(provenance) != count:
            raise StatefulEvaluatorStateError("accumulator record lengths differ")
        if len(set(provenance)) != len(provenance) or self._seen.intersection(provenance):
            raise StatefulEvaluatorOrderError("prediction provenance is duplicated")
        self._predictions.append(prediction.clone())
        self._targets.append(target.clone())
        self._learner_keys.extend(learner_keys)
        self._provenance.extend(provenance)
        self._seen.update(provenance)

    def finalize(self):
        if not self._predictions:
            raise StatefulEvaluatorStateError("cannot finalize an empty accumulator")
        return {
            "prediction": torch.cat(self._predictions),
            "target": torch.cat(self._targets),
            "learner_keys": tuple(self._learner_keys),
            "provenance": tuple(self._provenance),
        }

    def state_dict(self):
        return {
            "schema_version": self.SCHEMA_VERSION,
            "predictions": [value.clone() for value in self._predictions],
            "targets": [value.clone() for value in self._targets],
            "learner_keys": [key.state_dict() for key in self._learner_keys],
            "provenance": [list(value) for value in self._provenance],
        }

    @classmethod
    def from_state_dict(cls, payload):
        if int(payload.get("schema_version", -1)) != cls.SCHEMA_VERSION:
            raise StatefulEvaluatorStateError("unsupported accumulator schema")
        accumulator = cls()
        predictions = payload.get("predictions", [])
        targets = payload.get("targets", [])
        learner_keys = [
            StatefulEvaluationKey.from_state_dict(value)
            for value in payload.get("learner_keys", [])
        ]
        provenance = [tuple(value) for value in payload.get("provenance", [])]
        offset = 0
        for prediction, target in zip(predictions, targets):
            count = prediction.numel()
            accumulator.append(
                {
                    "prediction": prediction,
                    "target": target,
                    "learner_keys": learner_keys[offset : offset + count],
                    "provenance": provenance[offset : offset + count],
                }
            )
            offset += count
        if len(predictions) != len(targets) or offset != len(learner_keys) or offset != len(provenance):
            raise StatefulEvaluatorStateError("serialized accumulator lengths differ")
        return accumulator


def serialize_boundary_complete_evaluation(store, accumulator):
    if not isinstance(store, UIDBoundaryCompleteEvaluationStore):
        raise TypeError("store type changed")
    if not isinstance(accumulator, StatefulEvaluationAccumulator):
        raise TypeError("accumulator type changed")
    buffer = io.BytesIO()
    torch.save(
        {
            "schema_version": 1,
            "store": store.state_dict(),
            "accumulator": accumulator.state_dict(),
        },
        buffer,
    )
    return buffer.getvalue()


def deserialize_boundary_complete_evaluation(serialized, *, map_location="cpu"):
    payload = torch.load(
        io.BytesIO(serialized), map_location=map_location, weights_only=True
    )
    if int(payload.get("schema_version", -1)) != 1:
        raise StatefulEvaluatorStateError("unsupported evaluation checkpoint")
    return (
        UIDBoundaryCompleteEvaluationStore.from_state_dict(payload["store"]),
        StatefulEvaluationAccumulator.from_state_dict(payload["accumulator"]),
    )


__all__ = [
    "StatefulEvaluationAccumulator",
    "StatefulEvaluationKey",
    "StatefulEvaluatorError",
    "StatefulEvaluatorOrderError",
    "StatefulEvaluatorScopeError",
    "StatefulEvaluatorStateError",
    "StatefulEvaluatorTransactionError",
    "UIDBoundaryCompleteEvaluationStore",
    "assemble_boundary_complete_states",
    "deserialize_boundary_complete_evaluation",
    "evaluate_boundary_complete_uid_batch",
    "serialize_boundary_complete_evaluation",
    "split_boundary_complete_state",
]
