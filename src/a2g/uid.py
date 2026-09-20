"""UID metadata sidecar and depth-topological batching for sliced KT data."""

import csv
from dataclasses import dataclass, replace
import hashlib
import json
import random

import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate


class UIDLoaderContractError(RuntimeError):
    """Base class for fail-closed UID loader errors."""


class UIDFoldIsolationError(UIDLoaderContractError):
    """Raised when one dataset-local UID is assigned to multiple folds."""


class UIDRowOrderError(UIDLoaderContractError):
    """Raised when one UID reappears after its contiguous row block closed."""


class UIDSegmentShapeError(UIDLoaderContractError):
    """Raised when segment masks or indices violate the slicing contract."""


@dataclass(frozen=True, order=True)
class UIDSegmentMetadata:
    dataset_id: str
    split: str
    fold: int
    learner_uid: str
    segment_index: int
    valid_length: int
    sequence_width: int
    dataset_index: int
    csv_row_index: int
    is_first: bool
    is_last: bool

    @property
    def learner_key(self):
        return self.dataset_id, self.split, self.fold, self.learner_uid

    def state_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "fold": self.fold,
            "learner_uid": self.learner_uid,
            "segment_index": self.segment_index,
            "valid_length": self.valid_length,
            "sequence_width": self.sequence_width,
            "dataset_index": self.dataset_index,
            "csv_row_index": self.csv_row_index,
            "is_first": self.is_first,
            "is_last": self.is_last,
        }


def _require_nonempty_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _parse_fold(value):
    if isinstance(value, bool):
        raise UIDFoldIsolationError("boolean fold is invalid")
    try:
        fold = int(value)
    except (TypeError, ValueError) as error:
        raise UIDFoldIsolationError(f"invalid fold {value!r}") from error
    if fold < 0:
        raise UIDFoldIsolationError("negative fold is outside train/validation scope")
    return fold


def _parse_selectmask(value, sequence_width):
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",")]
    else:
        tokens = [str(token).strip() for token in value]
    if len(tokens) != sequence_width:
        raise UIDSegmentShapeError(
            f"selectmask width {len(tokens)} does not match {sequence_width}"
        )
    seen_padding = False
    valid_length = 0
    for token in tokens:
        if token in {"1", "1.0"}:
            if seen_padding:
                raise UIDSegmentShapeError("valid mask token appears after padding")
            valid_length += 1
        elif token in {"-1", "-1.0"}:
            seen_padding = True
        else:
            raise UIDSegmentShapeError(f"unexpected selectmask token {token!r}")
    if valid_length < 2:
        raise UIDSegmentShapeError("segment must contain at least two interactions")
    return valid_length


def build_uid_segment_metadata(
    rows,
    *,
    dataset_id,
    split,
    allowed_folds,
    sequence_width,
    expected_dataset_length=None,
):
    """Build sidecar rows in the exact order used by KTDataset fold filtering."""

    dataset_id = _require_nonempty_text(dataset_id, "dataset_id")
    split = _require_nonempty_text(split, "split")
    if isinstance(sequence_width, bool) or not isinstance(sequence_width, int):
        raise ValueError("sequence_width must be an integer")
    if sequence_width < 2:
        raise ValueError("sequence_width must be at least two")
    allowed_folds = frozenset(_parse_fold(value) for value in allowed_folds)
    if not allowed_folds:
        raise ValueError("allowed_folds must not be empty")

    materialized = list(rows)
    fold_by_uid = {}
    for row in materialized:
        uid = _require_nonempty_text(str(row["uid"]), "uid")
        fold = _parse_fold(row["fold"])
        prior = fold_by_uid.setdefault(uid, fold)
        if prior != fold:
            raise UIDFoldIsolationError(
                f"UID {uid!r} appears in folds {prior} and {fold}"
            )

    provisional = []
    counts = {}
    closed = set()
    previous_uid = None
    for csv_row_index, row in enumerate(materialized):
        fold = _parse_fold(row["fold"])
        if fold not in allowed_folds:
            continue
        uid = _require_nonempty_text(str(row["uid"]), "uid")
        if uid != previous_uid:
            if previous_uid is not None:
                closed.add(previous_uid)
            if uid in closed:
                raise UIDRowOrderError(
                    f"UID {uid!r} reappears after its row block closed"
                )
            previous_uid = uid
        segment_index = counts.get(uid, 0)
        counts[uid] = segment_index + 1
        provisional.append(
            UIDSegmentMetadata(
                dataset_id=dataset_id,
                split=split,
                fold=fold,
                learner_uid=uid,
                segment_index=segment_index,
                valid_length=_parse_selectmask(row["selectmasks"], sequence_width),
                sequence_width=sequence_width,
                dataset_index=len(provisional),
                csv_row_index=csv_row_index,
                is_first=segment_index == 0,
                is_last=False,
            )
        )

    if expected_dataset_length is not None and len(provisional) != int(
        expected_dataset_length
    ):
        raise UIDSegmentShapeError(
            f"metadata rows {len(provisional)} do not match dataset length "
            f"{expected_dataset_length}"
        )
    if not provisional:
        raise UIDSegmentShapeError("fold filtering produced no metadata rows")

    final = []
    for index, metadata in enumerate(provisional):
        is_last = counts[metadata.learner_uid] == metadata.segment_index + 1
        if not is_last and metadata.valid_length != sequence_width:
            raise UIDSegmentShapeError(
                "every non-final learner segment must be full width"
            )
        final.append(replace(metadata, is_last=is_last))
    return tuple(final)


def load_uid_segment_metadata_csv(
    path,
    *,
    dataset_id,
    split,
    allowed_folds,
    sequence_width,
    expected_dataset_length=None,
):
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"uid", "fold", "selectmasks"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise UIDLoaderContractError(
                f"metadata CSV must contain {sorted(required)}"
            )
        rows = [
            {
                "uid": row["uid"],
                "fold": row["fold"],
                "selectmasks": row["selectmasks"],
            }
            for row in reader
        ]
    return build_uid_segment_metadata(
        rows,
        dataset_id=dataset_id,
        split=split,
        allowed_folds=allowed_folds,
        sequence_width=sequence_width,
        expected_dataset_length=expected_dataset_length,
    )


def _validate_metadata(metadata):
    if not metadata:
        raise UIDSegmentShapeError("metadata must not be empty")
    seen_indices = set()
    groups = {}
    for position, record in enumerate(metadata):
        if not isinstance(record, UIDSegmentMetadata):
            raise TypeError("metadata rows must be UIDSegmentMetadata")
        if record.dataset_index != position:
            raise UIDSegmentShapeError("metadata must align with dataset indices")
        if record.dataset_index in seen_indices:
            raise UIDSegmentShapeError("duplicate dataset index")
        seen_indices.add(record.dataset_index)
        groups.setdefault(record.learner_key, []).append(record)
    for records in groups.values():
        indices = [record.segment_index for record in records]
        if indices != list(range(len(records))):
            raise UIDSegmentShapeError("learner segment indices must be contiguous")
        if sum(record.is_first for record in records) != 1 or not records[0].is_first:
            raise UIDSegmentShapeError("learner must have exactly one first segment")
        if sum(record.is_last for record in records) != 1 or not records[-1].is_last:
            raise UIDSegmentShapeError("learner must have exactly one last segment")
        if any(record.valid_length != record.sequence_width for record in records[:-1]):
            raise UIDSegmentShapeError("non-final segment is not full width")
    return groups


class UIDSegmentDataset(Dataset):
    """Attach state metadata without changing the underlying sequence tensors."""

    def __init__(self, base_dataset, metadata):
        self.base_dataset = base_dataset
        self.metadata = tuple(metadata)
        _validate_metadata(self.metadata)
        if len(base_dataset) != len(self.metadata):
            raise UIDSegmentShapeError("base dataset and metadata lengths differ")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.base_dataset[index], self.metadata[index]


def collate_uid_segments(batch):
    if not batch:
        raise UIDSegmentShapeError("cannot collate an empty batch")
    sequences, metadata = zip(*batch)
    learner_keys = [record.learner_key for record in metadata]
    if len(learner_keys) != len(set(learner_keys)):
        raise UIDSegmentShapeError("one batch contains repeated learner state keys")
    segment_indices = {record.segment_index for record in metadata}
    if len(segment_indices) != 1:
        raise UIDSegmentShapeError("one batch mixes segment depths")
    return {
        "dcur": default_collate(sequences),
        "state_meta": {
            "dataset_id": [record.dataset_id for record in metadata],
            "split": [record.split for record in metadata],
            "fold": torch.tensor(
                [record.fold for record in metadata], dtype=torch.long
            ),
            "learner_uid": [record.learner_uid for record in metadata],
            "segment_index": torch.tensor(
                [record.segment_index for record in metadata], dtype=torch.long
            ),
            "valid_length": torch.tensor(
                [record.valid_length for record in metadata], dtype=torch.long
            ),
            "dataset_index": torch.tensor(
                [record.dataset_index for record in metadata], dtype=torch.long
            ),
            "is_first": torch.tensor(
                [record.is_first for record in metadata], dtype=torch.bool
            ),
            "is_last": torch.tensor(
                [record.is_last for record in metadata], dtype=torch.bool
            ),
        },
    }


class UIDDepthBatchSampler(Sampler):
    """Batch one chronological depth at a time with unique learners per batch."""

    def __init__(self, metadata, batch_size, *, seed, drop_last=False):
        self.metadata = tuple(metadata)
        self.groups = _validate_metadata(self.metadata)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if drop_last:
            raise ValueError("drop_last is forbidden for state-continuity batches")
        self.batch_size = batch_size
        self.seed = int(seed)
        self.epoch = 0
        self._first_position = {
            key: records[0].dataset_index for key, records in self.groups.items()
        }

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def _schedule(self):
        learner_order = sorted(self.groups, key=self._first_position.__getitem__)
        random.Random(self.seed + self.epoch).shuffle(learner_order)
        maximum_depth = max(len(records) for records in self.groups.values())
        batches = []
        for depth in range(maximum_depth):
            wave = [
                self.groups[key][depth].dataset_index
                for key in learner_order
                if len(self.groups[key]) > depth
            ]
            for start in range(0, len(wave), self.batch_size):
                batches.append(wave[start : start + self.batch_size])
        audit_uid_depth_schedule(self.metadata, batches)
        return batches

    def __iter__(self):
        yield from self._schedule()

    def __len__(self):
        return len(self._schedule())

    def state_dict(self):
        batches = self._schedule()
        encoded = json.dumps(batches, separators=(",", ":")).encode("ascii")
        return {
            "schema_version": 1,
            "seed": self.seed,
            "epoch": self.epoch,
            "batch_size": self.batch_size,
            "batch_count": len(batches),
            "schedule_sha256": hashlib.sha256(encoded).hexdigest(),
            "resume_cursor_supported": False,
        }


def audit_uid_depth_schedule(metadata, batches):
    metadata = tuple(metadata)
    _validate_metadata(metadata)
    observed = []
    last_segment = {}
    for batch in batches:
        if not batch:
            raise UIDSegmentShapeError("empty batch is forbidden")
        records = [metadata[index] for index in batch]
        keys = [record.learner_key for record in records]
        if len(keys) != len(set(keys)):
            raise UIDSegmentShapeError("batch repeats one learner key")
        depths = {record.segment_index for record in records}
        if len(depths) != 1:
            raise UIDSegmentShapeError("batch mixes segment depths")
        for record in records:
            expected = last_segment.get(record.learner_key, -1) + 1
            if record.segment_index != expected:
                raise UIDSegmentShapeError(
                    "schedule violates per-learner segment order"
                )
            last_segment[record.learner_key] = record.segment_index
            observed.append(record.dataset_index)
    if sorted(observed) != list(range(len(metadata))):
        raise UIDSegmentShapeError("schedule does not cover every row exactly once")
    return {
        "batch_count": len(batches),
        "row_count": len(metadata),
        "learner_count": len({record.learner_key for record in metadata}),
        "maximum_depth": max(record.segment_index for record in metadata),
    }


__all__ = [
    "UIDDepthBatchSampler",
    "UIDFoldIsolationError",
    "UIDLoaderContractError",
    "UIDRowOrderError",
    "UIDSegmentDataset",
    "UIDSegmentMetadata",
    "UIDSegmentShapeError",
    "audit_uid_depth_schedule",
    "build_uid_segment_metadata",
    "collate_uid_segments",
    "load_uid_segment_metadata_csv",
]
