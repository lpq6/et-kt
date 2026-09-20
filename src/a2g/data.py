"""Server train/validation dataset and deterministic batching."""

import csv
import hashlib
import random
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate

FIELDS = {
    "questions": "qseqs",
    "concepts": "cseqs",
    "responses": "rseqs",
    "timestamps": "tseqs",
    "usetimes": "utseqs",
    "selectmasks": "smasks",
}


class TrainValidationDataset(Dataset):
    def __init__(self, path, config, folds):
        path = Path(path)
        if path.name != "train_valid_sequences.csv":
            raise ValueError("only train_valid_sequences.csv is permitted")
        folds = frozenset(folds)
        if (
            not folds
            or not folds <= {0, 1, 2, 3, 4}
            or (0 in folds and len(folds) != 1)
        ):
            raise ValueError("training and validation folds must remain isolated")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            row_count = sum((int(row["fold"]) in folds for row in reader))
        if row_count == 0:
            raise ValueError("empty split")
        selected_fields = [
            field
            for field in FIELDS
            if field in fieldnames
            and (
                field not in ("questions", "concepts") or field in config["input_type"]
            )
        ]
        buffers = {
            field: np.empty(
                (row_count, config["maxlen"]),
                dtype=np.float32 if field == "responses" else np.int64,
            )
            for field in selected_fields
        }
        cursor = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["fold"]) not in folds:
                    continue
                for field, buffer in buffers.items():
                    values = row[field].split(",")
                    if len(values) != config["maxlen"]:
                        raise ValueError(f"sequence width mismatch: {field}")
                    buffer[cursor] = [
                        int(float(value)) if field == "usetimes" else int(value)
                        for value in values
                    ]
                cursor += 1
        if cursor != row_count:
            raise ValueError("CSV changed while loading")
        selection = buffers["selectmasks"]
        if not np.isin(selection, [-1, 1]).all():
            raise ValueError("invalid selection mask")
        valid = selection == 1
        for field, capacity in (
            ("concepts", config["num_c"]),
            ("questions", config["num_q"]),
        ):
            if field not in buffers:
                continue
            identifiers = buffers[field]
            if not np.array_equal(identifiers >= 0, valid):
                raise ValueError(f"ID padding disagrees with mask: {field}")
            floor = 1 if folds == {0} else 2
            if np.any(identifiers[valid] < floor) or np.any(
                identifiers[valid] >= capacity
            ):
                raise ValueError(f"ID outside reserved-ID vocabulary: {field}")
        if not np.isin(buffers["responses"][valid], [0, 1]).all():
            raise ValueError("responses must be binary")
        self.dori = {
            name: torch.from_numpy(buffers[field])
            if field in buffers
            else torch.empty(0, dtype=torch.long)
            for (field, name) in FIELDS.items()
        }
        self.dori["masks"] = (self.dori["cseqs"][:, :-1] != -1) & (
            self.dori["cseqs"][:, 1:] != -1
        )
        self.dori["smasks"] = self.dori["smasks"][:, 1:] != -1
        if not torch.equal(self.dori["masks"], self.dori["smasks"]):
            raise ValueError("scored interactions must have an observed predecessor")
        self.scored_interactions = int(self.dori["smasks"].sum())

    def __len__(self):
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        mask = self.dori["masks"][index]
        current = {}
        for key, tensor in self.dori.items():
            if key in ("masks", "smasks"):
                continue
            if not tensor.numel():
                current[key] = tensor
                current["shft_" + key] = tensor
            else:
                current[key] = tensor[index, :-1] * mask
                current["shft_" + key] = tensor[index, 1:] * mask
        current["masks"] = mask
        current["smasks"] = self.dori["smasks"][index]
        return current


def collate_ordered_segments(batch):
    (sequences, metadata) = zip(*batch)
    return {
        "dcur": default_collate(sequences),
        "state_meta": {
            "learner_uid": [entry.learner_uid for entry in metadata],
            "dataset_index": [entry.dataset_index for entry in metadata],
        },
    }


class FixedOrderBatchSampler(Sampler):
    def __init__(self, metadata, batch_size, seed, epoch, shuffle=False):
        self.size = len(metadata)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.shuffle = bool(shuffle)
        self.signature = hashlib.sha256(
            json.dumps(
                {
                    "batch_size": batch_size,
                    "shuffle": bool(shuffle),
                    "order": [
                        [entry.csv_row_index, entry.learner_uid] for entry in metadata
                    ],
                },
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def __iter__(self):
        # v46: seeded per-epoch shuffle for training (deterministic, reproducible).
        indices = list(range(self.size))
        if self.shuffle:
            rng = random.Random((self.seed * 1000003 + self.epoch) & 0xFFFFFFFF)
            rng.shuffle(indices)
        for start in range(0, self.size, self.batch_size):
            yield indices[start : min(start + self.batch_size, self.size)]

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def state_dict(self):
        return {
            "schema_version": 1,
            "seed": self.seed,
            "epoch": self.epoch,
            "batch_size": self.batch_size,
            "batch_count": len(self),
            "shuffle": self.shuffle,
            "schedule_sha256": self.signature,
            "order": "fixed_csv_filtered_order",
            "state_carry_between_segments": False,
        }
