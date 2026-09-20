from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT))


def audit():
    from a2g_uid_topological_loader_20260807 import (
        UIDFoldIsolationError,
        UIDSegmentDataset,
        load_uid_segment_metadata_csv,
    )
    from legacy_data_loader import KTDataset
    from strict_sequence_data import (
        FixedOrderBatchSampler,
        TrainValidationDataset,
        collate_ordered_segments,
    )

    torch.set_num_threads(1)
    config = {
        "num_c": 5,
        "num_q": 7,
        "maxlen": 4,
        "input_type": ["questions", "concepts"],
    }
    rows = [
        {
            "uid": "alpha",
            "fold": 1,
            "questions": "2,3,4,6",
            "concepts": "2,3,4,2",
            "responses": "0,1,0,1",
            "selectmasks": "1,1,1,1",
        },
        {
            "uid": "alpha",
            "fold": 1,
            "questions": "4,5,-1,-1",
            "concepts": "3,4,-1,-1",
            "responses": "1,0,-1,-1",
            "selectmasks": "1,1,-1,-1",
        },
        {
            "uid": "beta",
            "fold": 2,
            "questions": "6,4,3,-1",
            "concepts": "4,2,3,-1",
            "responses": "1,1,0,-1",
            "selectmasks": "1,1,1,-1",
        },
        {
            "uid": "valid",
            "fold": 0,
            "questions": "1,6,2,-1",
            "concepts": "1,4,3,-1",
            "responses": "1,0,1,-1",
            "selectmasks": "1,1,1,-1",
        },
    ]
    checks = {}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train_valid_sequences.csv"

        def emit():
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

        emit()
        train = TrainValidationDataset(path, config, {1, 2, 3, 4})
        validation = TrainValidationDataset(path, config, {0})
        checks["no_cache_files_created"] = not list(
            path.parent.glob("*.pkl")
        )
        checks["cpu_only"] = all(
            not tensor.is_cuda for tensor in train.dori.values()
        )
        legacy_train = KTDataset(
            str(path),
            config["input_type"],
            {1, 2, 3, 4},
        )
        legacy_valid = KTDataset(
            str(path),
            config["input_type"],
            {0},
        )
        checks["exact_legacy_tensor_parity"] = all(
            torch.equal(dataset[index][key], legacy[index][key].cpu())
            for dataset, legacy in (
                (train, legacy_train),
                (validation, legacy_valid),
            )
            for index in range(len(dataset))
            for key in dataset[index]
        )
        metadata = load_uid_segment_metadata_csv(
            path,
            dataset_id="synthetic",
            split="train",
            allowed_folds={1, 2, 3, 4},
            sequence_width=4,
            expected_dataset_length=3,
        )
        wrapped = UIDSegmentDataset(train, metadata)
        combined = collate_ordered_segments(
            [wrapped[index] for index in range(3)]
        )
        checks["mixed_depth_repeated_learner_batches_allowed"] = (
            combined["state_meta"]["learner_uid"]
            == ["alpha", "alpha", "beta"]
            and combined["state_meta"]["dataset_index"] == [0, 1, 2]
        )
        first = FixedOrderBatchSampler(metadata, 2, 42, 0)
        second = FixedOrderBatchSampler(metadata, 2, 42, 1)
        checks["fixed_order_exact_coverage"] = (
            list(first) == list(second) == [[0, 1], [2]]
        )
        checks["fixed_schedule_hash"] = (
            first.state_dict()["schedule_sha256"]
            == second.state_dict()["schedule_sha256"]
        )
        checks["scored_count"] = (
            train.scored_interactions == 6
            and validation.scored_interactions == 2
        )
        embedding = torch.nn.Embedding(2 * config["num_c"], 3)
        highest = torch.tensor([config["num_c"] - 1])
        checks["baseline_embedding_capacity_safe"] = (
            embedding(highest + config["num_c"]).shape == (1, 3)
        )
        checks["unknown_validation_id_retained"] = (
            int(validation[0]["cseqs"][0]) == 1
        )
        try:
            TrainValidationDataset(path, config, {0, 1})
        except ValueError:
            checks["mixed_train_valid_rejected"] = True
        else:
            checks["mixed_train_valid_rejected"] = False
        rows[-1]["uid"] = "alpha"
        emit()
        try:
            load_uid_segment_metadata_csv(
                path,
                dataset_id="synthetic",
                split="train",
                allowed_folds={1, 2, 3, 4},
                sequence_width=4,
            )
        except UIDFoldIsolationError:
            checks["learner_overlap_rejected"] = True
        else:
            checks["learner_overlap_rejected"] = False
        rows[-1]["uid"] = "valid"
        rows[0]["concepts"] = "2,3,5,2"
        emit()
        try:
            TrainValidationDataset(path, config, {1, 2, 3, 4})
        except ValueError:
            checks["out_of_capacity_id_rejected"] = True
        else:
            checks["out_of_capacity_id_rejected"] = False
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "scope": "synthetic_only_no_production_dataset_access",
        "test_access": False,
        "window_test_access": False,
        "source_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "strict_sequence_data.py",
                "legacy_data_loader.py",
                "audit_data_contract.py",
            )
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)
