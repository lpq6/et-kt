"""Describe only Assist2017 training-fold historical timestamp differences."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED = "2fa792134739275c7e27db0d91f2200d9669e6691480ac28e53d2d2f9c614861"


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def profile(rows, width):
    counts = Counter()
    learners = set()
    deltas = []
    for row in rows:
        fold = int(row["fold"])
        if fold == 0:
            counts["validation_rows_skipped"] += 1
            continue
        if fold not in {1, 2, 3, 4}:
            raise ValueError("only training folds1-4 and skipped validation fold0 are permitted")
        timestamps = [int(value) for value in row["timestamps"].split(",")]
        masks = [int(value) for value in row["selectmasks"].split(",")]
        if len(timestamps) != width or len(masks) != width or not set(masks) <= {-1, 1}:
            raise ValueError("unexpected metadata width or selection")
        valid = [index for index, mask in enumerate(masks) if mask == 1]
        if valid != list(range(len(valid))):
            raise ValueError("selected events must be a prefix")
        counts["training_rows"] += 1
        learners.add(row["uid"])
        counts["scored_targets"] += max(len(valid) - 1, 0)
        for target in range(2, len(valid)):
            earlier, later = timestamps[target - 2:target]
            counts["targets_with_two_predecessors"] += 1
            if earlier < 0 or later < 0:
                counts["missing_history_time"] += 1
                continue
            if later < earlier:
                counts["negative_history_gap"] += 1
                continue
            delta = later - earlier
            deltas.append(delta)
            counts["positive_history_gap" if delta > 0 else "zero_history_gap"] += 1
            legacy = float(np.float32(later) - np.float32(earlier))
            counts["float32_epoch_subtraction_changes_gap"] += int(legacy != delta)
            counts["float32_epoch_subtraction_loses_positive_gap"] += int(delta > 0 and legacy == 0)
    if not deltas:
        raise ValueError("no usable historical time pairs")
    return {
        "counts": dict(sorted(counts.items())),
        "training_learners": len(learners),
        "gap_seconds_quantiles": dict(zip(
            ["min", "q25", "median", "q75", "q95", "max"],
            np.quantile(np.asarray(deltas, dtype=np.float64) / 1000.0, [0, .25, .5, .75, .95, 1]).tolist(),
        )),
        "scored_target_coverage_rule": "target t>=2 uses only timestamps[t-1] and timestamps[t-2]",
        "response_tokens_parsed": False,
        "validation_tokens_parsed": False,
        "feature_normalization_fitted": False,
        "prediction_metrics_computed": False,
        "gpu_used": False,
        "model_fitting": False,
        "test_access": False,
        "full_goal_complete": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.csv.name != "train_valid_sequences.csv" or args.csv.parent.name != "assist2017":
        raise ValueError("only the declared Assist2017 train/validation source is permitted")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite the descriptive profile")
    if digest(args.csv) != EXPECTED:
        raise ValueError("Assist2017 source differs from the immutable binding")
    with args.csv.open(encoding="utf-8", newline="") as handle:
        report = profile(csv.DictReader(handle), 200)
    if digest(args.csv) != EXPECTED:
        raise ValueError("Assist2017 source changed while profiling")
    report.update({
        "status": "train_only_history_time_profile_complete",
        "source": str(args.csv), "source_sha256": EXPECTED,
        "script_sha256": digest(Path(__file__)),
        "limitations": "Descriptive metadata only; no training, predictive effect or generalization is established.",
    })
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)
