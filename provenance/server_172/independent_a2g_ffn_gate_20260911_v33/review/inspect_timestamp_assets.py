"""Inspect timestamp availability, without fitting or reading response tokens."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


DATASETS = {
    "assist2009_corrected_collapsed", "assist2012", "assist2015", "assist2017",
    "junyi2015", "nips_task34", "slepemapy", "statics2011",
}
TRAIN_FOLDS = {1, 2, 3, 4}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_rows(reader, maxlen):
    fields = list(reader.fieldnames)
    if "timestamps" not in fields:
        return {"fields": fields, "status": "no_timestamp_column", "sampled_train_rows": 0}
    skipped = 0
    for row in reader:
        fold = int(row["fold"])
        if fold not in TRAIN_FOLDS:
            if fold != 0:
                raise ValueError("unexpected fold in the train/validation file")
            skipped += 1
            continue
        timestamps = [int(value) for value in row["timestamps"].split(",")]
        selected = [int(value) for value in row["selectmasks"].split(",")]
        if len(timestamps) != maxlen or len(selected) != maxlen:
            raise ValueError("metadata width differs from the frozen configuration")
        if not set(selected) <= {-1, 1}:
            raise ValueError("unexpected selection mask")
        valid = [time for time, mask in zip(timestamps, selected) if mask == 1]
        deltas = [right - left for left, right in zip(valid, valid[1:])]
        usable = len(valid) >= 2 and min(valid) >= 0 and all(x >= 0 for x in deltas)
        return {
            "fields": fields,
            "status": "nonconstant_ordered_timestamp_sample" if usable and any(x > 0 for x in deltas)
            else "sample_does_not_establish_timestamp_availability",
            "sampled_train_rows": 1,
            "sample_fold": fold,
            "validation_rows_skipped_without_token_parsing": skipped,
            "sample_valid_events": len(valid),
            "sample_timestamp_min": min(valid) if valid else None,
            "sample_timestamp_max": max(valid) if valid else None,
            "sample_positive_gaps": sum(x > 0 for x in deltas),
            "sample_zero_gaps": sum(x == 0 for x in deltas),
            "sample_negative_gaps": sum(x < 0 for x in deltas),
            "sample_distinct_gaps": len(set(deltas)),
        }
    raise ValueError("timestamp-bearing file has no training row")


def inspect(config_path):
    if config_path.name != "data_config.train_fold_only.json":
        raise ValueError("a frozen train-fold data configuration is required")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if set(config) != DATASETS:
        raise ValueError("the original eight-dataset scope differs")
    reports = {}
    for name, entry in sorted(config.items()):
        if entry["train_valid_file"] != "train_valid_sequences.csv":
            raise ValueError("test or alternative files are not permitted")
        path = Path(entry["dpath"]) / entry["train_valid_file"]
        before = path.stat()
        with path.open(encoding="utf-8", newline="") as handle:
            report = inspect_rows(csv.DictReader(handle), entry["maxlen"])
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source changed during metadata inspection")
        reports[name] = {
            **report,
            "csv": str(path),
            "csv_bytes": before.st_size,
            "csv_mtime_ns": before.st_mtime_ns,
            "question_ids_available_by_config": "questions" in entry["input_type"] and entry["num_q"] > 0,
            "whole_csv_content_hash_reverified": False,
        }
    available = [name for name, report in reports.items() if report["status"] == "nonconstant_ordered_timestamp_sample"]
    with_items = [name for name in available if reports[name]["question_ids_available_by_config"]]
    return {
        "status": "metadata_sample_inspection_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": digest(config_path),
        "datasets": reports,
        "nonconstant_timestamp_samples": available,
        "nonconstant_timestamp_samples_with_question_ids": with_items,
        "minimum_required_same_datasets": 5,
        "response_tokens_parsed": False,
        "validation_tokens_parsed": False,
        "prediction_metrics_computed": False,
        "gpu_used": False,
        "model_fitting": False,
        "test_access": False,
        "candidate_admitted": False,
        "full_goal_complete": False,
        "script_sha256": digest(Path(__file__)),
        "limitations": (
            "One training-row metadata sample establishes neither all-row validity, "
            "timestamp units/provenance, prediction-time observability, nor predictive benefit. "
            "Question-ID availability is checked separately because V19's attempt module needs items."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite metadata evidence")
    report = inspect(args.config)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)
