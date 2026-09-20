"""Independently verify stored artifacts without granting paper-level admission."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_protocol import (
    audit_epochs,
    audit_training_protocol,
    bind_predictions,
    compare_metrics,
    independent_metrics,
    reconstruct_population,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, source_root, data_directory=None):
    directory = Path(directory)
    source_root = Path(source_root).resolve()
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    for filename, field in [
        ("selected_model.pt", "checkpoint_sha256"),
        ("validation_predictions.npz", "predictions_sha256"),
        ("run.json", "run_sha256"),
        ("epochs.jsonl", "epoch_log_sha256"),
    ]:
        if digest(directory / filename) != result[field]:
            raise ValueError(f"artifact digest mismatch: {filename}")
    if digest(directory / "initial_state.pt") != run["initial_state_sha256"]:
        raise ValueError("initialization digest mismatch")
    snapshot = directory / "source_snapshot"
    source_binding = snapshot if snapshot.is_dir() else source_root
    for name, expected in run["source"].items():
        if digest(source_binding / name) != expected:
            raise ValueError(f"source changed since training: {name}")
    config = run["config"]
    if (
        config["dataset"] != "assist2017"
        or config["seed"] != 42
        or config["train_folds"] != [1, 2, 3, 4]
        or config["validation_folds"] != [0]
        or run["pretrained_checkpoint"] is not None
        or run["optimizer_reused"]
        or result["test_access"]
    ):
        raise ValueError("run is outside the reference scope")
    if (directory / "incomplete.json").exists():
        raise ValueError("run has an incomplete marker")
    if run["mode"] not in ("train", "smoke"):
        raise ValueError("unknown run mode")
    protocol = audit_training_protocol(run)
    if result.get(
        "training_protocol", "assist2017_fold0_seed42_batch64_v1"
    ) != protocol:
        raise ValueError("result protocol disagrees with its config")
    data_directory = (
        Path(data_directory)
        if data_directory is not None
        else source_root.parents[1] / "data/assist2017"
    )
    csv_path = data_directory / "train_valid_sequences.csv"
    mapping_path = data_directory / "training_fold_id_maps.json"
    binding = config["data_binding"]
    if (
        digest(csv_path) != binding["csv_sha256"]
        or digest(mapping_path) != binding["mapping_sha256"]
    ):
        raise ValueError("frozen data binding mismatch")
    population, counts = reconstruct_population(csv_path, config["data"]["maxlen"])
    if counts != binding["counts"] or counts != run["data"]["counts"]:
        raise ValueError("data population differs from the frozen run")
    with np.load(
        directory / "validation_predictions.npz", allow_pickle=False
    ) as archive:
        values = {name: archive[name] for name in archive.files}
    required = {"label", "probability", "learner_uid", "csv_row_index", "position"}
    if set(values) != required:
        raise ValueError("prediction schema mismatch")
    length = len(values["label"])
    if any(value.ndim != 1 or len(value) != length for value in values.values()):
        raise ValueError("prediction fields are not aligned vectors")
    p = values["probability"].astype(np.float64)
    y = values["label"].astype(np.float64)
    if (
        not np.isfinite(p).all()
        or not np.isin(y, [0, 1]).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("invalid predictions")
    keys = set(zip(values["learner_uid"], values["csv_row_index"], values["position"]))
    if len(keys) != length:
        raise ValueError("duplicate scored interaction keys")
    full = run["mode"] == "train"
    bind_predictions(values, population, full=full)
    recomputed = independent_metrics(y, p)
    compare_metrics(recomputed, result["metrics"])
    auc = recomputed["auc"]
    epochs = [
        json.loads(line)
        for line in (directory / "epochs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    audit_epochs(epochs, result, config["training"], counts, full=full)
    if full:
        if result["status"] != "complete_local_independent_validation":
            raise ValueError("full run is not complete")
        if length != 153067 or len(np.unique(values["learner_uid"])) != 274:
            raise ValueError("validation population is incomplete")
        if protocol == "assist2017_fold0_seed42_batch256_direct_v1" and any(
            "optimizer_steps" not in row for row in epochs
        ):
            raise ValueError("direct batch256 is missing optimizer step evidence")
        if result["point_gate_pass"] != bool(auc > 0.8174):
            raise ValueError("point gate was reported incorrectly")
    elif result["point_gate_pass"]:
        raise ValueError("smoke run cannot pass the performance gate")
    return {
        "status": "stored_artifacts_verified",
        "run_mode": run["mode"],
        "training_protocol": protocol,
        "recomputed_auc": auc,
        "recomputed_metrics": recomputed,
        "interaction_count": length,
        "csv_bound_interaction_keys": True,
        "csv_population": counts,
        "point_gate_pass": bool(full and auc > 0.8174),
        "paper_goal_complete": False,
        "limits": [
            "Not an independently retrained model.",
            "Does not certify matched baselines or positive module ablations.",
            "Does not reload the model; use reevaluate_checkpoint.py for that check.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = audit(args.directory, root / "src/a2g", args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
