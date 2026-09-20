"""Independent CSV/metric/epoch checks; no torch or training code imports."""

import csv
import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_population(path, width):
    """Read valid targets in CSV order, including their original row indexes."""
    path = Path(path)
    if path.name != "train_valid_sequences.csv":
        raise ValueError("only the frozen train/validation CSV is permitted")
    folds_by_uid = {}
    counts = {
        name: {"segments": 0, "learners": set(), "scored_interactions": 0}
        for name in ("train", "validation")
    }
    labels, learners, rows, positions = [], [], [], []
    with path.open(encoding="utf-8", newline="") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            fold, uid = int(row["fold"]), row["uid"].strip()
            if fold not in range(5) or not uid:
                raise ValueError("invalid fold or UID")
            if folds_by_uid.setdefault(uid, fold) != fold:
                raise ValueError("learner crosses folds")
            mask = [int(value) for value in row["selectmasks"].split(",")]
            response = [int(value) for value in row["responses"].split(",")]
            if len(mask) != width or len(response) != width:
                raise ValueError("sequence width differs from the contract")
            valid = sum(value == 1 for value in mask)
            if valid < 2 or mask != [1] * valid + [-1] * (width - valid):
                raise ValueError("selection masks must be a nonempty valid prefix")
            if any(value not in (0, 1) for value in response[:valid]):
                raise ValueError("invalid response in an observed position")
            if any(value != -1 for value in response[valid:]):
                raise ValueError("response padding differs from the mask")
            split = "validation" if fold == 0 else "train"
            counts[split]["segments"] += 1
            counts[split]["learners"].add(uid)
            counts[split]["scored_interactions"] += valid - 1
            if split == "validation":
                labels.extend(response[1:valid])
                learners.extend([uid] * (valid - 1))
                rows.extend([row_index] * (valid - 1))
                positions.extend(range(1, valid))
    counts = {
        name: {**values, "learners": len(values["learners"])}
        for name, values in counts.items()
    }
    return {
        "label": np.asarray(labels, dtype=np.int8),
        "learner_uid": np.asarray(learners),
        "csv_row_index": np.asarray(rows, dtype=np.int64),
        "position": np.asarray(positions, dtype=np.int16),
    }, counts


def bind_predictions(values, population, *, full):
    size = len(values["label"])
    if size < 1 or size > len(population["label"]):
        raise ValueError("prediction population has invalid length")
    if full and size != len(population["label"]):
        raise ValueError("full validation population is incomplete")
    for name, expected in population.items():
        if not np.array_equal(values[name], expected[:size]):
            raise ValueError(f"prediction {name} disagrees with CSV sequence order")


def independent_metrics(labels, probabilities):
    labels = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or p.shape != labels.shape or len(labels) == 0:
        raise ValueError("labels/probabilities must be aligned nonempty vectors")
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("binary metrics require both classes")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0,1]")
    y = labels.astype(np.float64)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    bins = np.minimum(np.floor(15 * p).astype(np.int64), 14)
    ece = 0.0
    for index in range(15):
        chosen = np.flatnonzero(bins == index)
        if chosen.size:
            ece += float(abs(np.sum(y[chosen] - p[chosen])) / len(p))
    return {
        "auc": float(roc_auc_score(y, p)),
        "acc": float(np.count_nonzero((p >= 0.5) == y) / len(y)),
        "nll": float(np.mean(np.where(y == 1, -np.log(p), -np.log1p(-p)))),
        "brier": float(np.dot(p - y, p - y) / len(y)),
        "ece_15bin": ece,
        "interaction_count": len(y),
        "positive_count": int(np.count_nonzero(y)),
    }


def compare_metrics(actual, claimed):
    if set(actual) != set(claimed):
        raise ValueError("metric schema mismatch")
    for name, value in actual.items():
        other = claimed[name]
        if name.endswith("_count"):
            equal = type(other) is int and value == other
        else:
            equal = bool(np.isfinite(other) and abs(value - other) <= 1e-12)
        if not equal:
            raise ValueError(f"metric {name} differs: actual={value}, claimed={other}")


def metric_key(metrics):
    return (
        metrics["auc"],
        -metrics["nll"],
        -metrics["brier"],
        -metrics["ece_15bin"],
    )


def audit_training_protocol(run):
    """Validate the declared batch protocol without importing training code."""
    config = run["config"]
    reference = "assist2017_fold0_seed42_batch64_v1"
    direct256 = "assist2017_fold0_seed42_batch256_direct_v1"
    identifier = config.get("training_protocol", reference)
    batches = {reference: 64, direct256: 256}
    if identifier not in batches:
        raise ValueError("unknown training protocol")
    expected = {
        "batch_size": batches[identifier],
        "evaluation_batch_size": 128,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "max_epochs": 200,
        "patience": 20,
    }
    training = config["training"]
    if set(training) != set(expected) or any(
        type(training[key]) is bool or training[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("declared optimizer/batch/stopping protocol changed")
    recorded_protocol = run.get("training_protocol", reference)
    if recorded_protocol != identifier:
        raise ValueError("run protocol disagrees with its config")
    if (
        run.get("gradient_accumulation_steps", 1) != 1
        or run.get("microbatching", False)
    ):
        raise ValueError("direct batch protocol cannot use accumulation or microbatches")
    if identifier == direct256 and (
        "gradient_accumulation_steps" not in run
        or "microbatching" not in run
        or run["gradient_accumulation_steps"] != 1
        or run["microbatching"] is not False
    ):
        raise ValueError("direct batch256 must explicitly record its execution")
    if run["mode"] == "train" and (
        run["effective_batch_size"] != batches[identifier]
        or run["effective_evaluation_batch_size"] != 128
    ):
        raise ValueError("executed batch differs from the declared protocol")
    return identifier


def audit_epochs(epochs, result, training, counts, *, full):
    if not epochs or [row["epoch"] for row in epochs] != list(
        range(1, len(epochs) + 1)
    ):
        raise ValueError("epoch sequence is incomplete or duplicated")
    if result["epochs"] != len(epochs):
        raise ValueError("reported epoch count differs from the log")
    selected, previous_elapsed = None, -1.0
    for row in epochs:
        validation = row["validation"]
        if not np.isfinite(row["train_loss"]) or row["train_loss"] < 0:
            raise ValueError("invalid training loss")
        if (
            not np.isfinite(row["duration_seconds"])
            or row["duration_seconds"] < previous_elapsed
        ):
            raise ValueError("epoch timestamps are not monotonic")
        previous_elapsed = row["duration_seconds"]
        for name in ("auc", "acc", "nll", "brier", "ece_15bin"):
            if not np.isfinite(validation[name]) or validation[name] < 0:
                raise ValueError(f"invalid epoch metric: {name}")
        for name in ("auc", "acc", "brier", "ece_15bin"):
            if validation[name] > 1:
                raise ValueError(f"epoch metric out of range: {name}")
        if selected is None or metric_key(validation) > metric_key(
            selected["validation"]
        ):
            selected = row
        if (
            row["best_epoch"] != selected["epoch"]
            or row["best_auc"] != selected["validation"]["auc"]
        ):
            raise ValueError(
                "best checkpoint history does not follow the selection rule"
            )
        if bool(row["smoke_only"]) == full:
            raise ValueError("smoke/full declaration contradicts the run mode")
        if full:
            if "optimizer_steps" in row and row["optimizer_steps"] != (
                counts["train"]["segments"] + training["batch_size"] - 1
            ) // training["batch_size"]:
                raise ValueError("optimizer step count differs from direct batching")
            if row["train_interactions"] != counts["train"]["scored_interactions"]:
                raise ValueError("an epoch did not cover all training interactions")
            if (
                validation["interaction_count"]
                != counts["validation"]["scored_interactions"]
            ):
                raise ValueError("an epoch did not cover all validation interactions")
            if (
                row is not epochs[-1]
                and row["epoch"] - selected["epoch"] >= training["patience"]
            ):
                raise ValueError("training continued beyond the first early-stop event")
    if selected["epoch"] != result["best_epoch"]:
        raise ValueError("selected epoch differs from result")
    compare_metrics(selected["validation"], result["metrics"])
    if full:
        if len(epochs) > training["max_epochs"]:
            raise ValueError("training exceeds maximum epochs")
        if (
            len(epochs) < training["max_epochs"]
            and len(epochs) - selected["epoch"] != training["patience"]
        ):
            raise ValueError("training did not satisfy the stopping rule")
    return selected["epoch"]
