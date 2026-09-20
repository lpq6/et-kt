"""Verify independent artifacts and estimate paired learner-bootstrap effects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

VARIANTS = ("full", "no_evidence", "no_ssm", "no_attention")
METRICS = ("auc", "acc", "nll", "brier", "ece_15bin")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class WeightedMetrics:
    def __init__(self, labels, probabilities, learners):
        self.labels = labels.astype(np.float64)
        self.probability = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
        self.learners = learners
        self.count = int(learners.max()) + 1
        self.sizes = np.bincount(learners, minlength=self.count)
        _, self.ties = np.unique(self.probability, return_inverse=True)
        self.tie_count = int(self.ties.max()) + 1
        self.bin_index = np.minimum((self.probability * 15).astype(np.int64), 14)
        self.flat_bin = learners * 15 + self.bin_index
        self.bin_sizes = np.bincount(
            self.flat_bin, minlength=self.count * 15
        ).reshape(self.count, 15)
        self.bin_error = np.bincount(
            self.flat_bin,
            weights=self.labels - self.probability,
            minlength=self.count * 15,
        ).reshape(self.count, 15)
        individual = np.column_stack([
            (self.probability >= 0.5) == self.labels,
            -self.labels * np.log(self.probability)
            - (1 - self.labels) * np.log1p(-self.probability),
            (self.probability - self.labels) ** 2,
        ])
        self.cluster_sums = np.column_stack([
            np.bincount(learners, weights=individual[:, index], minlength=self.count)
            for index in range(individual.shape[1])
        ])

    def evaluate(self, multiplicity):
        weights = multiplicity[self.learners]
        positives = np.bincount(
            self.ties, weights=weights * self.labels, minlength=self.tie_count
        )
        negatives = np.bincount(
            self.ties, weights=weights * (1 - self.labels), minlength=self.tie_count
        )
        positive_total, negative_total = positives.sum(), negatives.sum()
        if positive_total == 0 or negative_total == 0:
            return None
        auc = np.dot(positives, np.cumsum(negatives) - negatives * 0.5) / (
            positive_total * negative_total
        )
        total = float(multiplicity @ self.sizes)
        proper = (multiplicity @ self.cluster_sums) / total
        ece = np.abs(multiplicity @ self.bin_error).sum() / total
        return np.asarray([auc, *proper, ece], dtype=np.float64)


def verify_implementation():
    generator = np.random.default_rng(20260907)
    cases = 0
    for index in range(12):
        labels = generator.integers(0, 2, 60)
        probabilities = generator.choice([0.0, 0.2, 0.5, 0.8, 1.0], 60)
        learners = np.tile(np.arange(10), 6)
        metric = WeightedMetrics(labels, probabilities, learners)
        multiplicity = generator.integers(0, 4, 10)
        estimated = metric.evaluate(multiplicity)
        expanded = np.repeat(np.arange(60), multiplicity[learners])
        target = labels[expanded].astype(float)
        prediction = np.clip(probabilities[expanded], 1e-7, 1 - 1e-7)
        direct = [
            roc_auc_score(target, prediction),
            np.mean((prediction >= 0.5) == target),
            np.mean(-target * np.log(prediction) - (1 - target) * np.log1p(-prediction)),
            np.mean((prediction - target) ** 2),
        ]
        bins = np.minimum((prediction * 15).astype(int), 14)
        direct.append(sum(
            abs((target[bins == bin_index] - prediction[bins == bin_index]).sum())
            for bin_index in range(15)
        ) / len(target))
        if not np.allclose(estimated, direct, atol=1e-12, rtol=1e-12):
            raise AssertionError(f"bootstrap metric mismatch in case {index}")
        cases += 1
    return {"status": "pass", "cases": cases, "reference": "explicit learner resampling and sklearn ROC AUC"}


def audit(root, replicates, seed, family_size):
    implementation = verify_implementation()
    full_arrays = None
    estimators, result_records, metric_values = {}, {}, {}
    init_hashes, split_signatures, keys_by_variant = {}, {}, {}
    for variant in VARIANTS:
        result_path = root / variant / "result.json"
        result = json.loads(result_path.read_text())
        if result["variant"] != variant or result.get("pretrained_checkpoint") is not None:
            raise ValueError(f"unexpected variant or initialization: {variant}")
        for key, filename in (
            ("checkpoint", "selected_model.pt"),
            ("predictions", "validation_predictions.npz"),
            ("epoch_log", "epochs.jsonl"),
        ):
            if digest(root / variant / filename) != result[key]["sha256"]:
                raise ValueError(f"{variant} {key} hash mismatch")
        arrays = dict(np.load(root / variant / "validation_predictions.npz", allow_pickle=False))
        if full_arrays is None:
            full_arrays = arrays
            learner_names, learner_index = np.unique(arrays["learner_uid"], return_inverse=True)
            full_keys = np.rec.fromarrays([
                arrays["learner_uid"], arrays["csv_row_index"], arrays["position"]
            ])
            if np.unique(full_keys).size != len(full_keys):
                raise ValueError("duplicate interaction keys")
        for key in ("label", "learner_uid", "csv_row_index", "position"):
            if not np.array_equal(arrays[key], full_arrays[key]):
                raise ValueError(f"paired alignment mismatch for {variant}/{key}")
        estimator = WeightedMetrics(arrays["label"], arrays["probability"], learner_index)
        point = estimator.evaluate(np.ones(len(learner_names), dtype=np.int64))
        recorded = np.asarray([result["metrics"][name] for name in METRICS])
        if not np.allclose(point, recorded, atol=1e-12, rtol=1e-12):
            raise ValueError(f"metric replay mismatch for {variant}: {point - recorded}")
        epoch_path = root / variant / "epochs.jsonl"
        history = [json.loads(line) for line in epoch_path.read_text().splitlines()]
        for epoch, entry in enumerate(history, 1):
            if entry["epoch"] != epoch or entry["variant"] != variant:
                raise ValueError("epoch history is not sequential")
        selection = max(history, key=lambda entry: (
            entry["validation"]["auc"],
            -entry["validation"]["nll"],
            -entry["validation"]["brier"],
            -entry["validation"]["ece_15bin"],
            -entry["epoch"],
        ))
        if selection["epoch"] != result["best_epoch"]:
            raise ValueError("checkpoint selection differs from declared rule")
        split_signatures[variant] = [
            entry["schedule"]["schedule_sha256"] for entry in history
        ]
        init_hashes[variant] = result["initial_state"]["sha256"]
        keys_by_variant[variant] = (result["dataset"], result["seed"])
        estimators[variant] = estimator
        metric_values[variant] = point
        result_records[variant] = {
            "path": str(result_path.resolve()), "sha256": digest(result_path),
            "checkpoint_sha256": result["checkpoint"]["sha256"],
            "prediction_sha256": result["predictions"]["sha256"],
            "epochs": result["epochs"], "best_epoch": result["best_epoch"],
        }
    if len(set(init_hashes.values())) != 1 or digest(root / "initial_state.pt") != init_hashes["full"]:
        raise ValueError("initialization mismatch")
    if len(set(keys_by_variant.values())) != 1:
        raise ValueError("dataset or seed mismatch")
    for variant, signatures in split_signatures.items():
        common = min(len(signatures), len(split_signatures["full"]))
        if signatures[:common] != split_signatures["full"][:common]:
            raise ValueError(f"training schedule mismatch: {variant}")
    generator = np.random.default_rng(seed)
    draws = {variant: [] for variant in VARIANTS[1:]}
    invalid = 0
    for index in range(replicates):
        sampled = generator.integers(0, len(learner_names), len(learner_names))
        multiplicity = np.bincount(sampled, minlength=len(learner_names))
        metrics = {
            variant: estimator.evaluate(multiplicity)
            for variant, estimator in estimators.items()
        }
        if any(value is None for value in metrics.values()):
            invalid += 1
            continue
        for variant in VARIANTS[1:]:
            draws[variant].append(metrics["full"] - metrics[variant])
        if (index + 1) % 250 == 0:
            print(json.dumps({"bootstrap_completed": index + 1, "total": replicates}), flush=True)
    if invalid or len(draws["no_evidence"]) != replicates:
        raise ValueError(f"invalid bootstrap draws: {invalid}")
    effects = {}
    for variant in VARIANTS[1:]:
        distribution = np.asarray(draws[variant])
        point = metric_values["full"] - metric_values[variant]
        effects[variant] = {
            name: {
                "full_minus_ablation": float(point[index]),
                "ci95": np.quantile(distribution[:, index], [0.025, 0.975]).tolist(),
                "simultaneous_bonferroni_ci": np.quantile(
                    distribution[:, index],
                    [0.05 / (2 * family_size), 1 - 0.05 / (2 * family_size)],
                ).tolist(),
                "beneficial_direction": "positive" if name in ("auc", "acc") else "negative",
            }
            for index, name in enumerate(METRICS)
        }
    return {
        "status": "complete_exploratory_paired_learner_audit",
        "implementation_self_check": implementation,
        "dataset_seed": list(keys_by_variant["full"]),
        "learner_count": len(learner_names),
        "interaction_count": len(full_arrays["label"]),
        "replicates": replicates, "seed": seed, "family_size": family_size,
        "metric_replay": "all_20_values_match",
        "interaction_alignment": "exact",
        "initial_state_sha256": init_hashes["full"],
        "training_schedule": "matched_on_all_shared_epochs",
        "metrics": {
            variant: dict(zip(METRICS, values.tolist()))
            for variant, values in metric_values.items()
        },
        "effects": effects,
        "result_records": result_records,
        "limitation": "Validation-selected checkpoints; bootstrap conditions on model selection and one training seed. Not held-out or repeated-training confirmation.",
        "all_auc_ci95_positive": all(effect["auc"]["ci95"][0] > 0 for effect in effects.values()),
        "all_auc_simultaneous_ci_positive": all(
            effect["auc"]["simultaneous_bonferroni_ci"][0] > 0 for effect in effects.values()
        ),
        "paper_goal_complete": False,
        "test_access": False,
        "auditor_sha256": digest(Path(__file__)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--family-size", type=int, default=15)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(verify_implementation(), sort_keys=True))
    else:
        if args.root is None or args.output is None:
            parser.error("--root and --output are required")
        report = audit(args.root, args.replicates, args.seed, args.family_size)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps(report, sort_keys=True))
