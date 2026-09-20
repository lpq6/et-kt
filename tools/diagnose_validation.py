"""CSV-bound, post-selection error analysis; never a replacement admission gate."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from audit_protocol import bind_predictions, independent_metrics, reconstruct_population
from audit_run import audit, digest


def rows(path, width):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            values = {
                key: [int(value) for value in row[key].split(",")]
                for key in ("questions", "concepts", "responses", "selectmasks")
            }
            if any(len(value) != width for value in values.values()):
                raise ValueError("diagnostic CSV width mismatch")
            yield index, int(row["fold"]), row["uid"].strip(), values


def band(value, upper_bounds, labels):
    for bound, label in zip(upper_bounds, labels):
        if value < bound:
            return label
    return labels[-1]


def diagnostic_features(path, width):
    population, counts = reconstruct_population(path, width)
    item_support = Counter()
    concept_support = Counter()
    for _, fold, _, row in rows(path, width):
        if fold == 0:
            continue
        valid = sum(value == 1 for value in row["selectmasks"])
        for position in range(1, valid):
            item_support[row["questions"][position]] += 1
            concept_support[row["concepts"][position]] += 1
    fields = {
        name: [] for name in (
            "position_band", "item_support_band", "concept_support_band",
            "concept_history_band", "item_history_band", "previous_response",
            "concept_transition", "question", "concept",
        )
    }
    support_bounds = (1, 5, 20, 50, 200)
    support_labels = ("0", "1-4", "5-19", "20-49", "50-199", "200+")
    for _, fold, _, row in rows(path, width):
        if fold != 0:
            continue
        valid = sum(value == 1 for value in row["selectmasks"])
        seen_items, seen_concepts = Counter(), Counter()
        for position in range(valid):
            question, concept = row["questions"][position], row["concepts"][position]
            if position:
                fields["question"].append(question)
                fields["concept"].append(concept)
                fields["position_band"].append(band(
                    position, (5, 20, 50, 100), ("1-4", "5-19", "20-49", "50-99", "100-199")
                ))
                fields["item_support_band"].append(band(
                    item_support[question], support_bounds, support_labels
                ))
                fields["concept_support_band"].append(band(
                    concept_support[concept], support_bounds, support_labels
                ))
                for name, key, seen in (
                    ("concept_history_band", concept, seen_concepts),
                    ("item_history_band", question, seen_items),
                ):
                    fields[name].append(
                        "OOV" if key < 2 else band(
                            seen[key], (1, 3, 10), ("first", "1-2", "3-9", "10+")
                        )
                    )
                fields["previous_response"].append(str(row["responses"][position - 1]))
                fields["concept_transition"].append(
                    "unknown" if concept < 2 or row["concepts"][position - 1] < 2 else
                    "same" if concept == row["concepts"][position - 1] else "changed"
                )
            if question >= 2:
                seen_items[question] += 1
            if concept >= 2:
                seen_concepts[concept] += 1
    result = {key: np.asarray(value) for key, value in fields.items()}
    if any(len(value) != len(population["label"]) for value in result.values()):
        raise ValueError("diagnostic features are not aligned to scored interactions")
    return population, result, counts


def metrics_for_group(y, p):
    if len(np.unique(y)) == 2:
        return independent_metrics(y, p)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return {
        "auc": None,
        "auc_status": "undefined_single_class",
        "interaction_count": len(y),
        "positive_count": int(np.sum(y)),
        "acc": float(np.mean((p >= 0.5) == y)),
        "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def summarize(values, features):
    y, p, learners = values["label"], values["probability"], values["learner_uid"]
    result = {}
    for field, groups in features.items():
        if field in ("question", "concept"):
            continue
        result[field] = []
        for group in np.unique(groups):
            selected = groups == group
            result[field].append({
                "group": str(group),
                "learner_count": len(np.unique(learners[selected])),
                "fraction_of_scored_population": float(np.mean(selected)),
                "metrics": metrics_for_group(y[selected], p[selected]),
            })
    return result


def diagnose(directory, source, data_directory):
    directory, data_directory = Path(directory), Path(data_directory)
    audit_report = audit(directory, source, data_directory)
    if audit_report["run_mode"] != "train":
        raise ValueError("diagnostics require completed full validation, not smoke")
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    population, features, counts = diagnostic_features(
        data_directory / "train_valid_sequences.csv", run["config"]["data"]["maxlen"]
    )
    with np.load(directory / "validation_predictions.npz", allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    bind_predictions(values, population, full=True)
    return {
        "status": "post_selection_validation_diagnostics",
        "run_sha256": digest(directory / "run.json"),
        "result_sha256": digest(directory / "result.json"),
        "prediction_sha256": digest(directory / "validation_predictions.npz"),
        "diagnostic_source_sha256": digest(__file__),
        "overall": audit_report["recomputed_metrics"],
        "csv_population": counts,
        "groups": summarize(values, features),
        "support_scope": "training folds 1-4 scored targets only; same as prior initializer",
        "history_scope": "strictly past observed events in each segment, including position0",
        "test_access": False,
        "performance_gate_changed": False,
        "paper_goal_complete": False,
        "limits": [
            "Post-selection exploratory diagnostics, not held-out evaluation.",
            "Single-class groups have no defined AUC; they are never reported as zero.",
            "Group errors do not establish causal module deficiencies or positive ablations.",
            "No individual learner identifiers or responses are exported.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = diagnose(args.directory, args.source_root, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "overall": report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
