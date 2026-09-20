"""Measure the history-window change on training folds only; no model selection."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from audit_protocol import sha256


def profile(path, config):
    path = Path(path)
    if path.name != "train_valid_sequences.csv":
        raise ValueError("only the frozen train/validation CSV is permitted")
    if sha256(path) != config["data_binding"]["csv_sha256"]:
        raise ValueError("frozen CSV hash mismatch")
    width = config["data"]["maxlen"]
    learners, segments, scored = set(), 0, 0
    counts = Counter()
    changes = Counter()
    fields = ("accuracy", "log_exposure", "smoothed_success", "smoothed_failure", "gap")
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["fold"]) not in config["train_folds"]:
                continue
            selection, concepts, responses = [
                [int(value) for value in row[name].split(",")]
                for name in ("selectmasks", "concepts", "responses")
            ]
            if any(len(value) != width for value in (selection, concepts, responses)):
                raise ValueError("training sequence width mismatch")
            valid = selection.count(1)
            if valid < 2 or selection != [1] * valid + [-1] * (width - valid):
                raise ValueError("invalid selection prefix")
            if any(c < 2 or c >= config["data"]["num_c"] for c in concepts[:valid]):
                raise ValueError("training concept outside known vocabulary")
            if any(response not in (0, 1) for response in responses[:valid]):
                raise ValueError("training responses must be binary")
            learners.add(row["uid"].strip())
            segments += 1
            seen, success, last = Counter(), Counter(), {}
            for t in range(1, valid):
                c, previous = concepts[t], concepts[t - 1]
                old_n, old_s, old_last = seen[c], success[c], last.get(c, 0)
                seen[previous] += 1
                success[previous] += responses[t - 1]
                last[previous] = t
                new_n, new_s, new_last = seen[c], success[c], last.get(c, 0)
                scored += 1
                if previous == c:
                    counts["latest_observation_matches_target_concept"] += 1
                    counts["old_window_appeared_unseen"] += int(old_n == 0)
                old = (
                    old_s / max(1, old_n),
                    math.log1p(old_n) / math.log1p(max(2, width)),
                    old_s / (old_n + 1),
                    (old_n - old_s) / (old_n + 1),
                    (t - old_last) / width,
                )
                new = (
                    new_s / max(1, new_n),
                    math.log1p(new_n) / math.log1p(max(2, width)),
                    new_s / (new_n + 1),
                    (new_n - new_s) / (new_n + 1),
                    (t - new_last) / width,
                )
                for name, before, after in zip(fields, old, new):
                    counts[f"{name}_changed"] += int(before != after)
                    changes[name] += abs(after - before)
    population = {
        "segments": segments, "learners": len(learners), "scored_interactions": scored,
    }
    if population != config["data_binding"]["counts"]["train"]:
        raise ValueError("training population differs from the frozen contract")
    return {
        "status": "training_only_history_window_profile",
        "csv_sha256": sha256(path),
        "population": population,
        "event_counts": dict(counts),
        "fraction_latest_concept_match": (
            counts["latest_observation_matches_target_concept"] / scored
        ),
        "mean_absolute_feature_change": {
            name: changes[name] / scored for name in fields
        },
        "validation_responses_parsed": False,
        "test_access": False,
        "auc_gain_claimed": False,
        "limits": [
            "Describes input statistics only; does not establish a causal AUC improvement.",
            "Counts reset at every segment; no cross-segment student state.",
            "Gap remains in shifted-history coordinates to isolate the mask change.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = profile(args.data_dir / "train_valid_sequences.csv", config)
    report["profiler_sha256"] = sha256(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
