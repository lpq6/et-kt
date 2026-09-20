import copy
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_protocol import (
    audit_epochs,
    bind_predictions,
    compare_metrics,
    independent_metrics,
    reconstruct_population,
)
from a2g.metrics import binary_metrics


class IndependentAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "train_valid_sequences.csv"
        self.rows = [
            {
                "uid": "train",
                "fold": "1",
                "responses": "0,1,0,-1",
                "selectmasks": "1,1,1,-1",
            },
            {
                "uid": "valid",
                "fold": "0",
                "responses": "1,0,1,0",
                "selectmasks": "1,1,1,1",
            },
        ]
        self.write()

    def write(self):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)

    def test_original_csv_row_and_position_order(self):
        population, counts = reconstruct_population(self.path, 4)
        self.assertEqual(population["label"].tolist(), [0, 1, 0])
        self.assertEqual(population["position"].tolist(), [1, 2, 3])
        self.assertEqual(population["csv_row_index"].tolist(), [1, 1, 1])
        self.assertEqual(counts["train"]["scored_interactions"], 2)
        bind_predictions(population, population, full=True)
        short = {name: array[:2] for name, array in population.items()}
        bind_predictions(short, population, full=False)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            bind_predictions(short, population, full=True)

    def test_rejects_key_or_label_mismatch(self):
        population, _ = reconstruct_population(self.path, 4)
        for name in population:
            with self.subTest(field=name):
                altered = {key: value.copy() for key, value in population.items()}
                altered[name][0] = (
                    "wrong" if name == "learner_uid" else int(altered[name][0]) + 1
                )
                with self.assertRaises(ValueError):
                    bind_predictions(altered, population, full=True)

    def test_mask_holes_and_learner_overlap_fail(self):
        self.rows[0]["selectmasks"] = "1,-1,1,-1"
        self.write()
        with self.assertRaisesRegex(ValueError, "valid prefix"):
            reconstruct_population(self.path, 4)
        self.rows[0]["selectmasks"] = "1,1,1,-1"
        self.rows[1]["uid"] = "train"
        self.write()
        with self.assertRaisesRegex(ValueError, "crosses folds"):
            reconstruct_population(self.path, 4)

    def test_independent_all_metrics_equal_training_scorer(self):
        random = np.random.default_rng(9)
        y = random.integers(0, 2, 2001)
        p = random.random(2001)
        p[:4] = [0, 1, 1e-10, 1 - 1e-10]
        actual = independent_metrics(y, p)
        compare_metrics(actual, binary_metrics(y, p))
        for name in actual:
            with self.subTest(metric=name):
                incorrect = {**actual, name: actual[name] + 0.01}
                with self.assertRaises(ValueError):
                    compare_metrics(actual, incorrect)

    def fixture(self):
        metrics = independent_metrics([0, 0, 1, 1], [0.1, 0.3, 0.2, 0.8])
        epochs = []
        for epoch, auc in enumerate([0.75, 0.7, 0.74], 1):
            epochs.append(
                {
                    "epoch": epoch,
                    "best_epoch": 1,
                    "best_auc": 0.75,
                    "train_loss": 0.5,
                    "train_interactions": 2,
                    "validation": {**metrics, "auc": auc},
                    "smoke_only": False,
                    "duration_seconds": epoch * 10,
                }
            )
        result = {"epochs": 3, "best_epoch": 1, "metrics": metrics}
        training = {"max_epochs": 200, "patience": 2}
        counts = {
            "train": {"scored_interactions": 2},
            "validation": {"scored_interactions": 4},
        }
        return epochs, result, training, counts

    def test_complete_early_stop_is_accepted(self):
        self.assertEqual(audit_epochs(*self.fixture(), full=True), 1)

    def test_premature_stop_is_rejected(self):
        epochs, result, training, counts = self.fixture()
        result["epochs"] = 2
        with self.assertRaisesRegex(ValueError, "stopping rule"):
            audit_epochs(epochs[:2], result, training, counts, full=True)

    def test_invalid_epoch_coverage_and_history_are_rejected(self):
        for field, value in [
            ("train_interactions", 1),
            ("best_epoch", 2),
            ("best_auc", 0.99),
            ("smoke_only", True),
            ("train_loss", float("nan")),
            ("duration_seconds", -1),
        ]:
            epochs, result, training, counts = self.fixture()
            epochs[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_epochs(epochs, result, training, counts, full=True)

    def test_must_stop_at_first_patience_boundary(self):
        epochs, result, training, counts = self.fixture()
        epochs.append({**copy.deepcopy(epochs[-1]), "epoch": 4, "duration_seconds": 40})
        result["epochs"] = 4
        with self.assertRaisesRegex(ValueError, "continued beyond"):
            audit_epochs(epochs, result, training, counts, full=True)


if __name__ == "__main__":
    unittest.main()
