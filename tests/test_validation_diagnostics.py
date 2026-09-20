import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("validation_diagnostics", TOOLS / "diagnose_validation.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class ValidationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.csv = Path(self.temporary.name) / "train_valid_sequences.csv"
        self.rows = [
            dict(uid="a", fold="1", questions="4,2,2,3", concepts="4,2,2,3",
                 responses="0,1,1,0", selectmasks="1,1,1,1"),
            dict(uid="b", fold="0", questions="2,2,1,3", concepts="2,2,1,3",
                 responses="0,1,1,0", selectmasks="1,1,1,1"),
            dict(uid="c", fold="0", questions="4,4,-1,-1", concepts="4,4,-1,-1",
                 responses="0,1,-1,-1", selectmasks="1,1,-1,-1"),
        ]
        self.write()

    def write(self):
        with self.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_support_matches_scored_training_targets_and_not_validation_counts(self):
        population, features, _ = diagnostics.diagnostic_features(self.csv, 4)
        self.assertEqual(population["label"].tolist(), [1, 1, 0, 1])
        self.assertEqual(features["item_support_band"].tolist(), ["1-4", "0", "1-4", "0"])
        self.assertEqual(features["concept_support_band"].tolist(), ["1-4", "0", "1-4", "0"])

    def test_history_is_strict_and_resets_at_each_segment(self):
        _, features, _ = diagnostics.diagnostic_features(self.csv, 4)
        self.assertEqual(features["concept_history_band"].tolist(), ["1-2", "OOV", "first", "1-2"])
        self.assertEqual(features["previous_response"].tolist(), ["0", "1", "1", "0"])
        self.assertEqual(features["concept_transition"].tolist(), ["same", "unknown", "unknown", "same"])

    def test_current_response_does_not_enter_its_own_features(self):
        _, before, _ = diagnostics.diagnostic_features(self.csv, 4)
        self.rows[1]["responses"] = "0,0,0,1"
        self.write()
        _, after, _ = diagnostics.diagnostic_features(self.csv, 4)
        for field in before:
            self.assertEqual(before[field][0], after[field][0], field)
        self.assertNotEqual(before["previous_response"][1], after["previous_response"][1])

    def test_aggregates_partition_population_without_inventing_single_class_auc(self):
        population, features, _ = diagnostics.diagnostic_features(self.csv, 4)
        values = {**population, "probability": np.array([0.7, 0.8, 0.4, 0.8])}
        report = diagnostics.summarize(values, features)
        for groups in report.values():
            self.assertEqual(sum(row["metrics"]["interaction_count"] for row in groups), 4)
            self.assertAlmostEqual(sum(row["fraction_of_scored_population"] for row in groups), 1)
        self.assertTrue(any(row["metrics"]["auc"] is None for row in report["item_support_band"]))

    def test_fold_overlap_is_rejected(self):
        self.rows[1]["uid"] = "a"
        self.write()
        with self.assertRaisesRegex(ValueError, "crosses folds"):
            diagnostics.diagnostic_features(self.csv, 4)


if __name__ == "__main__":
    unittest.main()
