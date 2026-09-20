import csv
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from profile_history_alignment import profile  # noqa: E402


class HistoryProfileTests(unittest.TestCase):
    def test_training_prefix_counts_and_validation_labels_are_not_parsed(self):
        rows = [
            dict(uid="a", fold=1, selectmasks="1,1,1,1", concepts="2,2,3,2", responses="1,0,1,0"),
            dict(uid="b", fold=2, selectmasks="1,1,-1,-1", concepts="3,3,-1,-1", responses="0,1,-1,-1"),
            dict(uid="v", fold=0, selectmasks="invalid", concepts="invalid", responses="not_parsed"),
        ]
        with tempfile.TemporaryDirectory(prefix="a2g_history_profile_") as temporary:
            path = Path(temporary) / "train_valid_sequences.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            config = {
                "data": {"num_c": 4, "maxlen": 4},
                "train_folds": [1, 2, 3, 4],
                "data_binding": {
                    "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "counts": {"train": {"segments": 2, "learners": 2, "scored_interactions": 4}},
                },
            }
            result = profile(path, config)
            self.assertEqual(result["event_counts"]["latest_observation_matches_target_concept"], 2)
            self.assertEqual(result["event_counts"]["old_window_appeared_unseen"], 2)
            self.assertEqual(result["event_counts"]["accuracy_changed"], 1)
            self.assertEqual(result["fraction_latest_concept_match"], 0.5)
            self.assertEqual(result["mean_absolute_feature_change"]["accuracy"], 0.25)
            self.assertFalse(result["validation_responses_parsed"])
            config["data_binding"]["csv_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                profile(path, config)


if __name__ == "__main__":
    unittest.main()
