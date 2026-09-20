import csv
import tempfile
from pathlib import Path
import unittest

import torch

from a2g.data import TrainValidationDataset
from a2g.experiment import initialize_prior, load_config
from a2g.model import A2G
from a2g.uid import load_uid_segment_metadata_csv, UIDFoldIsolationError


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "train_valid_sequences.csv"
        self.config = {
            "num_q": 6,
            "num_c": 5,
            "maxlen": 4,
            "input_type": ["questions", "concepts"],
        }
        self.rows = [
            {
                "uid": "train",
                "fold": "1",
                "questions": "2,3,2,-1",
                "concepts": "2,3,2,-1",
                "responses": "1,0,1,-1",
                "selectmasks": "1,1,1,-1",
                "timestamps": "1000,2000,3000,-1",
            },
            {
                "uid": "valid",
                "fold": "0",
                "questions": "1,2,3,4",
                "concepts": "1,2,3,2",
                "responses": "0,1,0,1",
                "selectmasks": "1,1,1,1",
                "timestamps": "1000,2000,3000,4000",
            },
        ]
        self.write()

    def write(self):
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)

    def test_shift_masks_reserved_ids_and_counts(self):
        train = TrainValidationDataset(self.path, self.config, [1, 2, 3, 4])
        valid = TrainValidationDataset(self.path, self.config, [0])
        self.assertEqual(train.scored_interactions, 2)
        self.assertEqual(valid.scored_interactions, 3)
        self.assertEqual(train[0]["qseqs"].tolist(), [2, 3, 0])
        self.assertEqual(train[0]["shft_qseqs"].tolist(), [3, 2, 0])
        self.assertEqual(valid[0]["qseqs"][0], 1)
        self.assertEqual(train[0]["tseqs"].dtype, torch.int64)

    def test_oov_in_training_fails(self):
        self.rows[0]["questions"] = "1,3,2,-1"
        self.write()
        with self.assertRaises(ValueError):
            TrainValidationDataset(self.path, self.config, [1])

    def test_mixed_fold_and_test_files_fail(self):
        with self.assertRaises(ValueError):
            TrainValidationDataset(self.path, self.config, [0, 1])
        with self.assertRaises(ValueError):
            TrainValidationDataset(
                self.path.with_name("test_sequences.csv"), self.config, [0]
            )

    def test_learner_fold_overlap_fails(self):
        self.rows[1]["uid"] = "train"
        self.write()
        with self.assertRaises(UIDFoldIsolationError):
            load_uid_segment_metadata_csv(
                self.path,
                dataset_id="assist2017",
                split="train",
                allowed_folds=[1, 2, 3, 4],
                sequence_width=4,
            )

    def test_prior_uses_only_shifted_training_targets(self):
        torch.manual_seed(42)
        model = A2G(5, 6, d_model=16, d_ff=32, n_blocks=2, num_attn_heads=2, seq_len=4)
        train = TrainValidationDataset(self.path, self.config, [1])
        report = initialize_prior(model, train)
        self.assertEqual(report["selected_interactions"], 2)
        self.assertEqual(model.item_support_count.tolist(), [0, 0, 1, 1, 0, 0, 0])
        before = model.item_prior.weight.detach().clone()
        self.rows[1]["responses"] = "1,0,1,0"
        self.write()
        initialize_prior(model, TrainValidationDataset(self.path, self.config, [1]))
        torch.testing.assert_close(model.item_prior.weight, before, rtol=0, atol=0)

    def test_both_production_configs_validate(self):
        root = Path(__file__).resolve().parents[1]
        for version in ("v33", "v43"):
            config = load_config(root / f"configs/assist2017_{version}.json")
            self.assertEqual(
                config["data_binding"]["counts"]["train"]["scored_interactions"], 606414
            )


if __name__ == "__main__":
    unittest.main()
