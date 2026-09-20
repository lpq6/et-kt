import copy
import csv
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from a2g.aligned_history import A2GAlignedHistory
from a2g.data import TrainValidationDataset
from a2g.experiment import initialize_prior


class NoItemDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix="a2g_no_item_")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "train_valid_sequences.csv"
        rows = [
            dict(uid="train", fold=1, concepts="2,2,3,2", responses="1,0,1,0", selectmasks="1,1,1,1"),
            dict(uid="valid", fold=0, concepts="1,2,3,-1", responses="0,1,0,-1", selectmasks="1,1,1,-1"),
        ]
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.data = TrainValidationDataset(
            self.path, {"num_c": 4, "num_q": 0, "maxlen": 4, "input_type": ["concepts"]}, [1]
        )
        torch.manual_seed(42)
        self.model = A2GAlignedHistory(
            4, 0, d_model=32, d_ff=64, n_blocks=4, num_attn_heads=4, seq_len=4
        )
        self.batch = next(iter(DataLoader(self.data, batch_size=1)))

    def test_concept_only_pipeline_has_finite_training_gradients(self):
        self.assertEqual(self.batch["qseqs"].numel(), 0)
        report = initialize_prior(self.model, self.data)
        self.assertEqual(report["selected_interactions"], 3)
        self.assertEqual(self.model.item_support_count.tolist(), [0.0])
        prediction = self.model(self.batch, train=True)[0]
        loss = torch.nn.functional.binary_cross_entropy(
            prediction[:, 1:].double(), self.batch["shft_rseqs"].double()
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None)
        )

    def test_item_attempt_is_provably_inactive_without_question_ids(self):
        with torch.no_grad():
            for parameter in self.model.attempt_readout.parameters():
                parameter.fill_(0.2)
        full = self.model.eval()
        control = copy.deepcopy(full)
        control.use_item_attempt_stage = False
        expected = full(self.batch)
        actual = control(self.batch)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        expected.sum().backward()
        for parameter in full.attempt_readout.parameters():
            if parameter.grad is not None:
                self.assertEqual(parameter.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
