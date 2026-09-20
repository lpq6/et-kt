import math
import unittest

import torch

from a2g.candidate import A2GModal
from a2g.modules.evidence import EvidenceMethods
from a2g.transfer import A2GTransfer
from test_model import aligned, build
from v27_audit_helpers import timed_batch


class TransferResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_disabled_transfer_is_v43_exact(self):
        parent = build(A2GModal)
        torch.manual_seed(42)
        candidate = A2GTransfer(
            6,
            9,
            d_model=32,
            d_ff=64,
            n_blocks=4,
            num_attn_heads=4,
            dropout=0.2,
            seq_len=200,
            item_residual_dropout=0.4,
        )
        with torch.no_grad():
            candidate.item_support_count.fill_(64)
            candidate.item_prior.weight.fill_(0.15)
            candidate.concept_prior.weight.fill_(-0.07)
        self.assertEqual(list(parent.state_dict()), list(candidate.state_dict()))
        candidate.load_state_dict(parent.state_dict(), strict=True)
        expected, expected_trace = aligned(parent, timed_batch("cpu"))
        actual, actual_trace = aligned(candidate, timed_batch("cpu"))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual_trace, expected_trace)

    def test_transfer_residual_matches_strict_past_oracle(self):
        target_q = torch.tensor([[1, 1, 3, 4]], dtype=torch.long)
        target_c = torch.tensor([[2, 2, 2, 2]], dtype=torch.long)
        hist_q = torch.tensor([[0, 1, 2, 1]], dtype=torch.long)
        hist_c = torch.tensor([[0, 2, 2, 2]], dtype=torch.long)
        hist_r = torch.tensor([[2, 1, 0, 1]], dtype=torch.long)
        hist_logit = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
        actual = EvidenceMethods._transfer_calibrated_concept_residual(
            target_q,
            target_c,
            hist_q,
            hist_c,
            hist_r,
            hist_logit,
        )
        expected = torch.zeros(1, 4, dtype=torch.float64)
        # At target index 3, the aligned history includes positions 1, 2, 3.
        # Every historical item is balanced by its prefix repeat count.
        mass = 0.5 + 1.0 + 1.0 / math.sqrt(2.0)
        signed = (
            0.5 * 0.5
            + 1.0 * -0.5
            + (1.0 / math.sqrt(2.0)) * 0.5
        )
        expected[0, 3] = signed / (1.0 + mass)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(hist_logit.grad).all())

    def test_transfer_only_reads_past_and_scalar_gets_gradient(self):
        model = build(A2GTransfer, use_transfer_residual=1).double().eval()
        batch = timed_batch("cpu")
        expected = model(batch)
        changed = {name: value.clone() for name, value in batch.items()}
        changed["rseqs"][:, 4:] = 1 - changed["rseqs"][:, 4:]
        changed["shft_rseqs"][:] = 1 - changed["shft_rseqs"]
        changed["qseqs"][:, 5:] = 3
        changed["shft_qseqs"][:, 4:] = 3
        changed["cseqs"][:, 5:] = 2
        changed["shft_cseqs"][:, 4:] = 2
        torch.testing.assert_close(
            expected[:, :5], model(changed)[:, :5], rtol=0, atol=0
        )
        model(batch).sum().backward()
        self.assertIsNotNone(model.transfer_gate.grad)
        self.assertTrue(torch.isfinite(model.transfer_gate.grad))
        self.assertGreater(model.transfer_gate.grad.abs().item(), 0)

    def test_transfer_changes_enabled_forward_without_changing_width(self):
        batch = timed_batch("cpu")
        reference = build(A2GTransfer).eval()(batch)
        candidate = build(A2GTransfer, use_transfer_residual=1).eval()
        prediction, fused = candidate(batch, qtest=True)
        self.assertEqual(fused.shape, (*prediction.shape, 38))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertGreater((prediction - reference).abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main()
