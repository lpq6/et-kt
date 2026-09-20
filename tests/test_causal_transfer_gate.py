import copy
import unittest

import torch

from a2g.experiment import load_config, make_model
from a2g.modules.evidence import EvidenceMethods
from a2g.transfer_gate import A2GCausalTransferGate
from test_model import aligned, build
from v27_audit_helpers import timed_batch


class CausalTransferGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_signal_matches_strict_past_and_is_bounded(self):
        target_q = torch.tensor([[1, 3, 4, 4]], dtype=torch.long)
        target_c = torch.tensor([[2, 2, 2, 2]], dtype=torch.long)
        hist_q = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        hist_c = torch.tensor([[0, 2, 2, 2]], dtype=torch.long)
        hist_r = torch.tensor([[2, 1, 0, 1]], dtype=torch.long)
        hist_logit = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
        actual = EvidenceMethods._causal_transfer_gate_signal(
            target_q, target_c, hist_q, hist_c, hist_r, hist_logit
        )

        # The last target receives only history positions 1 and 2:
        # position 3 is the current target's immediately preceding event,
        # but it is the same item and therefore excluded from transfer.
        weights = torch.tensor([0.5, 1.0, 1.0], dtype=torch.float64)
        residual = torch.tensor([0.5, -0.5, 0.5], dtype=torch.float64)
        mass = weights.sum()
        raw = (weights * residual).sum() / (1.0 + mass)
        novelty = torch.rsqrt(torch.tensor(1.0, dtype=torch.float64))
        expected = torch.zeros(1, 4, dtype=torch.float64)
        # At target index 1, the single preceding event is a different item
        # with the same concept.  At index 2 the two residuals cancel.
        one_mass = torch.tensor(1.0, dtype=torch.float64)
        one_signed = torch.tensor(0.5, dtype=torch.float64)
        expected[0, 1] = (
            one_signed / (1.0 + one_mass)
            * novelty
        )
        expected[0, 3] = raw * novelty
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
        self.assertTrue(torch.isfinite(actual).all())
        actual.sum().backward()
        self.assertTrue(torch.isfinite(hist_logit.grad).all())

    def test_enabled_gate_changes_logit_without_changing_width(self):
        batch = timed_batch("cpu")
        reference = build(A2GCausalTransferGate).eval()(batch)
        candidate = build(
            A2GCausalTransferGate, use_causal_transfer_gate=1
        ).eval()
        prediction, fused = candidate(batch, qtest=True)
        self.assertEqual(fused.shape, (*prediction.shape, 38))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertGreater((prediction - reference).abs().max().item(), 0)
        prediction.sum().backward()
        self.assertIsNotNone(candidate.causal_transfer_gate.grad)
        self.assertTrue(torch.isfinite(candidate.causal_transfer_gate.grad))
        self.assertGreater(candidate.causal_transfer_gate.grad.abs().item(), 0)

    def test_enabled_gate_reads_only_strict_past(self):
        model = build(
            A2GCausalTransferGate, use_causal_transfer_gate=1
        ).double().eval()
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

    def test_disabled_constructor_has_no_extra_scalar(self):
        model = build(A2GCausalTransferGate)
        self.assertNotIn("causal_transfer_gate", model.state_dict())

    def test_matched_control_reuses_full_initial_state(self):
        root = __import__("pathlib").Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/assist2017_v48_causal_transfer.json")
        config["model"].update(d_model=32, d_ff=64, num_attn_heads=4)
        config["data"].update(num_c=6, num_q=9)
        torch.manual_seed(42)
        full = make_model(copy.deepcopy(config), "full")
        torch.manual_seed(42)
        control = make_model(copy.deepcopy(config), "no_causal_transfer_gate")
        self.assertEqual(list(full.state_dict()), list(control.state_dict()))
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(full.use_causal_transfer_gate)
        self.assertFalse(control.use_causal_transfer_gate)
        expected, full_trace = aligned(full, timed_batch("cpu"))
        actual, control_trace = aligned(control, timed_batch("cpu"))
        self.assertEqual(full_trace, control_trace)
        self.assertGreater((expected - actual).abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main()
