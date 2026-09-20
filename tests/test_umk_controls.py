import copy
from pathlib import Path
import unittest

import torch

from a2g.experiment import ABLATIONS, load_config, make_model
from test_model import aligned
from v27_audit_helpers import timed_batch


class UMKControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        root = Path(__file__).resolve().parents[1]
        cls.config = load_config(root / "configs/assist2017_v45_umk_all.json")
        cls.config["model"].update(d_model=32, d_ff=64, num_attn_heads=4)
        cls.config["data"].update(num_c=6, num_q=9)

    def model(self, variant="full"):
        torch.manual_seed(42)
        return make_model(copy.deepcopy(self.config), variant)

    def test_all_controls_have_exact_full_initial_state(self):
        full = self.model().state_dict()
        variants = [
            name
            for name in ABLATIONS
            if name
            not in {
                "no_aligned_history_statistics",
                "no_causal_transfer_gate",
                "no_evidence_equivalent_residual",
                "no_shared_embeddings",
            }
        ]
        self.assertEqual(len(variants), 18)
        for variant in variants:
            control = self.model(variant).state_dict()
            with self.subTest(variant=variant):
                self.assertEqual(list(control), list(full))
                for name in full:
                    torch.testing.assert_close(full[name], control[name], rtol=0, atol=0)

    def test_umk_controls_disable_only_the_requested_path(self):
        for variant, target in (
            ("no_umk_ssm", "use_umk_ssm"),
            ("no_umk_attn", "use_umk_attn"),
            ("no_umk_rwce", "use_umk_rwce"),
        ):
            full, control = self.model(), self.model(variant)
            with self.subTest(variant=variant):
                for flag in ("use_umk_ssm", "use_umk_attn", "use_umk_rwce"):
                    self.assertEqual(getattr(control, flag), flag != target)
                expected, full_trace = aligned(full, timed_batch("cpu"))
                actual, control_trace = aligned(control, timed_batch("cpu"))
                self.assertEqual(full_trace, control_trace)
                self.assertGreater((actual - expected).abs().max().item(), 0)
                actual.square().sum().backward()
                for name, parameter in (
                    ("use_umk_ssm", control.ssm.umk_alpha),
                    ("use_umk_attn", control.umk_beta),
                ):
                    if name == target:
                        self.assertIsNone(parameter.grad)
                    else:
                        self.assertGreater(parameter.grad.abs().item(), 0)
                self.assertTrue(all(
                    torch.isfinite(p.grad).all() for p in control.parameters() if p.grad is not None
                ))

    def test_plain_all_off_configuration_still_has_no_extra_scalars(self):
        config = copy.deepcopy(self.config)
        config["model"].update(use_umk_ssm=0, use_umk_attn=0, use_umk_rwce=0)
        model = make_model(config, "full")
        self.assertNotIn("ssm.umk_alpha", model.state_dict())
        self.assertNotIn("umk_beta", model.state_dict())


if __name__ == "__main__":
    unittest.main()
