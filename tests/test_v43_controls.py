import copy
import json
from pathlib import Path
import unittest

import torch

from a2g.experiment import make_model, seed_all
from test_model import aligned
from v27_audit_helpers import timed_batch


class V43ControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        path = Path(__file__).resolve().parents[1] / "configs/assist2017_v43_lowmem.json"
        self.config = json.loads(path.read_text(encoding="utf-8"))
        self.config["data"].update(num_c=6, num_q=9)
        self.config["model"].update(d_model=32, d_ff=64, num_attn_heads=4)

    def build(self, variant):
        seed_all(42)
        model = make_model(self.config, variant)
        with torch.no_grad():
            model.item_support_count.fill_(64)
        return model

    def test_controls_regenerate_the_identical_full_initial_state(self):
        full = self.build("full")
        expected_rng = torch.get_rng_state()
        for variant in ("no_input_memory", "no_multimode_residual"):
            model = self.build(variant)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            for name, value in full.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)

    def test_each_control_removes_only_its_own_active_effect_and_gradient(self):
        full = self.build("full")
        with torch.no_grad():
            full.ssm.input_memory.weight.normal_(std=0.03)
            full.ssm.mode_input_weight.normal_(std=0.1)
            full.ssm.mode_read_weight.normal_(std=0.1)
        batch = timed_batch("cpu")
        expected, trace = aligned(full, batch)
        expected.sum().backward()
        full_params = dict(full.named_parameters())
        memory = ["ssm.input_memory.weight"]
        modal = ["ssm.mode_input_weight", "ssm.mode_read_weight"]
        for variant, removed, retained in (
            ("no_input_memory", memory, modal),
            ("no_multimode_residual", modal, memory),
        ):
            model = self.build(variant)
            model.load_state_dict(full.state_dict(), strict=True)
            actual, control_trace = aligned(model, batch)
            actual.sum().backward()
            self.assertEqual(trace, control_trace)
            self.assertGreater(float((expected - actual).abs().max()), 1e-7)
            parameters = dict(model.named_parameters())
            for name in removed:
                self.assertIsNone(parameters[name].grad, name)
                self.assertGreater(float(full_params[name].grad.abs().sum()), 0)
            for name in retained:
                self.assertIsNotNone(parameters[name].grad, name)
                self.assertGreater(float(parameters[name].grad.abs().sum()), 0)
            self.assertTrue(model.use_ssm_branch)
            self.assertTrue(model.use_attention_branch)

    def test_ssm_control_still_removes_both_nested_memory_paths(self):
        model = self.build("no_ssm")
        prediction, _ = aligned(model, timed_batch("cpu"))
        prediction.sum().backward()
        for name, parameter in model.ssm.named_parameters():
            self.assertIsNone(parameter.grad, name)

    def test_v33_rejects_v43_only_control_names(self):
        self.config = copy.deepcopy(self.config)
        self.config["architecture"] = "v33"
        for variant in ("no_input_memory", "no_multimode_residual"):
            with self.assertRaisesRegex(ValueError, "v43 architecture"):
                make_model(self.config, variant)


if __name__ == "__main__":
    unittest.main()
