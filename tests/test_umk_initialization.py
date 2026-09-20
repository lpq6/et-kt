import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_production_initialization import compare_production_states


class UMKInitializationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "architecture": "v45_umk",
            "model": {"use_umk_ssm": 1, "use_umk_attn": 1, "use_umk_rwce": 1},
        }
        self.parent = {"common": torch.tensor([0.1, 0.2])}
        self.saved = {
            "umk_beta": torch.tensor(0.5),
            "common": self.parent["common"].clone(),
            "ssm.umk_alpha": torch.tensor(0.1),
        }

    def test_only_declared_scalar_additions_are_accepted(self):
        self.assertEqual(
            compare_production_states(self.saved, self.parent, self.config),
            {"ssm.umk_alpha": 0.1, "umk_beta": 0.5},
        )

    def test_trained_common_weights_wrong_scalars_and_extra_state_fail(self):
        for name, value in (
            ("common", torch.zeros(2)),
            ("umk_beta", torch.tensor(0.51)),
            ("ssm.umk_alpha", torch.tensor([0.1])),
            ("undeclared", torch.tensor(1.0)),
        ):
            changed = {**self.saved, name: value}
            with self.subTest(name=name), self.assertRaises(ValueError):
                compare_production_states(changed, self.parent, self.config)
        for name in ("ssm.umk_alpha", "umk_beta"):
            changed = dict(self.saved)
            del changed[name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                compare_production_states(changed, self.parent, self.config)

    def test_legacy_and_all_off_require_exact_original_keys(self):
        for config in (
            {"architecture": "v43"},
            {"architecture": "v45_umk", "model": {}},
        ):
            self.assertEqual(compare_production_states(self.parent, self.parent, config), {})
            with self.assertRaises(ValueError):
                compare_production_states(self.saved, self.parent, config)

    def test_disabled_scalar_cannot_be_silently_retained(self):
        config = copy.deepcopy(self.config)
        config["model"]["use_umk_ssm"] = 0
        with self.assertRaises(ValueError):
            compare_production_states(self.saved, self.parent, config)


if __name__ == "__main__":
    unittest.main()
