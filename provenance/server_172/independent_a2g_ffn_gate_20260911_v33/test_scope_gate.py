"""Synthetic CPU-only regression tests for the user's pilot-first rule."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch_candidate as gate


def protocol():
    return {
        "candidate_id": "synthetic_untrained_candidate",
        "phase": "assist2017_full_only",
        "datasets": ["assist2017"], "dataset_order": ["assist2017"],
        "eligible_datasets": list(gate.FLOORS),
        "reference_auc": copy.deepcopy(gate.FLOORS),
        "seeds": [42], "train_folds": [1, 2, 3, 4], "validation_folds": [0],
        "controls": [], "execution_order": ["full"],
        "test_access": False, "window_test_access": False,
        "baseline_training": {"run": False, "incumbent_rerun": False},
        "variants": {
            "full": {"use_attention_branch": 1, "use_evidence_branch": 1, "use_ssm_branch": 1}
        },
        "global_admission": {
            "eligible_dataset_count": 8, "required_dataset_count": 5,
            "same_datasets_for_baselines_and_all_modules": True,
            "mandatory_assist2017_precondition": True,
        },
        "assist2017_first_gate": {
            "strict_auc_floor": 0.8174,
            "independent_terminal_audit_required": True,
            "other_datasets_executable_in_this_package": False,
            "expansion_requires_new_frozen_package": True,
        },
        "training": {
            "pretrained_checkpoint": None,
            "initialization": "random_all_model_parameters_plus_training_only_empirical_priors",
            "max_epochs": 200, "patience": 20, "optimizer": "Adam",
            "learning_rate": 0.0001, "weight_decay": 0.0,
            "batch_size": 64, "evaluation_batch_size": 128,
            "batch_order": "fixed_csv_filtered_order", "deterministic_algorithms": True,
        },
    }


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.value = protocol()

    def validate(self):
        gate.validate_protocol(self.value, "assist2017", 42, "full")

    def test_assist2017_only_is_accepted(self):
        self.validate()

    def test_every_other_dataset_is_rejected_before_file_access(self):
        for dataset in gate.FLOORS.keys() - {"assist2017"}:
            with self.subTest(dataset=dataset), self.assertRaisesRegex(
                gate.ScopeError, "Assist2017 must pass first"
            ):
                gate.validate_package("nonexistent-package", dataset)

    def test_other_seed_is_rejected(self):
        with self.assertRaises(gate.ScopeError):
            gate.validate_protocol(self.value, "assist2017", 43, "full")

    def test_ablation_is_not_automatically_launched(self):
        with self.assertRaises(gate.ScopeError):
            gate.validate_protocol(self.value, "assist2017", 42, "no_attention")

    def test_multidataset_package_is_rejected(self):
        self.value["datasets"].append("statics2011")
        with self.assertRaisesRegex(gate.ScopeError, "datasets"):
            self.validate()

    def test_multidataset_order_is_rejected(self):
        self.value["dataset_order"].append("statics2011")
        with self.assertRaisesRegex(gate.ScopeError, "dataset_order"):
            self.validate()

    def test_wrong_validation_fold_is_rejected(self):
        self.value["validation_folds"] = [1]
        with self.assertRaisesRegex(gate.ScopeError, "validation_folds"):
            self.validate()

    def test_every_reference_floor_is_immutable(self):
        for dataset in gate.FLOORS:
            with self.subTest(dataset=dataset):
                value = protocol()
                value["reference_auc"][dataset] -= 0.001
                with self.assertRaisesRegex(gate.ScopeError, "reference floors"):
                    gate.validate_protocol(value, "assist2017", 42, "full")

    def test_five_of_eight_objective_is_not_reduced(self):
        self.value["global_admission"]["required_dataset_count"] = 1
        with self.assertRaisesRegex(gate.ScopeError, "required_dataset_count"):
            self.validate()

    def test_other_datasets_remain_eligible_but_not_executable(self):
        self.value["eligible_datasets"].remove("assist2015")
        with self.assertRaisesRegex(gate.ScopeError, "eight-choose-five"):
            self.validate()

    def test_baseline_training_is_rejected(self):
        self.value["baseline_training"]["run"] = True
        with self.assertRaisesRegex(gate.ScopeError, "retraining"):
            self.validate()

    def test_existing_candidate_cannot_be_rerun(self):
        for candidate in gate.CLOSED_OR_REFERENCE:
            with self.subTest(candidate=candidate):
                self.value["candidate_id"] = candidate
                with self.assertRaisesRegex(gate.ScopeError, "new candidate"):
                    self.validate()

    def test_checkpoint_initialization_is_rejected(self):
        self.value["training"]["pretrained_checkpoint"] = "trained.pt"
        with self.assertRaisesRegex(gate.ScopeError, "pretrained_checkpoint"):
            self.validate()

    def test_schedule_change_is_rejected(self):
        self.value["training"]["patience"] = 30
        with self.assertRaisesRegex(gate.ScopeError, "patience"):
            self.validate()

    def test_raw_auc_is_not_sufficient_to_unlock_other_datasets(self):
        self.value["assist2017_first_gate"]["independent_terminal_audit_required"] = False
        with self.assertRaisesRegex(gate.ScopeError, "terminal_audit"):
            self.validate()

    def test_current_package_never_unblocks_cross_dataset_execution(self):
        self.value["assist2017_first_gate"]["other_datasets_executable_in_this_package"] = True
        with self.assertRaisesRegex(gate.ScopeError, "other_datasets"):
            self.validate()

    def test_test_access_is_rejected(self):
        self.value["test_access"] = True
        with self.assertRaisesRegex(gate.ScopeError, "test_access"):
            self.validate()

    def test_malformed_nested_protocol_is_rejected(self):
        for field in ("global_admission", "assist2017_first_gate", "training"):
            with self.subTest(field=field):
                value = protocol()
                value[field] = None
                with self.assertRaises(gate.ScopeError):
                    gate.validate_protocol(value, "assist2017", 42, "full")


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("protocol.json", protocol())
        for name in ("run_independent.py", "a2g_single_trunk_candidate.py", "gpu_guard.py"):
            (self.root / name).write_text("# synthetic non-executable fixture\n", encoding="utf-8")
        for name in ("cpu_contract", "data_contract", "gpu_guard_contract", "runner_contract"):
            self.write(f"{name}.json", {"status": "pass", "checks": {"synthetic": True}})
        self.freeze()
        guard = patch.object(gate, "GUARD_SHA256", gate.digest(self.root / "gpu_guard.py"))
        guard.start()
        self.addCleanup(guard.stop)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def freeze(self):
        self.manifest = {
            "status": "frozen_before_training",
            "candidate_id": "synthetic_untrained_candidate",
            "sha256": {
                path.name: gate.digest(path) for path in self.root.iterdir()
                if path.name != "manifest.json" and path.is_file()
            },
        }
        self.write("manifest.json", self.manifest)

    def test_valid_synthetic_package_only_checks_scope(self):
        with patch.object(gate.subprocess, "run", side_effect=AssertionError("training launched")):
            report = gate.validate_package(self.root)
        self.assertFalse(report["gpu_used"])
        self.assertFalse(report["training_launched"])
        self.assertFalse(report["paper_goal_complete"])
        self.assertFalse(report["other_datasets_executable"])

    def test_source_mutation_is_rejected(self):
        (self.root / "run_independent.py").write_text("# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(gate.ScopeError, "manifest binding"):
            gate.validate_package(self.root)

    def test_missing_contract_binding_is_rejected(self):
        del self.manifest["sha256"]["runner_contract.json"]
        self.write("manifest.json", self.manifest)
        with self.assertRaisesRegex(gate.ScopeError, "missing"):
            gate.validate_package(self.root)

    def test_failed_contract_is_rejected(self):
        self.write("runner_contract.json", {"status": "fail"})
        self.freeze()
        with self.assertRaisesRegex(gate.ScopeError, "has not passed"):
            gate.validate_package(self.root)

    def test_false_check_cannot_hide_behind_pass_status(self):
        self.write("cpu_contract.json", {"status": "pass", "checks": {"causal": False}})
        self.freeze()
        with self.assertRaisesRegex(gate.ScopeError, "failed checks"):
            gate.validate_package(self.root)

    def test_existing_pilot_artifacts_prevent_retry(self):
        (self.root / "artifacts/assist2017/seed42").mkdir(parents=True)
        with self.assertRaisesRegex(gate.ScopeError, "no silent retry"):
            gate.validate_package(self.root)

    def test_unfrozen_package_is_rejected(self):
        self.manifest["status"] = "draft"
        self.write("manifest.json", self.manifest)
        with self.assertRaisesRegex(gate.ScopeError, "pretraining manifest"):
            gate.validate_package(self.root)

    def test_resource_guard_mutation_is_rejected_even_after_refreezing(self):
        (self.root / "gpu_guard.py").write_text("# changed guard\n", encoding="utf-8")
        self.freeze()
        with self.assertRaisesRegex(gate.ScopeError, "ownership guard"):
            gate.validate_package(self.root)

    def test_external_manifest_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as external:
            path = Path(external) / "outside.py"
            path.write_text("# fixture\n", encoding="utf-8")
            self.manifest["sha256"][str(path)] = gate.digest(path)
            self.write("manifest.json", self.manifest)
            with self.assertRaisesRegex(gate.ScopeError, "manifest binding"):
                gate.validate_package(self.root)

    def test_check_only_cli_does_not_launch(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(gate.subprocess, "run", side_effect=AssertionError("training launched")),
        ):
            code = gate.main(["--package", str(self.root), "--check-only"])
        self.assertEqual(code, 0)

    def test_forbidden_execute_cli_never_reaches_subprocess(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(gate.subprocess, "run", side_effect=AssertionError("training launched")),
        ):
            code = gate.main([
                "--package", str(self.root), "--dataset", "statics2011", "--execute",
            ])
        self.assertEqual(code, 2)

    def test_malformed_json_cli_returns_blocked_without_launch(self):
        (self.root / "protocol.json").write_text("{", encoding="utf-8")
        stream = io.StringIO()
        with (
            contextlib.redirect_stdout(stream),
            patch.object(gate.subprocess, "run", side_effect=AssertionError("training launched")),
        ):
            code = gate.main(["--package", str(self.root), "--check-only"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stream.getvalue())["status"], "execution_scope_blocked")


if __name__ == "__main__":
    unittest.main()
