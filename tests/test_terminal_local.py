import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_terminal_local import verify_exit, verify_supporting_evidence  # noqa: E402
from audit_protocol import independent_metrics  # noqa: E402


class TerminalLocalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.version = self.root / "version"
        self.version.mkdir()
        self.run_dir = self.version / "runs/full"
        self.config = self.version / "config.json"
        self.config.write_text('{"seed":42}', encoding="utf-8")
        self.run = {
            "runtime": {"device": "cuda"}, "config": {"seed": 42},
            "initial_state_sha256": "initial", "prior": {"scope": "train"},
            "data": {"binding": "frozen"},
        }
        self.process = {
            "status": "exited", "return_code": 0, "automatic_retries": 0,
            "ended_at": "2026-09-18", "cwd": str(self.version),
            "command": [
                "python", "-u", str(self.version / "run.py"), "train",
                "--config", str(self.config), "--output", str(self.run_dir),
                "--device", "cuda", "--data-dir", str(self.root / "data"),
            ],
        }
        self.result = {
            "metrics": independent_metrics([0, 1, 0, 1], [0.2, 0.7, 0.6, 0.8]),
            "checkpoint_sha256": "model", "predictions_sha256": "predictions",
        }
        self.initialization = {
            "status": "server_original_production_initialization_exact",
            "initial_state_sha256": "initial", "trained_checkpoint_loaded": False,
            "gpu_used": False, "prior": self.run["prior"], "data_binding": self.run["data"],
        }
        self.reevaluation = {
            "status": "checkpoint_and_csv_bound_predictions_verified",
            "run_mode": "train", "data": self.run["data"],
            "exact_prediction_fields": {
                name: True for name in ("label", "probability", "learner_uid", "csv_row_index", "position")
            },
            "metrics": self.result["metrics"], "run_sha256": "run",
            "result_sha256": "result", "checkpoint_sha256": "model",
            "predictions_sha256": "predictions",
        }

    def check_evidence(self, initialization, reevaluation):
        verify_supporting_evidence(
            initialization, reevaluation, self.run, self.result,
            run_sha256="run", result_sha256="result",
        )

    def test_successful_exit_is_bound_to_the_actual_run(self):
        self.assertEqual(
            verify_exit(self.process, self.run_dir, self.version, self.run),
            (self.root / "data").resolve(),
        )

    def test_running_failed_and_wrong_commands_are_rejected(self):
        for key, value in (
            ("status", "running"), ("return_code", 1),
            ("return_code", False), ("automatic_retries", 1), ("ended_at", None),
        ):
            record = {**self.process, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_exit(record, self.run_dir, self.version, self.run)
        altered = copy.deepcopy(self.process)
        altered["command"][3] = "smoke"
        with self.assertRaises(ValueError):
            verify_exit(altered, self.run_dir, self.version, self.run)
        with self.assertRaises(ValueError):
            verify_exit(self.process, self.root / "another", self.version, self.run)

    def test_changed_command_configuration_is_rejected(self):
        self.config.write_text(json.dumps({"seed": 43}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "configuration"):
            verify_exit(self.process, self.run_dir, self.version, self.run)

    def test_independent_evidence_is_accepted_only_with_artifact_bindings(self):
        self.check_evidence(self.initialization, self.reevaluation)
        for key in ("run_sha256", "result_sha256", "checkpoint_sha256", "predictions_sha256"):
            altered = copy.deepcopy(self.reevaluation)
            del altered[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check_evidence(self.initialization, altered)
            altered[key] = "another_run"
            with self.assertRaises(ValueError):
                self.check_evidence(self.initialization, altered)

    def test_smoke_or_mismatched_initialization_evidence_is_rejected(self):
        altered = {**self.initialization, "initial_state_sha256": "different"}
        with self.assertRaises(ValueError):
            self.check_evidence(altered, self.reevaluation)
        altered = {**self.reevaluation, "run_mode": "smoke"}
        with self.assertRaises(ValueError):
            self.check_evidence(self.initialization, altered)
        altered = copy.deepcopy(self.reevaluation)
        altered["exact_prediction_fields"]["label"] = False
        with self.assertRaises(ValueError):
            self.check_evidence(self.initialization, altered)

    def test_umk_requires_separate_common_state_and_scalar_evidence(self):
        self.run["config"].update(
            architecture="v45_umk", model={"use_umk_ssm": 1, "use_umk_attn": 1}
        )
        with self.assertRaises(ValueError):
            self.check_evidence(self.initialization, self.reevaluation)
        initialization = {
            **self.initialization,
            "status": "server_original_plus_declared_umk_scalars_initialization_exact",
            "server_common_state_exact": True,
            "declared_new_scalar_parameters": {"ssm.umk_alpha": 0.1, "umk_beta": 0.5},
        }
        self.check_evidence(initialization, self.reevaluation)
        for key, value in (
            ("server_common_state_exact", False),
            ("declared_new_scalar_parameters", {"ssm.umk_alpha": 0.1}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check_evidence({**initialization, key: value}, self.reevaluation)


if __name__ == "__main__":
    unittest.main()
