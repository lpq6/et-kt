import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from a2g.experiment import load_config
from a2g.protocol import DIRECT_256, REFERENCE, validate_training_protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from audit_protocol import audit_epochs, audit_training_protocol  # noqa: E402
import test_audit_protocol  # noqa: E402


class TrainingProtocolTests(unittest.TestCase):
    def config(self, name="v43_lowmem"):
        return json.loads(
            (ROOT / f"configs/assist2017_{name}.json").read_text(encoding="utf-8")
        )

    def record(self, config):
        return {
            "config": config,
            "mode": "train",
            "training_protocol": config.get("training_protocol", REFERENCE),
            "effective_batch_size": config["training"]["batch_size"],
            "effective_evaluation_batch_size": 128,
            "gradient_accumulation_steps": 1,
            "microbatching": False,
        }

    def test_reference_and_explicit_batch256_are_distinct(self):
        reference, direct = self.config(), self.config("v43_lowmem_batch256")
        self.assertEqual(validate_training_protocol(reference), REFERENCE)
        self.assertEqual(validate_training_protocol(direct), DIRECT_256)
        self.assertEqual(
            {key: value for key, value in direct.items() if key != "training_protocol"},
            {
                **reference,
                "training": {**reference["training"], "batch_size": 256},
            },
        )
        for config in (reference, direct):
            record = self.record(config)
            self.assertEqual(
                validate_training_protocol(config), audit_training_protocol(record)
            )

    def test_batch_change_without_explicit_protocol_is_rejected(self):
        config = self.config()
        config["training"]["batch_size"] = 256
        for validator in (
            validate_training_protocol,
            lambda value: audit_training_protocol(self.record(value)),
        ):
            with self.assertRaises(ValueError):
                validator(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_optimizer_stopping_and_execution_drift_are_rejected(self):
        for key, value in [
            ("learning_rate", 0.0004),
            ("max_epochs", 800),
            ("evaluation_batch_size", 256),
            ("gradient_accumulation_steps", 4),
            ("batch_size", True),
        ]:
            config = self.config("v43_lowmem_batch256")
            config["training"][key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_training_protocol(config)
                with self.assertRaises(ValueError):
                    audit_training_protocol(self.record(config))
        for key, value in [
            ("effective_batch_size", 64),
            ("gradient_accumulation_steps", 4),
            ("microbatching", True),
            ("training_protocol", REFERENCE),
        ]:
            run = self.record(self.config("v43_lowmem_batch256"))
            run[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_training_protocol(run)

    def test_missing_and_unknown_protocol_are_rejected_for_batch256(self):
        for name in (None, "arbitrary_experiment"):
            config = self.config("v43_lowmem_batch256")
            config["training_protocol"] = name
            with self.assertRaises(ValueError):
                validate_training_protocol(config)
        record = self.record(self.config("v43_lowmem_batch256"))
        del record["gradient_accumulation_steps"]
        with self.assertRaises(ValueError):
            audit_training_protocol(record)

    def test_historical_batch64_run_keeps_its_original_contract(self):
        record = self.record(self.config())
        for key in ("training_protocol", "gradient_accumulation_steps", "microbatching"):
            del record[key]
        self.assertEqual(audit_training_protocol(record), REFERENCE)

    def test_each_epoch_records_the_actual_optimizer_step_count(self):
        for batch_size, expected in ((64, 56), (256, 14)):
            epochs, result, training, counts = (
                test_audit_protocol.IndependentAuditTests().fixture()
            )
            training["batch_size"] = batch_size
            counts["train"]["segments"] = 3582
            for row in epochs:
                row["optimizer_steps"] = expected
            audit_epochs(epochs, result, training, counts, full=True)
            altered = copy.deepcopy(epochs)
            altered[0]["optimizer_steps"] += 1
            with self.assertRaises(ValueError):
                audit_epochs(altered, result, training, counts, full=True)


if __name__ == "__main__":
    unittest.main()
