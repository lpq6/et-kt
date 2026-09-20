"""Reject out-of-scope execution and any V32/new-parameter initialization drift."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_independent as runner
from a2g_incumbent import A2GMambaKT as Incumbent
from concept_graph_candidate import A2GMambaKT as V32
from ffn_gate_candidate import A2GMambaKT as Candidate
from launch_candidate import ScopeError


class RunnerScopeTests(unittest.TestCase):
    def test_assist2017_only(self):
        self.assertEqual(runner.verify_dataset_order("assist2017")["order"], ["assist2017"])
        for name in runner.PROTOCOL["eligible_datasets"]:
            if name != "assist2017":
                with self.subTest(dataset=name), self.assertRaises(ScopeError):
                    runner.verify_dataset_order(name)

    def test_other_data_cannot_be_loaded(self):
        with self.assertRaisesRegex(RuntimeError, "only Assist2017"):
            runner.dataset_config("statics2011")

    def test_modified_multidataset_protocol_is_rejected(self):
        value = copy.deepcopy(runner.PROTOCOL)
        value["datasets"].append("assist2015")
        with patch.object(runner, "PROTOCOL", value), self.assertRaises(ScopeError):
            runner.verify_dataset_order("assist2017")

    def test_cpu_audit_rejects_other_seed_before_io(self):
        with patch.object(runner, "verify_manifest", side_effect=AssertionError("IO reached")), self.assertRaises(ScopeError):
            runner.audit_production_cpu("assist2017", 43)

    def test_training_rejects_other_dataset_and_controls_before_io(self):
        for dataset, variant in (("assist2015", "full"), ("assist2017", "no_ssm")):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "unstarted"
                with self.assertRaises(ScopeError):
                    runner.train_variant({}, dataset, {}, {}, 42, {}, {}, output, variant)
                self.assertFalse(output.exists())

    def test_initialization_dispatch_cannot_bypass_scope(self):
        with self.assertRaises(ScopeError):
            runner.verify_dataset_initialization({}, {}, {}, "statics2011", 42)

    def test_initialization_dispatch_checks_both_controls(self):
        with (
            patch.object(runner, "verify_incumbent_initialization", return_value={"old": True}) as old,
            patch.object(runner, "verify_best_candidate", return_value={"v32": True}) as v32,
        ):
            result = runner.verify_dataset_initialization({}, {}, {}, "assist2017", 42)
        old.assert_called_once_with({})
        v32.assert_called_once_with({})
        self.assertEqual(result, ({"old": True}, {"v32": True}))


class InitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = copy.deepcopy(runner.PROTOCOL)
        context = patch.object(runner, "PROTOCOL", self.protocol)
        context.start()
        self.addCleanup(context.stop)
        kwargs = dict(self.protocol["model_kwargs"])
        kwargs.update(d_model=32, d_ff=64, num_attn_heads=4)
        self.protocol["model_kwargs"] = dict(kwargs)
        self.initial = self.state(Candidate, kwargs)
        kwargs.pop("use_ffn_gate")
        self.reference = self.state(V32, kwargs)
        reference_path = self.root / "v32_initial_state.pt"
        torch.save(self.reference, reference_path)
        binding = self.protocol["best_candidate"]
        result_path = self.root / "v32_result.json"
        result_path.write_text(json.dumps({
            "status": "complete_independent_random_init_validation",
            "candidate_id": binding["candidate_id"], "metrics": {"auc": binding["auc"]},
        }), encoding="utf-8")
        binding.update(
            result_path=str(result_path), result_sha256=runner.digest(result_path),
            initial_state_path=str(reference_path), initial_state_sha256=runner.digest(reference_path),
        )
        kwargs.pop("use_factorized_input")
        old_path = self.root / "incumbent_initial_state.pt"
        torch.save(self.state(Incumbent, kwargs), old_path)
        self.protocol["incumbent"].update(
            initial_state_path=str(old_path), initial_state_sha256=runner.digest(old_path),
        )

    def state(self, cls, kwargs):
        torch.manual_seed(42)
        return {name: value.detach().clone() for name, value in cls(6, 9, **kwargs).state_dict().items()}

    def test_complete_state_matches_v32(self):
        report = runner.verify_best_candidate(self.initial)
        self.assertEqual(report["shared_tensors"], len(self.reference))
        self.assertEqual(report["new_tensors"], self.protocol["candidate_structure"]["new_state_tensors"])
        self.assertTrue(report["all_new_parameters_zero_initialized"])
        self.assertFalse(report["new_initialization_consumes_rng"])
        self.assertTrue(report["new_initial_parameters_match_seed42"])
        self.assertEqual(set(report["new_initial_tensor_sha256"]), set(report["new_tensors"]))
        self.assertEqual(report["new_parameter_count"], 6336)
        self.assertFalse(report["artifact_used_for_initialization"])

    def test_incumbent_common_state_is_preserved(self):
        report = runner.verify_incumbent_initialization(self.initial)
        self.assertEqual(len(report["added_tensors"]), 41)
        self.assertFalse(report["trained_checkpoint_loaded"])

    def test_changed_common_tensor_rejected(self):
        self.initial["concept_emb.weight"][1, 0] += 0.1
        with self.assertRaisesRegex(RuntimeError, "differs from v32"):
            runner.verify_best_candidate(self.initial)

    def test_missing_common_tensor_rejected(self):
        del self.initial["concept_emb.weight"]
        with self.assertRaisesRegex(RuntimeError, "differs from v32"):
            runner.verify_best_candidate(self.initial)

    def test_extra_tensor_rejected(self):
        self.initial["event_projection.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "differs from v32"):
            runner.verify_best_candidate(self.initial)

    def test_changed_attention_tensor_rejected(self):
        self.initial["blocks.0.attn.in_proj_weight"][0, 0] += 0.1
        with self.assertRaisesRegex(RuntimeError, "differs from v32"):
            runner.verify_best_candidate(self.initial)

    def test_dtype_and_shape_drift_rejected(self):
        name = "blocks.0.attn.in_proj_weight"
        original = self.initial[name]
        for changed in (original.double(), original.reshape(32, 96)):
            self.initial[name] = changed
            with self.assertRaisesRegex(RuntimeError, "differs from v32"):
                runner.verify_best_candidate(self.initial)

    def test_different_declared_parameter_is_not_allowed(self):
        self.protocol["candidate_structure"]["new_state_tensors"] = ["extra.weight"]
        self.initial["extra.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "exactly six declared"):
            runner.verify_best_candidate(self.initial)

    def test_each_new_parameter_wrong_initialization_rejected(self):
        for name in self.protocol["candidate_structure"]["new_state_tensors"]:
            original = self.initial[name].clone()
            for value in (0.1, float("nan"), 0.0):
                self.initial[name].fill_(value)
                if torch.equal(self.initial[name], original):
                    continue
                with self.subTest(name=name, value=value), self.assertRaisesRegex(RuntimeError, "FFN-gate"):
                    runner.verify_best_candidate(self.initial)
            self.initial[name].copy_(original)

    def test_new_parameter_shape_or_dtype_rejected(self):
        for name in self.protocol["candidate_structure"]["new_state_tensors"]:
            original = self.initial[name]
            wrong_shape = original.transpose(0, 1) if original.ndim == 2 else original.unsqueeze(-1)
            for changed in (original.double(), wrong_shape):
                with self.subTest(name=name, shape=changed.shape):
                    self.initial[name] = changed
                    with self.assertRaisesRegex(RuntimeError, "FFN-gate"):
                        runner.verify_best_candidate(self.initial)
                    self.initial[name] = original

    def test_each_new_parameter_is_required(self):
        for name in self.protocol["candidate_structure"]["new_state_tensors"]:
            initial = dict(self.initial)
            del initial[name]
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "differs from v32"):
                runner.verify_best_candidate(initial)

    def test_retained_newton_projection_cannot_drift(self):
        self.initial["prequential_newton.projection.weight"][0, 0] += 0.1
        with self.assertRaisesRegex(RuntimeError, "differs from v32"):
            runner.verify_best_candidate(self.initial)

    def test_zero_retained_projection_cannot_hide_wrong_initialization(self):
        self.initial["prequential_newton.projection.weight"].zero_()
        with self.assertRaisesRegex(RuntimeError, "differs from v32"):
            runner.verify_best_candidate(self.initial)

    def test_nonfinite_gate_projection_is_rejected(self):
        self.initial["blocks.0.ffn.gate_weight"][0, 0] = float("inf")
        with self.assertRaisesRegex(RuntimeError, "FFN-gate parameters must be finite"):
            runner.verify_best_candidate(self.initial)

    def test_finite_nonzero_gate_is_rejected(self):
        self.initial["blocks.0.ffn.gate_weight"][0, 0] += 0.001
        with self.assertRaisesRegex(RuntimeError, "all be zero initialized"):
            runner.verify_best_candidate(self.initial)

    def test_retained_graph_projection_cannot_drift(self):
        self.initial["concept_graph.event_projection.weight"][0, 0] += 0.001
        with self.assertRaisesRegex(RuntimeError, "differs from v32"):
            runner.verify_best_candidate(self.initial)

    def test_retained_history_pace_cannot_drift(self):
        self.initial["history_pace.scale"][0] += 0.1
        with self.assertRaisesRegex(RuntimeError, "differs from v32"):
            runner.verify_best_candidate(self.initial)

    def test_rng_preserved_by_auditor(self):
        before = torch.get_rng_state().clone()
        runner.verify_best_candidate(self.initial)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_initialization_check_does_not_seed_cuda(self):
        with patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("CUDA RNG touched")):
            runner.verify_best_candidate(self.initial)

    def test_changed_artifact_rejected_before_load(self):
        path = Path(self.protocol["best_candidate"]["initial_state_path"])
        path.write_bytes(path.read_bytes() + b"changed")
        with patch.object(torch, "load", side_effect=AssertionError("load reached")), self.assertRaisesRegex(RuntimeError, "initialization binding"):
            runner.verify_best_candidate(self.initial)

    def test_only_random_initial_state_is_loaded(self):
        expected = Path(self.protocol["best_candidate"]["initial_state_path"])
        real_load = torch.load
        loaded = []

        def restricted(path, **kwargs):
            loaded.append(Path(path))
            self.assertEqual(Path(path), expected)
            self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True})
            return real_load(path, **kwargs)

        with patch.object(torch, "load", side_effect=restricted):
            runner.verify_best_candidate(self.initial)
        self.assertEqual(loaded, [expected])

    def test_wrong_reference_identity_rejected(self):
        path = Path(self.protocol["best_candidate"]["result_path"])
        value = json.loads(path.read_text())
        value["candidate_id"] = "wrong"
        path.write_text(json.dumps(value), encoding="utf-8")
        self.protocol["best_candidate"]["result_sha256"] = runner.digest(path)
        with self.assertRaisesRegex(RuntimeError, "identity"):
            runner.verify_best_candidate(self.initial)


if __name__ == "__main__":
    unittest.main()
