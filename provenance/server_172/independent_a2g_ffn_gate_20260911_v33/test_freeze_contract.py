"""Reject protocol drift before the V33 experiment is frozen."""

import copy
import unittest

from freeze_package import load, validate_science_contract


class FreezeContractTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load("protocol.json")

    def test_registered_v33_is_valid(self):
        validate_science_contract(self.protocol)

    def test_each_of_nine_controls_is_required(self):
        for key in self.protocol["planned_ablation_variants"]:
            changed = copy.deepcopy(self.protocol)
            del changed["planned_ablation_variants"][key]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_science_contract(changed)

    def test_family_cannot_remain64(self):
        self.protocol["uncertainty"]["multiplicity_family_size"] = 64
        with self.assertRaisesRegex(RuntimeError, "72-contrast"):
            validate_science_contract(self.protocol)

    def test_parent_and_best_observed_roles_must_agree(self):
        self.protocol["best_observed_candidate"]["auc"] -= 0.001
        with self.assertRaisesRegex(RuntimeError, "control role"):
            validate_science_contract(self.protocol)

    def test_prior_control_cannot_replace_v32(self):
        self.protocol["structural_control"]["candidate_id"] = "v27"
        with self.assertRaisesRegex(RuntimeError, "control role"):
            validate_science_contract(self.protocol)

    def test_rank_or_normalization_tuning_is_rejected(self):
        for field, value in (("num_attn_heads", 4), ("prior_scale", 0.1)):
            changed = copy.deepcopy(self.protocol)
            changed["model_kwargs"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "model arguments"):
                validate_science_contract(changed)

    def test_retained_modules_cannot_be_disabled(self):
        for flag in ("use_item_attempt_stage", "use_history_pace", "use_prequential_newton", "use_concept_graph", "use_ffn_gate"):
            changed = copy.deepcopy(self.protocol)
            changed["model_kwargs"][flag] = 0
            with self.subTest(flag=flag), self.assertRaises(RuntimeError):
                validate_science_contract(changed)

    def test_production_parameter_or_tensor_counts_cannot_change(self):
        for field in ("new_parameters", "expected_assist2017_parameters", "expected_parent_state_tensors", "expected_total_state_tensors"):
            changed = copy.deepcopy(self.protocol)
            changed["candidate_structure"][field] += 1
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_science_contract(changed)

    def test_tensor_names_cannot_describe_the_previous_module(self):
        self.protocol["candidate_structure"]["new_state_tensors"] = ["prequential_newton.output_scale", "prequential_newton.projection.weight"]
        with self.assertRaises(RuntimeError):
            validate_science_contract(self.protocol)

    def test_whole_state_equality_cannot_be_claimed(self):
        self.protocol["candidate_structure"]["all_initial_state_tensors_equal_v32"] = True
        with self.assertRaises(RuntimeError):
            validate_science_contract(self.protocol)

    def test_same_training_schedule_is_required(self):
        self.protocol["training"]["auxiliary_loss_weight"] = 0.01
        with self.assertRaisesRegex(RuntimeError, "control protocol"):
            validate_science_contract(self.protocol)

    def test_older_v28_cannot_replace_immediate_closed_predecessor(self):
        self.protocol["closed_predecessor"]["candidate_id"] = "a2g_v27_prequential_newton_readout_20260910_v28"
        with self.assertRaisesRegex(RuntimeError, "immediate predecessor"):
            validate_science_contract(self.protocol)

    def test_closed_predecessor_metric_cannot_drift(self):
        self.protocol["closed_predecessor"]["auc"] += 0.001
        with self.assertRaisesRegex(RuntimeError, "immediate predecessor"):
            validate_science_contract(self.protocol)

    def test_gate_geometry_cannot_change(self):
        self.protocol["candidate_structure"]["gate_shapes"]["blocks.0.ffn.gate_weight"] = [256, 256]
        with self.assertRaisesRegex(RuntimeError, "structure"):
            validate_science_contract(self.protocol)

    def test_learned_edges_cannot_be_claimed_pedagogical_prerequisites(self):
        self.protocol["candidate_structure"]["pedagogical_prerequisites_claimed"] = True
        with self.assertRaisesRegex(RuntimeError, "structure"):
            validate_science_contract(self.protocol)

    def test_old_gaussian_overlap_cannot_be_reintroduced(self):
        self.protocol["model_kwargs"]["use_gaussian_overlap"] = 1
        with self.assertRaisesRegex(RuntimeError, "model arguments"):
            validate_science_contract(self.protocol)

    def test_parent_spare_embedding_cannot_become_a_graph_node(self):
        self.protocol["candidate_structure"]["assist2017_known_graph_nodes"] = 95
        with self.assertRaisesRegex(RuntimeError, "structure"):
            validate_science_contract(self.protocol)

    def test_all_new_tensors_must_be_zero_initialized(self):
        self.protocol["candidate_structure"]["zero_initialized_new_tensors"].pop()
        with self.assertRaisesRegex(RuntimeError, "structure"):
            validate_science_contract(self.protocol)

    def test_retained_graph_geometry_cannot_change(self):
        self.protocol["candidate_structure"]["retained_graph_shapes"]["event_projection.weight"] = [16, 256]
        with self.assertRaisesRegex(RuntimeError, "structure"):
            validate_science_contract(self.protocol)

    def test_attention_gate_dependency_cannot_be_hidden(self):
        self.protocol["candidate_structure"]["ablation_dependencies"] = {}
        with self.assertRaisesRegex(RuntimeError, "structure"):
            validate_science_contract(self.protocol)

    def test_added_capacity_cannot_be_claimed_parameter_matched(self):
        self.protocol["candidate_structure"]["parameter_matched_superiority_claimed"] = True
        with self.assertRaisesRegex(RuntimeError, "structure"):
            validate_science_contract(self.protocol)


if __name__ == "__main__":
    unittest.main()
