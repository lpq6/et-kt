"""Independent prefix solves and causal tests for the Newton readout."""

import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT))
from prequential_newton_candidate import PrequentialNewtonReadout, prefix_newton_scores


def direct_prefix_scores(logits, features, responses, observed):
    """Rebuild and solve each prefix; do not reuse the incremental inverse."""
    batch, length, rank = features.shape
    rows = []
    for learner in range(batch):
        values = []
        for target in range(length):
            selected = observed[learner, :target] & (
                responses[learner, :target].eq(0) | responses[learner, :target].eq(1)
            )
            design = features[learner, :target][selected]
            probability = logits[learner, :target][selected].sigmoid()
            outcome = responses[learner, :target][selected].to(logits.dtype)
            weight = probability * (1.0 - probability)
            precision = torch.eye(rank, device=features.device, dtype=features.dtype)
            precision = precision + design.T @ (weight.unsqueeze(-1) * design)
            score = design.T @ (outcome - probability)
            offset = torch.linalg.solve(precision, score)
            value = (features[learner, target] * offset).sum()
            values.append(torch.where(observed[learner, target], value, torch.zeros_like(value)))
        rows.append(torch.stack(values))
    return torch.stack(rows)


def synthetic(batch=2, length=7, rank=4, dtype=torch.float64):
    generator = torch.Generator().manual_seed(20260910)
    logits = torch.randn(batch, length, generator=generator, dtype=dtype)
    coordinates = torch.randn(batch, length, rank - 1, generator=generator, dtype=dtype)
    features = torch.cat([
        torch.ones(batch, length, 1, dtype=dtype),
        coordinates.tanh() / math.sqrt(max(rank - 1, 1)),
    ], dim=-1)
    responses = torch.randint(0, 2, (batch, length - 1), generator=generator)
    observed = torch.ones(batch, length, dtype=torch.bool)
    return logits, features, responses, observed


class PrefixNewtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_oracle(self, values, atol=2e-12, rtol=2e-12):
        actual = prefix_newton_scores(*values)
        expected = direct_prefix_scores(*values)
        self.assertTrue(torch.allclose(actual, expected, atol=atol, rtol=rtol))
        return actual

    def test_every_prefix_matches_direct_fisher_solve(self):
        self.assert_oracle(synthetic())

    def test_first_target_has_exactly_zero_correction(self):
        actual = prefix_newton_scores(*synthetic())
        self.assertTrue(torch.equal(actual[:, 0], torch.zeros_like(actual[:, 0])))

    def test_single_target_has_no_update(self):
        self.assert_oracle(synthetic(length=1))

    def test_scalar_intercept_matches_closed_form(self):
        length = 11
        values = (
            torch.zeros(1, length, dtype=torch.float64),
            torch.ones(1, length, 1, dtype=torch.float64),
            torch.ones(1, length - 1, dtype=torch.long),
            torch.ones(1, length, dtype=torch.bool),
        )
        count = torch.arange(length, dtype=torch.float64)
        expected = 0.5 * count / (1.0 + 0.25 * count)
        self.assertTrue(torch.allclose(prefix_newton_scores(*values)[0], expected, atol=1e-14, rtol=1e-14))

    def test_opposite_errors_cancel(self):
        values = list(synthetic(batch=1, length=3, rank=1))
        values[0].zero_()
        values[2][0] = torch.tensor([0, 1])
        actual = self.assert_oracle(values)
        self.assertEqual(float(actual[0, 2]), 0.0)

    def test_same_response_has_different_weight_when_surprising(self):
        values = list(synthetic(batch=2, length=2, rank=1))
        values[0][:, 0] = torch.logit(torch.tensor([0.2, 0.8], dtype=torch.float64))
        values[2].fill_(1)
        actual = self.assert_oracle(values)
        self.assertAlmostEqual(float(actual[0, 1]), 0.8 / 1.16, places=13)
        self.assertAlmostEqual(float(actual[1, 1]), 0.2 / 1.16, places=13)

    def test_surprising_failure_has_larger_negative_score(self):
        values = list(synthetic(batch=2, length=2, rank=1))
        values[0][:, 0] = torch.logit(torch.tensor([0.2, 0.8], dtype=torch.float64))
        values[2].zero_()
        actual = self.assert_oracle(values)
        self.assertLess(float(actual[1, 1]), float(actual[0, 1]))

    def test_current_and_future_responses_cannot_change_current_score(self):
        values = synthetic()
        expected = prefix_newton_scores(*values)
        for position in range(values[2].size(1)):
            changed = [value.clone() for value in values]
            changed[2][:, position:] = 1 - changed[2][:, position:]
            actual = prefix_newton_scores(*changed)
            self.assertTrue(torch.equal(actual[:, :position + 1], expected[:, :position + 1]))

    def test_current_and_future_reference_logits_are_not_history(self):
        values = synthetic()
        expected = prefix_newton_scores(*values)
        for position in range(values[0].size(1)):
            changed = [value.clone() for value in values]
            changed[0][:, position:] += 7.0
            actual = prefix_newton_scores(*changed)
            self.assertTrue(torch.equal(actual[:, :position + 1], expected[:, :position + 1]))

    def test_future_features_cannot_change_earlier_scores(self):
        values = synthetic()
        expected = prefix_newton_scores(*values)
        for position in range(1, values[0].size(1)):
            changed = [value.clone() for value in values]
            changed[1][:, position:] *= -3.0
            self.assertTrue(torch.equal(prefix_newton_scores(*changed)[:, :position], expected[:, :position]))

    def test_latest_observed_response_changes_next_score(self):
        values = synthetic()
        changed = [value.clone() for value in values]
        changed[2][:, -1] = 1 - changed[2][:, -1]
        self.assertFalse(torch.equal(prefix_newton_scores(*values)[:, -1], prefix_newton_scores(*changed)[:, -1]))

    def test_last_target_logit_never_enters_any_newton_score(self):
        values = synthetic()
        changed = [value.clone() for value in values]
        changed[0][:, -1] = 50.0
        self.assertTrue(torch.equal(prefix_newton_scores(*values), prefix_newton_scores(*changed)))

    def test_prefix_equivalence(self):
        values = synthetic()
        expected = prefix_newton_scores(*values)
        for length in range(1, values[0].size(1)):
            prefix = (values[0][:, :length], values[1][:, :length], values[2][:, :length - 1], values[3][:, :length])
            self.assertTrue(torch.equal(prefix_newton_scores(*prefix), expected[:, :length]))

    def test_missing_responses_do_not_update_fisher_or_score(self):
        values = list(synthetic())
        values[2][:, 1:4] = -1
        self.assert_oracle(values)

    def test_nonbinary_responses_are_not_observations(self):
        values = list(synthetic())
        values[2] = values[2].to(torch.float64)
        values[2][:, 1] = 2
        values[2][:, 3] = float("nan")
        actual = self.assert_oracle(values)
        self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_padding_masks_interior_and_tail_events(self):
        values = list(synthetic(length=9))
        values[3][0, 2:4] = False
        values[3][1, 6:] = False
        actual = self.assert_oracle(values)
        self.assertTrue(bool(actual[~values[3]].eq(0).all()))

    def test_masked_padding_values_cannot_affect_real_targets(self):
        values = list(synthetic(length=9))
        values[3][:, 2:4] = False
        changed = [value.clone() for value in values]
        changed[0][:, 2:4] = 100.0
        changed[1][:, 2:4] = 1e10
        changed[2][:, 2:4] = 1 - changed[2][:, 2:4]
        self.assertTrue(torch.equal(prefix_newton_scores(*values), prefix_newton_scores(*changed)))

    def test_all_padding_is_finite_zero(self):
        values = list(synthetic())
        values[3].zero_()
        actual = self.assert_oracle(values)
        self.assertTrue(bool(actual.eq(0).all()))

    def test_learners_are_independent(self):
        values = synthetic()
        changed = [value.clone() for value in values]
        changed[0][1] *= -10
        changed[1][1] *= 2
        changed[2][1] = 1 - changed[2][1]
        self.assertTrue(torch.equal(prefix_newton_scores(*values)[0], prefix_newton_scores(*changed)[0]))

    def test_learner_permutation_equivariance(self):
        values = synthetic(batch=3)
        order = torch.tensor([2, 0, 1])
        self.assertTrue(torch.equal(
            prefix_newton_scores(*(value[order] for value in values)),
            prefix_newton_scores(*values)[order],
        ))

    def test_inputs_are_not_mutated(self):
        values = synthetic()
        copies = tuple(value.clone() for value in values)
        prefix_newton_scores(*values)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(values, copies)))

    def test_state_does_not_carry_between_calls(self):
        values = synthetic()
        first = prefix_newton_scores(*values)
        changed = [value.clone() for value in values]
        changed[2] = 1 - changed[2]
        prefix_newton_scores(*changed)
        self.assertTrue(torch.equal(first, prefix_newton_scores(*values)))

    def test_last_prefix_is_order_invariant_for_fixed_reference_values(self):
        values = synthetic(length=7)
        order = torch.tensor([4, 1, 5, 0, 3, 2])
        changed = [value.clone() for value in values]
        changed[0][:, :-1] = values[0][:, order]
        changed[1][:, :-1] = values[1][:, order]
        changed[2] = values[2][:, order]
        changed[3][:, :-1] = values[3][:, order]
        self.assertTrue(torch.allclose(
            prefix_newton_scores(*values)[:, -1],
            prefix_newton_scores(*changed)[:, -1], atol=2e-12, rtol=2e-12,
        ))

    def test_full_fisher_coupling_is_not_a_diagonal_approximation(self):
        logits = torch.logit(torch.tensor([[0.4, 0.7, 0.5]], dtype=torch.float64))
        features = torch.tensor([[[1.0, 1.0], [1.0, -0.5], [1.0, 0.3]]], dtype=torch.float64)
        responses = torch.tensor([[1, 0]])
        observed = torch.ones(1, 3, dtype=torch.bool)
        actual = self.assert_oracle((logits, features, responses, observed))
        p = logits[0, :2].sigmoid()
        diagonal = 1 + (p[:, None] * (1-p[:, None]) * features[0, :2].square()).sum(0)
        score = (features[0, :2] * (responses[0] - p)[:, None]).sum(0)
        approximation = (features[0, 2] * score / diagonal).sum()
        self.assertGreater(abs(float(actual[0, 2] - approximation)), 1e-4)

    def test_gradients_match_independent_prefix_solves(self):
        values = list(synthetic(length=6))
        values[0].requires_grad_()
        values[1].requires_grad_()
        actual = prefix_newton_scores(*values)
        expected = direct_prefix_scores(*values)
        actual_grad = torch.autograd.grad(actual.square().sum(), values[:2], retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().sum(), values[:2])
        for left, right in zip(actual_grad, expected_grad):
            self.assertTrue(torch.allclose(left, right, atol=2e-10, rtol=2e-10))

    def test_current_prediction_logit_has_zero_newton_gradient(self):
        values = list(synthetic())
        values[0].requires_grad_()
        output = prefix_newton_scores(*values)
        for position in range(1, output.size(1)):
            gradient = torch.autograd.grad(output[:, position].sum(), values[0], retain_graph=True)[0]
            self.assertTrue(bool(gradient[:, position:].eq(0).all()))

    def test_production_length_rank_is_finite_and_matches_double_oracle(self):
        values = synthetic(batch=1, length=200, rank=33, dtype=torch.float32)
        actual = prefix_newton_scores(*values)
        expected = direct_prefix_scores(values[0].double(), values[1].double(), values[2], values[3])
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(torch.allclose(actual.double(), expected, atol=3e-5, rtol=3e-5))

    def test_saturated_reference_logits_remain_finite(self):
        values = list(synthetic())
        values[0][:, ::2] = 1000
        values[0][:, 1::2] = -1000
        actual = self.assert_oracle(values)
        self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_empty_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            prefix_newton_scores(
                torch.empty(2, 0), torch.empty(2, 0, 4),
                torch.empty(2, 0), torch.empty(2, 0, dtype=torch.bool),
            )

    def test_response_width_mismatch_is_rejected(self):
        values = list(synthetic())
        values[2] = values[2][:, :-1]
        with self.assertRaises(ValueError):
            prefix_newton_scores(*values)

    def test_feature_width_mismatch_is_rejected(self):
        values = list(synthetic())
        values[1] = values[1][:, :-1]
        with self.assertRaises(ValueError):
            prefix_newton_scores(*values)

    def test_event_mask_must_be_boolean(self):
        values = list(synthetic())
        values[3] = values[3].long()
        with self.assertRaises(ValueError):
            prefix_newton_scores(*values)

    def test_mixed_feature_dtype_is_rejected(self):
        values = list(synthetic())
        values[1] = values[1].float()
        with self.assertRaises(ValueError):
            prefix_newton_scores(*values)


class NewtonReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.module = PrequentialNewtonReadout(32, 4)
        self.target = torch.randn(2, 7, 32)
        self.logits = torch.zeros(2, 7)
        self.concepts = torch.ones(2, 7, dtype=torch.long) * 2
        self.responses = torch.ones(2, 6, dtype=torch.long)

    def test_feature_intercept_and_bound(self):
        features = self.module.features(self.target)
        self.assertEqual(features.shape, (2, 7, 9))
        self.assertTrue(bool(features[..., 0].eq(1).all()))
        self.assertTrue(bool(features.square().sum(-1).le(2).all()))

    def test_parameter_count_uses_one_attention_head(self):
        self.assertEqual(sum(p.numel() for p in self.module.parameters()), 257)
        model = PrequentialNewtonReadout(256, 8)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8193)
        self.assertEqual(set(dict(model.named_parameters())), {"projection.weight", "output_scale"})

    def test_zero_start_has_exactly_zero_effect(self):
        output = self.module(self.logits, self.target, self.concepts, self.responses)
        self.assertTrue(bool(output.eq(0).all()))

    def test_output_scale_receives_gradient_at_zero(self):
        output = self.module(self.logits, self.target, self.concepts, self.responses)
        output.sum().backward()
        self.assertGreater(abs(float(self.module.output_scale.grad)), 0)
        self.assertTrue(bool(self.module.projection.weight.grad.eq(0).all()))

    def test_projection_receives_gradient_after_activation(self):
        with torch.no_grad():
            self.module.output_scale.fill_(0.25)
        self.module(self.logits, self.target, self.concepts, self.responses).square().sum().backward()
        self.assertGreater(float(self.module.projection.weight.grad.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(self.module.projection.weight.grad).all()))

    def test_activated_readout_matches_explicit_formula(self):
        with torch.no_grad():
            self.module.output_scale.fill_(-0.2)
        expected = direct_prefix_scores(
            self.logits, self.module.features(self.target), self.responses, self.concepts.gt(0),
        )
        actual = self.module(self.logits, self.target, self.concepts, self.responses)
        self.assertTrue(torch.allclose(actual, -0.2 * expected, atol=2e-6, rtol=2e-6))

    def test_unknown_concepts_are_observed_but_padding_is_not(self):
        with torch.no_grad():
            self.module.output_scale.fill_(1)
        self.concepts.fill_(1)
        unknown = self.module(self.logits, self.target, self.concepts, self.responses)
        self.assertGreater(float(unknown[:, 1:].abs().sum()), 0)
        self.concepts.zero_()
        padding = self.module(self.logits, self.target, self.concepts, self.responses)
        self.assertTrue(bool(padding.eq(0).all()))

    def test_no_dropout_or_persistent_learner_state(self):
        self.assertFalse(any(isinstance(x, torch.nn.Dropout) for x in self.module.modules()))
        self.assertEqual(list(self.module.buffers()), [])

    def test_invalid_head_width_is_rejected(self):
        for width, heads in ((0, 4), (32, 0), (31, 4)):
            with self.assertRaises(ValueError):
                PrequentialNewtonReadout(width, heads)

    def test_target_width_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module.features(self.target[..., :-1])


if __name__ == "__main__":
    unittest.main()
