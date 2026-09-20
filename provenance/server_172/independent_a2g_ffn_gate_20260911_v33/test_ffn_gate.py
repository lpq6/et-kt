"""Independent FFN equations, intervention witnesses and V32 control checks."""

import copy
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT))
from concept_graph_candidate import A2GMambaKT as V32
from ffn_gate_candidate import A2GMambaKT as Candidate, InputGatedFeedForward
from test_concept_graph import activate_graph
from v19_audit_helpers import aligned, build, clone, common_gradients_match
from v27_audit_helpers import timed_batch
from v28_audit_helpers import has_gradient
from v32_audit_helpers import BRANCHES as V32_BRANCHES, activate_parent

GATE_NAMES = sorted(
    f"blocks.{block}.ffn.gate_{suffix}"
    for block in range(3) for suffix in ("weight", "bias")
)
BRANCHES = V32_BRANCHES + (("concept_graph", ("concept_graph.",), "use_concept_graph"),)


def base_ffn(width=4, hidden=6, dropout=0.2, dtype=torch.float64):
    torch.manual_seed(123)
    return nn.Sequential(
        nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, width), nn.Dropout(dropout),
    ).to(dtype=dtype)


def gate_parameters(model):
    return {name: parameter for name, parameter in model.named_parameters() if name in GATE_NAMES}


@torch.no_grad()
def activate_gates(model):
    for parameter in gate_parameters(model).values():
        index = torch.arange(parameter.numel(), device=parameter.device, dtype=parameter.dtype)
        parameter.copy_((0.08 * torch.sin(index + 1.0)).reshape_as(parameter))


@torch.no_grad()
def activate_retained(model):
    activate_parent(model)
    activate_graph(model.concept_graph)


def linear_oracle(inputs, weight, bias):
    return torch.stack([
        (inputs * row).sum(-1) + bias[index]
        for index, row in enumerate(weight)
    ], dim=-1)


def ffn_oracle(module, inputs):
    preactivation = linear_oracle(inputs, module[0].weight, module[0].bias)
    hidden = 0.5 * preactivation * (1.0 + torch.erf(preactivation / math.sqrt(2.0)))
    logits = linear_oracle(inputs, module.gate_weight, module.gate_bias)
    gated = hidden * (2.0 / (1.0 + torch.exp(-logits)))
    return linear_oracle(gated, module[3].weight, module[3].bias)


def gates_active(module):
    with torch.no_grad():
        for parameter in (module.gate_weight, module.gate_bias):
            index = torch.arange(parameter.numel(), dtype=parameter.dtype, device=parameter.device)
            parameter.copy_((0.11 * torch.cos(index + 1)).reshape_as(parameter))


class FeedForwardTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.original = base_ffn()
        self.module = InputGatedFeedForward(copy.deepcopy(self.original))
        self.inputs = torch.linspace(-1.7, 2.2, 40, dtype=torch.float64).reshape(2, 5, 4)

    def test_zero_parameters_and_five_original_modules(self):
        self.assertEqual(len(self.module), 5)
        self.assertEqual(set(self.module.state_dict()) - set(self.original.state_dict()), {"gate_weight", "gate_bias"})
        self.assertTrue(bool(self.module.gate_weight.eq(0).all()))
        self.assertTrue(bool(self.module.gate_bias.eq(0).all()))

    def test_constructor_consumes_no_rng(self):
        before = torch.get_rng_state().clone()
        InputGatedFeedForward(self.original)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_original_modules_are_reused(self):
        module = InputGatedFeedForward(self.original)
        self.assertTrue(all(module[index] is self.original[index] for index in range(5)))

    def test_dtype_and_device_are_preserved(self):
        for parameter in (self.module.gate_weight, self.module.gate_bias):
            self.assertEqual(parameter.dtype, self.original[0].weight.dtype)
            self.assertEqual(parameter.device, self.original[0].weight.device)

    def test_initial_eval_exact(self):
        self.original.eval()
        self.module.eval()
        self.assertTrue(torch.equal(self.original(self.inputs), self.module(self.inputs)))

    def test_initial_train_output_and_dropout_rng_exact(self):
        torch.manual_seed(77)
        expected = self.original(self.inputs)
        after = torch.get_rng_state().clone()
        torch.manual_seed(77)
        actual = self.module(self.inputs)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(after, torch.get_rng_state()))

    def test_active_formula_matches_independent_erf_and_row_sum(self):
        self.module.eval()
        gates_active(self.module)
        torch.testing.assert_close(self.module(self.inputs), ffn_oracle(self.module, self.inputs), rtol=1e-12, atol=1e-12)

    def test_active_gradients_match_independent_formula(self):
        self.module.eval()
        gates_active(self.module)
        inputs = self.inputs.clone().requires_grad_(True)
        parameters = tuple(self.module.parameters())
        actual = torch.autograd.grad(self.module(inputs).square().sum(), (inputs, *parameters))
        expected = torch.autograd.grad(ffn_oracle(self.module, inputs).square().sum(), (inputs, *parameters))
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)

    def test_nonzero_gates_change_output(self):
        self.original.eval()
        self.module.eval()
        gates_active(self.module)
        self.assertFalse(torch.equal(self.original(self.inputs), self.module(self.inputs)))

    def test_disabled_nonzero_gates_exact(self):
        self.original.eval()
        self.module.eval()
        gates_active(self.module)
        self.module.enabled = False
        self.assertTrue(torch.equal(self.original(self.inputs), self.module(self.inputs)))

    def test_disabled_gates_have_no_gradient(self):
        self.module.enabled = False
        self.module(self.inputs).sum().backward()
        self.assertIsNone(self.module.gate_weight.grad)
        self.assertIsNone(self.module.gate_bias.grad)

    def test_gate_can_use_information_outside_a_neurons_preactivation(self):
        original = base_ffn(width=2, hidden=1, dropout=0.0)
        with torch.no_grad():
            original[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
            original[0].bias.zero_()
            original[3].weight.fill_(1.0)
            original[3].bias.zero_()
        module = InputGatedFeedForward(copy.deepcopy(original))
        with torch.no_grad():
            module.gate_weight.copy_(torch.tensor([[0.0, 1.0]]))
        inputs = torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=torch.float64)
        self.assertTrue(torch.equal(original(inputs)[0], original(inputs)[1]))
        self.assertFalse(torch.equal(module(inputs)[0], module(inputs)[1]))

    def test_not_a_shared_scalar_branch_scale(self):
        original = base_ffn(width=2, hidden=2, dropout=0.0)
        with torch.no_grad():
            for layer in (original[0], original[3]):
                layer.weight.copy_(torch.eye(2))
                layer.bias.zero_()
        module = InputGatedFeedForward(copy.deepcopy(original))
        with torch.no_grad():
            module.gate_bias.copy_(torch.tensor([1.0, -1.0]))
        value = torch.ones(1, 2, dtype=torch.float64)
        ratios = module(value) / original(value)
        self.assertNotEqual(float(ratios[0, 0]), float(ratios[0, 1]))

    def test_pointwise_permutation_and_independent_formula(self):
        self.module.eval()
        gates_active(self.module)
        permutation = torch.tensor([3, 0, 4, 2, 1])
        permuted = self.module(self.inputs[:, permutation])
        torch.testing.assert_close(permuted, self.module(self.inputs)[:, permutation], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(permuted, ffn_oracle(self.module, self.inputs[:, permutation]), rtol=1e-12, atol=1e-12)

    def test_no_cross_token_or_learner_gradients(self):
        self.module.eval()
        gates_active(self.module)
        inputs = self.inputs.clone().requires_grad_(True)
        self.module(inputs)[0, 2].sum().backward()
        self.assertTrue(bool(inputs.grad[1].eq(0).all()))
        self.assertTrue(bool(inputs.grad[0, [0, 1, 3, 4]].eq(0).all()))
        self.assertTrue(bool(inputs.grad[0, 2].ne(0).any()))

    def test_rejects_an_already_wrapped_module(self):
        with self.assertRaises(ValueError):
            InputGatedFeedForward(self.module)

    def test_rejects_changed_topology(self):
        alternatives = (
            nn.Identity(), nn.Sequential(*list(self.original.children())[:4]),
            nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Dropout(), nn.Linear(6, 4), nn.Dropout()),
        )
        for value in alternatives:
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                InputGatedFeedForward(value)

    def test_rejects_changed_geometry(self):
        alternatives = (
            nn.Sequential(nn.Linear(4, 6, bias=False), nn.GELU(), nn.Dropout(), nn.Linear(6, 4), nn.Dropout()),
            nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Dropout(), nn.Linear(7, 4), nn.Dropout()),
            nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Dropout(), nn.Linear(6, 5), nn.Dropout()),
        )
        for value in alternatives:
            with self.subTest(value=value), self.assertRaises(ValueError):
                InputGatedFeedForward(value)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.device = torch.device("cpu")
        self.batch = timed_batch(self.device)
        self.parent = build(V32, self.device)
        self.candidate = build(Candidate, self.device)

    def activate(self):
        activate_retained(self.parent)
        activate_retained(self.candidate)
        activate_gates(self.candidate)
        self.parent.eval()
        self.candidate.eval()

    def test_declared_parameters_only(self):
        old, new = self.parent.state_dict(), self.candidate.state_dict()
        self.assertEqual(sorted(set(new) - set(old)), GATE_NAMES)
        self.assertEqual((len(old), len(new)), (105, 111))
        self.assertEqual(sum(value.numel() for value in gate_parameters(self.candidate).values()), 6336)
        self.assertTrue(all(torch.equal(value, new[name]) for name, value in old.items()))
        self.assertTrue(all(bool(value.eq(0).all()) for value in gate_parameters(self.candidate).values()))

    def test_forward_and_all_parent_methods_inherited(self):
        for name in (
            "forward", "_attend", "_stats", "_sequences", "_boundary_weights",
            "_factorized_token", "_modulate_history",
        ):
            self.assertIs(getattr(Candidate, name), getattr(V32, name))

    def test_constructor_preserves_rng(self):
        build(V32, self.device)
        expected = torch.get_rng_state().clone()
        build(Candidate, self.device)
        self.assertTrue(torch.equal(expected, torch.get_rng_state()))

    def test_original_attention_and_three_ffn_shapes(self):
        self.assertEqual(sum("ffn" in block for block in self.candidate.blocks), 3)
        self.assertTrue(all(type(block["attn"]) is nn.MultiheadAttention for block in self.candidate.blocks))
        self.assertNotIn("ffn", self.candidate.blocks[-1])
        for block in self.candidate.blocks[:-1]:
            self.assertIsInstance(block["ffn"], InputGatedFeedForward)
            self.assertEqual(len(block["ffn"]), 5)

    def test_initial_eval_exact_v32(self):
        self.parent.eval()
        self.candidate.eval()
        self.assertTrue(torch.equal(self.parent(self.batch), self.candidate(self.batch)))

    def test_initial_train_dropout_and_common_gradients_exact(self):
        expected, before_trace = aligned(self.parent, self.batch)
        actual, after_trace = aligned(self.candidate, self.batch)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(before_trace, after_trace)
        for prediction in (expected, actual):
            F.binary_cross_entropy(prediction[:, 1:], self.batch["shft_rseqs"].float()).backward()
        self.assertTrue(common_gradients_match(self.parent, self.candidate))
        self.assertTrue(all(has_gradient(self.candidate, (name,)) for name in GATE_NAMES))

    def test_active_gates_change_predictions(self):
        self.activate()
        self.assertFalse(torch.equal(self.parent(self.batch), self.candidate(self.batch)))

    def test_switch_updates_all_wrappers(self):
        self.candidate.use_ffn_gate = 0
        self.assertFalse(self.candidate.use_ffn_gate)
        self.assertTrue(all(not block["ffn"].enabled for block in self.candidate.blocks if "ffn" in block))
        self.candidate.use_ffn_gate = 1
        self.assertTrue(all(block["ffn"].enabled for block in self.candidate.blocks if "ffn" in block))

    def test_disabled_constructor_exact_v32(self):
        candidate = build(Candidate, self.device, use_ffn_gate=0).eval()
        self.parent.eval()
        self.assertTrue(torch.equal(self.parent(self.batch), candidate(self.batch)))

    def test_disabled_nonzero_gates_eval_and_train_exact_v32(self):
        self.activate()
        self.candidate.use_ffn_gate = False
        self.assertTrue(torch.equal(self.parent(self.batch), self.candidate(self.batch)))
        self.parent.train()
        self.candidate.train()
        before, before_trace = aligned(self.parent, self.batch)
        after, after_trace = aligned(self.candidate, self.batch)
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(before_trace, after_trace)
        for prediction in (before, after):
            prediction.sum().backward()
        self.assertTrue(common_gradients_match(self.parent, self.candidate))
        self.assertTrue(all(parameter.grad is None for parameter in gate_parameters(self.candidate).values()))

    def test_current_response_and_future_boundaries(self):
        self.activate()
        with torch.no_grad():
            expected = self.candidate(self.batch)
            for boundary in range(self.batch["rseqs"].size(1)):
                changed = clone(self.batch)
                changed["rseqs"][:, boundary:] = 1 - changed["rseqs"][:, boundary:]
                self.assertTrue(torch.equal(expected[:, :boundary + 1], self.candidate(changed)[:, :boundary + 1]))

    def test_shifted_labels_and_durations_are_unused(self):
        self.activate()
        changed = clone(self.batch)
        changed["shft_rseqs"] = 1 - changed["shft_rseqs"]
        for key in ("shft_tseqs", "utseqs", "shft_utseqs"):
            changed[key].fill_(987654321)
        with torch.no_grad():
            self.assertTrue(torch.equal(self.candidate(self.batch), self.candidate(changed)))

    def test_learner_isolation_and_no_persistent_state(self):
        self.activate()
        changed = clone(self.batch)
        changed["rseqs"][1] = 1 - changed["rseqs"][1]
        with torch.no_grad():
            expected = self.candidate(self.batch)
            actual = self.candidate(changed)
            self.assertTrue(torch.equal(expected[0], actual[0]))
            self.assertTrue(torch.equal(expected, self.candidate(self.batch)))

    def test_other_module_removals_retain_gate_effect(self):
        for branch, _, flag in BRANCHES:
            if branch == "attention":
                continue
            with self.subTest(branch=branch):
                parent = build(V32, self.device, **{flag: 0}).eval()
                candidate = build(Candidate, self.device, **{flag: 0}).eval()
                for model in (parent, candidate):
                    activate_retained(model)
                activate_gates(candidate)
                with torch.no_grad():
                    self.assertFalse(torch.equal(parent(self.batch), candidate(self.batch)))

    def test_no_attention_removes_the_ffn_and_gate_effect(self):
        self.activate()
        self.parent.use_attention_branch = False
        self.candidate.use_attention_branch = False
        self.assertTrue(torch.equal(self.parent(self.batch), self.candidate(self.batch)))
        self.candidate(self.batch).sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in gate_parameters(self.candidate).values()))

    def test_gate_effect_does_not_require_nonzero_attention_outputs(self):
        self.activate()
        with torch.no_grad():
            for model in (self.parent, self.candidate):
                for block in model.blocks:
                    block["attn"].out_proj.weight.zero_()
                    block["attn"].out_proj.bias.zero_()
            self.assertFalse(torch.equal(self.parent(self.batch), self.candidate(self.batch)))

    def test_active_gate_and_retained_module_gradients(self):
        self.activate()
        F.binary_cross_entropy(self.candidate(self.batch)[:, 1:], self.batch["shft_rseqs"].float()).backward()
        for name in GATE_NAMES:
            self.assertTrue(has_gradient(self.candidate, (name,)), name)
        for branch, prefixes, _ in BRANCHES:
            self.assertTrue(has_gradient(self.candidate, prefixes), branch)
        self.assertTrue(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in self.candidate.parameters()))

    def test_qtest_signature(self):
        self.activate()
        with torch.no_grad():
            actual, fused = self.candidate(self.batch, qtest=True)
            self.assertTrue(torch.equal(actual, self.candidate(self.batch)))
            self.assertEqual(fused.shape[:2], (2, 8))

    def test_missing_items_unknown_and_padding_remain_finite(self):
        self.activate()
        batches = []
        unknown, empty, missing = (clone(self.batch) for _ in range(3))
        for key in ("qseqs", "cseqs", "shft_qseqs", "shft_cseqs"):
            unknown[key].fill_(1)
            empty[key].zero_()
        for key in ("qseqs", "shft_qseqs"):
            missing[key] = missing[key][:, :0]
        batches.extend((unknown, empty, missing))
        with torch.no_grad():
            for batch in batches:
                self.assertTrue(bool(torch.isfinite(self.candidate(batch)).all()))
            self.assertTrue(bool(torch.isfinite(build(Candidate, self.device, n_pid=0).eval()(self.batch)).all()))


if __name__ == "__main__":
    unittest.main()
