import copy
import math
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from a2g.candidate import A2GModal
from a2g.experiment import load_config, make_model
from a2g.modules.core import SelectiveSSMBlock
from a2g.modules.evidence import EvidenceMethods
from a2g.modules.input_memory import InputMemorySSM
from a2g.modules.modal import MultiModeResidualInputMemorySSM
from a2g.modules.umk import concept_last_seen_gaps, concept_rate, umk_attention_bias
from a2g.umk import A2GUMK
from test_model import aligned, build
from v27_audit_helpers import timed_batch


FLAGS = ("use_umk_ssm", "use_umk_attn", "use_umk_rwce")


class UMKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_disabled_state_rng_and_training_are_exactly_v43(self):
        parent = build(A2GModal)
        rng = torch.get_rng_state().clone()
        candidate = build(A2GUMK)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(parent.state_dict()), list(candidate.state_dict()))
        candidate.load_state_dict(parent.state_dict(), strict=True)
        expected, expected_trace = aligned(parent, timed_batch("cpu"))
        actual, actual_trace = aligned(candidate, timed_batch("cpu"))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual_trace, expected_trace)
        expected.sum().backward()
        actual.sum().backward()
        for name, parameter in parent.named_parameters():
            other = dict(candidate.named_parameters())[name]
            if parameter.grad is None:
                self.assertIsNone(other.grad)
            else:
                torch.testing.assert_close(other.grad, parameter.grad, rtol=0, atol=0)

    def test_enabled_additions_are_only_two_scalar_parameters(self):
        parent = build(A2GModal)
        rng = torch.get_rng_state().clone()
        candidate = build(A2GUMK, **dict.fromkeys(FLAGS, 1))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        state = candidate.state_dict()
        extra = state.keys() - parent.state_dict().keys()
        self.assertEqual(extra, {"ssm.umk_alpha", "umk_beta"})
        for name in extra:
            self.assertEqual(state[name].shape, torch.Size([]))
        self.assertAlmostEqual(state["ssm.umk_alpha"].item(), 0.3)
        self.assertEqual(state["umk_beta"].item(), 1.0)
        for name, value in parent.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_rate_formula_and_dtype(self):
        prior = torch.tensor([[-2.0, 0.0, 2.0]], dtype=torch.float64, requires_grad=True)
        actual = concept_rate(prior, 0.1)
        torch.testing.assert_close(actual, 0.1 * (1.0 - prior.sigmoid()), rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float64)
        actual.sum().backward()
        self.assertTrue((prior.grad < 0).all())
        for invalid in (-1.0, 0.0, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build(A2GUMK, umk_lambda0=invalid)

    def test_selective_retention_matches_manual_recurrence(self):
        torch.manual_seed(42)
        model = SelectiveSSMBlock(4, dropout=0, use_umk_ssm=1).double().eval()
        x = torch.randn(2, 5, 4, dtype=torch.float64)
        prior = torch.randn(2, 5, dtype=torch.float64)
        scope = torch.full_like(x, 0.7)
        h, outputs = torch.zeros_like(x[:, 0]), []
        for index in range(x.size(1)):
            candidate, gate, delta = model.in_proj(x[:, index]).chunk(3, -1)
            decay = torch.exp(-math.log(2) * 0.5 * F.softplus(delta)).clamp(0.0001, 0.9999)
            decay = decay.pow(1 + model.umk_alpha * 0.3 * (1 - prior[:, index, None].sigmoid()))
            update = 1 - decay
            h = (1 - update) * h + update * candidate.tanh()
            gate = gate.sigmoid() * scope[:, index]
            outputs.append(gate * h + (1 - gate) * x[:, index])
        expected = model.norm(torch.stack(outputs, 1))
        actual = model(x, scope, concept_prior=prior)
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)

    def test_ssm_wrappers_preserve_prior_and_scalar_in_all_routes(self):
        for input_memory in (False, True):
            for modal in (False, True):
                with self.subTest(input_memory=input_memory, modal=modal):
                    model = SelectiveSSMBlock(4, dropout=0, use_umk_ssm=1)
                    alpha = model.umk_alpha
                    model = InputMemorySSM.from_existing(model, input_memory)
                    model = MultiModeResidualInputMemorySSM.from_existing(model, 2, modal)
                    self.assertIs(alpha, model.umk_alpha)
                    x = torch.randn(2, 6, 4)
                    prior = torch.zeros(2, 6, requires_grad=True)
                    model(x, concept_prior=prior).square().sum().backward()
                    self.assertTrue(torch.isfinite(alpha.grad))
                    self.assertGreater(alpha.grad.abs().item(), 0)
                    self.assertGreater(prior.grad.abs().sum().item(), 0)
                    with self.assertRaises(ValueError):
                        model(x)
                    with self.assertRaises(ValueError):
                        model(x, concept_prior=prior[:, :-1])

    def test_last_seen_is_concept_relative_and_strictly_past(self):
        concepts = torch.tensor([[2, 3, 2, 4, 3, 0], [0, 0, 0, 0, 0, 0]])
        gaps, seen = concept_last_seen_gaps(concepts)
        for batch in range(2):
            for i in range(6):
                for j in range(6):
                    history = [
                        k for k in range(i)
                        if concepts[batch, k] == concepts[batch, j] and concepts[batch, j] > 0
                    ]
                    self.assertEqual(seen[batch, i, j].item(), bool(history))
                    if history:
                        self.assertEqual(gaps[batch, i, j].item(), i - history[-1])
        self.assertEqual(gaps[0, 4, 0], gaps[0, 4, 2])
        self.assertEqual(gaps[0, 4, 0].item(), 2)

    def test_attention_bias_masks_missing_history_and_keeps_empty_rows_finite(self):
        concepts = torch.tensor([[2, 3, 2, 4, 3], [0, 0, 0, 0, 0]])
        prior = torch.zeros(2, 5, dtype=torch.float64, requires_grad=True)
        beta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        bias = umk_attention_bias(concepts, prior, beta, 0.1)
        self.assertEqual(bias.shape, (2, 5, 5))
        self.assertEqual(bias.dtype, torch.float64)
        self.assertTrue(torch.isfinite(torch.softmax(bias, -1)).all())
        self.assertTrue(torch.isneginf(bias[:, torch.ones(5, 5).triu(1).bool()]).all())
        self.assertTrue(torch.isneginf(bias[0, 1, 1]))
        self.assertEqual(bias[0, 0, 0].item(), 0)
        self.assertAlmostEqual(bias[0, 4, 0].item(), -0.5 * 0.05 * 2)
        bias[0, 4, 0].backward()
        self.assertLess(beta.grad.item(), 0)
        self.assertGreater(prior.grad[0, 0].item(), 0)

    def test_rwce_matches_unshifted_history_oracle_in_float64(self):
        target = torch.tensor([[2, 2, 3, 2, 3]])
        hist = torch.tensor([[0, 2, 2, 3, 2]])
        answers = torch.tensor([[2, 1, 0, 1, 0]])
        prior = torch.tensor([[0.0, 0.1, 0.3, -0.2, 0.4]], dtype=torch.float64, requires_grad=True)
        result = EvidenceMethods._recency_weighted_concept_evidence_components(
            target, hist, answers, concept_prior=prior, use_umk_rwce=1, umk_lambda0=0.1
        )
        for i in range(5):
            mass, signed, count = 0.0, 0.0, 0
            for j in range(i + 1):
                if hist[0, j] > 0 and hist[0, j] == target[0, i] and answers[0, j] in (0, 1):
                    weight = math.exp(-0.1 * (1.0 - torch.sigmoid(prior[0, i])).item() * (i - j))
                    mass += weight
                    signed += weight * (2 * answers[0, j].item() - 1)
                    count += 1
            self.assertAlmostEqual(result[0][0, i].item(), signed / (1 + mass), places=14)
            self.assertAlmostEqual(result[1][0, i].item(), mass, places=14)
            self.assertEqual(result[2][0, i].item(), count)
        self.assertEqual(result[0].dtype, torch.float64)
        result[0].sum().backward()
        self.assertGreater(prior.grad.abs().sum().item(), 0)
        old = EvidenceMethods._recency_weighted_concept_evidence_components(target, hist, answers)
        off = EvidenceMethods._recency_weighted_concept_evidence_components(
            target, hist, answers, concept_prior=prior, use_umk_rwce=0
        )
        for expected, actual in zip(old, off):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_single_switches_and_joint_model_have_finite_double_gradients(self):
        batch = timed_batch("cpu")
        reference = build(A2GModal).double().eval()(batch)
        for flags in [{name: 1} for name in FLAGS] + [dict.fromkeys(FLAGS, 1)]:
            with self.subTest(flags=flags):
                model = build(A2GUMK, **flags).double().eval()
                prediction, fused = model(batch, qtest=True)
                self.assertEqual(prediction.dtype, torch.float64)
                self.assertEqual(fused.shape, (*prediction.shape, 38))
                self.assertGreater((prediction - reference).abs().max().item(), 0)
                prediction.square().sum().backward()
                self.assertTrue(all(
                    torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
                ))
                for name, p in model.named_parameters():
                    if name in {"ssm.umk_alpha", "umk_beta"}:
                        self.assertGreater(p.grad.abs().item(), 0)
                restored = build(A2GUMK, **flags).double().eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                torch.testing.assert_close(prediction, restored(batch), rtol=0, atol=0)

    def test_current_and_future_labels_timestamps_and_concepts_do_not_leak(self):
        batch = timed_batch("cpu")
        current = 3
        for flags in [{name: 1} for name in FLAGS] + [dict.fromkeys(FLAGS, 1)]:
            with self.subTest(flags=flags):
                model = build(A2GUMK, **flags).eval()
                expected = model(batch)
                changed = {name: value.clone() for name, value in batch.items()}
                changed["rseqs"][:, current:] = 1 - changed["rseqs"][:, current:]
                changed["shft_rseqs"][:] = 1 - changed["shft_rseqs"]
                changed["tseqs"][:, current:] += 123456789
                changed["cseqs"][:, current + 1:] = 2
                changed["shft_cseqs"][:, current:] = 2
                torch.testing.assert_close(
                    expected[:, :current + 1], model(changed)[:, :current + 1], rtol=0, atol=0
                )

    def test_attention_supports_every_existing_normalization_topology(self):
        for topology in ("control", "attention_postnorm", "full_postnorm"):
            with self.subTest(topology=topology):
                model = build(A2GUMK, use_umk_attn=1, normalization_topology=topology).eval()
                result = model(timed_batch("cpu"))
                self.assertTrue(torch.isfinite(result).all())
                result.sum().backward()
                self.assertGreater(model.umk_beta.grad.abs().item(), 0)

    def test_v45_configs_keep_the_fixed_training_contract(self):
        root = Path(__file__).resolve().parents[1]
        parent = load_config(root / "configs/assist2017_v43_lowmem.json")
        for suffix in ("off", "ssm", "attn", "rwce", "all"):
            config = load_config(root / f"configs/assist2017_v45_umk_{suffix}.json")
            other = copy.deepcopy(config)
            self.assertEqual(other.pop("architecture"), "v45_umk")
            for flag in FLAGS:
                self.assertEqual(other["model"].pop(flag), int(suffix in ("all", flag.removeprefix("use_umk_"))))
            self.assertEqual(other["model"].pop("umk_lambda0"), 0.3)
            self.assertEqual(other, {key: value for key, value in parent.items() if key != "architecture"})
            self.assertIsInstance(make_model(config, "full"), A2GUMK)


if __name__ == "__main__":
    unittest.main()
