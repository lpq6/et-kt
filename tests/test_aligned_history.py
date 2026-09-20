import copy
import math
from pathlib import Path
import unittest

import torch

from a2g.aligned_history import A2GAlignedHistory
from a2g.candidate import A2GModal
from a2g.experiment import load_config, make_model
from test_model import aligned, build
from test_ffn_gate import activate_retained, activate_gates
from v27_audit_helpers import timed_batch


class AlignedHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_legacy_excludes_latest_observation_but_candidate_includes_it(self):
        target = torch.tensor([[2, 2, 2]])
        history = torch.tensor([[0, 2, 2]])
        response = torch.tensor([[2, 1, 0]])
        prior = torch.zeros(1, 3, 1)
        old = build(A2GModal)._stats(target, history, response, prior, prior)
        new = build(A2GAlignedHistory)._stats(target, history, response, prior, prior)
        self.assertEqual(old[0, :, 1].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(new[0, :, 1].tolist(), [0.0, 1.0, 0.5])
        self.assertEqual(new[0, 1, 5].item(), 0.0)

    def test_all_statistics_match_an_unshifted_prefix_oracle(self):
        model = build(A2GAlignedHistory)
        concepts = torch.tensor([[2, 3, 2, 2, 1, 1, 0], [4, 4, 3, 4, 3, 4, 4]])
        responses = torch.tensor([[1, 0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 1, 0]])
        history = torch.cat([torch.zeros_like(concepts[:, :1]), concepts[:, :-1]], 1)
        answers = torch.cat([torch.full_like(responses[:, :1], 2), responses[:, :-1]], 1)
        item = torch.linspace(-0.2, 0.3, concepts.numel()).reshape(*concepts.shape, 1)
        concept = torch.full_like(item, 0.13)
        actual = model._stats(concepts, history, answers, item, concept)
        expected = torch.zeros_like(actual)
        width = concepts.size(1)
        for learner in range(concepts.size(0)):
            for target in range(width):
                observed = [
                    j for j in range(target)
                    if concepts[learner, j] > 0
                    and concepts[learner, j] == concepts[learner, target]
                ]
                success = sum(int(responses[learner, j]) == 1 for j in observed)
                failure = sum(int(responses[learner, j]) == 0 for j in observed)
                count = len(observed)
                # Gap uses shifted-history coordinates; latest observation has gap 0.
                last_history_index = observed[-1] + 1 if observed else 0
                expected[learner, target] = torch.tensor([
                    torch.sigmoid(-(item[learner, target, 0] + concept[learner, target, 0])),
                    success / max(count, 1),
                    math.log1p(count) / math.log1p(max(2, width)),
                    success / (count + 1),
                    failure / (count + 1),
                    (target - last_history_index) / width,
                    item[learner, target, 0].tanh(),
                    concept[learner, target, 0].tanh(),
                ])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_initial_state_and_rng_equal_parent_without_new_parameters(self):
        parent = build(A2GModal)
        rng = torch.get_rng_state().clone()
        candidate = build(A2GAlignedHistory)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(parent.state_dict()), list(candidate.state_dict()))
        self.assertEqual(
            sum(p.numel() for p in parent.parameters()),
            sum(p.numel() for p in candidate.parameters()),
        )
        for name, value in parent.state_dict().items():
            torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)

    def test_disabled_candidate_matches_parent_three_training_steps(self):
        parent = build(A2GModal, graph_checkpoint_steps=3)
        control = build(
            A2GAlignedHistory, use_aligned_history_statistics=0, graph_checkpoint_steps=3
        )
        activate_retained(parent)
        activate_gates(parent)
        with torch.no_grad():
            parent.ssm.input_memory.weight.normal_(std=0.02)
            parent.ssm.mode_input_weight.normal_(std=0.02)
        control.load_state_dict(parent.state_dict())
        optimizers = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in (parent, control)]
        for _ in range(3):
            outputs, traces = [], []
            for model, optimizer in zip((parent, control), optimizers):
                optimizer.zero_grad(set_to_none=True)
                prediction, trace = aligned(model, timed_batch("cpu"))
                prediction.square().sum().backward()
                outputs.append(prediction)
                traces.append(trace)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            self.assertEqual(traces[0], traces[1])
            for first, second in zip(parent.parameters(), control.parameters()):
                if first.grad is None:
                    self.assertIsNone(second.grad)
                else:
                    torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
            for optimizer in optimizers:
                optimizer.step()
            for name, value in parent.state_dict().items():
                torch.testing.assert_close(value, control.state_dict()[name], rtol=0, atol=0)

    def test_enabled_candidate_changes_predictions_with_finite_gradients(self):
        parent, candidate = build(A2GModal), build(A2GAlignedHistory)
        expected, parent_trace = aligned(parent, timed_batch("cpu"))
        actual, candidate_trace = aligned(candidate, timed_batch("cpu"))
        self.assertEqual(parent_trace, candidate_trace)
        self.assertGreater((expected - actual).abs().max().item(), 1e-7)
        actual.sum().backward()
        gradients = [p.grad for p in candidate.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)

    def test_current_and_future_responses_do_not_affect_current_prediction(self):
        model = build(A2GAlignedHistory).eval()
        activate_retained(model)
        activate_gates(model)
        batch = timed_batch("cpu")
        expected = model(batch)
        current = 3
        changed = {name: value.clone() for name, value in batch.items()}
        changed["rseqs"][:, current:] = 1 - changed["rseqs"][:, current:]
        changed["shft_rseqs"][:] = 1 - changed["shft_rseqs"]
        changed["tseqs"][:, current:] += 123456789
        actual = model(changed)
        torch.testing.assert_close(
            expected[:, :current + 1], actual[:, :current + 1], rtol=0, atol=0
        )
        self.assertGreater((expected[:, current + 1:] - actual[:, current + 1:]).abs().max(), 0)

    def test_no_evidence_removes_the_alignment_effect(self):
        parent = build(A2GModal, use_evidence_branch=0).eval()
        candidate = build(A2GAlignedHistory, use_evidence_branch=0).eval()
        torch.testing.assert_close(
            parent(timed_batch("cpu")), candidate(timed_batch("cpu")), rtol=0, atol=0
        )

    def test_configuration_changes_only_architecture_and_control_is_explicit(self):
        root = Path(__file__).resolve().parents[1]
        parent = load_config(root / "configs/assist2017_v43_lowmem.json")
        candidate = load_config(root / "configs/assist2017_v44_aligned_history.json")
        self.assertEqual(candidate, {**parent, "architecture": "v44_aligned_history"})
        small = copy.deepcopy(candidate)
        small["model"].update(d_model=32, d_ff=64, num_attn_heads=4)
        small["data"].update(num_c=6, num_q=9)
        model = make_model(small, "no_aligned_history_statistics")
        self.assertFalse(model.use_aligned_history_statistics)
        for architecture in ("v33", "v43"):
            small["architecture"] = architecture
            with self.assertRaisesRegex(ValueError, "v44_aligned_history"):
                make_model(small, "no_aligned_history_statistics")
        small["architecture"] = "v44_aligned_history"
        for variant in ("no_input_memory", "no_multimode_residual"):
            make_model(small, variant)


if __name__ == "__main__":
    unittest.main()
