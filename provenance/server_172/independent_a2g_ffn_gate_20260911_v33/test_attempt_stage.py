"""Independent event-loop oracles for item-local ordinal state."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT))
from attempt_stage_candidate import ItemAttemptReadout, item_attempt_indices


def shifted(questions, responses, device="cpu"):
    q = torch.tensor(questions, dtype=torch.long, device=device)
    r = torch.tensor(responses, dtype=torch.long, device=device)
    if q.ndim == 1:
        q, r = q.unsqueeze(0), r.unsqueeze(0)
    hq = torch.cat([torch.zeros_like(q[:, :1]), q[:, :-1]], dim=1)
    hr = torch.cat([torch.full_like(r[:, :1], 2), r[:, :-1]], dim=1)
    return q, hq, hr


def loop_oracle(target_q, hist_q, hist_r):
    outputs = [torch.zeros_like(target_q) for _ in range(4)]
    q, hq, hr = (value.cpu().tolist() for value in (target_q, hist_q, hist_r))
    for batch, sequence in enumerate(q):
        for t, target in enumerate(sequence):
            if target < 2:
                continue
            available = [j for j in range(1, t + 1) if hq[batch][j] == target and hr[batch][j] in (0, 1)]
            outputs[0][batch, t] = len(available)
            outputs[1][batch, t] = sum(hr[batch][j] == 1 for j in available)
            for field, allowed in ((2, (0, 1)), (3, (0,))):
                length, j = 0, t
                while j >= 1 and hq[batch][j] == target and hr[batch][j] in allowed:
                    length += 1
                    j -= 1
                outputs[field][batch, t] = length
    return tuple(outputs)


def oracle_checks(device):
    sequences = [
        ([2, 2, 2, 2, 3, 2, 2, 2], [0, 0, 1, 0, 1, 0, 0, 1]),
        ([3, 4, 3, 3, 3, 4, 0, 0], [1, 1, 0, 1, 0, 1, 2, 2]),
        ([1, 1, 2, 2, 1, 2, 3, 3], [0, 0, 0, 2, 1, 1, 0, 1]),
    ]
    tensors = shifted([entry[0] for entry in sequences], [entry[1] for entry in sequences], device)
    actual = item_attempt_indices(*tensors)
    expected = loop_oracle(*tensors)
    checks = {f"ordinal_{i}_matches_independent_event_loop": torch.equal(left, right) for i, (left, right) in enumerate(zip(actual, expected))}
    checks["ordinal_prefix_invariant"] = all(
        all(torch.equal(result[:, :length], prefix) for result, prefix in zip(
            actual, item_attempt_indices(*(value[:, :length] for value in tensors)),
        ))
        for length in range(1, tensors[0].size(1))
    )
    return checks


class AttemptIndexTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_independent_oracle(self):
        for name, passed in oracle_checks("cpu").items():
            with self.subTest(check=name):
                self.assertTrue(passed)

    def test_exact_retries_and_failure_reset(self):
        result = item_attempt_indices(*shifted([2] * 5, [0, 0, 1, 0, 1]))
        for actual, expected in zip(result, ([0, 1, 2, 3, 4], [0, 0, 0, 1, 1], [0, 1, 2, 3, 4], [0, 1, 2, 0, 1])):
            self.assertEqual(actual.tolist(), [expected])

    def test_return_to_item_retains_total_but_resets_run(self):
        result = item_attempt_indices(*shifted([2, 2, 3, 2, 2], [0, 0, 1, 0, 1]))
        self.assertEqual(result[0].tolist(), [[0, 1, 0, 2, 3]])
        self.assertEqual(result[2].tolist(), [[0, 1, 0, 0, 1]])
        self.assertEqual(result[3].tolist(), [[0, 1, 0, 0, 1]])

    def test_global_failure_streak_is_not_an_item_failure_streak(self):
        result = item_attempt_indices(*shifted([2, 3, 4, 4], [0, 0, 0, 1]))
        self.assertEqual(result[3].tolist(), [[0, 0, 0, 1]])

    def test_unknown_ids_cannot_match_one_another(self):
        result = item_attempt_indices(*shifted([1, 1, 1, 1], [0, 1, 0, 1]))
        self.assertTrue(all(bool(value.eq(0).all()) for value in result))

    def test_padding_cannot_create_counts(self):
        result = item_attempt_indices(*shifted([2, 2, 0, 0], [0, 1, 0, 0]))
        self.assertTrue(all(bool(value[:, 2:].eq(0).all()) for value in result))

    def test_unknown_response_does_not_advance_attempt(self):
        result = item_attempt_indices(*shifted([2] * 4, [0, 2, 1, 0]))
        self.assertEqual(result[0].tolist(), [[0, 1, 1, 2]])
        self.assertEqual(result[2].tolist(), [[0, 1, 0, 1]])

    def test_maximum_count_is_199(self):
        result = item_attempt_indices(*shifted([2] * 200, [0] * 200))
        self.assertEqual(result[0][0, -1].item(), 199)
        self.assertEqual(result[3][0, -1].item(), 199)

    def test_one_event_has_no_history(self):
        result = item_attempt_indices(*shifted([2], [1]))
        self.assertTrue(all(bool(value.eq(0).all()) for value in result))

    def test_batch_members_are_independent(self):
        args = shifted([[2] * 4, [3, 2, 3, 2]], [[0] * 4, [1] * 4])
        full = item_attempt_indices(*args)
        for batch in range(2):
            individual = item_attempt_indices(*(value[batch:batch + 1] for value in args))
            self.assertTrue(all(torch.equal(a[batch:batch + 1], b) for a, b in zip(full, individual)))

    def test_current_and_future_response_changes_do_not_change_state(self):
        questions, responses = [2] * 6, [0, 1, 0, 0, 1, 0]
        base = item_attempt_indices(*shifted(questions, responses))
        for t in range(6):
            changed = [r if i < t else 1 - r for i, r in enumerate(responses)]
            actual = item_attempt_indices(*shifted(questions, changed))
            self.assertTrue(all(torch.equal(a[:, :t + 1], b[:, :t + 1]) for a, b in zip(base, actual)))

    def test_mismatched_shapes_rejected(self):
        args = shifted([2] * 3, [0] * 3)
        with self.assertRaisesRegex(ValueError, "identical"):
            item_attempt_indices(args[0], args[1][:, :2], args[2])

    def test_empty_sequence_rejected(self):
        empty = torch.empty((1, 0), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "at least one"):
            item_attempt_indices(empty, empty, empty)


class AttemptReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.module = ItemAttemptReadout(16, 4, 200).double()
        self.args = shifted([2, 2, 2, 2, 3, 2], [0, 0, 1, 0, 1, 0])
        self.sequence = torch.randn(1, 6, 16, dtype=torch.float64)
        self.target = torch.randn(1, 6, 16, dtype=torch.float64)

    def activate(self):
        with torch.no_grad():
            self.module.output.weight.copy_(torch.linspace(-0.2, 0.2, 12, dtype=torch.float64).view(1, 12))

    def test_zero_start_has_no_logit_effect(self):
        self.assertTrue(bool(self.module(self.sequence, self.target, *self.args).eq(0).all()))

    def test_zero_start_output_receives_gradient(self):
        self.module(self.sequence, self.target, *self.args).sum().backward()
        self.assertGreater(self.module.output.weight.grad.abs().sum().item(), 0)

    def test_zero_count_rows_are_zero(self):
        for name, module in self.module.named_modules():
            if name.endswith("_embedding"):
                self.assertTrue(bool(module.weight[0].eq(0).all()))

    def test_gate_blocks_cold_start_after_activation(self):
        self.activate()
        delta = self.module(self.sequence, self.target, *self.args)
        self.assertEqual(delta[0, 0].item(), 0)
        self.assertEqual(delta[0, 4].item(), 0)
        self.assertGreater(delta.abs().sum().item(), 0)

    def test_activated_exact_formula(self):
        self.activate()
        counts = loop_oracle(*self.args)
        state = sum(table(index) for table, index in zip((
            self.module.attempt_embedding, self.module.success_embedding,
            self.module.run_embedding, self.module.failure_embedding,
        ), counts)) / 2
        target = torch.tanh(self.module.target_projection(self.module.target_norm(self.target)))
        sequence = torch.tanh(self.module.sequence_projection(self.module.sequence_norm(self.sequence)))
        weights = self.module.output.weight[0].split(self.module.rank)
        expected = (
            (state * weights[0]).sum(-1)
            + (state * target * weights[1]).sum(-1)
            + (state * sequence * weights[2]).sum(-1)
        ) * counts[0].gt(0)
        actual = self.module(self.sequence, self.target, *self.args)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_activated_state_reaches_both_context_projections(self):
        self.activate()
        self.module(self.sequence, self.target, *self.args).square().sum().backward()
        for parameter in (self.module.target_projection.weight, self.module.sequence_projection.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_embedding_padding_rows_have_no_gradient(self):
        self.activate()
        self.module(self.sequence, self.target, *self.args).sum().backward()
        for name, module in self.module.named_modules():
            if name.endswith("_embedding"):
                self.assertTrue(bool(module.weight.grad[0].eq(0).all()))

    def test_local_context_has_no_future_gradient(self):
        self.activate()
        sequence = self.sequence.clone().requires_grad_()
        target = self.target.clone().requires_grad_()
        self.module(sequence, target, *self.args)[:, :3].sum().backward()
        self.assertTrue(bool(sequence.grad[:, 3:].eq(0).all()))
        self.assertTrue(bool(target.grad[:, 3:].eq(0).all()))

    def test_readout_prefix_equivalence(self):
        self.activate()
        full = self.module(self.sequence, self.target, *self.args)
        for length in range(1, 6):
            prefix = self.module(self.sequence[:, :length], self.target[:, :length], *(value[:, :length] for value in self.args))
            self.assertTrue(torch.allclose(full[:, :length], prefix, atol=1e-12, rtol=1e-12))

    def test_unknown_item_readout_is_zero(self):
        self.activate()
        self.assertTrue(bool(self.module(self.sequence, self.target, *shifted([1] * 6, [0] * 6)).eq(0).all()))

    def test_width_above_frozen_table_is_rejected(self):
        module = ItemAttemptReadout(16, 4, 3).double()
        with self.assertRaisesRegex(ValueError, "width"):
            module(self.sequence, self.target, *self.args)

    def test_parameter_count_is_dimension_derived(self):
        d, heads, width = 256, 8, 200
        module = ItemAttemptReadout(d, heads, width)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 43104)

    def test_readout_has_no_dropout(self):
        self.assertFalse(any(isinstance(m, torch.nn.Dropout) for m in self.module.modules()))


if __name__ == "__main__":
    unittest.main()
