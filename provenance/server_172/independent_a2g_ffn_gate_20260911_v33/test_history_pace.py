import csv
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "sources"))
from history_pace_candidate import HistoricalPaceModulation, historical_pace_feature


def scalar_oracle(timestamps, hist_c):
    result = torch.zeros(*hist_c.shape, 1, dtype=torch.float32, device=hist_c.device)
    for row in range(hist_c.size(0)):
        for target in range(2, hist_c.size(1)):
            left, right = (int(timestamps[row, index]) for index in (target - 2, target - 1))
            if int(hist_c[row, target - 1]) > 0 and int(hist_c[row, target]) > 0 and 0 <= left <= right:
                result[row, target, 0] = math.log1p((right - left) / 1000.0)
    return result


class HistoryPaceTests(unittest.TestCase):
    def sample(self):
        times = torch.tensor([
            [1500000000000, 1500000001000, 1500000004000, 1500003604000, 1500003604000],
            [1500000000000, 1500000000300, 1500000000600, 1500000000900, 0],
        ], dtype=torch.int64)
        hist_c = torch.tensor([[0, 2, 2, 3, 3, 4], [0, 2, 1, 3, 3, 0]])
        return times, hist_c

    def test_scalar_oracle(self):
        times, hist_c = self.sample()
        torch.testing.assert_close(historical_pace_feature(times, hist_c), scalar_oracle(times, hist_c))

    def test_first_two_targets_have_no_time_difference(self):
        times, hist_c = self.sample()
        self.assertTrue(bool(historical_pace_feature(times, hist_c)[:, :2].eq(0).all()))

    def test_epoch_millisecond_subtraction_retains_small_gaps(self):
        times, hist_c = self.sample()
        result = historical_pace_feature(times, hist_c)
        self.assertAlmostEqual(float(result[0, 2, 0]), math.log(2), places=6)
        self.assertAlmostEqual(float(result[1, 2, 0]), math.log1p(.3), places=6)
        self.assertEqual(float(times[0, 1].float() - times[0, 0].float()), 0)

    def test_epoch_offset_invariance(self):
        times, hist_c = self.sample()
        shifted = times + 3000000000000
        self.assertTrue(torch.equal(
            historical_pace_feature(times, hist_c), historical_pace_feature(shifted, hist_c),
        ))

    def test_current_or_future_times_do_not_change_past_features(self):
        times, hist_c = self.sample()
        expected = historical_pace_feature(times, hist_c)
        for current in range(times.size(1)):
            changed = times.clone()
            changed[:, current:] += 123456789
            self.assertTrue(torch.equal(
                expected[:, :current + 1],
                historical_pace_feature(changed, hist_c)[:, :current + 1],
            ))

    def test_variable_prefix_equivalence(self):
        times, hist_c = self.sample()
        expected = historical_pace_feature(times, hist_c)
        for length in range(1, hist_c.size(1)):
            self.assertTrue(torch.equal(
                expected[:, :length],
                historical_pace_feature(times[:, :length - 1], hist_c[:, :length]),
            ))

    def test_missing_timestamp_field_is_exactly_zero(self):
        _, hist_c = self.sample()
        self.assertTrue(bool(historical_pace_feature(None, hist_c).eq(0).all()))

    def test_empty_timestamp_field_is_exactly_zero(self):
        _, hist_c = self.sample()
        self.assertTrue(bool(historical_pace_feature(torch.empty(2, 0, dtype=torch.int64), hist_c).eq(0).all()))

    def test_padding_does_not_supply_timing(self):
        times, hist_c = self.sample()
        hist_c.zero_()
        self.assertTrue(bool(historical_pace_feature(times, hist_c).eq(0).all()))

    def test_unknown_concepts_are_real_observed_events(self):
        times, hist_c = self.sample()
        hist_c[:, 1:] = 1
        self.assertGreater(float(historical_pace_feature(times, hist_c)[1, 2, 0]), 0)

    def test_invalid_or_backward_pairs_are_unavailable_not_clipped_positive(self):
        times = torch.tensor([[1000, -1, 4000, 3000]], dtype=torch.int64)
        hist_c = torch.tensor([[0, 2, 2, 2, 2]])
        self.assertTrue(bool(historical_pace_feature(times, hist_c).eq(0).all()))

    def test_extreme_valid_integer_gap_is_finite(self):
        times = torch.tensor([[0, torch.iinfo(torch.int64).max]], dtype=torch.int64)
        result = historical_pace_feature(times, torch.tensor([[0, 2, 2]]))
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertGreater(float(result[0, 2, 0]), 0)

    def test_float_timestamps_are_rejected(self):
        times, hist_c = self.sample()
        with self.assertRaisesRegex(ValueError, "int64"):
            historical_pace_feature(times.float(), hist_c)

    def test_wrong_timestamp_width_is_rejected(self):
        times, hist_c = self.sample()
        with self.assertRaisesRegex(ValueError, "fewer column"):
            historical_pace_feature(times[:, :-1], hist_c)

    def test_empty_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            historical_pace_feature(None, torch.empty(2, 0, dtype=torch.int64))

    def test_inputs_are_not_mutated(self):
        times, hist_c = self.sample()
        before = (times.clone(), hist_c.clone())
        historical_pace_feature(times, hist_c)
        self.assertTrue(torch.equal(times, before[0]) and torch.equal(hist_c, before[1]))

    def test_other_learners_do_not_affect_first_learner(self):
        times, hist_c = self.sample()
        expected = historical_pace_feature(times, hist_c)
        times[1] = 0
        self.assertTrue(torch.equal(expected[0], historical_pace_feature(times, hist_c)[0]))

    def test_modulator_consumes_no_random_numbers(self):
        before = torch.get_rng_state().clone()
        module = HistoricalPaceModulation(8)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(set(module.state_dict()), {"scale", "shift"})
        self.assertEqual(sum(p.numel() for p in module.parameters()), 16)

    def test_initial_modulation_is_exact_identity(self):
        history = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8) / 13
        feature = torch.tensor([[[0], [.5], [9]], [[1], [2], [3]]])
        module = HistoricalPaceModulation(8)
        self.assertTrue(torch.equal(module(history, feature), history))

    def test_both_parameter_vectors_receive_gradient_at_zero(self):
        history = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8) / 13
        feature = torch.ones(2, 3, 1)
        module = HistoricalPaceModulation(8)
        module(history, feature).square().sum().backward()
        self.assertTrue(all(p.grad is not None and bool(p.grad.ne(0).all()) for p in module.parameters()))

    def test_absent_feature_gives_identity_even_after_activation(self):
        history = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8) / 13
        module = HistoricalPaceModulation(8)
        with torch.no_grad():
            module.scale.fill_(.2)
            module.shift.fill_(-.3)
        self.assertTrue(torch.equal(module(history, torch.zeros(2, 3, 1)), history))

    def test_modulation_matches_explicit_coordinate_formula(self):
        history = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 7
        feature = torch.tensor([[[0.0], [1.0], [2.0]]])
        module = HistoricalPaceModulation(4)
        with torch.no_grad():
            module.scale.copy_(torch.tensor([.1, -.2, .3, -.4]))
            module.shift.copy_(torch.tensor([.4, -.3, .2, -.1]))
        expected = history.clone()
        for position in range(3):
            for channel in range(4):
                expected[0, position, channel] = (
                    history[0, position, channel] * (1 + feature[0, position, 0] * module.scale[channel])
                    + feature[0, position, 0] * module.shift[channel]
                )
        self.assertTrue(torch.equal(module(history, feature), expected))

    def test_modulation_alignment_is_checked(self):
        with self.assertRaisesRegex(ValueError, "align"):
            HistoricalPaceModulation(4)(torch.zeros(2, 3, 4), torch.zeros(2, 4, 1))

    def test_invalid_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            HistoricalPaceModulation(0)


class TimestampLoaderTests(unittest.TestCase):
    def setUp(self):
        from strict_sequence_data import TrainValidationDataset

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "train_valid_sequences.csv"
        self.config = {"num_c": 5, "num_q": 7, "maxlen": 4, "input_type": ["questions", "concepts"]}
        rows = [
            {
                "uid": "train", "fold": 1, "questions": "2,3,4,5", "concepts": "2,3,4,2",
                "responses": "0,1,0,1", "timestamps": "1500000000000,1500000001000,1500000004000,1500000009000",
                "usetimes": "1000,3000,5000,1000", "selectmasks": "1,1,1,1",
            },
            {
                "uid": "train", "fold": 1, "questions": "2,3,4,-1", "concepts": "2,3,4,-1",
                "responses": "0,1,0,-1", "timestamps": "1500000010000,1500000012000,1500000015000,-1",
                "usetimes": "2000,3000,1000,-1", "selectmasks": "1,1,1,-1",
            },
            {
                "uid": "valid", "fold": 0, "questions": "1,3,4,-1", "concepts": "1,3,4,-1",
                "responses": "1,0,1,-1", "timestamps": "1500000000000,1500000002000,1500000003000,-1",
                "usetimes": "2000,1000,1000,-1", "selectmasks": "1,1,1,-1",
            },
        ]
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        self.train = TrainValidationDataset(self.path, self.config, {1, 2, 3, 4})
        self.valid = TrainValidationDataset(self.path, self.config, {0})

    def features(self, dataset, index):
        batch = {key: value.unsqueeze(0) for key, value in dataset[index].items()}
        hist_c = torch.cat([torch.zeros_like(batch["cseqs"][:, :1]), batch["cseqs"]], dim=1)
        return historical_pace_feature(batch["tseqs"], hist_c)

    def test_loader_retains_int64_epoch_precision(self):
        batch = self.train[0]
        self.assertEqual(batch["tseqs"].dtype, torch.int64)
        self.assertEqual(int(batch["tseqs"][1] - batch["tseqs"][0]), 1000)

    def test_loaded_feature_uses_two_predecessors_not_current_target_time(self):
        expected = torch.tensor([[[0.0], [0.0], [math.log(2)], [math.log(4)]]])
        torch.testing.assert_close(self.features(self.train, 0), expected)

    def test_masked_padding_cannot_create_a_time_feature(self):
        self.assertEqual(int(self.train[1]["tseqs"][-1]), 0)
        self.assertEqual(float(self.features(self.train, 1)[0, -1, 0]), 0)

    def test_segment_boundary_resets_available_history(self):
        for index in range(len(self.train)):
            self.assertTrue(bool(self.features(self.train, index)[:, :2].eq(0).all()))

    def test_validation_unknown_concept_keeps_observed_time(self):
        self.assertEqual(int(self.valid[0]["cseqs"][0]), 1)
        self.assertAlmostEqual(float(self.features(self.valid, 0)[0, 2, 0]), math.log(3), places=6)

    def test_timestamps_match_historical_loader(self):
        from legacy_data_loader import KTDataset

        legacy = KTDataset(str(self.path), self.config["input_type"], {1, 2, 3, 4})
        for index in range(len(self.train)):
            for field in ("tseqs", "shft_tseqs", "utseqs", "shft_utseqs"):
                self.assertTrue(torch.equal(self.train[index][field], legacy[index][field].cpu()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
