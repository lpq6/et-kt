import unittest

from profile_train_history_time import profile


def row(times, fold=1, masks="1,1,1,1", uid="learner"):
    return {
        "fold": str(fold), "uid": uid, "timestamps": times,
        "selectmasks": masks, "responses": "never-parse-this-label-field",
    }


class HistoryTimeProfileTests(unittest.TestCase):
    def test_validation_and_response_tokens_are_not_parsed(self):
        result = profile([row("invalid", fold=0), row("1000,2000,4000,9000")], 4)
        self.assertEqual(result["counts"]["scored_targets"], 3)
        self.assertEqual(result["counts"]["targets_with_two_predecessors"], 2)
        self.assertEqual(result["counts"]["validation_rows_skipped"], 1)
        self.assertFalse(result["response_tokens_parsed"])
        self.assertFalse(result["validation_tokens_parsed"])

    def test_last_timestamp_is_never_a_feature(self):
        first = profile([row("1000,2000,4000,9000")], 4)
        second = profile([row("1000,2000,4000,9000000000000")], 4)
        self.assertEqual(first, second)
        self.assertEqual(first["gap_seconds_quantiles"]["max"], 2)

    def test_epoch_float32_precision_loss_is_detected(self):
        result = profile([row("1500000000000,1500000001000,1500000002000,1500000003000")], 4)
        self.assertEqual(result["counts"]["float32_epoch_subtraction_loses_positive_gap"], 2)
        self.assertEqual(result["gap_seconds_quantiles"]["median"], 1)

    def test_nonprefix_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prefix"):
            profile([row("1,2,3,4", masks="1,-1,1,1")], 4)

    def test_test_fold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fold"):
            profile([row("1,2,3,4", fold=-1)], 4)


if __name__ == "__main__":
    unittest.main()
