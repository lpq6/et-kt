import csv
import io
import unittest

from inspect_timestamp_assets import inspect_rows


def reader(rows, fields=("fold", "timestamps", "selectmasks", "responses")):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    stream.seek(0)
    return csv.DictReader(stream)


def row(fold, times, masks="1,1,1"):
    return {"fold": str(fold), "timestamps": times, "selectmasks": masks, "responses": "do-not-parse"}


class TimestampAssetTests(unittest.TestCase):
    def test_validation_tokens_are_never_parsed(self):
        result = inspect_rows(reader([row(0, "poison"), row(1, "100,110,130")]), 3)
        self.assertEqual(result["sample_fold"], 1)
        self.assertEqual(result["validation_rows_skipped_without_token_parsing"], 1)
        self.assertEqual(result["sample_positive_gaps"], 2)

    def test_only_first_training_row_is_parsed(self):
        result = inspect_rows(reader([row(2, "100,110,130"), row(3, "poison")]), 3)
        self.assertEqual(result["sampled_train_rows"], 1)

    def test_missing_column_does_not_claim_availability(self):
        result = inspect_rows(reader([{"fold": "1"}], fields=("fold",)), 3)
        self.assertEqual(result["status"], "no_timestamp_column")

    def test_constant_time_does_not_claim_availability(self):
        result = inspect_rows(reader([row(1, "0,0,0")]), 3)
        self.assertEqual(result["status"], "sample_does_not_establish_timestamp_availability")

    def test_backward_time_does_not_claim_availability(self):
        result = inspect_rows(reader([row(1, "100,90,130")]), 3)
        self.assertEqual(result["sample_negative_gaps"], 1)
        self.assertEqual(result["status"], "sample_does_not_establish_timestamp_availability")

    def test_padding_does_not_create_negative_gaps(self):
        result = inspect_rows(reader([row(1, "100,110,-1", "1,1,-1")]), 3)
        self.assertEqual(result["sample_valid_events"], 2)
        self.assertEqual(result["sample_negative_gaps"], 0)

    def test_test_fold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexpected fold"):
            inspect_rows(reader([row(-1, "100,110,130")]), 3)

    def test_width_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "width"):
            inspect_rows(reader([row(1, "100,110")]), 3)


if __name__ == "__main__":
    unittest.main()
