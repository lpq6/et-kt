import unittest

import numpy as np

from a2g.metrics import binary_metrics, exceeds_floor, selection_key


class MetricsTests(unittest.TestCase):
    def test_floor_is_strict_without_rounding(self):
        self.assertFalse(exceeds_floor(0.8174))
        self.assertTrue(exceeds_floor(0.817400001))
        self.assertFalse(exceeds_floor(float("nan")))

    def test_known_auc(self):
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertEqual(result["auc"], 0.75)
        self.assertEqual(result["interaction_count"], 4)
        self.assertEqual(result["positive_count"], 2)

    def test_ties_and_endpoints(self):
        result = binary_metrics([0, 1], [0.5, 0.5])
        self.assertEqual(result["auc"], 0.5)
        self.assertAlmostEqual(binary_metrics([0, 1], [0, 1])["ece_15bin"], 1e-7)
        better = {**result, "nll": result["nll"] - 0.01}
        self.assertGreater(selection_key(better), selection_key(result))

    def test_server_clipping_precedes_auc(self):
        result = binary_metrics([0, 1], [0.0, 1e-9])
        self.assertEqual(result["auc"], 0.5)

    def test_invalid_metrics_fail_closed(self):
        for labels, pred in [
            ([], []),
            ([1, 1], [0.4, 0.8]),
            ([0, 1], [np.nan, 0.8]),
            ([0, 1], [-0.1, 0.8]),
            ([0, 1], [0.5]),
        ]:
            with self.assertRaises(ValueError):
                binary_metrics(labels, pred)


if __name__ == "__main__":
    unittest.main()
