from __future__ import annotations

import unittest

from calibration_metrics import (
    brier_score,
    calibration_summary,
    expected_calibration_error,
    selective_metrics,
)


class CalibrationMetricTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"confidence": 0.9, "correct": True},
            {"confidence": 0.8, "correct": True},
            {"confidence": 0.2, "correct": False},
            {"confidence": 0.1, "correct": False},
        ]

    def test_brier_score(self):
        self.assertAlmostEqual(
            brier_score(self.samples),
            (0.01 + 0.04 + 0.04 + 0.01) / 4,
        )

    def test_ece_is_deterministic(self):
        self.assertAlmostEqual(
            expected_calibration_error(self.samples, bins=2),
            0.0,
        )

    def test_selective_accuracy_improves_at_high_threshold(self):
        values = selective_metrics(self.samples, thresholds=(0.5, 0.9))
        self.assertEqual(values[0]["accuracy"], 1.0)
        self.assertEqual(values[1]["coverage"], 0.25)

    def test_calibration_summary_is_unavailable_without_labels(self):
        result = calibration_summary([{"confidence": 0.9}])
        self.assertFalse(result["available"])
        self.assertIn("no_explicit", result["reason"])

    def test_calibration_summary_available_with_explicit_labels(self):
        result = calibration_summary(self.samples, bins=2)
        self.assertTrue(result["available"])
        self.assertEqual(result["sample_count"], 4)
        self.assertIn("reliability_bins", result)
        self.assertIn("selective", result)


if __name__ == "__main__":
    unittest.main()
