import unittest
from datetime import datetime, timedelta, timezone

from source_concordance_v1_0 import compare_source_datasets


def source_rows(n=1200, aware=True, price_shift=0.0, start=3000.0):
    rows = []
    price = start
    for i in range(n):
        stamp = datetime(2026, 1, 1) + timedelta(minutes=5 * i)
        if aware:
            stamp = stamp.replace(tzinfo=timezone.utc)
        close = price + (0.2 if i % 3 else -0.1) + price_shift
        rows.append({
            "timestamp": stamp,
            "open": price + price_shift,
            "high": max(price, close) + 0.1,
            "low": min(price, close) - 0.1,
            "close": close,
            "volume": 10.0 + i % 5,
        })
        price = close
    return rows


class SourceConcordanceTests(unittest.TestCase):
    def test_naive_utc_and_aware_utc_align_without_merging_sources(self):
        primary = source_rows(1200, aware=True)
        secondary = source_rows(1200, aware=False, price_shift=0.01)
        report, sample = compare_source_datasets(primary, secondary)
        self.assertEqual(report["status"], "CROSS_SOURCE_DIAGNOSTIC_ONLY")
        self.assertEqual(report["common_timestamps"], 1200)
        self.assertEqual(report["sample_rows"], 250)
        self.assertEqual(len(sample), 250)
        self.assertFalse(report["provider_provenance_independently_verified"])
        self.assertFalse(report["upstream_data_rights_independently_verified"])
        self.assertEqual(report["decision_output"], "NO_TRADE")
        self.assertEqual(report["comparison_mode"], "exact_timestamp_overlap_only_no_dataset_concatenation")
        self.assertGreater(report["absolute_close_difference_basis_points"]["median"], 0)

    def test_partial_overlap_is_counted_explicitly(self):
        primary = source_rows(1200, aware=True)
        secondary = source_rows(800, aware=False)
        report, _ = compare_source_datasets(primary, secondary)
        self.assertEqual(report["status"], "INSUFFICIENT_OVERLAP")
        self.assertEqual(report["common_timestamps"], 800)
        self.assertEqual(report["primary_only_timestamps"], 400)

    def test_duplicate_timestamps_fail_closed(self):
        primary = source_rows(40, aware=True)
        secondary = source_rows(40, aware=True)
        secondary[1]["timestamp"] = secondary[0]["timestamp"]
        with self.assertRaisesRegex(ValueError, "duplicate_timestamp_in_secondary"):
            compare_source_datasets(primary, secondary)

    def test_source_price_fields_are_compared_without_volume_assumption(self):
        primary = source_rows(1200, aware=True)
        secondary = source_rows(1200, aware=False, price_shift=0.01)
        report, _ = compare_source_datasets(primary, secondary)
        self.assertIn("open", report["absolute_price_difference_units"])
        self.assertIn("high", report["absolute_price_difference_units"])
        self.assertIn("low", report["absolute_price_difference_units"])
        self.assertIn("close", report["absolute_price_difference_units"])
        self.assertIn("volume semantics", report["volume_comparison"])


    def test_material_price_basis_disagreement_is_not_mislabeled_as_clean_diagnostic(self):
        primary = source_rows(1200, aware=True)
        secondary = source_rows(1200, aware=False, price_shift=10.0)
        report, _ = compare_source_datasets(primary, secondary)
        self.assertEqual(report["status"], "SOURCE_DISAGREEMENT_REVIEW_REQUIRED")
        self.assertTrue(report["source_disagreement_screen"]["material_disagreement_flag"])
        self.assertGreater(report["absolute_close_difference_basis_points"]["p95"], 20.0)
        self.assertIn("signed_close_difference_units_primary_minus_secondary", report)
        self.assertTrue(report["source_disagreement_screen"]["thresholds_are_heuristics_not_execution_price_truth"])

    def test_invalid_disagreement_threshold_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "max_median_close_difference_bps"):
            compare_source_datasets(source_rows(30), source_rows(30), max_median_close_difference_bps=float("nan"))

if __name__ == "__main__":
    unittest.main()
