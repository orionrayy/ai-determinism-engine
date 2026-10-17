import csv
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import multi_source_benchmark as multi


def make_rows(count=120, start=None, price_offset=0.0):
    start = start or datetime(2026, 3, 1, tzinfo=timezone.utc)
    rows = []
    price = 3000.0 + price_offset
    for index in range(count):
        stamp = start + timedelta(minutes=5 * index)
        open_price = price
        close = price + (0.4 if index % 3 else -0.25)
        rows.append({
            "timestamp": stamp, "open": open_price,
            "high": max(open_price, close) + 0.2,
            "low": min(open_price, close) - 0.2,
            "close": close, "volume": 100.0 + index,
        })
        price = close
    return rows


def make_csv(count=120):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["datetime", "open", "high", "low", "close", "volume"])
    start = datetime(2026, 3, 1)
    price = 3000.0
    for index in range(count):
        stamp = start + timedelta(minutes=5 * index)
        open_price = price
        close = price + (0.4 if index % 3 else -0.25)
        writer.writerow([
            stamp.isoformat(), open_price, max(open_price, close) + 0.2,
            min(open_price, close) - 0.2, close, 100.0 + index,
        ])
        price = close
    return stream.getvalue().encode("utf-8")


class MultiSourceBenchmarkTests(unittest.TestCase):
    def test_catalog_uses_unique_fixed_https_allowlisted_sources(self):
        ids = [source["id"] for source in multi.SOURCES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 3)
        for source in multi.SOURCES:
            prefix = "https://raw.githubusercontent.com/" + source["repository"] + "/"
            self.assertTrue(source["url"].startswith(prefix))
            self.assertGreater(source["max_bytes"], 0)
            self.assertGreaterEqual(source["min_rows"], 1000)
            self.assertFalse(source.get("source_independence_verified", False))

    def test_naive_timestamps_fail_without_explicit_timezone(self):
        rows = [{"timestamp": datetime(2026, 1, 1)}]
        with self.assertRaisesRegex(multi.SourceError, "naive_timestamp"):
            multi.normalize_utc(rows, None)

    def test_naive_timestamps_are_explicitly_normalized_to_utc(self):
        rows = [{"timestamp": datetime(2026, 1, 1)}]
        multi.normalize_utc(rows, "UTC")
        self.assertEqual(rows[0]["timestamp"].utcoffset(), timedelta(0))
        self.assertEqual(rows[0]["timestamp"].tzinfo, timezone.utc)

    def test_chronology_audit_reports_gaps_without_silently_filling(self):
        rows = make_rows(8)
        rows[4]["timestamp"] += timedelta(minutes=15)
        for index in range(5, len(rows)):
            rows[index]["timestamp"] += timedelta(minutes=15)
        quality = multi.audit_quality(rows, 300)
        self.assertGreater(quality["intervals_over_1_5x_expected"], 0)
        self.assertFalse(quality["gaps_filled"])
        self.assertTrue(quality["strictly_increasing_timestamps"])

    def test_cross_source_overlap_is_diagnostic_and_does_not_merge(self):
        rows_a = make_rows(1200)
        rows_b = make_rows(1200, price_offset=0.2)
        spec_a = {"id": "a", "interval_seconds": 300}
        spec_b = {"id": "b", "interval_seconds": 300}
        result = multi.compare_m5(spec_a, rows_a, spec_b, rows_b)
        self.assertEqual(result["exact_timestamp_matches"], 1200)
        self.assertFalse(result["datasets_merged"])
        self.assertIn("UNVERIFIED", result["independence_status"])
        self.assertAlmostEqual(result["close_abs_delta_price_units"]["median"], 0.2, places=6)
        self.assertEqual(result["close_relative_delta"]["compatibility_status"], "COMPARABLE")
        probe = result["temporal_alignment_probe"]
        self.assertIn(0, probe["tested_offsets_seconds"])
        self.assertFalse(probe["automatic_timestamp_shifting"])
        self.assertGreaterEqual(len(probe["results"]), 10)
        self.assertIsNotNone(result["close_to_close_direction_agreement"])

    def test_large_price_level_divergence_is_flagged_even_when_direction_matches(self):
        rows_a = make_rows(1200)
        rows_b = make_rows(1200, price_offset=100.0)
        spec_a = {"id": "a", "interval_seconds": 300}
        spec_b = {"id": "b", "interval_seconds": 300}
        result = multi.compare_m5(spec_a, rows_a, spec_b, rows_b)
        self.assertEqual(result["exact_timestamp_matches"], 1200)
        self.assertEqual(result["close_relative_delta"]["compatibility_status"], "DIVERGENT")
        self.assertGreater(result["close_relative_delta"]["median_fraction"], 0.0025)
        self.assertFalse(result["cross_source_confirmation_allowed"])
        self.assertFalse(result["datasets_merged"])
        probe = result["temporal_alignment_probe"]
        self.assertEqual(probe["diagnosis"], "NO_OBVIOUS_BOUNDED_SHIFT_FIX")
        self.assertFalse(probe["material_shift_candidate"])
        self.assertIsNone(probe["best_offset_seconds_diagnostic_only"])
        self.assertGreaterEqual(probe["relative_median_improvement_fraction"], 0.0)
        self.assertLess(probe["relative_median_improvement_fraction"], 0.2)

    def test_overlap_fails_closed_when_independent_match_count_is_too_small(self):
        spec_a = {"id": "a", "interval_seconds": 300}
        spec_b = {"id": "b", "interval_seconds": 300}
        with self.assertRaisesRegex(multi.SourceError, "insufficient_exact_timestamp_overlap"):
            multi.compare_m5(spec_a, make_rows(50), spec_b, make_rows(50))

    def test_overlap_rejects_mismatched_timeframes(self):
        spec_a = {"id": "a", "interval_seconds": 60}
        spec_b = {"id": "b", "interval_seconds": 300}
        with self.assertRaisesRegex(multi.SourceError, "same_timeframe_m5_required"):
            multi.compare_m5(spec_a, make_rows(1200), spec_b, make_rows(1200))

    def test_single_dataset_benchmark_runs_without_enabling_live_signals(self):
        spec = {
            "id": "fixture_only", "repository": "example/fixture",
            "url": "https://raw.githubusercontent.com/example/fixture/main/data.csv",
            "license_url": "https://example.invalid/license",
            "license_status": "synthetic unit-test fixture; no market evidence",
            "semantics": "synthetic fixture only",
            "interval_seconds": 300, "naive_timezone": "UTC",
            "min_rows": 100, "max_bytes": 1024 * 1024,
        }
        result, rows = multi.run_one(spec, make_csv(160), cases=30, bootstrap_replicates=20)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["real_data_benchmark"]["rule_comparisons"], 150)
        self.assertEqual(result["full_history_replay"]["decision_opportunities"], 99)
        self.assertEqual(result["decision_output"], "NO_TRADE")
        self.assertFalse(result["predictive_model_validated"])
        self.assertFalse(result["live_signal_enabled"])
        self.assertEqual(len(rows), 160)

    def test_report_writer_produces_valid_json_and_real_markdown_newlines(self):
        overlap = {
            "status": "FAIL", "source_a": "a", "source_b": "b",
            "exact_timestamp_matches": 0, "close_abs_delta_median": 0.0,
            "close_abs_delta_p95": 0.0, "close_abs_delta_max": 0.0,
            "close_to_close_direction_agreement": None,
            "datasets_merged": False, "independence_status": "UNVERIFIED",
        }
        report = multi.build_report([], overlap, min_total_rows=100000)
        with tempfile.TemporaryDirectory() as temporary:
            paths = multi.write_report_files(report, __import__("pathlib").Path(temporary))
            loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
            markdown = paths["markdown"].read_text(encoding="utf-8")
        self.assertEqual(loaded["status"], "FAIL")
        self.assertGreater(markdown.count(chr(10)), 2)
        self.assertNotIn(chr(92) + "n", markdown.splitlines()[0])

    def test_report_semantics_distinguish_pipeline_success_from_model_success(self):
        # Fail-closed aggregate contract; empty evidence is never labeled PASS.
        result = multi.build_report([], {
            "status": "FAIL", "exact_timestamp_matches": 0,
        }, min_total_rows=100000)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["predictive_model_validated"])
        self.assertFalse(result["model_calibrated_for_deployment"])
        self.assertFalse(result["live_order_execution_enabled"])
        self.assertFalse(result["production_ready"])
        self.assertFalse(result["cross_source_confirmation_allowed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
