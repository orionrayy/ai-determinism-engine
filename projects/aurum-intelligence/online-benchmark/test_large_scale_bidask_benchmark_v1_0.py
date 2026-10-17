import csv
import io
import unittest
from datetime import datetime, timezone

import numpy as np

import large_scale_bidask_benchmark_v1_0 as b


DTYPE = np.dtype([
    ("ts", "i8"),
    ("bo", "f8"), ("bh", "f8"), ("bl", "f8"), ("bc", "f8"),
    ("ao", "f8"), ("ah", "f8"), ("al", "f8"), ("ac", "f8"),
])


def fixture(n=2500):
    start = int(datetime(2023, 1, 3, tzinfo=timezone.utc).timestamp() * 1000)
    d = np.empty(n, dtype=DTYPE)
    x = np.arange(n, dtype=np.float64)
    mid_close = 2000.0 + 0.0005 * x + 0.7 * np.sin(x / 13.0) + 0.2 * np.sin(x / 3.0)
    mid_open = np.concatenate(([mid_close[0]], mid_close[:-1]))
    mid_high = np.maximum(mid_open, mid_close) + 0.25 + 0.05 * np.sin(x / 7.0) ** 2
    mid_low = np.minimum(mid_open, mid_close) - 0.25 - 0.05 * np.cos(x / 5.0) ** 2
    d["ts"] = start + np.arange(n, dtype=np.int64) * 60_000
    for side_prefix, shift in (("b", -0.10), ("a", 0.10)):
        d[side_prefix + "o"] = mid_open + shift
        d[side_prefix + "h"] = mid_high + shift
        d[side_prefix + "l"] = mid_low + shift
        d[side_prefix + "c"] = mid_close + shift
    return d


def month_csv(year=2023, month=1, prices=(2000.0, 2000.5, 1999.5, 2000.1)):
    ts = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["timestamp", "open", "high", "low", "close"])
    writer.writerow([ts, *prices])
    return stream.getvalue().encode("utf-8")


class LargeScaleBidAskBenchmarkTests(unittest.TestCase):
    def test_month_parser_accepts_audited_utc_epoch_milliseconds(self):
        payload = month_csv()
        stamps, prices = b.parse_month_csv(payload, 2023, 1, "bid", expected_lines=2)
        self.assertEqual(stamps.tolist(), [int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)])
        self.assertEqual(prices.shape, (1, 4))
        self.assertAlmostEqual(prices[0, 1], 2000.5)

    def test_month_parser_rejects_invalid_ohlc_and_duplicate_times(self):
        with self.assertRaisesRegex(b.BenchmarkError, "invalid_ohlc_geometry"):
            b.parse_month_csv(month_csv(prices=(2001, 2000, 1999, 2000)), 2023, 1, "bid")
        ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        duplicate = (
            "timestamp,open,high,low,close\n"
            f"{ts},2000,2001,1999,2000\n"
            f"{ts},2000,2001,1999,2000\n"
        ).encode()
        with self.assertRaisesRegex(b.BenchmarkError, "timestamps_not_strictly_increasing"):
            b.parse_month_csv(duplicate, 2023, 1, "ask")

    def test_published_audit_gate_requires_ok_status_and_matching_calendar_counts(self):
        payload = (
            "Name,Bytes,Lines,CalendarExpectedLines,Status,Note,FirstUtc,LastUtc,GapCount,MaxGapMinutes\n"
            "xauusd_bid_m1_2023_01.csv,123,44641,44641,ok,calendar,2023-01-01,2023-01-31,0,0\n"
        ).encode()
        audit = b.parse_audit(payload, "bid")
        entry = b.audit_month_entry(audit, 2023, 1, "bid")
        self.assertEqual(entry["lines"], 44641)
        self.assertEqual(entry["status"], "ok")
        bad = dict(audit)
        bad[(2023, 1)] = dict(entry, status="bad")
        with self.assertRaisesRegex(b.BenchmarkError, "audit_status_not_ok"):
            b.audit_month_entry(bad, 2023, 1, "bid")

    def test_market_quality_and_spread_metrics_handle_large_candle_arrays(self):
        d = fixture(1000)
        q = b.market_quality(d)
        self.assertEqual(q["row_count"], 1000)
        self.assertEqual(q["invalid_or_crossed_quote_rows"], 0)
        self.assertEqual(q["non_minute_intervals_including_closed_sessions"], 0)
        self.assertGreater(q["spread_close_price_units"]["median"], 0)
        self.assertGreater(q["spread_close_bps_mid"]["p95"], 0)

    def test_candidate_windows_are_unique_contiguous_and_minute_aligned(self):
        d = fixture(2200)
        anchors, eligible = b.candidate_anchor_indices(d, required_count=250)
        self.assertEqual(len(anchors), 250)
        self.assertEqual(len(np.unique(anchors)), 250)
        self.assertGreaterEqual(eligible, 250)
        self.assertTrue(np.all(np.diff(d["ts"][anchors]) > 0))
        self.assertTrue(np.all(d["ts"][anchors] - d["ts"][anchors - b.WARMUP_BARS] == b.WARMUP_BARS * 60_000))
        self.assertTrue(np.all(d["ts"][anchors + b.HORIZON_BARS] - d["ts"][anchors] == b.HORIZON_BARS * 60_000))

    def test_scenario_matrix_produces_two_sides_times_three_atr_targets(self):
        d = fixture(800)
        features = b.indicators(d)
        anchors = np.array([100, 180, 260, 340], dtype=np.int64)
        labels, counts, scenario_count, _ = b.scenario_matrix(d, anchors, features)
        self.assertEqual(scenario_count, 4 * 2 * len(b.TP_MULTIPLIERS))
        self.assertEqual(labels.shape, (4, 2))
        self.assertEqual(sum(v["scenarios"] for v in counts.values()), scenario_count)
        self.assertEqual(len(counts), 2 * len(b.TP_MULTIPLIERS))

    def test_evaluate_path_reports_same_bar_collision_without_optimistic_resolution(self):
        d = fixture(300)
        features = b.indicators(d)
        idx = 120
        features["atr14"][idx] = 0.5
        entry = float(d["ao"][idx + 1])
        d["bh"][idx + 1] = entry + 0.8
        d["bl"][idx + 1] = entry - 0.8
        d["ah"][idx + 1] = d["bh"][idx + 1] + 0.2
        d["al"][idx + 1] = d["bl"][idx + 1] + 0.2
        result = b.evaluate_path(d, features, idx, 1, 1.0, horizon=3)
        self.assertEqual(result["status"], "ambiguous_same_bar")
        self.assertIsNone(result["label"])

    def test_prequential_final_test_priors_are_frozen(self):
        n = 120
        anchors = np.arange(100, 100 + 90 * n, 90, dtype=np.int64)
        signals = {name: np.ones(n, dtype=np.int8) for name in b.STRATEGIES}
        labels = np.empty((n, 2), dtype=np.int8)
        for i in range(n):
            labels[i, 0] = i % 2
            labels[i, 1] = (i // 2) % 2
        evaluation, rows = b.prequential_evaluation(fixture(11000), anchors, signals, labels, minimum_scored_events=100)
        final_rows = [row for row in rows if row["split"] == "final_holdout"]
        self.assertGreater(len(final_rows), 100)
        for rule in b.STRATEGIES:
            per_rule = [row for row in final_rows if row["rule"] == rule and row["side"] == "long"]
            self.assertTrue(per_rule)
            self.assertEqual(len({row["prior_matured_labels"] for row in per_rule}), 1)
            self.assertTrue(all(row["prior_matured_labels"] >= 0 for row in per_rule))
        self.assertTrue(evaluation["chronological_partitions"]["final_holdout"]["final_test_priors_frozen"])

    def test_metrics_and_block_bootstrap_are_deterministic_and_descriptive(self):
        p = [0.4, 0.6] * 100
        y = [0, 1] * 100
        metrics = b._metric_pack(p, y)
        self.assertEqual(metrics["n"], 200)
        self.assertIsNotNone(metrics["reliability"]["ece_descriptive"])
        a = b._moving_block_ci([0.01, -0.01, 0.02, 0.0] * 100, replicates=40, block_length=20, seed=7)
        c = b._moving_block_ci([0.01, -0.01, 0.02, 0.0] * 100, replicates=40, block_length=20, seed=7)
        self.assertEqual(a, c)
        self.assertEqual(a["status"], "DIAGNOSTIC_ONLY")


if __name__ == "__main__":
    unittest.main()
