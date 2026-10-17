import math
import random
import unittest
from datetime import datetime, timedelta, timezone

import barrier_oos_benchmark as barrier


def make_market_rows(count=1800, seed=20261017):
    rng = random.Random(seed)
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    price = 3000.0
    rows = []
    for i in range(count):
        op = price
        close = max(1.0, op + rng.gauss(0.0, 0.65))
        wick_up = rng.uniform(0.03, 1.15)
        wick_down = rng.uniform(0.03, 1.15)
        high = max(op, close) + wick_up
        low = min(op, close) - wick_down
        rows.append({
            "timestamp": stamp + timedelta(minutes=5 * i),
            "open": op,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0 + (i % 23),
        })
        price = close
    return rows


class BarrierOOSBenchmarkTests(unittest.TestCase):
    def test_feature_values_do_not_depend_on_future_bars(self):
        rows = make_market_rows(120)
        changed = [dict(row) for row in rows]
        index = 55
        for j in range(index + 1, len(changed)):
            changed[j]["open"] *= 1.08
            changed[j]["high"] *= 1.08
            changed[j]["low"] *= 1.08
            changed[j]["close"] *= 1.08
        original_features = barrier.prepare_features(rows)
        changed_features = barrier.prepare_features(changed)
        self.assertEqual(original_features[index], changed_features[index])
        self.assertTrue(all(math.isfinite(x) for x in original_features[index]))

    def test_same_bar_double_touch_uses_conservative_stop_first(self):
        rows = make_market_rows(40)
        index = 8
        entry = rows[index + 1]["open"]
        rows[index + 1]["high"] = entry + 4.0
        rows[index + 1]["low"] = entry - 4.0
        long_result = barrier.label_barrier(rows, index, 1.0, "long")
        short_result = barrier.label_barrier(rows, index, 1.0, "short")
        self.assertEqual(long_result["outcome"], "SL_FIRST_SAME_BAR_AMBIGUITY")
        self.assertEqual(short_result["outcome"], "SL_FIRST_SAME_BAR_AMBIGUITY")
        self.assertEqual(long_result["label"], 0)
        self.assertEqual(short_result["label"], 0)
        self.assertEqual(long_result["exit_index"], index + 1)

    def test_gap_through_barrier_is_checked_at_open_before_ohlc_range(self):
        rows = make_market_rows(40)
        index = 8
        rows[index + 1]["open"] = rows[index]["close"] - 2.0
        rows[index + 1]["high"] = rows[index + 1]["open"] + 0.3
        rows[index + 1]["low"] = rows[index + 1]["open"] - 0.3
        result = barrier.label_barrier(rows, index, 1.0, "long")
        self.assertEqual(result["outcome"], "SL_FIRST")
        self.assertEqual(result["label"], 0)

    def test_chronological_split_purges_unmatured_labels_at_each_boundary(self):
        rows = [
            {"decision_index": 58, "target_exit_index": 59, "actual_tp": 0},
            {"decision_index": 59, "target_exit_index": 60, "actual_tp": 1},
            {"decision_index": 60, "target_exit_index": 79, "actual_tp": 0},
            {"decision_index": 70, "target_exit_index": 80, "actual_tp": 1},
            {"decision_index": 80, "target_exit_index": 81, "actual_tp": 0},
        ]
        parts = barrier.split_matured_samples(rows, 100)
        self.assertEqual([x["decision_index"] for x in parts["train"]], [58])
        self.assertEqual([x["decision_index"] for x in parts["calibration"]], [60])
        self.assertEqual([x["decision_index"] for x in parts["holdout"]], [80])
        self.assertEqual(parts["train_purged_unmatured"], 1)
        self.assertEqual(parts["calibration_purged_unmatured"], 1)
        self.assertLess(max(x["target_exit_index"] for x in parts["train"]), 60)
        self.assertLess(max(x["target_exit_index"] for x in parts["calibration"]), 80)

    def test_long_and_short_labels_are_separate_and_mature_after_decision(self):
        rows = make_market_rows(300)
        samples = barrier.build_samples(rows)
        self.assertGreater(len(samples["long"]), 0)
        self.assertEqual(len(samples["long"]), len(samples["short"]))
        self.assertNotEqual(samples["long"][0]["outcome"], "NOT_DEFINED")
        for side in ("long", "short"):
            for item in samples[side]:
                self.assertGreater(item["target_exit_index"], item["decision_index"])
                self.assertIn(item["actual_tp"], (0, 1))
                self.assertEqual(len(item["features"]), len(barrier.FEATURE_NAMES))

    def test_model_and_calibration_report_never_enables_deployment(self):
        result = barrier.run_barrier_oos(make_market_rows(1800), source_id="synthetic_test_only", bootstrap_replicates=20)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(result["decision_gate"], "NO_TRADE")
        self.assertFalse(result["predictive_model_validated"])
        self.assertFalse(result["deployment_calibrated"])
        self.assertFalse(result["live_signal_enabled"])
        self.assertFalse(result["live_order_execution_enabled"])
        self.assertEqual(result["production_approval"], "BLOCKED")
        for side in ("long", "short"):
            item = result["sides"][side]
            self.assertEqual(item["status"], "DIAGNOSTIC_ONLY")
            self.assertTrue(item["model_training"]["labels_matured_before_calibration_period"])
            self.assertTrue(item["calibration"]["all_calibration_labels_matured_before_holdout"])
            self.assertGreaterEqual(item["holdout"]["rows"], 100)
            self.assertGreaterEqual(item["raw_holdout_metrics"]["brier_score"], 0.0)
            self.assertLessEqual(item["raw_holdout_metrics"]["brier_score"], 1.0)
            self.assertFalse(item["deployment_calibrated"])
            self.assertFalse(item["live_signal_enabled"])

    def test_insufficient_history_fails_closed(self):
        result = barrier.run_barrier_oos(make_market_rows(100), source_id="tiny_fixture")
        self.assertEqual(result["status"], "INSUFFICIENT_HISTORY")
        self.assertFalse(result["deployment_calibrated"])
        self.assertFalse(result["live_signal_enabled"])
        self.assertFalse(result["live_order_execution_enabled"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
