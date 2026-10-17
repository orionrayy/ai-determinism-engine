import math
import unittest
from datetime import datetime, timedelta, timezone

import evaluation_protocol_v1_4 as p


def bars(n=2000):
    rows = []
    price = 3000.0
    steps = (0.32, -0.24, 0.18, -0.41, 0.55, -0.12, -0.36, 0.27)
    for i in range(n):
        opening = price
        close = opening + steps[i % len(steps)]
        wick = 0.08 + (i % 5) * 0.025
        rows.append({
            "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i),
            "open": opening,
            "high": max(opening, close) + wick,
            "low": min(opening, close) - wick,
            "close": close,
            "volume": float(100 + (i % 17)),
        })
        price = close
    return rows


def alternating_signal(rows, index, rule):
    return 1 if index % 4 in (0, 1) else -1


class EvaluationProtocolV14Tests(unittest.TestCase):
    def test_long_tp_hit_is_labeled_tp_first(self):
        rows = bars(50)
        decision = 25
        entry_index = decision + 1
        rows[entry_index]["open"] = rows[decision]["close"]
        atr = p._atr_at_decision(rows, decision, 14)
        entry = rows[entry_index]["open"]
        rows[entry_index]["high"] = entry + 1.8 * atr
        rows[entry_index]["low"] = entry - 0.1 * atr
        rows[entry_index]["close"] = entry + 0.2 * atr
        result = p.label_tp_before_sl(rows, decision, 1, horizon_bars=12)
        self.assertEqual(result["actual_tp"], 1)
        self.assertEqual(result["resolution"], "take_profit_first")
        self.assertEqual(result["target_exit_index"], entry_index)
        self.assertAlmostEqual(result["gross_r"], 1.5)

    def test_same_bar_collision_is_ambiguous_not_optimistic(self):
        rows = bars(50)
        decision = 25
        entry_index = decision + 1
        atr = p._atr_at_decision(rows, decision, 14)
        entry = rows[entry_index]["open"]
        rows[entry_index]["high"] = entry + 2 * atr
        rows[entry_index]["low"] = entry - 2 * atr
        result = p.label_tp_before_sl(rows, decision, 1, horizon_bars=12)
        self.assertIsNone(result["actual_tp"])
        self.assertIsNone(result["gross_r"])
        self.assertEqual(result["resolution"], "ambiguous_same_bar")

    def test_short_stop_and_timeout_semantics(self):
        rows = bars(60)
        decision = 25
        entry_index = decision + 1
        atr = p._atr_at_decision(rows, decision, 14)
        entry = rows[entry_index]["open"]
        rows[entry_index]["high"] = entry + 1.1 * atr
        rows[entry_index]["low"] = entry - 0.2 * atr
        stop = p.label_tp_before_sl(rows, decision, -1, horizon_bars=12)
        self.assertEqual(stop["actual_tp"], 0)
        self.assertEqual(stop["resolution"], "stop_loss_first")
        self.assertAlmostEqual(stop["gross_r"], -1.0)

        quiet = bars(60)
        for i in range(entry_index, entry_index + 12):
            quiet[i]["open"] = quiet[i]["close"] = 3000.0
            quiet[i]["high"] = 3000.05
            quiet[i]["low"] = 2999.95
        timeout = p.label_tp_before_sl(quiet, decision, 1, horizon_bars=12)
        self.assertEqual(timeout["actual_tp"], 0)
        self.assertEqual(timeout["resolution"], "horizon_timeout_no_tp")

    def test_incomplete_horizon_fails_closed(self):
        rows = bars(28)
        result = p.label_tp_before_sl(rows, 20, 1, horizon_bars=12)
        self.assertIsNone(result["actual_tp"])
        self.assertEqual(result["resolution"], "truncated_horizon")

    def test_dataset_quality_does_not_claim_timezone_or_source_index_provenance(self):
        report = p.audit_dataset_quality(bars(50))
        self.assertTrue(report["timestamps_strictly_increasing"])
        self.assertEqual(report["source_index_continuity"], "not_available_in_source_schema")
        self.assertFalse(report["provider_timezone_semantics_verified"])
        self.assertIn("median_interval_seconds", report)
        self.assertIn("robust_price_jumps", report)

    def test_prequential_forecasts_require_matured_history_and_no_live_enablement(self):
        rows = bars(1800)
        protocol, events = p.run_tp_sl_protocol(
            rows, alternating_signal, ("test_rule",), min_matured_labels=10, bootstrap_replicates=20
        )
        self.assertEqual(protocol["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(protocol["decision_output"], "NO_TRADE")
        self.assertFalse(protocol["deployment_calibrated"])
        self.assertFalse(protocol["validated"])
        self.assertFalse(protocol["live_signal_enabled"])
        self.assertEqual(protocol["eligible_decisions"], len(rows) - p.MAX_HORIZON_BARS - p.WARMUP_BARS)
        self.assertTrue(events)
        self.assertTrue(all(row["p_tp_prequential"] is None or
                            row["training_labels_available"] >= 10 for row in events))
        self.assertTrue(all(row["target_exit_index"] is None or row["target_exit_index"] < len(rows)
                            for row in events))
        for side in protocol["rules"]["test_rule"].values():
            evaluation = side["oos_probability_evaluation"]
            self.assertFalse(evaluation["live_signal_enabled"])
            self.assertFalse(evaluation["deployment_calibrated"])

    def test_future_data_mutation_cannot_change_earlier_event_labels_or_forecasts(self):
        original_rows = bars(1600)
        changed_rows = [dict(row) for row in original_rows]
        for row in changed_rows[900:]:
            for key in ("open", "high", "low", "close"):
                row[key] *= 1.04
        before, before_events = p.run_tp_sl_protocol(
            original_rows, alternating_signal, ("test_rule",), min_matured_labels=10, bootstrap_replicates=20
        )
        after, after_events = p.run_tp_sl_protocol(
            changed_rows, alternating_signal, ("test_rule",), min_matured_labels=10, bootstrap_replicates=20
        )
        cut = 850
        early_before = [row for row in before_events if row["decision_index"] < cut]
        early_after = [row for row in after_events if row["decision_index"] < cut]
        self.assertEqual(len(early_before), len(early_after))
        for left, right in zip(early_before, early_after):
            self.assertEqual(left["decision_index"], right["decision_index"])
            self.assertEqual(left["actual_tp"], right["actual_tp"])
            self.assertEqual(left["target_exit_index"], right["target_exit_index"])
            self.assertEqual(left["training_labels_available"], right["training_labels_available"])
            self.assertEqual(left["p_tp_prequential"], right["p_tp_prequential"])

    def test_protocol_defines_three_chronological_partitions_and_purges_cross_boundary_labels(self):
        rows = bars(1800)
        report, events = p.run_tp_sl_protocol(
            rows, alternating_signal, ("test_rule",), min_matured_labels=10, bootstrap_replicates=20
        )
        partitions = report["chronological_partitions"]
        self.assertEqual(set(partitions), {"development", "calibration", "final_holdout"})
        self.assertEqual(partitions["development"]["end_index_exclusive"],
                         partitions["calibration"]["start_index"])
        self.assertEqual(partitions["calibration"]["end_index_exclusive"],
                         partitions["final_holdout"]["start_index"])
        self.assertTrue(report["probability_contract"]["calibration_labels_with_exit_after_holdout_start_are_purged"])
        holdout_start = partitions["final_holdout"]["start_index"]
        for side in report["rules"]["test_rule"].values():
            evaluation = side["oos_probability_evaluation"]
            self.assertEqual(evaluation["holdout_start_index"], holdout_start)
            self.assertFalse(evaluation["validated"])

    def test_input_validation_rejects_bad_signal_side_and_bad_parameters(self):
        rows = bars(100)
        with self.assertRaisesRegex(ValueError, "side_must"):
            p.label_tp_before_sl(rows, 50, 0)
        with self.assertRaisesRegex(ValueError, "horizon_bars"):
            p.label_tp_before_sl(rows, 50, 1, horizon_bars=0)
        with self.assertRaisesRegex(ValueError, "expected_interval"):
            p.audit_dataset_quality(rows, 0)


if __name__ == "__main__":
    unittest.main()
