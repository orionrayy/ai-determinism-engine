import csv
import io
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import ci_benchmark as b

def fixture(n=100):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["datetime", "open", "high", "low", "close", "volume"])
    start, price = datetime(2026, 1, 1), 3000.0
    for i in range(n):
        ts = start + timedelta(minutes=5*i)
        o, c = price, price + (0.4 if i % 3 else -0.25)
        writer.writerow([ts.isoformat(), o, max(o,c)+.2, min(o,c)-.2, c, 100+i])
        price = c
    return stream.getvalue().encode()

class OnlineBenchmarkTests(unittest.TestCase):
    def test_dataset_schema_and_spread_out_cases(self):
        rows = b.parse_csv(fixture(), 30)
        self.assertEqual(len(rows), 100)
        indices = b.choose_indices(35954, 50)
        self.assertEqual(len(set(indices)), 50)
        self.assertGreater(indices[-1], 35000)

    def test_invalid_timestamp_and_ohlc_fail_closed(self):
        with self.assertRaisesRegex(b.DatasetError, "strictly increasing"):
            b.parse_csv(b"timestamp,open,high,low,close,volume\n2026-01-01T00:05:00,1,2,1,1,1\n2026-01-01T00:00:00,1,2,1,1,1\n", 2)
        with self.assertRaisesRegex(b.DatasetError, "invalid OHLCV"):
            b.parse_csv(b"timestamp,open,high,low,close,volume\n2026-01-01T00:00:00,1,.5,.9,1,1\n", 1)

    def test_timezone_mixing_is_rejected(self):
        with self.assertRaisesRegex(b.DatasetError, "mixed timezone"):
            b.parse_csv(b"timestamp,open,high,low,close,volume\n2026-01-01T00:00:00Z,1,2,1,1,1\n2026-01-01T00:05:00,1,2,1,1,1\n", 2)

    def test_30_cases_produce_150_rule_evaluations(self):
        rows = b.parse_csv(fixture(), 30)
        result = b.case_replay(rows, 30)
        self.assertEqual(len(result), 150)
        self.assertEqual(len({x["decision_time"] for x in result}), 30)
        self.assertTrue(all(x["cost_is_assumption"] for x in result))

    def test_480_market_paths_and_240_software_stress_cases(self):
        rows = b.parse_csv(fixture(), 30)
        market = b.market_scenarios(rows, 2)
        software = b.software_scenarios(rows)
        self.assertEqual(len(market), 480)
        self.assertTrue(all(x["passed"] for x in market), [x for x in market if not x["passed"]][:4])
        self.assertEqual(len(software), 240)
        self.assertTrue(all(x["passed"] for x in software), [x for x in software if not x["passed"]][:4])


    def test_full_history_replay_covers_all_rows_in_five_disjoint_folds(self):
        rows = b.parse_csv(fixture(180), 30)
        result = b.run_full_history_replay(rows, warmup_bars=30)
        self.assertEqual(result["decision_opportunities"], 149)
        self.assertEqual(result["tail_holdout_start_index"], 144)
        self.assertFalse(result["rules_fitted_or_tuned"])
        self.assertEqual(sum(f["opportunities"] for f in result["chronological_folds"]), 149)
        self.assertEqual(result["chronological_folds"][0]["start_index"], 30)
        self.assertEqual(result["chronological_folds"][-1]["start_index"], result["tail_holdout_start_index"])
        self.assertEqual(result["chronological_folds"][-1]["end_index_exclusive"], 179)
        self.assertEqual(result["rules"]["ema5_vs_ema10"]["chronological_fold_5"], result["rules"]["ema5_vs_ema10"]["chronological_tail_holdout"])
        for left, right in zip(result["chronological_folds"], result["chronological_folds"][1:]):
            self.assertEqual(left["end_index_exclusive"], right["start_index"])
        for rule in b.RULES:
            segments = result["rules"][rule]
            self.assertEqual(segments["full_history"]["opportunities"], 149)
            self.assertEqual(segments["early_segment"]["opportunities"] + segments["chronological_tail_holdout"]["opportunities"], 149)

    def test_optimized_rule_signals_match_legacy_implementation(self):
        rows = b.parse_csv(fixture(160), 30)
        import statistics
        def legacy(i, rule):
            closes = [x["close"] for x in rows[:i+1]]
            p = closes[-1]
            if rule == "flat_no_trade": return 0
            if rule == "last_return_sign": return int(p > closes[-2]) - int(p < closes[-2])
            if rule == "ema5_vs_ema10":
                xs = closes[-60:]
                return int(b.ema(xs, 5) > b.ema(xs, 10)) - int(b.ema(xs, 5) < b.ema(xs, 10))
            if rule == "mean_reversion_z10":
                xs = closes[-10:]
                sd = statistics.pstdev(xs)
                z = (p-statistics.mean(xs))/sd if sd else 0
                return -1 if z > 1 else 1 if z < -1 else 0
            previous = rows[max(0, i-20):i]
            return 1 if p > max(x["high"] for x in previous) else -1 if p < min(x["low"] for x in previous) else 0
        for i in (30, 51, 79, 120, 158):
            for rule in b.RULES:
                self.assertEqual(b.get_signal(rows, i, rule), legacy(i, rule))

    def test_prequential_probability_baselines_do_not_leak_future_labels(self):
        rows = b.parse_csv(fixture(220), 30)
        original = list(b.iter_prequential_probability_rows(rows, 30, 40))
        altered = [dict(row) for row in rows]
        for i in range(150, len(altered)):
            for field in ("open", "high", "low", "close"):
                altered[i][field] *= 1.001
        changed = list(b.iter_prequential_probability_rows(altered, 30, 40))
        self.assertEqual(len(original), len(changed))
        for before, after in zip(original[:110], changed[:110]):
            self.assertEqual(before["p_expanding_laplace"], after["p_expanding_laplace"])
            self.assertEqual(before["p_rolling_laplace"], after["p_rolling_laplace"])
            self.assertEqual(before["target_index"], after["target_index"])

    def test_probability_metrics_include_reliability_and_block_intervals(self):
        rows = b.parse_csv(fixture(900), 30)
        result = b.run_prequential_probability_diagnostics(rows, 30, 120, 50)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(result["predictions"], 869)
        self.assertFalse(result["validated_model"])
        self.assertFalse(result["calibrated_model"])
        self.assertEqual(set(result["methods"]), {"constant_50pct", "expanding_laplace_base_rate", "rolling_120_bar_laplace_base_rate"})
        for method in result["methods"].values():
            self.assertGreaterEqual(method["brier_score"], 0)
            self.assertGreaterEqual(method["log_loss"], 0)
            self.assertEqual(sum(item["n"] for item in method["reliability"]["bins"]), result["predictions"])
            self.assertIsNotNone(method["reliability"]["ece_descriptive"])
            self.assertIsNotNone(method["reliability"]["mce_descriptive"])
        for interval in result["uncertainty"].values():
            self.assertLessEqual(interval["lower_95"], interval["upper_95"])
            self.assertEqual(interval["replicates"], 50)

    def test_probability_short_history_fails_closed(self):
        rows = b.parse_csv(fixture(80), 30)
        result = b.run_prequential_probability_diagnostics(rows, 60)
        self.assertEqual(result["status"], "insufficient_history")
        self.assertFalse(result["calibrated_model"])
        self.assertEqual(result["predictions"], 19)

    def test_end_to_end_outputs_replayable_reports_and_no_trade(self):
        with tempfile.TemporaryDirectory() as td:
            result = b.run(fixture(), Path(td), 30, 30, 2, "test-fixture")
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["real_data_evaluations"], 150)
            self.assertEqual(result["full_history_decision_opportunities"], 39)
            self.assertEqual(result["prequential_probability_predictions"], 39)
            self.assertEqual(result["market_scenarios"], 480)
            self.assertEqual(result["software_scenarios"], 240)
            self.assertEqual(result["decision_output"], "NO_TRADE")
            for name in ("online_dataset_benchmark.json", "online_dataset_benchmark.md", "real_data_case_results.csv", "market_scenario_results.csv", "software_scenario_results.csv"):
                self.assertTrue((Path(td)/name).exists(), name)

if __name__ == "__main__":
    unittest.main()
