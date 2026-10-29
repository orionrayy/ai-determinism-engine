import json
import os
import tempfile
import unittest

from sol_sentinel import (
    Candle, DEFAULT_CONFIG, DataError, Journal, MarketSnapshot, PaperTrader,
    analyze, atr, completed_candles, ema, features, load_config, parse_candles, rsi,
)


class IndicatorTests(unittest.TestCase):
    def test_ema(self):
        self.assertEqual(ema([1, 2, 3], 1), [1.0, 2.0, 3.0])
        self.assertAlmostEqual(ema([1, 2, 3], 3)[-1], 2.25)

    def test_rsi_up_only_and_down_only(self):
        self.assertEqual(rsi([1, 2, 3, 4], 2), 100.0)
        self.assertEqual(rsi([4, 3, 2, 1], 2), 0.0)
        self.assertEqual(rsi([1, 1, 1, 1], 2), 50.0)

    def test_candle_parse_and_invalid_rows(self):
        raw = [
            [1_800_000_000, 100, 105, 98, 107, 10, 1000],
            [1_800_000_300, 105, 103, 102, 106, 12, 1200],
            ["bad", 1, 2, 1, 2, 5, 10],
        ]
        parsed = parse_candles(raw)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].high, 107)
        self.assertEqual(parsed[1].close, 103)

    def test_completed_candle_drops_open_interval(self):
        candles = [Candle(300, 1, 2, 0.5, 1.5, 3), Candle(600, 1.5, 2, 1, 1.8, 4)]
        self.assertEqual([c.ts for c in completed_candles(candles, sampled_at=899)], [300])
        self.assertEqual([c.ts for c in completed_candles(candles, sampled_at=900)], [300, 600])

    def test_feature_count_guard(self):
        candles = [Candle(i*300, 100+i, 101+i, 99+i, 100.5+i, 10) for i in range(20)]
        self.assertFalse(features(candles)["ready"])
        self.assertIsNone(atr(candles, 14))


class AnalysisTests(unittest.TestCase):
    def snapshot(self, candles=None, spread=10.0, sampled_at=1_800_000_000):
        candles = candles or []
        return MarketSnapshot(sampled_at, "test", "SOL/IDR", 99, 1_820_000, 1_819_000,
                              1_821_000, spread, 0.2, 10_000_000, candles,
                              chart_age_seconds=0, errors=[])

    def test_no_chart_means_no_signal(self):
        report = analyze(self.snapshot(), dict(DEFAULT_CONFIG), sampled_at=1_800_000_000)
        self.assertEqual(report["status"], "NO_DATA")
        self.assertTrue(any("candles" in x for x in report["blockers"]))

    def test_wide_spread_is_blocked(self):
        candles = [Candle(1_799_000_000+i*300, 100+i, 102+i, 99+i, 101+i, 10+i) for i in range(40)]
        snap = self.snapshot(candles, spread=100)
        report = analyze(snap, dict(DEFAULT_CONFIG), sampled_at=1_800_000_000)
        self.assertTrue(any("spread" in x for x in report["blockers"]))

    def test_config_limits_risk(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write(json.dumps({"risk_per_trade_pct": 0.5}))
            name = handle.name
        try:
            with self.assertRaises(ValueError):
                load_config(name)
        finally:
            os.unlink(name)

    def test_expected_edge_accounts_for_costs(self):
        candles = [Candle(1_799_000_000+i*300, 100+i, 102+i, 99+i, 101+i, 10+i) for i in range(40)]
        report = analyze(self.snapshot(candles), dict(DEFAULT_CONFIG), sampled_at=1_800_000_000)
        self.assertGreaterEqual(report["modeled_round_trip_cost_pct"],
                                DEFAULT_CONFIG["buy_fee_pct"] + DEFAULT_CONFIG["sell_fee_pct"])


class PaperJournalTests(unittest.TestCase):
    def test_initial_status_paper_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(os.path.join(tmp, "test.sqlite3"))
            status = journal.status(dict(DEFAULT_CONFIG))
            self.assertEqual(status["state"]["cash_idr"], 300_000.0)
            self.assertIsNone(status["state"]["position"])
            journal.close()


if __name__ == "__main__":
    unittest.main()
