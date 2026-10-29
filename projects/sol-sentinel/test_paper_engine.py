import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

import paper_engine as engine


BASE_MS = 1_760_000_100_000


def make_bar(index, close=None, volume=10.0):
    price = float(close if close is not None else 1000.0 + index * 0.01)
    open_ms = BASE_MS + index * engine.ONE_MINUTE_MS
    return engine.Bar(
        open_time_ms=open_ms,
        open=price - 0.01,
        high=price + 0.30,
        low=price - 0.30,
        close=price,
        volume=volume,
        close_time_ms=open_ms + engine.ONE_MINUTE_MS - 1,
    )


class DataAndIndicatorTests(unittest.TestCase):
    def test_timestamp_parses_milliseconds_seconds_and_iso(self):
        self.assertEqual(engine.parse_timestamp_ms("1760000000000"), 1_760_000_000_000)
        self.assertEqual(engine.parse_timestamp_ms("1760000000"), 1_760_000_000_000)
        self.assertEqual(engine.parse_timestamp_ms("2025-10-09T08:53:20+00:00"), 1_760_000_000_000)

    def test_ema_period_one_tracks_input(self):
        self.assertEqual(engine.ema_series([1, 3, 2], 1), [1.0, 3.0, 2.0])

    def test_rsi_insufficient_history_is_none(self):
        self.assertIsNone(engine.rsi_at([10, 11, 12], 2, 14))

    def test_aggregation_keeps_only_complete_buckets(self):
        bars = [make_bar(i) for i in range(31)]
        five_minute = engine.aggregate_bars(bars, 5)
        self.assertEqual(len(five_minute), 6)
        self.assertEqual(five_minute[0].volume, 50.0)
        self.assertEqual(five_minute[0].open, bars[0].open)
        self.assertEqual(five_minute[0].close, bars[4].close)

    def test_config_rejects_invalid_risk_fraction(self):
        with self.assertRaises(ValueError):
            engine.BacktestConfig(risk_fraction=1.2).validate()

    def test_csv_reader_rejects_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume"])
                for i in range(101):
                    if i == 50:
                        continue
                    bar = make_bar(i)
                    writer.writerow([bar.open_time_ms, bar.close_time_ms, bar.open, bar.high, bar.low, bar.close, bar.volume])
            with self.assertRaisesRegex(ValueError, "missing/non-1m candle"):
                engine.read_bars_csv(path)


class BacktestAndLedgerTests(unittest.TestCase):
    def test_monotonic_overbought_series_fails_closed_without_trade(self):
        # Monotonic price makes RSI saturate; the strategy must reject it rather than chase it.
        bars = [make_bar(i, close=1000 + i * 0.01) for i in range(600)]
        report = engine.backtest(bars, engine.BacktestConfig(), dataset_sha256="test-dataset")
        self.assertEqual(report["mode"], "OFFLINE_BACKTEST_ONLY")
        self.assertFalse(report["execution_enabled"])
        self.assertEqual(report["closed_trades"], 0)
        self.assertEqual(report["final_equity_idr"], report["initial_equity_idr"])
        self.assertGreater(report["bars_5m_complete"], 30)
        self.assertGreater(report["bars_15m_complete"], 30)

    def test_persisting_same_run_is_idempotent(self):
        bars = [make_bar(i, close=1000 + i * 0.01) for i in range(600)]
        report = engine.backtest(bars, engine.BacktestConfig(), dataset_sha256="stable-data")
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.sqlite3"
            engine.persist_ledger(db_path, report, "stable-run")
            engine.persist_ledger(db_path, report, "stable-run")
            with sqlite3.connect(db_path) as db:
                run_count = db.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
                trade_count = db.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
            self.assertEqual(run_count, 1)
            self.assertEqual(trade_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
