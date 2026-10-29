import csv
import json
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
    def test_source_market_metadata_recognizes_native_indodax_idr_and_disables_fx(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "INDODAX_SOLIDR_1m.csv"
            sidecar = Path(str(csv_path) + ".meta.json")
            sidecar.write_text(json.dumps({
                "venue": "INDODAX",
                "pair": "SOL/IDR",
                "input_price_currency": "IDR",
                "dataset_sha256": "fixture",
            }), encoding="utf-8")
            market = engine.resolve_input_market_metadata(csv_path)
            rate, source = engine.resolve_quote_to_idr(csv_path, source_metadata=market)
        self.assertEqual(market, {"venue": "INDODAX", "symbol": "SOL/IDR", "price_currency": "IDR"})
        self.assertEqual(rate, 1.0)
        self.assertIn("FX conversion disabled", source)

    def test_source_market_metadata_rejects_mismatched_overrides_and_unknown_currencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "native.csv"
            sidecar = Path(str(csv_path) + ".meta.json")
            sidecar.write_text(json.dumps({
                "venue": "INDODAX", "pair": "SOL/IDR", "input_price_currency": "IDR"
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicts with capture metadata"):
                engine.resolve_input_market_metadata(csv_path, input_price_currency_override="USDT")
            sidecar.write_text(json.dumps({
                "venue": "Other", "pair": "SOL/EUR", "input_price_currency": "EUR"
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported source quote currency"):
                engine.resolve_input_market_metadata(csv_path)

    def test_idr_normalization_is_identity_and_usdt_is_converted_once(self):
        bar = make_bar(0, close=1_800_000.0)
        native = engine.normalize_bars_to_idr([bar], input_price_currency="IDR", quote_to_idr_rate=1.0)[0]
        reference = engine.normalize_bars_to_idr([bar], input_price_currency="USDT", quote_to_idr_rate=16_000.0)[0]
        self.assertEqual(native.open, bar.open)
        self.assertEqual(native.close, bar.close)
        self.assertEqual(reference.open, bar.open * 16_000.0)
        self.assertEqual(reference.close, bar.close * 16_000.0)
        with self.assertRaisesRegex(ValueError, "native IDR bars require quote_to_idr_rate=1.0"):
            engine.normalize_bars_to_idr([bar], input_price_currency="IDR", quote_to_idr_rate=16_000.0)

    def test_native_idr_backtest_config_requires_indodax_and_custom_fee_model(self):
        with self.assertRaisesRegex(ValueError, "native IDR candles require INDODAX"):
            engine.BacktestConfig(input_price_currency="IDR", quote_to_idr_rate=1.0).validate()
        with self.assertRaisesRegex(ValueError, "require explicit INDODAX_CUSTOM"):
            engine.BacktestConfig(
                input_venue="INDODAX", input_symbol="SOL/IDR",
                input_price_currency="IDR", quote_to_idr_rate=1.0,
            ).validate()

    def test_fx_resolution_prefers_explicit_override(self):
        rate, source = engine.resolve_quote_to_idr("does-not-need-to-exist.csv", 16_500.0, "test override")
        self.assertEqual(rate, 16_500.0)
        self.assertEqual(source, "test override")

    def test_fx_resolution_reads_fresh_capture_sidecar(self):
        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "SOLUSDT_1m.csv"
            sidecar = Path(str(csv_path) + ".meta.json")
            sidecar.write_text(json.dumps({
                "quote_to_idr_rate": 16_250.0,
                "quote_to_idr_date": datetime.now(timezone.utc).date().isoformat(),
                "quote_to_idr_source": "Frankfurter test reference",
            }), encoding="utf-8")
            rate, source = engine.resolve_quote_to_idr(csv_path)
            self.assertEqual(rate, 16_250.0)
            self.assertIn("Frankfurter test reference", source)

    def test_fx_resolution_rejects_missing_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "No quote-to-IDR rate supplied"):
                engine.resolve_quote_to_idr(Path(tmp) / "missing.csv")

    def test_timestamp_parses_milliseconds_seconds_and_iso(self):
        self.assertEqual(engine.parse_timestamp_ms("1760000000000"), 1_760_000_000_000)
        self.assertEqual(engine.parse_timestamp_ms("1760000000"), 1_760_000_000_000)
        self.assertEqual(engine.parse_timestamp_ms("2025-10-09T08:53:20+00:00"), 1_760_000_000_000)

    def test_ema_period_one_tracks_input(self):
        self.assertEqual(engine.ema_series([1, 3, 2], 1), [1.0, 3.0, 2.0])

    def test_ema_matches_independent_hand_calculation(self):
        actual = engine.ema_series([10.0, 12.0, 11.0], period=2)
        self.assertAlmostEqual(actual[0], 10.0, places=10)
        self.assertAlmostEqual(actual[1], 11.333333333333334, places=10)
        self.assertAlmostEqual(actual[2], 11.111111111111112, places=10)

    def test_rsi_matches_simple_rolling_gain_loss_fixture(self):
        # Deltas +1, -1, +1 give average gain 2/3 and average loss 1/3.
        actual = engine.rsi_at([1.0, 2.0, 1.0, 2.0], index=3, period=3)
        self.assertAlmostEqual(actual, 66.66666666666667, places=10)

    def test_atr_matches_independent_true_range_fixture(self):
        bars = [
            engine.Bar(BASE_MS, 10.0, 11.0, 9.0, 10.0, 1.0, BASE_MS + 59_999),
            engine.Bar(BASE_MS + 60_000, 10.0, 12.0, 9.0, 11.0, 1.0, BASE_MS + 119_999),
            engine.Bar(BASE_MS + 120_000, 11.0, 13.0, 10.0, 12.0, 1.0, BASE_MS + 179_999),
        ]
        # Both selected true ranges are 3.0, so the two-bar arithmetic ATR is 3.0.
        self.assertAlmostEqual(engine.atr_at(bars, index=2, period=2), 3.0, places=10)

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

    def test_config_rejects_negative_spread_assumption(self):
        with self.assertRaisesRegex(ValueError, "spread_bps_round_trip"):
            engine.BacktestConfig(spread_bps_round_trip=-1.0).validate()

    def test_config_rejects_negative_minimum_order_notional(self):
        with self.assertRaisesRegex(ValueError, "minimum_order_notional_idr"):
            engine.BacktestConfig(minimum_order_notional_idr=-1.0).validate()

    def test_minimum_order_notional_rejects_invalid_and_under_minimum_orders(self):
        self.assertTrue(engine.notional_meets_minimum(0.02, 1_800_000.0, 30_000.0))
        self.assertFalse(engine.notional_meets_minimum(0.01, 1_800_000.0, 30_000.0))
        self.assertFalse(engine.notional_meets_minimum(0.0, 1_800_000.0, 30_000.0))
        self.assertFalse(engine.notional_meets_minimum(float("nan"), 1_800_000.0, 30_000.0))
        self.assertFalse(engine.notional_meets_minimum(1.0, float("inf"), 30_000.0))

    def test_csv_reader_rejects_close_timestamp_outside_one_minute_candle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume"])
                for i in range(101):
                    bar = make_bar(i)
                    close_time = bar.open_time_ms + 2 * engine.ONE_MINUTE_MS - 1 if i == 25 else bar.close_time_ms
                    writer.writerow([bar.open_time_ms, close_time, bar.open, bar.high, bar.low, bar.close, bar.volume])
            with self.assertRaisesRegex(ValueError, "close timestamp does not match 1m candle interval"):
                engine.read_bars_csv(path)

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
    def test_backtest_rejects_close_timestamp_outside_one_minute_candle(self):
        bars = [make_bar(i) for i in range(120)]
        original = bars[25]
        bars[25] = engine.Bar(
            open_time_ms=original.open_time_ms, open=original.open, high=original.high, low=original.low,
            close=original.close, volume=original.volume,
            close_time_ms=original.open_time_ms + 2 * engine.ONE_MINUTE_MS - 1,
        )
        with self.assertRaisesRegex(ValueError, "close timestamp does not match 1m candle interval"):
            engine.backtest(bars, engine.BacktestConfig(), dataset_sha256="bad-close-time")

    def test_backtest_rejects_nonfinite_ohlcv_from_direct_bar_input(self):
        bars = [make_bar(i) for i in range(120)]
        original = bars[25]
        bars[25] = engine.Bar(
            open_time_ms=original.open_time_ms, open=original.open, high=original.high,
            low=original.low, close=float("nan"), volume=original.volume, close_time_ms=original.close_time_ms,
        )
        with self.assertRaisesRegex(ValueError, "non-finite OHLCV"):
            engine.backtest(bars, engine.BacktestConfig(), dataset_sha256="nan-close")

    def test_gap_through_stop_fills_at_open_not_stop_price(self):
        bar = engine.Bar(open_time_ms=BASE_MS + 60_000, open=95.0, high=100.0, low=94.0, close=96.0, volume=10.0, close_time_ms=BASE_MS + 120_000 - 1)
        reason, reference = engine.resolve_exit(bar, stop_price=98.0, target_price=105.0, bars_held=1, maximum_hold_bars=30)
        self.assertEqual(reason, "stop_loss")
        self.assertEqual(reference, bar.open)
        self.assertLess(reference, 98.0)

    def test_same_bar_stop_and_target_uses_conservative_stop(self):
        bar = engine.Bar(open_time_ms=BASE_MS + 60_000, open=100.0, high=106.0, low=97.0, close=102.0, volume=10.0, close_time_ms=BASE_MS + 120_000 - 1)
        reason, reference = engine.resolve_exit(bar, stop_price=98.0, target_price=105.0, bars_held=1, maximum_hold_bars=30)
        self.assertEqual(reason, "stop_and_target_same_bar_stop_first")
        self.assertEqual(reference, 98.0)

    def test_mark_equity_deducts_estimated_exit_friction(self):
        cfg = engine.BacktestConfig(exit_fee_bps=100.0, spread_bps_round_trip=20.0, slippage_bps_per_side=10.0)
        pos = engine.Position(
            signal_time_ms=BASE_MS, entry_time_ms=BASE_MS, entry_reference_price=100.0,
            entry_fill_price=100.2, quantity=2.0, entry_fee_idr=0.0, stop_price=99.0,
            target_price=102.0, entry_atr=1.0,
        )
        marked = engine.mark_equity(0.0, pos, 100.0, cfg)
        self.assertLess(marked, pos.quantity * 100.0)
        self.assertAlmostEqual(marked, 197.604, places=3)

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
        self.assertGreater(report["signal_evaluations"], 0)
        self.assertEqual(report["signal_candidates"], 0)
        self.assertEqual(report["entry_order_attempts"], 0)
        self.assertEqual(report["simulated_entry_fills"], 0)
        self.assertEqual(report["positions_opened"], 0)
        self.assertEqual(report["positions_closed"], 0)
        self.assertIsNone(report["expectancy_net_per_trade_idr"])
        self.assertIsNone(report["win_rate_pct"])
        self.assertIsNone(report["profit_factor"])
        self.assertGreater(sum(report["signal_rejection_counts"].values()), 0)
        self.assertGreater(report["signal_filter_rejection_events"], 0)
        self.assertTrue(report["signal_rejection_combinations"])
        self.assertIn("1m_volume_confirmation_missing", report["signal_rejection_counts"])

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

    def test_persisting_same_run_with_trades_is_idempotent(self):
        report = {
            "schema_version": engine.SCHEMA_VERSION,
            "dataset_sha256": "sample-dataset",
            "initial_equity_idr": 300_000.0,
            "final_equity_idr": 300_010.0,
            "net_pnl_idr": 10.0,
            "trades": [{"trade_id": "trade-1", "net_pnl_idr": 10.0}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.sqlite3"
            engine.persist_ledger(db_path, report, "stable-trade-run")
            engine.persist_ledger(db_path, report, "stable-trade-run")
            with sqlite3.connect(db_path) as db:
                run_count = db.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
                trades = db.execute(
                    "SELECT trade_index, trade_id, net_pnl_idr FROM paper_trades WHERE run_id=?",
                    ("stable-trade-run",),
                ).fetchall()
            self.assertEqual(run_count, 1)
            self.assertEqual(trades, [(0, "trade-1", 10.0)])

    def test_persisting_conflicting_content_for_same_run_id_fails_closed(self):
        original = {
            "schema_version": engine.SCHEMA_VERSION,
            "dataset_sha256": "sample-dataset",
            "initial_equity_idr": 300_000.0,
            "final_equity_idr": 300_010.0,
            "net_pnl_idr": 10.0,
            "trades": [{"trade_id": "trade-1", "net_pnl_idr": 10.0}],
        }
        changed = {**original, "net_pnl_idr": 11.0}
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.sqlite3"
            engine.persist_ledger(db_path, original, "conflicting-run")
            with self.assertRaisesRegex(ValueError, "different report content"):
                engine.persist_ledger(db_path, changed, "conflicting-run")
            with sqlite3.connect(db_path) as db:
                stored_pnl = db.execute(
                    "SELECT net_pnl_idr FROM backtest_runs WHERE run_id=?",
                    ("conflicting-run",),
                ).fetchone()[0]
                trades = db.execute(
                    "SELECT COUNT(*) FROM paper_trades WHERE run_id=?",
                    ("conflicting-run",),
                ).fetchone()[0]
            self.assertEqual(stored_pnl, 10.0)
            self.assertEqual(trades, 1)

    def test_persisting_same_report_detects_missing_trade_rows(self):
        report = {
            "schema_version": engine.SCHEMA_VERSION,
            "dataset_sha256": "sample-dataset",
            "initial_equity_idr": 300_000.0,
            "final_equity_idr": 300_010.0,
            "net_pnl_idr": 10.0,
            "trades": [{"trade_id": "trade-1", "net_pnl_idr": 10.0}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.sqlite3"
            engine.persist_ledger(db_path, report, "damaged-run")
            with sqlite3.connect(db_path) as db:
                db.execute("DELETE FROM paper_trades WHERE run_id=?", ("damaged-run",))
            with self.assertRaisesRegex(ValueError, "trade rows are inconsistent"):
                engine.persist_ledger(db_path, report, "damaged-run")
            with sqlite3.connect(db_path) as db:
                run_count = db.execute("SELECT COUNT(*) FROM backtest_runs WHERE run_id=?", ("damaged-run",)).fetchone()[0]
                trade_count = db.execute("SELECT COUNT(*) FROM paper_trades WHERE run_id=?", ("damaged-run",)).fetchone()[0]
            self.assertEqual(run_count, 1)
            self.assertEqual(trade_count, 0)

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
