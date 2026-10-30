import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paper_engine import Bar, ONE_MINUTE_MS
from strategy_research import (
    audit_dataset,
    block_bootstrap_ci,
    initial_stop_price,
    simulate_segment,
    run_research,
    round_quantity_down,
    signal_at,
    prepare_features,
    wilson_interval,
)


def fixture_bars(count=3000):
    rows = []
    base_raw = 1_700_000_000_000
    base = base_raw - (base_raw % (15 * 60_000))
    for i in range(count):
        close = 1_700_000.0 + i * 0.2 + (i % 9) * 0.03
        opened = base + i * ONE_MINUTE_MS
        rows.append(Bar(
            open_time_ms=opened,
            open=close - 0.05,
            high=close + 0.20,
            low=close - 0.20,
            close=close,
            volume=10.0 + (i % 5),
            close_time_ms=opened + ONE_MINUTE_MS - 1,
        ))
    return rows


class StrategyResearchTests(unittest.TestCase):
    def test_native_dataset_audit_verifies_hash_pair_currency_and_contiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.csv"
            meta_path = Path(tmp) / "native.csv.meta.json"
            bars = fixture_bars(120)
            lines = ["open_time_ms,close_time_ms,open,high,low,close,volume"]
            lines.extend(
                f"{b.open_time_ms},{b.close_time_ms},{b.open},{b.high},{b.low},{b.close},{b.volume}"
                for b in bars
            )
            raw = ("\n".join(lines) + "\n").encode()
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            meta_path.write_text(json.dumps({
                "data_status": "OK",
                "venue": "INDODAX",
                "pair": "SOL/IDR",
                "input_price_currency": "IDR",
                "fx_conversion_applied": False,
                "dataset_sha256": digest,
                "source_endpoint": "https://indodax.com/tradingview/history_v2",
                "count": len(bars),
                "captured_at_utc": "2026-10-30T00:00:00+00:00",
                "market_filters": {
                    "price_tick": "1",
                    "quantity_step": "0.00000001",
                    "minimum_base_quantity": "0.00000001",
                },
                "market_quote": {},
                "market_quote_status": "ERROR",
            }), encoding="utf-8")
            _, quality = audit_dataset(path, meta_path)
            self.assertEqual(quality["records"], 120)
            self.assertEqual(quality["missing_intervals"], 0)
            self.assertEqual(quality["duplicate_timestamps"], 0)
            self.assertEqual(quality["sha256"], digest)

    def test_native_dataset_audit_rejects_fx_conversion_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.csv"
            meta_path = Path(tmp) / "native.csv.meta.json"
            path.write_text(
                "open_time_ms,close_time_ms,open,high,low,close,volume\n"
                "1700000000000,1700000059999,10,11,9,10,1\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            base = {
                "data_status": "OK", "venue": "INDODAX", "pair": "SOL/IDR",
                "input_price_currency": "IDR", "dataset_sha256": digest,
                "source_endpoint": "https://indodax.com/tradingview/history_v2", "count": 1,
            }
            meta_path.write_text(json.dumps({**base, "fx_conversion_applied": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "converted"):
                audit_dataset(path, meta_path)
            meta_path.write_text(json.dumps({**base, "fx_conversion_applied": False, "dataset_sha256": "bad"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                audit_dataset(path, meta_path)

    def test_indicators_are_prepared_with_complete_timeframe_buckets(self):
        bars = fixture_bars(3000)
        features = prepare_features(bars)
        self.assertEqual(len(features["bars5"]), 600)
        self.assertEqual(len(features["bars15"]), 200)
        self.assertEqual(len(features["f1"]["atr"]), len(bars))
        # A decision at index i is aligned only to higher-timeframe bars already closed.
        index = 1200
        from bisect import bisect_right
        self.assertLessEqual(
            features["times15"][bisect_right(features["times15"], bars[index].close_time_ms) - 1],
            bars[index].close_time_ms,
        )

    def test_initial_stop_is_below_market_reference_even_when_costs_exceed_atr(self):
        reference = 1_700_000.0
        adverse_entry_cost = reference * 22.5 / 10_000
        fill = reference + adverse_entry_cost
        stop = initial_stop_price(reference, 10.0)
        self.assertGreater(fill, reference)
        self.assertLess(stop, reference)
        self.assertLess(stop, fill)

    def test_entry_bar_does_not_hit_stop_created_above_reference_by_cost_inflated_fill(self):
        bars = fixture_bars(3000)
        features = prepare_features(bars)
        start = 600
        signal = {
            "accepted": True,
            "reasons": [],
            "regime": "UPTREND",
            "atr": 10.0,
            "target_price": None,
            "stop_atr": 1.0,
            "target_atr": 1.6,
            "max_hold": 2,
            "target_mode": "atr",
            "selected_substrategy": "trend_breakout",
        }
        def one_signal(_name, _features, index):
            return signal if index == start else {"accepted": False, "reasons": ["test_no_repeat"], "regime": "UPTREND"}

        from unittest.mock import patch
        with patch("strategy_research.signal_at", side_effect=one_signal):
            result = simulate_segment(
                "trend_breakout",
                bars,
                features,
                start,
                start + 5,
                entry_fee_bps=20.555,
                exit_fee_bps=41.555,
                spread_bps_round_trip=25.6669,
                slippage_bps_per_side=10.0,
                min_notional_idr=25_000.0,
            )
        trade = result["closed_trade_details"][0]
        self.assertEqual(trade["exit_reason"], "time_stop")
        self.assertEqual(trade["bars_held"], 2)
        self.assertGreater(trade["entry_notional_idr"], 25_000.0)

    def test_quantity_rounding_never_exceeds_computed_risk_size(self):
        self.assertAlmostEqual(round_quantity_down(0.0123456789, "0.0001"), 0.0123)
        self.assertEqual(round_quantity_down(0.00009, "0.0001"), 0.0)

    def test_bootstrap_and_wilson_report_small_samples_honestly(self):
        self.assertEqual(block_bootstrap_ci([1.0, -1.0])["status"], "INSUFFICIENT_SAMPLE")
        self.assertIsNone(wilson_interval(0, 0))
        interval = wilson_interval(5, 10)
        self.assertLess(interval[0], 0.5)
        self.assertGreater(interval[1], 0.5)

    def test_candidate_evaluation_is_offline_and_keeps_no_trade_unpromoted(self):
        bars = fixture_bars(3000)
        report = run_research(
            bars,
            {
                "sha256": "fixture",
                "records": len(bars),
                "same_run_quote": {"spread_bps": 15.0},
                "sidecar_market_filters": {
                    "price_tick": "1",
                    "quantity_step": "0.00000001",
                    "minimum_base_quantity": "0.00000001",
                },
            },
        )
        self.assertFalse(report["execution_enabled"])
        self.assertFalse(report["paper_state_accessed"])
        self.assertEqual(report["promoted_candidate"], None)
        self.assertEqual(report["target_feasibility"]["monte_carlo"]["status"], "INSUFFICIENT_SAMPLE")
        self.assertEqual(report["chronological_splits"]["ratios"]["out_of_sample"], 0.20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
