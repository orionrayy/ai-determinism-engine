from __future__ import annotations
import unittest
from datetime import datetime, timedelta, timezone
from quote_path_replay_v1 import BidAskBar, simulate_bracket_trade


def bar(minute, *, bo=100, bh=101, bl=99, bc=100, ao=None, ah=None, al=None, ac=None):
    ao = bo + 0.2 if ao is None else ao
    ah = bh + 0.2 if ah is None else ah
    al = bl + 0.2 if al is None else al
    ac = bc + 0.2 if ac is None else ac
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return BidAskBar(stamp, bo, bh, bl, bc, ao, ah, al, ac)


class QuotePathReplayTests(unittest.TestCase):
    def test_long_uses_ask_entry_and_bid_exit(self):
        bars = [bar(0), bar(1, bo=100, bh=100.5, bl=99.5, bc=100), bar(2, bo=100.1, bh=101.3, bl=100, bc=101)]
        result = simulate_bracket_trade(bars, decision_index=0, side="long", atr=1, horizon_bars=2)
        self.assertEqual(result["entry_quote_side_price"], "100.2")
        self.assertEqual(result["exit_quote_side_price"], "101.2")
        self.assertEqual(result["status"], "take_profit")
        self.assertEqual(result["gross_quote_pnl_price_units_per_instrument_unit"], "1.0")
        self.assertIsNone(result["net_pnl"])

    def test_short_uses_bid_entry_and_ask_exit(self):
        bars = [bar(0), bar(1, bo=100, bh=100.5, bl=99.5, bc=100), bar(2, bo=99.5, bh=100, bl=98.5, bc=99)]
        result = simulate_bracket_trade(bars, decision_index=0, side="short", atr=1, horizon_bars=2)
        self.assertEqual(result["entry_quote_side_price"], "100")
        self.assertEqual(result["exit_quote_side_price"], "99.0")
        self.assertEqual(result["status"], "take_profit")
        self.assertEqual(result["gross_quote_pnl_price_units_per_instrument_unit"], "1.0")

    def test_gap_through_stop_uses_adverse_open(self):
        bars = [bar(0), bar(1, bo=100, bh=100.5, bl=99.5, bc=100), bar(2, bo=97, bh=98, bl=96, bc=97)]
        result = simulate_bracket_trade(bars, decision_index=0, side="long", atr=1, horizon_bars=2)
        self.assertEqual(result["status"], "stop_loss_at_open")
        self.assertEqual(result["exit_quote_side_price"], "97")
        self.assertEqual(result["gross_quote_pnl_price_units_per_instrument_unit"], "-3.2")

    def test_same_bar_collision_reports_bounds_and_no_point_pnl(self):
        bars = [bar(0), bar(1, bo=100, bh=100.5, bl=99.5, bc=100), bar(2, bo=100, bh=102, bl=98, bc=100)]
        result = simulate_bracket_trade(bars, decision_index=0, side="long", atr=1, horizon_bars=2)
        self.assertEqual(result["status"], "ambiguous_same_bar")
        self.assertIsNone(result["gross_quote_pnl_price_units_per_instrument_unit"])
        self.assertEqual(result["pnl_lower_bound_price_units_per_unit"], "-1.0")
        self.assertEqual(result["pnl_upper_bound_price_units_per_unit"], "1.0")

    def test_timeout_uses_executable_quote_close(self):
        bars = [bar(0), bar(1, bo=100, bh=100.4, bl=99.8, bc=100), bar(2, bo=100, bh=100.5, bl=99.5, bc=100.5)]
        result = simulate_bracket_trade(bars, decision_index=0, side="long", atr=5, horizon_bars=2)
        self.assertEqual(result["status"], "horizon_timeout")
        self.assertEqual(result["exit_quote_side_price"], "100.5")
        self.assertEqual(result["outcome_label_tp_before_sl"], 0)

    def test_stop_inside_entry_spread_is_blocked_for_both_sides(self):
        bars = [
            bar(0),
            bar(1, bo=100, bh=103, bl=99, bc=100, ao=102, ah=103.2, al=100.2, ac=102),
            bar(2, bo=100, bh=101, bl=99, bc=100, ao=100.2, ah=101.2, al=99.2, ac=100.2),
        ]
        for side in ("long", "short"):
            result = simulate_bracket_trade(
                bars, decision_index=0, side=side, atr=1, horizon_bars=2
            )
            self.assertEqual(result["status"], "invalid_stop_geometry_at_entry")
            self.assertIsNone(result["outcome_label_tp_before_sl"])
            self.assertIsNone(result["gross_quote_pnl_price_units_per_instrument_unit"])
            self.assertFalse(result["stop_geometry_valid"])
            self.assertEqual(result["entry_spread_to_stop_distance_ratio"], "2")

    def test_gap_in_horizon_blocks_outcome_instead_of_skipping(self):
        bars = [bar(0), bar(1, bo=100, bh=100.3, bl=99.5, bc=100), bar(3)]
        result = simulate_bracket_trade(bars, decision_index=0, side="long", atr=1, horizon_bars=2)
        self.assertEqual(result["status"], "incomplete_path_gap")
        self.assertIsNone(result["outcome_label_tp_before_sl"])

    def test_rejects_naive_timestamps(self):
        bad = BidAskBar("2026-01-01T00:00:00", 100, 101, 99, 100, 100.2, 101.2, 99.2, 100.2)
        with self.assertRaisesRegex(ValueError, "timezone_aware"):
            bad.normalized()

    def test_rejects_crossed_quote(self):
        bad = BidAskBar("2026-01-01T00:00:00Z", 100, 101, 99, 100, 99.8, 101.2, 99.2, 100.2)
        with self.assertRaisesRegex(ValueError, "crossed_quote"):
            bad.normalized()

    def test_rejects_bad_ohlc(self):
        bad = BidAskBar("2026-01-01T00:00:00Z", 100, 99, 99, 100, 100.2, 101.2, 99.2, 100.2)
        with self.assertRaisesRegex(ValueError, "high_below_open_or_close"):
            bad.normalized()

    def test_rejects_duplicate_or_unsorted_timestamps(self):
        bars = [bar(0), bar(1), bar(1)]
        with self.assertRaisesRegex(ValueError, "strictly_increasing"):
            simulate_bracket_trade(bars, decision_index=0, side="long", atr=1, horizon_bars=2)

    def test_rejects_invalid_inputs(self):
        bars = [bar(0), bar(1), bar(2)]
        with self.assertRaisesRegex(ValueError, "side_must_be"):
            simulate_bracket_trade(bars, decision_index=0, side="buy", atr=1, horizon_bars=2)
        with self.assertRaisesRegex(ValueError, "atr_and_barrier"):
            simulate_bracket_trade(bars, decision_index=0, side="long", atr=0, horizon_bars=2)
        with self.assertRaisesRegex(ValueError, "out_of_range"):
            simulate_bracket_trade(bars, decision_index=1, side="long", atr=1, horizon_bars=2)


if __name__ == "__main__":
    unittest.main()
