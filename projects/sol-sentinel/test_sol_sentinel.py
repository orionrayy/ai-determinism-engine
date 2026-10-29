import importlib.util
import pathlib
import unittest

MODULE_PATH = pathlib.Path(__file__).with_name("sol_sentinel.py")
SPEC = importlib.util.spec_from_file_location("sol_sentinel", MODULE_PATH)
sentinel = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(sentinel)


def candle(i, close, high=None, low=None, volume=10.0, close_time=None):
    return sentinel.Candle(
        open_time_ms=i * 60_000, open=close,
        high=high if high is not None else close + 1.0,
        low=low if low is not None else close - 1.0,
        close=close, volume=volume,
        close_time_ms=close_time if close_time is not None else i * 60_000 + 59_999,
    )


class IndicatorTests(unittest.TestCase):
    def test_ema_period_one_tracks_input(self):
        self.assertEqual(sentinel.ema([1, 3, 2], 1), [1.0, 3.0, 2.0])

    def test_rsi_monotonic_increase_is_100(self):
        self.assertEqual(sentinel.rsi(list(range(20)), 14), 100.0)

    def test_rsi_insufficient_history(self):
        self.assertIsNone(sentinel.rsi([1, 2, 3], 14))

    def test_atr_is_positive_for_nonflat_series(self):
        candles = [candle(i, 100 + i, high=102 + i, low=99 + i) for i in range(20)]
        value = sentinel.atr(candles, 14)
        self.assertIsNotNone(value)
        self.assertGreater(value, 0)

    def test_relative_volume_handles_zero_baseline(self):
        candles = [candle(i, 100 + i, volume=0) for i in range(30)]
        self.assertIsNone(sentinel.relative_volume(candles, 20))

    def test_parse_klines_drops_open_candle_and_bad_row(self):
        rows = [
            [0, "10", "11", "9", "10.5", "5", 50_000, "52", 2],
            [60_000, "10.5", "12", "10", "11.5", "6", 119_999, "69", 3],
            ["bad"],
        ]
        parsed = sentinel.parse_klines(rows, now_ms=100_000)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].close, 10.5)

    def test_parse_klines_drops_nonfinite(self):
        rows = [[0, "10", "11", "9", "NaN", "5", 50_000, "52", 2]]
        self.assertEqual(sentinel.parse_klines(rows, now_ms=100_000), [])

    def test_missing_timeframe_fails_closed(self):
        out = sentinel.evaluate_signal({"1m": {}})
        self.assertEqual(out["signal"], "NO_TRADE")
        self.assertIn("missing_timeframe_metrics", out["reasons"])

    def test_fee_cost_model_is_explicit(self):
        out = sentinel.evaluate_signal(
            {
                "1m": {"close_above_prior_20_high": False, "relative_volume20": 1, "rsi14": 50},
                "5m": {"atr14": 1, "close": 100, "trend_up": True, "ema9": 99, "rsi14": 55},
                "15m": {"trend_up": True, "close": 101, "ema21": 100},
            },
            fee_bps_round_trip=40, slippage_bps_round_trip=10, min_net_edge_bps=25,
        )
        self.assertEqual(out["cost_model"]["total_cost_bps"], 50)
        self.assertTrue(out["cost_model"]["assumption_only"])
        self.assertEqual(out["signal"], "NO_TRADE")

    def test_candle_validation_rejects_impossible_high_low(self):
        rows = [[0, "10", "9", "8", "10.5", "5", 50_000, "52", 2]]
        self.assertEqual(sentinel.parse_klines(rows, now_ms=100_000), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
