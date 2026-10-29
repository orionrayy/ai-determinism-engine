import unittest

import evaluate_splits
from paper_engine import BacktestConfig, Bar, ONE_MINUTE_MS


BASE_MS = 1_760_000_100_000


def make_bars(count):
    result = []
    for index in range(count):
        price = 1000.0 + index * 0.01
        opened = BASE_MS + index * ONE_MINUTE_MS
        result.append(Bar(
            open_time_ms=opened, open=price - 0.01, high=price + 0.30,
            low=price - 0.30, close=price, volume=10.0,
            close_time_ms=opened + ONE_MINUTE_MS - 1,
        ))
    return result


class ChronologicalSplitTests(unittest.TestCase):
    def test_native_idr_evaluation_preserves_native_price_denomination(self):
        bars = make_bars(3000)
        config = BacktestConfig(
            initial_capital_idr=300_000.0,
            quote_to_idr_rate=1.0,
            quote_to_idr_source="native INDODAX SOL/IDR fixture; FX disabled",
            input_venue="INDODAX",
            input_symbol="SOL/IDR",
            input_price_currency="IDR",
            fee_model_name="INDODAX_CUSTOM",
            fee_model_source="fixture costs only",
            entry_fee_bps=10.0,
            exit_fee_bps=20.0,
            spread_bps_round_trip=15.0,
            slippage_bps_per_side=5.0,
            minimum_order_notional_idr=25_000.0,
        )
        report = evaluate_splits.evaluate(bars, config, dataset_sha256="fixture-native-idr")
        self.assertEqual(report["source_quote_currency"], "IDR")
        self.assertEqual(report["symbol"], "SOL/IDR")
        self.assertEqual(report["input_venue"], "INDODAX")
        self.assertEqual(report["quote_to_idr_rate"], 1.0)
        self.assertIn("FX conversion is disabled", report["quote_to_idr_assumption"])
        self.assertEqual(report["fee_model_name"], "INDODAX_CUSTOM")
        self.assertEqual(report["config"]["minimum_order_notional_idr"], 25_000.0)

    def test_split_is_chronological_disjoint_and_exact(self):
        bars = make_bars(3000)
        splits = evaluate_splits.split_chronologically(bars)
        self.assertEqual(len(splits["development"]), 1800)
        self.assertEqual(len(splits["validation"]), 600)
        self.assertEqual(len(splits["out_of_sample"]), 600)
        self.assertLess(splits["development"][-1].open_time_ms, splits["validation"][0].open_time_ms)
        self.assertLess(splits["validation"][-1].open_time_ms, splits["out_of_sample"][0].open_time_ms)
        self.assertEqual(splits["development"][0], bars[0])
        self.assertEqual(splits["out_of_sample"][-1], bars[-1])

    def test_split_rejects_insufficient_history(self):
        with self.assertRaisesRegex(ValueError, "at least 600"):
            evaluate_splits.split_chronologically(make_bars(1000))

    def test_split_rejects_ratios_not_summing_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            evaluate_splits.split_chronologically(make_bars(3000), ratios=(0.5, 0.2, 0.2))

    def test_split_rejects_timestamp_gap(self):
        bars = make_bars(3000)
        bars[10] = Bar(
            open_time_ms=bars[10].open_time_ms + ONE_MINUTE_MS,
            open=bars[10].open, high=bars[10].high, low=bars[10].low,
            close=bars[10].close, volume=bars[10].volume,
            close_time_ms=bars[10].close_time_ms + ONE_MINUTE_MS,
        )
        with self.assertRaisesRegex(ValueError, "timestamp gap"):
            evaluate_splits.split_chronologically(bars)


if __name__ == "__main__":
    unittest.main(verbosity=2)
