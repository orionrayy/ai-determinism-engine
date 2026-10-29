import unittest

import evaluate_splits
from paper_engine import Bar, ONE_MINUTE_MS


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
