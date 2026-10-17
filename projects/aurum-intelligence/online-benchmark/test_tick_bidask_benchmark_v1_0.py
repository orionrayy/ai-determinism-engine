import unittest
from datetime import datetime, timedelta, timezone

import tick_bidask_benchmark_v1_0 as tick


class TickBidAskBenchmarkTests(unittest.TestCase):
    def test_spread_summary_records_observed_values_not_user_costs(self):
        result = tick.spread_diagnostics([2000.0, 2000.0, 2000.0], [2000.1, 2000.2, 2000.3])
        self.assertEqual(result["tick_count"], 3)
        self.assertAlmostEqual(result["spread_price_units"]["median"], 0.2, places=8)
        self.assertIsNotNone(result["spread_basis_points_of_mid"]["p95"])
        self.assertFalse(result["execution_cost_is_user_broker_measured"])

    def test_crossed_or_invalid_quote_fails_closed(self):
        with self.assertRaisesRegex(tick.TickDatasetError, "invalid_bid_ask_quote"):
            tick.spread_diagnostics([2000.1], [2000.0])
        with self.assertRaisesRegex(tick.TickDatasetError, "invalid_bid_ask_quote"):
            tick.spread_diagnostics([0.0], [2000.0])

    def test_exact_m5_mid_quote_match_is_diagnostic_and_never_promoted(self):
        primary, bid, mid, ask = [], {}, {}, {}
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        for i in range(1001):
            stamp = start + timedelta(minutes=5 * i)
            p = 2000.0 + i * 0.1
            mid_row = {"open": p, "high": p + 0.5, "low": p - 0.5, "close": p + 0.2}
            bid[i] = {key: value - 0.1 for key, value in mid_row.items()}
            mid[i] = dict(mid_row)
            ask[i] = {key: value + 0.1 for key, value in mid_row.items()}
            primary.append({"timestamp": stamp, **mid_row})
        quote_bars = {
            "bid": {start + timedelta(minutes=5*i): bid[i] for i in bid},
            "mid": {start + timedelta(minutes=5*i): mid[i] for i in mid},
            "ask": {start + timedelta(minutes=5*i): ask[i] for i in ask},
        }
        comparison, rows = tick.compare_price_bars(primary, quote_bars)
        self.assertEqual(comparison["status"], "THIRD_SOURCE_DIAGNOSTIC_ONLY")
        self.assertEqual(comparison["shared_timestamps"], 1001)
        self.assertEqual(comparison["basis_comparisons"]["mid"]["exact_ohlc_match_count"], 1001)
        self.assertEqual(comparison["basis_comparisons"]["bid"]["exact_ohlc_match_count"], 0)
        self.assertFalse(comparison["automatic_timestamp_shift"])
        self.assertFalse(comparison["automatic_price_rescale"])
        self.assertFalse(comparison["automatic_source_promotion"])
        self.assertEqual(len(rows), 1001)


    def test_near_exact_bid_match_flags_possible_shared_lineage_not_independence(self):
        result = tick.classify_lineage_overlap(4391, 4392)
        self.assertEqual(result["status"], "POTENTIAL_SHARED_LINEAGE_OR_PRICE_BASIS_MISLABEL_REVIEW_REQUIRED")
        self.assertAlmostEqual(result["exact_ohlc_match_fraction"], 4391 / 4392)
        self.assertFalse(result["source_independence_verified"])
        self.assertFalse(result["independent_source_confirmation"])
        self.assertFalse(result["automatic_source_promotion"])

    def test_lineage_diagnostic_does_not_call_nonmatching_sources_independent(self):
        result = tick.classify_lineage_overlap(1, 1001)
        self.assertEqual(result["status"], "NO_NEAR_EXACT_OHLC_MATCH_EVIDENCE_NOT_PROOF_OF_INDEPENDENCE")
        self.assertFalse(result["source_independence_verified"])

    def test_lineage_diagnostic_rejects_invalid_counts_and_small_overlap(self):
        self.assertEqual(
            tick.classify_lineage_overlap(1, 10)["status"],
            "INSUFFICIENT_OVERLAP_FOR_LINEAGE_DIAGNOSTIC",
        )
        with self.assertRaisesRegex(tick.TickDatasetError, "invalid_lineage_overlap_counts"):
            tick.classify_lineage_overlap(3, 2)


if __name__ == "__main__":
    unittest.main()
