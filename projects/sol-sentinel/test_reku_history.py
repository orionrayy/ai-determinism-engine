import io
import unittest
import urllib.error
from datetime import date

import reku_history as history


class RekuHistoryTests(unittest.TestCase):
    def test_month_sequence_crosses_year_boundary(self):
        self.assertEqual(
            history.month_sequence(date(2025, 12, 17), date(2026, 2, 2)),
            [(2025, 12), (2026, 1), (2026, 2)],
        )

    def test_month_sequence_enforces_maximum_months(self):
        with self.assertRaisesRegex(ValueError, "maximum is 3"):
            history.month_sequence(date(2025, 1, 1), date(2025, 5, 1))

    def test_parse_activity_rows_converts_seconds_and_filters_invalid_rows(self):
        start_ms = 1_800_000_000_000
        end_ms = start_ms + 300_000
        rows = [
            [start_ms / 1000, 0, 1_800_000, 180_000, 0.1],
            [start_ms + 60_000, 1, 1_805_000, 180_500, 0.1],
            [start_ms / 1000 - 1, 0, 1_800_000, 100, 0.1],
            [start_ms / 1000 + 2, 7, 1_800_000, 100, 0.1],
            [start_ms / 1000 + 3, 0, "NaN", 100, 0.1],
            ["bad", 0, 1_800_000, 100, 0.1],
            [start_ms / 1000 + 4, 1, -1, 100, 0.1],
        ]
        trades = history.parse_activity_rows(rows, start_ms, end_ms)
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0].timestamp_ms, start_ms)
        self.assertEqual(trades[0].transaction_type, 0)
        self.assertEqual(trades[1].timestamp_ms, start_ms + 60_000)
        self.assertEqual(trades[1].transaction_type, 1)

    def test_aggregate_trades_builds_contiguous_idr_bars_and_marks_empty_minutes(self):
        start = 1_800_000_000_000
        trades = []
        # Provide 101 active minutes so aggregation satisfies minimum history without network calls.
        for minute in range(101):
            price = 1_800_000.0 + minute * 100.0
            trades.append(history.Trade(
                timestamp_ms=start + minute * history.ONE_MINUTE_MS + 5_000,
                transaction_type=0,
                price_idr=price,
                notional_idr=price * 0.1,
                quantity_coin=0.1,
            ))
        # Add a second price in minute 10 to validate OHLC aggregation and quantity summation.
        trades.append(history.Trade(
            timestamp_ms=start + 10 * history.ONE_MINUTE_MS + 30_000,
            transaction_type=1,
            price_idr=1_801_250.0,
            notional_idr=180_125.0,
            quantity_coin=0.1,
        ))
        bars, synthetic = history.aggregate_trades_to_1m(trades)
        self.assertEqual(len(bars), 101)
        self.assertEqual(synthetic, 0)
        for left, right in zip(bars, bars[1:]):
            self.assertEqual(right.open_time_ms - left.open_time_ms, history.ONE_MINUTE_MS)
        bar10 = bars[10]
        self.assertEqual(bar10.open, 1_801_000.0)
        self.assertEqual(bar10.high, 1_801_250.0)
        self.assertEqual(bar10.low, 1_801_000.0)
        self.assertEqual(bar10.close, 1_801_250.0)
        self.assertEqual(bar10.trades, 2)
        self.assertAlmostEqual(bar10.volume, 0.2)

    def test_aggregate_trades_rejects_empty_or_short_history(self):
        with self.assertRaisesRegex(ValueError, "No in-range"):
            history.aggregate_trades_to_1m([])
        short = [
            history.Trade(1_800_000_000_000 + i * 60_000, 0, 100.0, 100.0, 1.0)
            for i in range(10)
        ]
        with self.assertRaisesRegex(ValueError, "at least 100"):
            history.aggregate_trades_to_1m(short)

    def test_parse_activity_rejects_non_list_payload(self):
        with self.assertRaisesRegex(ValueError, "must be a list"):
            history.parse_activity_rows({"result": []}, 1_800_000_000_000, 1_800_000_300_000)

    def test_activity_request_includes_http_error_body_for_404_diagnostics(self):
        err = urllib.error.HTTPError(
            "https://api.reku.id/v2/activitytrans",
            404,
            "Not Found",
            {},
            io.BytesIO(b'{"message":"legacy route removed"}'),
        )
        from unittest.mock import patch
        with patch.object(history.urllib.request, "urlopen", side_effect=err) as mocked:
            with self.assertRaisesRegex(RuntimeError, "HTTP 404.*legacy route removed"):
                history.request_activity_month(51, 2026, 10, attempts=1)
        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
