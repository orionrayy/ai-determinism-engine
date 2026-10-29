import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import capture_indodax_history as capture
from execution_adapters import MarketCandle, MarketSpec


class FakeIndodaxAdapter:
    def __init__(self, bars):
        self.bars = bars
        self.market = MarketSpec(
            venue="INDODAX", symbol="SOL/IDR", venue_symbol="solidr",
            base_asset="SOL", quote_asset="IDR", price_tick="1000",
            quantity_step="0.000001", min_notional_quote="10000",
            base_decimals=6, quote_decimals=0,
        )
        self.calls = []

    def get_market_spec(self, symbol):
        if symbol != "SOL/IDR":
            raise AssertionError("unexpected symbol")
        return self.market

    def get_history(self, market, start_ms, end_ms, *, interval_minutes, now_ms):
        self.calls.append((market, start_ms, end_ms, interval_minutes, now_ms))
        return self.bars


def make_bars(start_ms, count):
    return [
        MarketCandle(
            open_time_ms=start_ms + i * 60_000,
            close_time_ms=start_ms + i * 60_000 + 59_999,
            interval_minutes=1,
            open=Decimal("1800000"),
            high=Decimal("1801000"),
            low=Decimal("1799000"),
            close=Decimal("1800500"),
            volume=Decimal("0.25"),
            quote_currency="IDR",
        )
        for i in range(count)
    ]


class IndodaxNativeCaptureTests(unittest.TestCase):
    def test_capture_uses_completed_native_idr_window_and_exact_candle_count(self):
        now = 1_800_000_060_000
        end_ms = now - now % 60_000
        start_ms = end_ms - 86_400_000
        adapter = FakeIndodaxAdapter(make_bars(start_ms, 1440))
        bars, market, actual_start, actual_end, captured_at = capture.capture_history(
            "SOL/IDR", days=1, adapter=adapter, now_ms=now,
        )
        self.assertEqual(len(bars), 1440)
        self.assertEqual(market.quote_asset, "IDR")
        self.assertEqual(actual_start, start_ms)
        self.assertEqual(actual_end, end_ms)
        self.assertEqual(adapter.calls[0][3], 1)
        self.assertEqual(adapter.calls[0][4], now)
        self.assertTrue(datetime.fromisoformat(captured_at).tzinfo)

    def test_capture_rejects_invalid_days_and_other_symbols_before_network(self):
        adapter = FakeIndodaxAdapter([])
        for days in (0, 31, True):
            with self.subTest(days=days), self.assertRaisesRegex(ValueError, "days must"):
                capture.capture_history("SOL/IDR", days=days, adapter=adapter, now_ms=1_800_000_060_000)
        with self.assertRaisesRegex(ValueError, "restricted to SOL/IDR"):
            capture.capture_history("SOL/USDT", days=1, adapter=adapter, now_ms=1_800_000_060_000)
        self.assertEqual(adapter.calls, [])

    def test_write_history_records_native_idr_provenance_and_digest(self):
        start = 1_800_000_000_000
        bars = make_bars(start, 100)
        market = MarketSpec(
            venue="INDODAX", symbol="SOL/IDR", venue_symbol="solidr",
            base_asset="SOL", quote_asset="IDR", price_tick="1000",
            quantity_step="0.000001", min_notional_quote="10000",
            base_decimals=6, quote_decimals=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "native.csv"
            meta_path = Path(tmp) / "native.meta.json"
            metadata = capture.write_history(
                csv_path, meta_path, bars, market,
                requested_days=1, requested_start_ms=start,
                requested_end_ms=start + 1440 * 60_000,
                captured_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 100)
        self.assertEqual(rows[0]["open"], "1800000")
        self.assertEqual(metadata["input_price_currency"], "IDR")
        self.assertFalse(metadata["fx_conversion_applied"])
        self.assertFalse(metadata["orders_enabled"])
        self.assertEqual(metadata["dataset_sha256"], hashlib.sha256(csv_path.read_bytes()).hexdigest() if csv_path.exists() else sidecar["dataset_sha256"])
        self.assertEqual(sidecar["venue"], "INDODAX")
        self.assertEqual(sidecar["quote_asset"], "IDR")


if __name__ == "__main__":
    unittest.main(verbosity=2)
