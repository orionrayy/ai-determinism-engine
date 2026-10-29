#!/usr/bin/env python3
"""Capture venue-native INDODAX SOL/IDR OHLC history for offline research only.

This uses public market-history endpoints. It makes no authenticated requests,
and cannot create/cancel orders, withdraw assets, sign, or submit transactions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from execution_adapters import IndodaxPublicMarketDataAdapter, MarketCandle, MarketSpec, VenueDataError

ONE_MINUTE_MS = 60_000
MAX_DAYS = 30


def write_history(
    output_path: str | Path,
    metadata_path: str | Path,
    bars: Sequence[MarketCandle],
    market: MarketSpec,
    *,
    requested_days: int,
    requested_start_ms: int,
    requested_end_ms: int,
    captured_at_utc: str,
) -> dict:
    if not bars:
        raise ValueError("refusing to write empty INDODAX history")
    out = Path(output_path).expanduser()
    meta_path = Path(metadata_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume"])
        for bar in bars:
            writer.writerow([
                bar.open_time_ms, bar.close_time_ms,
                str(bar.open), str(bar.high), str(bar.low), str(bar.close), str(bar.volume),
            ])

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    metadata = {
        "schema_version": "sol-sentinel-indodax-native-history-v1",
        "mode": "READ_ONLY_NATIVE_MARKET_HISTORY",
        "data_status": "OK",
        "venue": "INDODAX",
        "pair": market.symbol,
        "venue_pair_id": market.venue_symbol,
        "input_price_currency": "IDR",
        "base_asset": market.base_asset,
        "quote_asset": market.quote_asset,
        "interval": "1m",
        "interval_minutes": 1,
        "requested_days": requested_days,
        "requested_start_ms": requested_start_ms,
        "requested_end_ms_exclusive": requested_end_ms,
        "first_open_time_ms": bars[0].open_time_ms,
        "last_open_time_ms": bars[-1].open_time_ms,
        "first_open_time_utc": datetime.fromtimestamp(bars[0].open_time_ms / 1000, timezone.utc).isoformat(),
        "last_close_time_utc": datetime.fromtimestamp(bars[-1].close_time_ms / 1000, timezone.utc).isoformat(),
        "captured_at_utc": captured_at_utc,
        "count": len(bars),
        "dataset_sha256": digest,
        "source_endpoint": "https://indodax.com/tradingview/history_v2",
        "pair_metadata_endpoint": "https://indodax.com/api/pairs",
        "market_filters": {
            "price_tick": str(market.price_tick),
            "quantity_step": str(market.quantity_step),
            "api_min_notional_quote": str(market.min_notional_quote),
            "minimum_base_quantity": str(market.min_quantity_base),
        },
        "read_only": True,
        "authenticated": False,
        "orders_enabled": False,
        "fx_conversion_applied": False,
        "warning": "Native IDR candles are not a fee model or proof of executable fills. Supply verified venue/mode-specific fees before backtest interpretation.",
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def capture_history(
    symbol: str = "SOL/IDR",
    days: int = 30,
    *,
    adapter: IndodaxPublicMarketDataAdapter | None = None,
    now_ms: int | None = None,
) -> tuple[list[MarketCandle], MarketSpec, int, int, str]:
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= MAX_DAYS:
        raise ValueError(f"days must be an integer in 1..{MAX_DAYS}")
    if not isinstance(symbol, str) or symbol.strip().upper().replace("/", "") != "SOLIDR":
        raise ValueError("this initial native-history capture is restricted to SOL/IDR")
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if isinstance(now, bool) or not isinstance(now, int) or now <= 0:
        raise ValueError("now_ms must be a positive integer")
    end_ms = now - (now % ONE_MINUTE_MS)
    start_ms = end_ms - days * 86_400_000
    source = adapter or IndodaxPublicMarketDataAdapter()
    market = source.get_market_spec("SOL/IDR")
    bars = source.get_history(market, start_ms, end_ms, interval_minutes=1, now_ms=now)
    if len(bars) != days * 1440:
        raise VenueDataError(f"expected {days * 1440} contiguous 1m candles, received {len(bars)}")
    return bars, market, start_ms, end_ms, datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SOL/IDR", help="supported native history market: SOL/IDR")
    parser.add_argument("--days", type=int, default=30, help="completed 1m history window, 1..30 days")
    parser.add_argument("--output", default="~/.sol-sentinel/INDODAX_SOLIDR_1m.csv")
    parser.add_argument("--metadata-output", default="", help="optional metadata sidecar path")
    args = parser.parse_args(argv)
    out = Path(args.output).expanduser()
    metadata_path = Path(args.metadata_output).expanduser() if args.metadata_output else Path(str(out) + ".meta.json")
    try:
        bars, market, start_ms, end_ms, captured_at = capture_history(args.symbol, args.days)
        metadata = write_history(out, metadata_path, bars, market, requested_days=args.days,
                                 requested_start_ms=start_ms, requested_end_ms=end_ms, captured_at_utc=captured_at)
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        error = {
            "schema_version": "sol-sentinel-indodax-native-history-v1",
            "mode": "READ_ONLY_NATIVE_MARKET_HISTORY",
            "data_status": "ERROR",
            "venue": "INDODAX",
            "pair": args.symbol,
            "input_price_currency": "IDR",
            "read_only": True,
            "authenticated": False,
            "orders_enabled": False,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(error, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(error, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
