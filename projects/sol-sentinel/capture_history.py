#!/usr/bin/env python3
"""Capture bounded public Binance OHLCV history for offline research only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://data-api.binance.vision/api/v3/klines"
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000}
USER_AGENT = "sol-sentinel-history/0.1 read-only"


def fetch_page(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    query = urllib.parse.urlencode({
        "symbol": symbol, "interval": interval,
        "startTime": start_ms, "endTime": end_ms, "limit": 1000,
    })
    request = urllib.request.Request(
        BASE + "?" + query,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError("HTTP status " + str(response.status))
                rows = json.loads(response.read(2_000_000).decode("utf-8"))
            if not isinstance(rows, list):
                raise ValueError("Expected a list of candles")
            return rows
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (418, 429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {exc.code} from {exc.url}: {exc.reason}") from exc
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = min(5.0, max(0.5, float(retry_after))) if retry_after else 0.5 * (2 ** attempt)
            except ValueError:
                delay = 0.5 * (2 ** attempt)
            if attempt < 2:
                time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.25 * (2 ** attempt))
    raise RuntimeError("Public market-data request failed after bounded retries: " + str(last_error))


def fetch_quote_to_idr_reference(timeout: float = 8.0, attempts: int = 3, pause_seconds: float = 0.5) -> dict:
    """Fetch a dated USD/IDR reference with bounded retries; never treat it as a live trading quote."""
    if not 0.1 <= timeout <= 20.0:
        raise ValueError("FX timeout must be between 0.1 and 20 seconds")
    if not 1 <= attempts <= 5:
        raise ValueError("FX attempts must be between 1 and 5")
    if not 0.0 <= pause_seconds <= 5.0:
        raise ValueError("FX retry pause must be between 0 and 5 seconds")
    url = "https://api.frankfurter.dev/v2/rate/USD/IDR"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    data = None
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError("FX reference HTTP status " + str(response.status))
                payload = response.read(100_000)
            data = json.loads(payload.decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (408, 425, 429, 500, 502, 503, 504):
                raise RuntimeError(f"FX reference HTTP {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(min(2.0, pause_seconds * (2 ** attempt)))
    if data is None:
        raise RuntimeError(f"FX reference request failed after {attempts} attempts: {last_error}")
    if not isinstance(data, dict) or data.get("base") != "USD" or data.get("quote") != "IDR":
        raise ValueError("Unexpected USD/IDR reference response")
    rate = float(data.get("rate", 0))
    rate_date = str(data.get("date", ""))
    parsed_date = datetime.strptime(rate_date, "%Y-%m-%d").date()
    age_days = (datetime.now(timezone.utc).date() - parsed_date).days
    if not math.isfinite(rate) or not 1_000 <= rate <= 50_000:
        raise ValueError("USD/IDR reference rate is outside safety bounds")
    if age_days < 0 or age_days > 5:
        raise ValueError(f"USD/IDR reference is stale ({age_days} days old)")
    return {
        "quote_to_idr_rate": rate,
        "quote_to_idr_date": rate_date,
        "quote_to_idr_source": "Frankfurter daily USD/IDR mid-market reference; USDT treated as USD proxy",
        "quote_to_idr_source_url": url,
        "quote_to_idr_age_days": age_days,
        "quote_to_idr_is_trading_quote": False,
    }


def capture(symbol: str, interval: str, days: int, max_pages: int) -> tuple[list, dict]:
    if interval not in INTERVAL_MS:
        raise ValueError("Unsupported interval")
    if not 1 <= days <= 30:
        raise ValueError("days must be in range 1..30")
    if not 1 <= max_pages <= 60:
        raise ValueError("max-pages must be in range 1..60")
    now = int(time.time() * 1000)
    step = INTERVAL_MS[interval]
    start = ((now - days * 86_400_000) // step) * step
    cursor = start
    by_open = {}
    pages = 0
    while cursor < now and pages < max_pages:
        rows = fetch_page(symbol, interval, cursor, now)
        pages += 1
        if not rows:
            break
        last_open = int(rows[-1][0])
        for row in rows:
            if len(row) < 9:
                continue
            open_ms, close_ms = int(row[0]), int(row[6])
            if close_ms >= now:
                continue
            values = [float(row[i]) for i in (1, 2, 3, 4, 5)]
            o, h, low, close, volume = values
            if not all(map(lambda x: x == x and abs(x) != float("inf"), values)):
                continue
            if min(o, h, low, close) <= 0 or volume < 0 or h < max(o, close, low) or low > min(o, close, h):
                continue
            by_open[open_ms] = {
                "open_time_ms": open_ms, "close_time_ms": close_ms,
                "open": o, "high": h, "low": low, "close": close,
                "volume": volume, "quote_volume": float(row[7]), "trades": int(row[8]),
            }
        next_cursor = last_open + step
        if next_cursor <= cursor:
            raise RuntimeError("Pagination did not advance; stopping safely")
        cursor = next_cursor
        if len(rows) < 1000:
            break
        time.sleep(0.25)
    if pages >= max_pages and cursor < now:
        raise RuntimeError("Page cap reached before completing requested interval")
    candles = [by_open[k] for k in sorted(by_open)]
    if len(candles) < 100:
        raise ValueError("Insufficient completed candles: " + str(len(candles)))
    for previous, current in zip(candles, candles[1:]):
        if current["open_time_ms"] - previous["open_time_ms"] != step:
            raise ValueError("History contains a gap; capture another window or investigate source data")
    meta = {
        "schema_version": "sol-sentinel-capture-v1",
        "source": "Binance public spot klines",
        "symbol": symbol, "interval": interval, "requested_days": days,
        "page_count": pages, "count": len(candles),
        "first_open_time_utc": datetime.fromtimestamp(candles[0]["open_time_ms"] / 1000, timezone.utc).isoformat(),
        "last_close_time_utc": datetime.fromtimestamp(candles[-1]["close_time_ms"] / 1000, timezone.utc).isoformat(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True, "orders_enabled": False,
    }
    try:
        meta.update(fetch_quote_to_idr_reference())
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        meta["quote_to_idr_rate"] = None
        meta["quote_to_idr_error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
    return candles, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--interval", choices=sorted(INTERVAL_MS), default="1m")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--max-pages", type=int, default=60)
    parser.add_argument("--output", default="~/.sol-sentinel/SOLUSDT_1m.csv")
    args = parser.parse_args()
    if not args.symbol.isalnum():
        parser.error("--symbol must use letters and numbers only")
    if args.interval != "1m":
        parser.error("The current paper engine expects 1m input; capture 1m history for backtest")
    output = Path(args.output).expanduser()
    if output == Path("~/.sol-sentinel/SOLUSDT_1m.csv").expanduser() and args.symbol != "SOLUSDT":
        output = output.with_name(args.symbol + "_1m.csv")
    try:
        bars, meta = capture(args.symbol.upper(), args.interval, args.days, args.max_pages)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "open_time_ms", "close_time_ms", "open", "high", "low", "close",
                "volume", "quote_volume", "trades",
            ])
            writer.writeheader()
            writer.writerows(bars)
        meta["csv_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        meta["output_path"] = str(output)
        meta_path = output.with_suffix(output.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(json.dumps({"error": str(exc), "orders_enabled": False}), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
