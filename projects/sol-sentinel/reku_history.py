#!/usr/bin/env python3
"""Capture public Reku SOL/IDR transaction history and aggregate to contiguous 1m bars.

Read-only public API only. No login, API secret, order placement, or wallet operations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from reku_market import BASE_URL, USER_AGENT, find_coin, request_json

MAX_RESPONSE_BYTES = 25_000_000
MAX_MONTHS = 3
ONE_MINUTE_MS = 60_000


@dataclass(frozen=True)
class Trade:
    timestamp_ms: int
    transaction_type: int
    price_idr: float
    notional_idr: float
    quantity_coin: float


@dataclass(frozen=True)
class Bar:
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int


def utc_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).isoformat()


def month_sequence(start_date: date, end_date: date) -> list[tuple[int, int]]:
    cursor = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)
    months: list[tuple[int, int]] = []
    while cursor <= end_month:
        months.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    if len(months) > MAX_MONTHS:
        raise ValueError(f"requested range spans {len(months)} months; maximum is {MAX_MONTHS}")
    return months


def request_activity_month(coin_id: Any, year: int, month: int, timeout: float = 15.0,
                           attempts: int = 3, pause_seconds: float = 0.5) -> Any:
    """POST the documented read-only activitytrans query with bounded retries."""
    if not 1 <= month <= 12:
        raise ValueError("month must be in 1..12")
    if not 1 <= attempts <= 5:
        raise ValueError("attempts must be in 1..5")
    url = f"{BASE_URL}/activitytrans"
    body = urllib.parse.urlencode({
        "coinid": str(coin_id),
        "bulan": str(month),
        "tahun": str(year),
    }).encode("ascii")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"Reku activity HTTP {response.status}")
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError(f"Reku activity response exceeded {MAX_RESPONSE_BYTES} bytes")
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("result", payload.get("data"))
            if not isinstance(payload, list):
                raise ValueError("Unexpected Reku /activitytrans response structure")
            return payload
        except urllib.error.HTTPError as exc:
            try:
                response_body = exc.read(2048).decode("utf-8", "replace").strip()
            except OSError:
                response_body = ""
            response_body = " ".join(response_body.split())
            detail = f"; response={response_body[:240]}" if response_body else ""
            error = RuntimeError(f"Reku activity HTTP {exc.code}: {exc.reason}{detail}")
            last_error = error
            if exc.code not in (408, 425, 429, 500, 502, 503, 504):
                raise error from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(min(3.0, pause_seconds * (2 ** attempt)))
    raise RuntimeError(f"Reku activity request failed for {year}-{month:02d} after {attempts} attempts: {last_error}")


def parse_activity_rows(payload: Any, start_ms: int, end_ms: int) -> list[Trade]:
    """Parse documented rows [unix_timestamp, side, price_IDR, total_IDR, coin_quantity]."""
    if not isinstance(payload, list):
        raise ValueError("Activity payload must be a list")
    trades: list[Trade] = []
    for row in payload:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        try:
            timestamp_raw = float(row[0])
            timestamp_ms = int(timestamp_raw if timestamp_raw >= 100_000_000_000 else timestamp_raw * 1000.0)
            side_value = float(row[1])
            if not math.isfinite(side_value) or not side_value.is_integer():
                continue
            transaction_type = int(side_value)
            price_idr = float(row[2])
            notional_idr = float(row[3])
            quantity_coin = float(row[4])
        except (TypeError, ValueError, OverflowError):
            continue
        values = (price_idr, notional_idr, quantity_coin)
        if not all(math.isfinite(value) for value in values):
            continue
        if timestamp_ms < start_ms or timestamp_ms >= end_ms:
            continue
        if transaction_type not in (0, 1):
            continue
        if price_idr <= 0 or notional_idr <= 0 or quantity_coin <= 0:
            continue
        trades.append(Trade(timestamp_ms, transaction_type, price_idr, notional_idr, quantity_coin))
    # Stable ordering preserves API order for trades carrying identical second-resolution timestamps.
    trades.sort(key=lambda trade: trade.timestamp_ms)
    return trades


def aggregate_trades_to_1m(trades: Sequence[Trade]) -> tuple[list[Bar], int]:
    """Build 1m IDR OHLCV bars and fill no-trade minutes with the last known close/zero volume."""
    if not trades:
        raise ValueError("No in-range valid Reku trades to aggregate")
    grouped: dict[int, list[Trade]] = {}
    for trade in trades:
        minute = (trade.timestamp_ms // ONE_MINUTE_MS) * ONE_MINUTE_MS
        grouped.setdefault(minute, []).append(trade)

    first_minute = min(grouped)
    last_minute = max(grouped)
    result: list[Bar] = []
    current_minute = first_minute
    previous_close: float | None = None
    synthetic_no_trade_minutes = 0
    while current_minute <= last_minute:
        rows = grouped.get(current_minute)
        if rows:
            rows.sort(key=lambda trade: trade.timestamp_ms)
            prices = [trade.price_idr for trade in rows]
            close = prices[-1]
            result.append(Bar(
                open_time_ms=current_minute,
                close_time_ms=current_minute + ONE_MINUTE_MS - 1,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=close,
                volume=sum(trade.quantity_coin for trade in rows),
                quote_volume=sum(trade.notional_idr for trade in rows),
                trades=len(rows),
            ))
            previous_close = close
        elif previous_close is not None:
            result.append(Bar(
                open_time_ms=current_minute,
                close_time_ms=current_minute + ONE_MINUTE_MS - 1,
                open=previous_close,
                high=previous_close,
                low=previous_close,
                close=previous_close,
                volume=0.0,
                quote_volume=0.0,
                trades=0,
            ))
            synthetic_no_trade_minutes += 1
        current_minute += ONE_MINUTE_MS

    if len(result) < 100:
        raise ValueError(f"Only {len(result)} 1m bars generated; at least 100 are required")
    for previous, current in zip(result, result[1:]):
        if current.open_time_ms - previous.open_time_ms != ONE_MINUTE_MS:
            raise ValueError("Internal aggregation produced a timestamp gap")
    return result, synthetic_no_trade_minutes


def write_history(path: str | Path, bars: Sequence[Bar], trades: Sequence[Trade], symbol: str,
                  coin_id: Any, requested_days: int, requested_start_ms: int, captured_at: str,
                  month_counts: dict[str, int], synthetic_no_trade_minutes: int,
                  api_errors: list[str]) -> dict[str, Any]:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume", "quote_volume", "trades"])
        for bar in bars:
            writer.writerow([
                bar.open_time_ms, bar.close_time_ms, bar.open, bar.high,
                bar.low, bar.close, bar.volume, bar.quote_volume, bar.trades,
            ])

    first_trade_ms = trades[0].timestamp_ms
    last_trade_ms = trades[-1].timestamp_ms
    now_ms = int(time.time() * 1000)
    requested_span_ms = max(1, now_ms - requested_start_ms)
    covered_span_ms = max(0, last_trade_ms - first_trade_ms)
    coverage_fraction = min(1.0, covered_span_ms / requested_span_ms)
    first_gap_ms = max(0, first_trade_ms - requested_start_ms)
    last_age_seconds = max(0.0, (now_ms - last_trade_ms) / 1000.0)
    warnings = [
        "Native Reku activity trades are used to aggregate IDR candles; minutes with no trades are forward-filled at last close with zero volume.",
        "The public API documentation is legacy and returned data must be validated for live freshness and coverage.",
        "No private API, account login, order endpoint, or credential is used.",
    ]
    if synthetic_no_trade_minutes:
        warnings.append(f"{synthetic_no_trade_minutes} no-trade minutes were filled with last known close and zero volume.")
    if coverage_fraction < 0.80:
        warnings.append("Coverage spans less than 80% of requested period; do not use for the full requested-window backtest.")
    if first_gap_ms > 15 * ONE_MINUTE_MS:
        warnings.append("First valid trade begins more than 15 minutes after requested start.")
    if last_age_seconds > 30 * 60:
        warnings.append("Last public activity trade is older than 30 minutes.")
    status = "OK" if coverage_fraction >= 0.80 and first_gap_ms <= 15 * ONE_MINUTE_MS and last_age_seconds <= 30 * 60 and not api_errors else "PARTIAL"
    if api_errors:
        status = "PARTIAL"
    meta = {
        "schema_version": "sol-sentinel-reku-history-v1",
        "mode": "READ_ONLY_PUBLIC_ACTIVITY_HISTORY",
        "orders_enabled": False,
        "authenticated": False,
        "venue": "Reku",
        "symbol": symbol.upper(),
        "pair": f"{symbol.upper()}/IDR",
        "input_price_currency": "IDR",
        "source_endpoint": "POST https://api.reku.id/v2/activitytrans",
        "source_documentation": "https://api.reku.id/",
        "coin_id": coin_id,
        "requested_days": requested_days,
        "requested_start_utc": utc_iso(requested_start_ms),
        "requested_end_utc": captured_at,
        "captured_at_utc": captured_at,
        "months_requested": month_counts,
        "valid_trade_rows": len(trades),
        "generated_1m_bars": len(bars),
        "first_trade_utc": utc_iso(first_trade_ms),
        "last_trade_utc": utc_iso(last_trade_ms),
        "last_trade_age_seconds": last_age_seconds,
        "coverage_fraction": coverage_fraction,
        "synthetic_no_trade_minutes": synthetic_no_trade_minutes,
        "api_errors": api_errors,
        "warnings": warnings,
        "data_status": status,
        "quote_to_idr_rate": 1.0,
        "quote_to_idr_date": datetime.now(timezone.utc).date().isoformat(),
        "quote_to_idr_age_days": 0,
        "quote_to_idr_source": "Native Reku SOL/IDR prices; no currency conversion applied",
        "quote_to_idr_is_trading_quote": False,
    }
    meta["csv_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    meta["csv_path"] = str(output)
    return meta


def capture_history(symbol: str = "SOL", days: int = 30) -> tuple[list[Bar], list[Trade], dict[str, Any]]:
    if not 1 <= days <= 62:
        raise ValueError("days must be between 1 and 62")
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    start_ms = now_ms - days * 86_400_000
    start_date = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end_date = now.date()
    month_keys = month_sequence(start_date, end_date)
    coin = find_coin(request_json("/coins"), symbol)
    coin_id = coin.get("id")
    if coin_id is None:
        raise ValueError(f"Reku coin ID is missing for {symbol.upper()}")

    all_trades: list[Trade] = []
    month_counts: dict[str, int] = {}
    api_errors: list[str] = []
    for index, (year, month) in enumerate(month_keys):
        key = f"{year}-{month:02d}"
        try:
            payload = request_activity_month(coin_id, year, month)
            trades = parse_activity_rows(payload, start_ms, now_ms)
            month_counts[key] = len(trades)
            all_trades.extend(trades)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            # A definitive 404 means this legacy route no longer exists; stop
            # immediately rather than repeating the same unsupported request.
            if "HTTP 404" in str(exc):
                raise RuntimeError(
                    f"Legacy Reku /activitytrans endpoint unavailable (HTTP 404) for {key}; "
                    "stop collection until a currently supported public history endpoint is verified."
                ) from exc
            month_counts[key] = 0
            api_errors.append(f"{key}: {type(exc).__name__}: {str(exc)[:180]}")
        if index + 1 < len(month_keys):
            time.sleep(0.25)

    # De-duplicate exact rows that may be repeated by an API range response.
    unique: dict[tuple[int, int, float, float, float], Trade] = {}
    for trade in all_trades:
        key = (trade.timestamp_ms, trade.transaction_type, trade.price_idr, trade.notional_idr, trade.quantity_coin)
        unique[key] = trade
    trades = sorted(unique.values(), key=lambda trade: trade.timestamp_ms)
    if not trades:
        raise ValueError("Reku /activitytrans returned no valid SOL transactions in requested range; " + " | ".join(api_errors))
    bars, synthetic_count = aggregate_trades_to_1m(trades)
    captured_at = utc_now()
    meta = {
        "requested_start_ms": start_ms,
        "month_counts": month_counts,
        "api_errors": api_errors,
        "synthetic_no_trade_minutes": synthetic_count,
    }
    return bars, trades, {
        "requested_start_ms": start_ms,
        "captured_at_utc": captured_at,
        "month_counts": month_counts,
        "api_errors": api_errors,
        "synthetic_no_trade_minutes": synthetic_count,
        "coin_id": coin_id,
        "bars": bars,
        "trades": trades,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SOL")
    parser.add_argument("--days", type=int, default=30, help="trailing calendar window, 1..62 days")
    parser.add_argument("--output", default="~/.sol-sentinel/REKU_SOLIDR_1m.csv")
    parser.add_argument("--metadata-output", default="", help="optional sidecar JSON path")
    args = parser.parse_args(argv)
    if not args.symbol.isalnum():
        parser.error("--symbol must use letters and numbers only")

    output = Path(args.output).expanduser()
    metadata_path = Path(args.metadata_output).expanduser() if args.metadata_output else Path(str(output) + ".meta.json")
    try:
        bars, trades, detail = capture_history(args.symbol.upper(), args.days)
        meta = write_history(
            output, bars, trades, args.symbol, detail["coin_id"], args.days,
            detail["requested_start_ms"], detail["captured_at_utc"], detail["month_counts"],
            detail["synthetic_no_trade_minutes"], detail["api_errors"],
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        print(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if meta["data_status"] == "OK" else 2
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        error_meta = {
            "schema_version": "sol-sentinel-reku-history-v1",
            "mode": "READ_ONLY_PUBLIC_ACTIVITY_HISTORY",
            "orders_enabled": False,
            "authenticated": False,
            "venue": "Reku",
            "symbol": args.symbol.upper(),
            "pair": f"{args.symbol.upper()}/IDR",
            "data_status": "ERROR",
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "error_category": "endpoint_not_found" if "HTTP 404" in str(exc) else "collection_failed",
            "next_action": (
                "Legacy /activitytrans returned HTTP 404; keep native Reku history disabled until a currently supported official history source is verified."
                if "HTTP 404" in str(exc)
                else "Inspect the bounded read-only API diagnostics; do not substitute incomplete chart data as historical OHLCV."
            ),
            "source_endpoint": "POST https://api.reku.id/v2/activitytrans",
            "source_documentation": "https://api.reku.id/",
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(error_meta, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        print(json.dumps(error_meta, ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
