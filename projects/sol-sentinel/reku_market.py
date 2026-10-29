#!/usr/bin/env python3
"""Read-only Reku SOL/IDR market snapshot using documented public endpoints.

No account credentials, private order endpoints, order placement, or transactions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

BASE_URL = "https://api.reku.id/v2"
USER_AGENT = "sol-sentinel-research/0.2 (public read-only market data)"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_json(path: str, params: dict[str, Any] | None = None, timeout: float = 8.0) -> Any:
    query = urllib.parse.urlencode(params or {})
    url = f"{BASE_URL}/{path.lstrip('/')}"
    if query:
        url += "?" + query
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"Reku public API HTTP {response.status} for {path}")
        raw = response.read(2_000_000)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Reku public API returned invalid JSON for {path}") from exc


def rows_from_coins(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("result", payload.get("data", []))
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError("Unexpected Reku /coins response structure")
    return [row for row in rows if isinstance(row, dict)]


def row_symbol(row: dict[str, Any]) -> str:
    return str(row.get("accountcode") or row.get("code") or row.get("cd") or "").strip().upper()


def find_coin(coins_payload: Any, symbol: str) -> dict[str, Any]:
    matches = [row for row in rows_from_coins(coins_payload) if row_symbol(row) == symbol.upper()]
    if not matches:
        raise ValueError(f"{symbol.upper()} was not found in Reku's public coin catalogue")
    # Refuse ambiguous duplicate symbol IDs rather than choosing an arbitrary listing.
    ids = {str(row.get("id")) for row in matches}
    if len(ids) != 1:
        raise ValueError(f"Reku public coin catalogue contains ambiguous IDs for {symbol.upper()}")
    return matches[0]


def price_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("result", payload.get("data", []))
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError("Unexpected Reku /price response structure")
    return [row for row in rows if isinstance(row, dict)]


def find_price(payload: Any, symbol: str, coin_id: Any) -> dict[str, Any]:
    for row in price_rows(payload):
        code = str(row.get("cd") or row.get("code") or row.get("accountcode") or "").strip().upper()
        if code == symbol.upper() or str(row.get("id")) == str(coin_id):
            return row
    raise ValueError(f"{symbol.upper()} price was not found in Reku's public /price response")


def find_bidask(payload: Any, symbol: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        rows = payload.get("result", payload.get("data", []))
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError("Unexpected Reku /bidask response structure")
    for row in rows:
        if isinstance(row, dict):
            code = str(row.get("code") or row.get("cd") or row.get("accountcode") or "").strip().upper()
            if code == symbol.upper():
                return row
    raise ValueError(f"{symbol.upper()} bid/ask was not found in Reku's public /bidask response")


def parse_orderbook(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Unexpected Reku order-book response structure")
    parsed: dict[str, list[dict[str, float]]] = {"bids": [], "asks": []}
    for source_key, target_key in (("b", "bids"), ("s", "asks")):
        levels = payload.get(source_key, [])
        if not isinstance(levels, list):
            raise ValueError(f"Unexpected Reku order-book {source_key} levels")
        for row in levels:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            try:
                total_idr, price_idr, quantity = (float(row[0]), float(row[1]), float(row[2]))
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(x) for x in (total_idr, price_idr, quantity)):
                continue
            if total_idr <= 0 or price_idr <= 0 or quantity <= 0:
                continue
            parsed[target_key].append({
                "total_idr": total_idr,
                "price_idr": price_idr,
                "quantity_coin": quantity,
            })
    parsed["bids"].sort(key=lambda level: level["price_idr"], reverse=True)
    parsed["asks"].sort(key=lambda level: level["price_idr"])
    if not parsed["bids"] or not parsed["asks"]:
        raise ValueError("Reku order book does not contain both valid bid and ask levels")
    return parsed


def spread_metrics(bid: float, ask: float) -> dict[str, float]:
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask <= 0:
        raise ValueError("Bid and ask must be finite positive prices")
    if ask < bid:
        raise ValueError("Reku quotes are crossed (ask < bid); reject snapshot")
    mid = (bid + ask) / 2.0
    spread = ask - bid
    return {
        "bid_idr": bid,
        "ask_idr": ask,
        "mid_idr": mid,
        "spread_idr": spread,
        "spread_bps": (spread / mid) * 10_000.0,
    }


def parse_chart_payload(payload: Any, interval_minutes: int, now_ms: int | None = None) -> dict[str, Any]:
    """Parse the documented Reku chart row: [timestamp, open, close, low, high, volume_coin, volume_idr]."""
    if not isinstance(payload, list):
        raise ValueError("Unexpected Reku /chart response structure")
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    bars: dict[int, dict[str, float | int]] = {}
    for row in payload:
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            continue
        try:
            raw_timestamp = float(row[0])
            timestamp_ms = int(raw_timestamp if raw_timestamp >= 100_000_000_000 else raw_timestamp * 1000)
            opened, close, low, high, volume_coin, volume_idr = [float(row[i]) for i in range(1, 7)]
        except (TypeError, ValueError, OverflowError):
            continue
        values = (opened, close, low, high, volume_coin, volume_idr)
        if not all(math.isfinite(value) for value in values):
            continue
        if min(opened, close, low, high) <= 0 or volume_coin < 0 or volume_idr < 0:
            continue
        if high < max(opened, close, low) or low > min(opened, close, high):
            continue
        bars[timestamp_ms] = {
            "timestamp_ms": timestamp_ms,
            "open_idr": opened,
            "close_idr": close,
            "low_idr": low,
            "high_idr": high,
            "volume_coin": volume_coin,
            "volume_idr": volume_idr,
        }
    ordered = [bars[key] for key in sorted(bars)]
    if not ordered:
        raise ValueError("Reku /chart returned no valid OHLCV bars")
    last = ordered[-1]
    age_seconds = (now - int(last["timestamp_ms"])) / 1000.0
    allowed_age_seconds = max(interval_minutes * 60 * 3, 900.0)
    status = "OK" if -5.0 <= age_seconds <= allowed_age_seconds else "STALE"
    return {
        "data_status": status,
        "interval_minutes": interval_minutes,
        "points_returned": len(ordered),
        "first_time_utc": datetime.fromtimestamp(int(ordered[0]["timestamp_ms"]) / 1000.0, timezone.utc).isoformat(),
        "last_time_utc": datetime.fromtimestamp(int(last["timestamp_ms"]) / 1000.0, timezone.utc).isoformat(),
        "last_age_seconds": age_seconds,
        "last_bar": last,
        "warning": "Chart timestamps/interval semantics must be confirmed against current Reku behavior; this is a public-data probe, not proof of full historical coverage.",
    }


def fetch_chart_probe(coin_id: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Probe 1m chart first, then documented 5m interval; report errors without inventing candles."""
    errors: list[str] = []
    for interval in (1, 5):
        try:
            payload = request_json("/chart", {"t": 2, "f": interval, "id": coin_id})
            parsed = parse_chart_payload(payload, interval)
            if parsed["data_status"] == "OK":
                parsed["endpoint"] = f"/v2/chart?t=2&f={interval}&id={coin_id}"
                return parsed, None
            errors.append(f"f={interval}: chart data stale (age_seconds={parsed['last_age_seconds']:.1f})")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            errors.append(f"f={interval}: {type(exc).__name__}: {str(exc)[:130]}")
    return None, " | ".join(errors)[:400]


def compare_reference(reku_close_idr: float, binance_close_usdt: float, quote_to_idr_rate: float) -> dict[str, Any]:
    values = (reku_close_idr, binance_close_usdt, quote_to_idr_rate)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Reference prices and FX rate must be finite and positive")
    binance_proxy_idr = binance_close_usdt * quote_to_idr_rate
    return {
        "reku_close_idr": reku_close_idr,
        "binance_close_usdt": binance_close_usdt,
        "quote_to_idr_rate": quote_to_idr_rate,
        "binance_usdt_usd_proxy_idr": binance_proxy_idr,
        "reku_vs_binance_proxy_pct": (reku_close_idr / binance_proxy_idr - 1.0) * 100.0,
        "reku_vs_binance_proxy_bps": (reku_close_idr / binance_proxy_idr - 1.0) * 10_000.0,
        "warning": "Different venues and timestamps; USD/IDR approximates USDT/IDR. This is a reference comparison, not an executable arbitrage signal.",
    }


def load_binance_reference(csv_path: str | Path) -> tuple[float, dict[str, Any]]:
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise ValueError(f"Binance reference CSV does not exist: {path}")
    sidecar = Path(str(path) + ".meta.json")
    if not sidecar.is_file():
        raise ValueError("Binance CSV metadata sidecar is required for quote-currency consistency")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Binance metadata sidecar: {exc}") from exc
    fx = metadata.get("quote_to_idr_rate")
    rate_date = str(metadata.get("quote_to_idr_date", ""))
    if not fx:
        raise ValueError("Binance metadata sidecar has no USD/IDR reference")
    if metadata.get("quote_to_idr_is_trading_quote") is not False:
        raise ValueError("Unexpected FX reference semantics")
    try:
        parsed_date = datetime.strptime(rate_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("Binance metadata has no valid FX reference date") from exc
    age_days = (datetime.now(timezone.utc).date() - parsed_date).days
    if age_days < 0 or age_days > 5:
        raise ValueError(f"USD/IDR reference is stale ({age_days} days old)")
    last_row = None
    import csv
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            last_row = row
    if not last_row:
        raise ValueError("Binance reference CSV has no candles")
    close = float(last_row.get("close", "nan"))
    if not math.isfinite(close) or close <= 0:
        raise ValueError("Invalid last SOLUSDT close in Binance reference CSV")
    return close, {
        "quote_to_idr_rate": float(fx),
        "quote_to_idr_date": rate_date,
        "quote_to_idr_age_days": age_days,
        "source": metadata.get("quote_to_idr_source"),
        "binance_last_close_time_ms": last_row.get("close_time_ms"),
        "binance_last_close_usdt": close,
        "binance_csv_sha256": metadata.get("csv_sha256"),
    }


def fetch_snapshot(symbol: str = "SOL") -> dict[str, Any]:
    started = utc_now()
    coin = find_coin(request_json("/coins"), symbol)
    coin_id = coin.get("id")
    if coin_id is None:
        raise ValueError(f"Reku coin ID missing for {symbol.upper()}")

    prices = find_price(request_json("/price"), symbol, coin_id)
    price_idr = float(prices.get("c", 0))
    if not math.isfinite(price_idr) or price_idr <= 0:
        raise ValueError("Reku price response contains an invalid IDR close")

    bidask = find_bidask(request_json("/bidask"), symbol)
    direct_spread = spread_metrics(float(bidask.get("bid", 0)), float(bidask.get("ask", 0)))

    orderbook_status = "OK"
    orderbook_error = None
    orderbook_endpoint_used = None
    book = None
    endpoint_errors: list[str] = []
    for endpoint in ("orderbook", "orderbookall"):
        try:
            book = parse_orderbook(request_json(f"/{endpoint}", {"id": coin_id}))
            orderbook_endpoint_used = endpoint
            break
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            endpoint_errors.append(f"{endpoint}: {type(exc).__name__}: {str(exc)[:140]}")
    if book is None:
        orderbook_status = "ERROR"
        orderbook_error = " | ".join(endpoint_errors)[:360]
        book_stats = None
    else:
        book_spread = spread_metrics(
            book["bids"][0]["price_idr"],
            book["asks"][0]["price_idr"],
        )
        top_bid = book["bids"][0]
        top_ask = book["asks"][0]
        book_stats = {
            "endpoint": orderbook_endpoint_used,
            "best_bid_idr": book_spread["bid_idr"],
            "best_ask_idr": book_spread["ask_idr"],
            "mid_idr": book_spread["mid_idr"],
            "spread_idr": book_spread["spread_idr"],
            "spread_bps": book_spread["spread_bps"],
            "bid_levels_returned": len(book["bids"]),
            "ask_levels_returned": len(book["asks"]),
            "top_bid_notional_idr": top_bid["total_idr"],
            "top_ask_notional_idr": top_ask["total_idr"],
            "top_bid_quantity_coin": top_bid["quantity_coin"],
            "top_ask_quantity_coin": top_ask["quantity_coin"],
            "bids_top5_notional_idr": sum(x["total_idr"] for x in book["bids"][:5]),
            "asks_top5_notional_idr": sum(x["total_idr"] for x in book["asks"][:5]),
        }

    warnings = []
    if book_stats is not None:
        direct_mid = direct_spread["mid_idr"]
        book_mid = book_stats["mid_idr"]
        midpoint_deviation_bps = abs(direct_mid / book_mid - 1.0) * 10_000.0
        if midpoint_deviation_bps > 50.0:
            warnings.append("direct_bidask_vs_orderbook_mid_mismatch_over_50bps")
        if price_idr > 0 and abs(price_idr / direct_mid - 1.0) * 10_000.0 > 500.0:
            warnings.append("last_price_vs_quote_mid_mismatch_over_500bps")

    chart_probe, chart_probe_error = fetch_chart_probe(coin_id)
    if chart_probe is None:
        warnings.append("reku_chart_probe_unavailable_or_stale")

    return {
        "schema_version": "sol-sentinel-reku-market-v1",
        "mode": "READ_ONLY_PUBLIC_MARKET_DATA",
        "execution_enabled": False,
        "authenticated": False,
        "orders_enabled": False,
        "venue": "Reku",
        "symbol": symbol.upper(),
        "pair": f"{symbol.upper()}/IDR",
        "captured_at_start_utc": started,
        "captured_at_end_utc": utc_now(),
        "source_docs": "https://api.reku.id/",
        "source_endpoints": [
            "GET https://api.reku.id/v2/coins",
            "GET https://api.reku.id/v2/price",
            "GET https://api.reku.id/v2/bidask",
            f"GET https://api.reku.id/v2/orderbook?id={coin_id}",
            f"GET https://api.reku.id/v2/orderbookall?id={coin_id} (fallback)",
            f"GET https://api.reku.id/v2/chart?t=2&f=1&id={coin_id} (1-minute chart probe)",
            f"GET https://api.reku.id/v2/chart?t=2&f=5&id={coin_id} (5-minute fallback)",
        ],
        "coin_id": coin_id,
        "coin_catalogue": {
            "symbol": row_symbol(coin),
            "name": coin.get("accountname") or coin.get("name"),
            "enablebuy": coin.get("enablebuy"),
            "enablesell": coin.get("enablesell"),
        },
        "last_price_idr": price_idr,
        "price_fields": {
            "open_idr": prices.get("o"),
            "high_idr": prices.get("h"),
            "low_idr": prices.get("l"),
            "volume_coin": prices.get("v"),
            "change_pct": prices.get("cp"),
            "last_trade_type": prices.get("tt"),
        },
        "direct_bidask": direct_spread,
        "orderbook_status": orderbook_status,
        "orderbook_endpoint_used": orderbook_endpoint_used,
        "orderbook_error": orderbook_error,
        "orderbook": book_stats,
        "chart_status": chart_probe["data_status"] if chart_probe else "ERROR",
        "chart_probe": chart_probe,
        "chart_probe_error": chart_probe_error,
        "warnings": warnings,
        "data_status": "OK" if orderbook_status == "OK" and not warnings else "PARTIAL",
        "warning": "Public API documentation may be older than the current platform. This adapter is read-only and must not be used to infer order-placement support.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SOL", help="Reku coin symbol (default SOL)")
    parser.add_argument("--binance-csv", default="", help="optional captured SOLUSDT CSV to compare venue price references")
    parser.add_argument("--output", default="", help="optional JSON report path")
    args = parser.parse_args(argv)
    if not args.symbol.isalnum():
        parser.error("--symbol must contain letters and digits only")

    try:
        report = fetch_snapshot(args.symbol.upper())
        if args.binance_csv:
            try:
                binance_close, reference_metadata = load_binance_reference(args.binance_csv)
                report["binance_reference"] = reference_metadata
                report["cross_venue_comparison"] = compare_reference(
                    report["last_price_idr"],
                    binance_close,
                    reference_metadata["quote_to_idr_rate"],
                )
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                report["cross_venue_comparison"] = None
                report["cross_venue_comparison_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                report["warnings"].append("cross_venue_comparison_unavailable")
        status_code = 0 if report["data_status"] == "OK" else 2
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": "sol-sentinel-reku-market-v1",
            "mode": "READ_ONLY_PUBLIC_MARKET_DATA",
            "execution_enabled": False,
            "authenticated": False,
            "orders_enabled": False,
            "venue": "Reku",
            "symbol": args.symbol.upper(),
            "pair": f"{args.symbol.upper()}/IDR",
            "captured_at_utc": utc_now(),
            "data_status": "ERROR",
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            "warning": "No fallback price or order was fabricated.",
        }
        status_code = 2

    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
    print(encoded)
    return status_code


if __name__ == "__main__":
    raise SystemExit(main())
