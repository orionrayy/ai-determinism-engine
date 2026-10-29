#!/usr/bin/env python3
"""Read-only SOL public-market scanner. No orders, wallet keys, or trade execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

BINANCE_BASE = "https://api.binance.com"
USER_AGENT = "sol-sentinel-research/0.1 (read-only public market data)"
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000}


@dataclass(frozen=True)
class Candle:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int
    quote_volume: float = 0.0
    trades: int = 0


def utc_iso(ts_ms: int | float) -> str:
    return datetime.fromtimestamp(float(ts_ms) / 1000, tz=timezone.utc).isoformat()


def ema(values: Sequence[float], period: int) -> list[float]:
    if period < 1:
        raise ValueError("period must be >= 1")
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def rsi(values: Sequence[float], period: int = 14) -> float | None:
    if period < 1 or len(values) < period + 1:
        return None
    deltas = [float(values[i]) - float(values[i - 1]) for i in range(1, len(values))]
    window = deltas[-period:]
    gains = sum(max(delta, 0.0) for delta in window) / period
    losses = sum(max(-delta, 0.0) for delta in window) / period
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return 100.0 - 100.0 / (1.0 + rs)


def atr(candles: Sequence[Candle], period: int = 14) -> float | None:
    if period < 1 or len(candles) < period + 1:
        return None
    true_ranges = []
    for previous, current in zip(candles[-period - 1:-1], candles[-period:]):
        true_ranges.append(max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        ))
    return sum(true_ranges) / len(true_ranges) if true_ranges else None


def relative_volume(candles: Sequence[Candle], lookback: int = 20) -> float | None:
    if lookback < 1 or len(candles) < lookback + 1:
        return None
    baseline = sum(c.volume for c in candles[-lookback - 1:-1]) / lookback
    if baseline <= 0:
        return None
    return candles[-1].volume / baseline


def parse_klines(rows: Any, now_ms: int | None = None) -> list[Candle]:
    if not isinstance(rows, list):
        raise ValueError("Kline payload must be a list")
    now = int(time.time() * 1000) if now_ms is None else now_ms
    candles: list[Candle] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 9:
            continue
        try:
            close_time = int(row[6])
            if close_time >= now:
                continue
            candle = Candle(
                open_time_ms=int(row[0]), open=float(row[1]), high=float(row[2]),
                low=float(row[3]), close=float(row[4]), volume=float(row[5]),
                close_time_ms=close_time, quote_volume=float(row[7]), trades=int(row[8]),
            )
        except (TypeError, ValueError, OverflowError):
            continue
        values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
        if not all(math.isfinite(v) for v in values):
            continue
        if min(candle.open, candle.high, candle.low, candle.close) <= 0 or candle.volume < 0:
            continue
        if candle.high < max(candle.open, candle.close, candle.low):
            continue
        if candle.low > min(candle.open, candle.close, candle.high):
            continue
        candles.append(candle)
    candles.sort(key=lambda c: c.open_time_ms)
    deduped: dict[int, Candle] = {c.open_time_ms: c for c in candles}
    return [deduped[key] for key in sorted(deduped)]


def fetch_candles(symbol: str, interval: str, limit: int = 100, timeout: float = 8.0) -> list[Candle]:
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported interval: {interval}")
    query = urllib.parse.urlencode({"symbol": symbol.upper(), "interval": interval, "limit": limit})
    request = urllib.request.Request(
        f"{BINANCE_BASE}/api/v3/klines?{query}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"market data HTTP status {response.status}")
        raw = response.read(1_000_000)
    rows = json.loads(raw.decode("utf-8"))
    candles = parse_klines(rows)
    if len(candles) < 30:
        raise ValueError(f"{interval}: insufficient completed candles ({len(candles)}/30)")
    now_ms = int(time.time() * 1000)
    age_ms = now_ms - candles[-1].close_time_ms
    if age_ms > max(2 * INTERVAL_MS[interval], 120_000):
        raise ValueError(f"{interval}: latest candle stale ({age_ms} ms)")
    for left, right in zip(candles, candles[1:]):
        if right.open_time_ms <= left.open_time_ms:
            raise ValueError(f"{interval}: candle timestamps are not increasing")
    return candles


def timeframe_metrics(candles: Sequence[Candle], interval: str) -> dict[str, Any]:
    if len(candles) < 30:
        raise ValueError(f"{interval}: at least 30 completed candles are required")
    closes = [c.close for c in candles]
    e9, e21 = ema(closes, 9), ema(closes, 21)
    rsi14, atr14 = rsi(closes, 14), atr(candles, 14)
    rv20, latest = relative_volume(candles, 20), candles[-1]
    previous_high = max(c.high for c in candles[-21:-1])
    atr_pct = (atr14 / latest.close * 100.0) if atr14 is not None else None
    return {
        "interval": interval, "closed_at": utc_iso(latest.close_time_ms),
        "age_ms": max(0, int(time.time() * 1000) - latest.close_time_ms),
        "close": latest.close, "ema9": e9[-1], "ema21": e21[-1],
        "rsi14": rsi14, "atr14": atr14, "atr_pct": atr_pct,
        "relative_volume20": rv20, "prior_20_high": previous_high,
        "close_above_prior_20_high": latest.close > previous_high,
        "trend_up": e9[-1] > e21[-1],
    }


def evaluate_signal(
    metrics: dict[str, dict[str, Any]], fee_bps_round_trip: float = 42.11,
    spread_bps_round_trip: float = 10.0, slippage_bps_round_trip: float = 20.0, min_net_edge_bps: float = 25.0,
    volume_ratio_min: float = 1.5,
) -> dict[str, Any]:
    required = {"1m", "5m", "15m"}
    if not required.issubset(metrics):
        return {"signal": "NO_TRADE", "reasons": ["missing_timeframe_metrics"]}
    reasons: list[str] = []
    m1, m5, m15 = metrics["1m"], metrics["5m"], metrics["15m"]
    costs_bps = fee_bps_round_trip + spread_bps_round_trip + slippage_bps_round_trip
    atr14, close5 = m5.get("atr14"), m5.get("close")
    gross_target_bps = (
        float(atr14) / float(close5) * 10_000.0 * 1.5
        if atr14 is not None and close5 and close5 > 0 else 0.0
    )
    net_edge_bps = gross_target_bps - costs_bps

    if not m15.get("trend_up") or m15.get("close", 0) <= m15.get("ema21", math.inf):
        reasons.append("15m_trend_not_confirmed")
    if not m5.get("trend_up") or m5.get("close", 0) <= m5.get("ema9", math.inf):
        reasons.append("5m_momentum_not_confirmed")
    rsi5 = m5.get("rsi14")
    if rsi5 is None or not 48.0 <= rsi5 <= 70.0:
        reasons.append("5m_rsi_outside_entry_range")
    if not m1.get("close_above_prior_20_high"):
        reasons.append("1m_breakout_not_confirmed")
    volume_ratio = m1.get("relative_volume20")
    if volume_ratio is None or volume_ratio < volume_ratio_min:
        reasons.append("1m_volume_confirmation_missing")
    rsi1 = m1.get("rsi14")
    if rsi1 is None or rsi1 >= 78.0:
        reasons.append("1m_momentum_exhaustion_or_missing")
    if net_edge_bps < min_net_edge_bps:
        reasons.append("estimated_net_edge_below_threshold")
    return {
        "signal": "PAPER_CANDIDATE" if not reasons else "NO_TRADE",
        "reasons": reasons,
        "cost_model": {
            "fee_bps_round_trip": fee_bps_round_trip,
            "spread_bps_round_trip": spread_bps_round_trip,
            "slippage_bps_round_trip": slippage_bps_round_trip,
            "total_cost_bps": costs_bps, "min_net_edge_bps": min_net_edge_bps,
            "assumption_only": True,
        },
        "edge_model": {
            "atr_target_multiple": 1.5, "gross_move_budget_bps": round(gross_target_bps, 2),
            "estimated_net_edge_bps": round(net_edge_bps, 2), "not_a_return_forecast": True,
        },
    }


def scan_once(args: argparse.Namespace) -> dict[str, Any]:
    started_ms = int(time.time() * 1000)
    record: dict[str, Any] = {
        "schema_version": "sol-sentinel-scan-v1",
        "run_id": hashlib.sha256(f"{args.symbol}:{started_ms}".encode()).hexdigest()[:20],
        "observed_at": utc_iso(started_ms), "mode": "READ_ONLY_RESEARCH",
        "symbol": args.symbol.upper(), "market_data_source": "Binance public spot klines",
        "execution_venue": None, "order_placement_enabled": False,
        "account_balance_source": None, "metrics": {},
    }
    try:
        candles_by_interval = {iv: fetch_candles(args.symbol, iv) for iv in ("1m", "5m", "15m")}
        metrics = {iv: timeframe_metrics(cs, iv) for iv, cs in candles_by_interval.items()}
        record.update({
            "data_status": "OK", "metrics": metrics,
            **evaluate_signal(
                metrics, fee_bps_round_trip=args.fee_bps_round_trip,
                spread_bps_round_trip=args.spread_bps_round_trip,
                slippage_bps_round_trip=args.slippage_bps_round_trip,
                min_net_edge_bps=args.min_net_edge_bps, volume_ratio_min=args.volume_ratio_min,
            ),
        })
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        record.update({
            "data_status": "ERROR", "signal": "NO_TRADE",
            "reasons": [f"market_data_unavailable_or_invalid: {type(exc).__name__}: {str(exc)[:240]}"],
        })
    return record


def emit(record: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        return
    print(f"SOL SENTINEL | {record['observed_at']} | {record['symbol']} | {record['mode']}")
    print(f"Data status : {record.get('data_status', 'UNKNOWN')}")
    print(f"Signal      : {record.get('signal', 'NO_TRADE')}")
    for interval in ("15m", "5m", "1m"):
        item = record.get("metrics", {}).get(interval)
        if item:
            print(
                f"{interval:>3} close={item['close']:.6f} EMA9={item['ema9']:.6f} "
                f"EMA21={item['ema21']:.6f} RSI14={item['rsi14']:.2f} ATR%={item['atr_pct']:.3f}"
            )
    edge = record.get("edge_model")
    if edge:
        print(
            f"Cost budget={record['cost_model']['total_cost_bps']:.2f} bps "
            f"(fees={record['cost_model']['fee_bps_round_trip']:.2f}, "
            f"spread={record['cost_model']['spread_bps_round_trip']:.2f}, "
            f"slippage={record['cost_model']['slippage_bps_round_trip']:.2f}); "
            f"gross ATR budget={edge['gross_move_budget_bps']:.2f} bps; "
            f"estimated net edge={edge['estimated_net_edge_bps']:.2f} bps"
        )
    for reason in record.get("reasons", []):
        print(f"  - {reason}")
    print("SAFETY: This scanner cannot place orders. Signal is a paper-testing candidate only.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SOLUSDT", help="public Binance spot market symbol (default: SOLUSDT)")
    parser.add_argument("--fee-bps-round-trip", type=float, default=42.11,
                        help="assumed total round-trip venue fee in basis points; verify actual fees")
    parser.add_argument("--spread-bps-round-trip", type=float, default=10.0,
                        help="assumed quoted spread cost for a full entry/exit round trip")
    parser.add_argument("--slippage-bps-round-trip", type=float, default=20.0,
                        help="assumed total round-trip slippage budget in basis points")
    parser.add_argument("--min-net-edge-bps", type=float, default=25.0,
                        help="minimum modeled net edge after estimated costs")
    parser.add_argument("--volume-ratio-min", type=float, default=1.5,
                        help="minimum 1m volume vs mean of prior 20 completed candles")
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    parser.add_argument("--log-path", default="sol_sentinel_runs.jsonl",
                        help="append scan output as JSONL (use a private local path)")
    parser.add_argument("--watch-seconds", type=int, default=0,
                        help="repeat scans; minimum 15 seconds, 0 means run once")
    args = parser.parse_args(argv)
    for field in ("fee_bps_round_trip", "spread_bps_round_trip", "slippage_bps_round_trip", "min_net_edge_bps", "volume_ratio_min"):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{field.replace('_', '-')} must be finite and non-negative")
    if args.watch_seconds != 0 and args.watch_seconds < 15:
        parser.error("--watch-seconds must be 0 or at least 15")
    if not args.symbol.isalnum():
        parser.error("--symbol must contain letters and digits only")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    while True:
        record = scan_once(args)
        emit(record, args.json)
        if args.log_path:
            try:
                path = Path(args.log_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            except OSError as exc:
                print(f"WARNING: could not append audit log: {exc}", file=sys.stderr)
        if not args.watch_seconds:
            return 0 if record.get("data_status") == "OK" else 2
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
