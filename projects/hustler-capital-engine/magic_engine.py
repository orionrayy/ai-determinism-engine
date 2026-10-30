#!/usr/bin/env python3
"""Hustler Capital Engine — MAGICUSDT USDⓈ-M Futures research module.

Read-only public market-data collector + deterministic analysis + SQLite evidence ledger.
No authenticated endpoints, order placement, cancel, leverage changes, or withdrawals exist.
Python standard library only; suitable for constrained Termux environments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional, Sequence

ENGINE_VERSION = "0.1.0"
BASE_URL = "https://fapi.binance.com"
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
DEFAULT_CONFIG: Dict[str, Any] = {
    "symbol": "MAGICUSDT", "market_type": "USD_M_PERPETUAL",
    "entry_notional_usdt": 1.0, "leverage_scenarios": [23, 25, 30],
    "analysis_margin_mode": "ISOLATED", "live_trading_enabled": False,
    "paper_execution_enabled": False, "max_open_positions": 1,
    "risk_per_trade_fraction": 0.005, "daily_loss_limit_fraction": 0.02,
    "max_spread_bps": 20.0, "max_quote_age_seconds": 15.0,
    "max_funding_rate_abs": 0.003, "required_intervals": ["5m", "15m", "1h", "4h"],
    "history_limit": 200, "fee_taker_bps": None, "fee_maker_bps": None,
    "fees_verified": False, "slippage_bps_per_side": None, "slippage_verified": False,
    "net_edge_buffer_bps": 10.0, "holding_hours_for_cost_model": 4.0,
}


class DataUnavailable(RuntimeError):
    pass


class DataIntegrityError(RuntimeError):
    pass


class LedgerConflict(RuntimeError):
    pass


def _finite(value: Any, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataIntegrityError(f"{field}: not numeric") from exc
    if not math.isfinite(out):
        raise DataIntegrityError(f"{field}: not finite")
    return out


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SymbolRules:
    symbol: str
    status: str
    contract_type: str
    quote_asset: str
    margin_asset: str
    tick_size: float
    min_qty: float
    step_size: float
    market_min_qty: float
    market_step_size: float
    min_notional: float
    max_notional: Optional[float]


def parse_symbol_rules(exchange_info: Dict[str, Any], symbol: str) -> SymbolRules:
    if not isinstance(exchange_info, dict) or not isinstance(exchange_info.get("symbols"), list):
        raise DataIntegrityError("exchangeInfo missing symbols array")
    rows = [row for row in exchange_info["symbols"] if isinstance(row, dict) and row.get("symbol") == symbol]
    if len(rows) != 1:
        raise DataIntegrityError(f"symbol {symbol} absent or duplicated in exchangeInfo")
    row = rows[0]
    filters = row.get("filters")
    if not isinstance(filters, list):
        raise DataIntegrityError("symbol filters missing")
    by_type = {f.get("filterType"): f for f in filters if isinstance(f, dict) and f.get("filterType")}
    price = by_type.get("PRICE_FILTER", {})
    lot = by_type.get("LOT_SIZE", {})
    market_lot = by_type.get("MARKET_LOT_SIZE", lot)
    notional_filter = by_type.get("MIN_NOTIONAL") or by_type.get("NOTIONAL") or {}
    min_notional = notional_filter.get("notional", notional_filter.get("minNotional", price.get("notional")))
    if min_notional is None:
        raise DataIntegrityError("minimum notional not published in symbol filters")
    tick = _finite(price.get("tickSize"), "tickSize")
    min_qty = _finite(lot.get("minQty"), "minQty")
    step = _finite(lot.get("stepSize"), "stepSize")
    market_min_qty = _finite(market_lot.get("minQty", min_qty), "market minQty")
    market_step = _finite(market_lot.get("stepSize", step), "market stepSize")
    minimum = _finite(min_notional, "minNotional")
    maximum = notional_filter.get("maxNotional")
    maximum_f = _finite(maximum, "maxNotional") if maximum is not None else None
    if min(tick, min_qty, step, market_min_qty, market_step, minimum) <= 0:
        raise DataIntegrityError("symbol filters contain non-positive increments")
    return SymbolRules(
        symbol=symbol, status=str(row.get("status", "UNKNOWN")),
        contract_type=str(row.get("contractType", "UNKNOWN")),
        quote_asset=str(row.get("quoteAsset", "")), margin_asset=str(row.get("marginAsset", "")),
        tick_size=tick, min_qty=min_qty, step_size=step,
        market_min_qty=market_min_qty, market_step_size=market_step,
        min_notional=minimum, max_notional=maximum_f,
    )


def floor_step(value: float, step: float) -> float:
    if value < 0 or step <= 0 or not math.isfinite(value) or not math.isfinite(step):
        raise ValueError("invalid value/step")
    d_value, d_step = Decimal(str(value)), Decimal(str(step))
    return float((d_value / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step)


def size_order(notional_usdt: float, price: float, rules: SymbolRules, market: bool = False) -> Dict[str, Any]:
    notional = _finite(notional_usdt, "entry_notional_usdt")
    px = _finite(price, "price")
    if notional <= 0 or px <= 0:
        raise DataIntegrityError("notional and price must be positive")
    min_qty = rules.market_min_qty if market else rules.min_qty
    step = rules.market_step_size if market else rules.step_size
    raw_qty = notional / px
    qty = floor_step(raw_qty, step)
    actual_notional = qty * px
    reasons: List[str] = []
    if qty < min_qty:
        reasons.append("QUANTITY_BELOW_EXCHANGE_MINIMUM")
    if actual_notional < rules.min_notional:
        reasons.append("NOTIONAL_BELOW_EXCHANGE_MINIMUM")
    if rules.max_notional is not None and actual_notional > rules.max_notional:
        reasons.append("NOTIONAL_ABOVE_EXCHANGE_MAXIMUM")
    return {
        "requested_notional_usdt": round(notional, 8), "raw_quantity": raw_qty,
        "rounded_quantity": qty, "actual_notional_usdt": round(actual_notional, 8),
        "min_qty": min_qty, "step_size": step, "min_notional_usdt": rules.min_notional,
        "max_notional_usdt": rules.max_notional, "valid": not reasons, "reasons": reasons,
    }


class BinanceUSDMClient:
    """Public-only USDⓈ-M Futures API client. No API-key support by design."""
    def __init__(self, base_url: str = BASE_URL, timeout: float = 8.0, retries: int = 2, max_bytes: int = 2_000_000):
        if not base_url.startswith("https://"):
            raise ValueError("HTTPS required")
        self.base_url = base_url.rstrip("/")
        self.timeout = max(2.0, min(float(timeout), 20.0))
        self.retries = max(0, min(int(retries), 3))
        self.max_bytes = max(1024, min(int(max_bytes), 4_000_000))

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not path.startswith("/fapi/") and not path.startswith("/futures/data/"):
            raise ValueError("endpoint not on allowlist")
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
        url = self.base_url + path + ("?" + query if query else "")
        request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "HustlerCapitalEngine-MAGIC/0.1 read-only"})
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    status = getattr(response, "status", 200)
                    if status != 200:
                        raise DataUnavailable(f"HTTP {status} for {path}")
                    payload = response.read(self.max_bytes + 1)
                    if len(payload) > self.max_bytes:
                        raise DataUnavailable(f"response exceeded {self.max_bytes} bytes")
                try:
                    return json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise DataIntegrityError(f"invalid JSON from {path}") from exc
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in (418, 429, 500, 502, 503, 504) or attempt >= self.retries:
                    raise DataUnavailable(f"HTTP {exc.code} for {path}") from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = min(2 ** attempt, 4)
                if retry_after:
                    try:
                        delay = min(max(float(retry_after), 0.25), 5.0)
                    except ValueError:
                        pass
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise DataUnavailable(f"network error for {path}: {type(exc).__name__}") from exc
                time.sleep(min(2 ** attempt, 3))
        raise DataUnavailable(f"request failed: {last_error}")

    def exchange_info(self) -> Dict[str, Any]:
        return self.get("/fapi/v1/exchangeInfo")

    def klines(self, symbol: str, interval: str, limit: int = 200) -> List[List[Any]]:
        if interval not in INTERVAL_MS:
            raise ValueError(f"unsupported interval {interval}")
        return self.get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": min(max(int(limit), 60), 1000)})

    def collect(self, symbol: str = "MAGICUSDT", history_limit: int = 200) -> Dict[str, Any]:
        collected_start = _now_ms()
        info = self.exchange_info()
        rules = parse_symbol_rules(info, symbol)
        ticker = self.get("/fapi/v1/ticker/price", {"symbol": symbol})
        ticker_received_ms = _now_ms()
        premium = self.get("/fapi/v1/premiumIndex", {"symbol": symbol})
        premium_received_ms = _now_ms()
        depth = self.get("/fapi/v1/depth", {"symbol": symbol, "limit": 20})
        depth_received_ms = _now_ms()
        ticker_24h = self.get("/fapi/v1/ticker/24hr", {"symbol": symbol})
        ticker_24h_received_ms = _now_ms()
        candles = {tf: self.klines(symbol, tf, history_limit) for tf in ("5m", "15m", "1h", "4h")}
        optional: Dict[str, Any] = {}
        optional_specs = {
            "open_interest": ("/fapi/v1/openInterest", {"symbol": symbol}),
            "open_interest_hist": ("/futures/data/openInterestHist", {"symbol": symbol, "period": "15m", "limit": 10}),
            "global_long_short_15m": ("/futures/data/globalLongShortAccountRatio", {"symbol": symbol, "period": "15m", "limit": 10}),
            "top_long_short_position_15m": ("/futures/data/topLongShortPositionRatio", {"symbol": symbol, "period": "15m", "limit": 10}),
            "taker_long_short_15m": ("/futures/data/takerlongshortRatio", {"symbol": symbol, "period": "15m", "limit": 10}),
            "funding_history": ("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 10}),
        }
        optional_errors: Dict[str, str] = {}
        for name, (path, params) in optional_specs.items():
            try:
                optional[name] = self.get(path, params)
            except (DataUnavailable, DataIntegrityError) as exc:
                optional_errors[name] = str(exc)
        server_time_response = self.get("/fapi/v1/time")
        server_time_received_ms = _now_ms()
        server_time_ms = int(server_time_response.get("serverTime")) if isinstance(server_time_response, dict) and server_time_response.get("serverTime") is not None else None
        collected_end = _now_ms()
        return {
            "schema_version": 1, "engine_version": ENGINE_VERSION, "source": "binance_usdm_public_rest",
            "symbol": symbol, "collected_start_ms": collected_start, "collected_end_ms": collected_end,
            "collected_at_utc": _utc_now(), "exchange_server_time_ms": server_time_ms,
            "server_time_received_ms": server_time_received_ms,
            "quote_received_ms": {"ticker": ticker_received_ms, "premium_index": premium_received_ms, "depth": depth_received_ms, "ticker_24h": ticker_24h_received_ms},
            "rules": rules.__dict__, "ticker": ticker, "ticker_24h": ticker_24h, "premium_index": premium, "depth": depth,
            "candles": candles, "optional": optional, "optional_errors": optional_errors,
        }


def parse_candles(raw: Sequence[Sequence[Any]], interval: str, now_ms: Optional[int] = None, min_bars: int = 60) -> List[Dict[str, float]]:
    if interval not in INTERVAL_MS or not isinstance(raw, (list, tuple)):
        raise DataIntegrityError("invalid candle collection")
    now = _now_ms() if now_ms is None else int(now_ms)
    interval_ms = INTERVAL_MS[interval]
    parsed: List[Dict[str, float]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            raise DataIntegrityError(f"{interval} candle {i}: malformed row")
        try:
            open_ms = int(row[0]); close_ms = int(row[6])
        except (TypeError, ValueError) as exc:
            raise DataIntegrityError(f"{interval} candle {i}: invalid timestamp") from exc
        vals = [_finite(row[j], f"{interval}[{i}].{name}") for j, name in zip((1,2,3,4,5), ("open","high","low","close","volume"))]
        o, h, l, c, v = vals
        if min(o, h, l, c) <= 0 or v < 0 or l > min(o,c) or h < max(o,c) or h < l:
            raise DataIntegrityError(f"{interval} candle {i}: impossible OHLCV")
        if close_ms < open_ms or close_ms - open_ms + 1 != interval_ms:
            raise DataIntegrityError(f"{interval} candle {i}: interval/close-time mismatch")
        if close_ms >= now:
            continue
        parsed.append({"open_time_ms": open_ms, "close_time_ms": close_ms, "open": o, "high": h, "low": l, "close": c, "volume": v})
    for previous, current in zip(parsed, parsed[1:]):
        if current["open_time_ms"] <= previous["open_time_ms"]:
            raise DataIntegrityError(f"{interval}: duplicate or non-monotonic candles")
        if current["open_time_ms"] - previous["open_time_ms"] != interval_ms:
            raise DataIntegrityError(f"{interval}: candle gap detected")
    if len(parsed) < min_bars:
        raise DataIntegrityError(f"{interval}: insufficient completed bars ({len(parsed)} < {min_bars})")
    return parsed


def ema(values: Sequence[float], period: int) -> float:
    if period <= 0 or len(values) < period:
        raise DataIntegrityError(f"EMA{period}: insufficient history")
    alpha = 2.0 / (period + 1.0)
    result = sum(values[:period]) / period
    for value in values[period:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def rsi_wilder(values: Sequence[float], period: int = 14) -> float:
    if period <= 0 or len(values) < period + 1:
        raise DataIntegrityError("RSI: insufficient history")
    deltas = [b - a for a, b in zip(values, values[1:])]
    gains = [max(x, 0.0) for x in deltas[:period]]
    losses = [max(-x, 0.0) for x in deltas[:period]]
    avg_gain, avg_loss = sum(gains)/period, sum(losses)/period
    for delta in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def atr(candles: Sequence[Dict[str, float]], period: int = 14) -> float:
    if len(candles) < period + 1:
        raise DataIntegrityError("ATR: insufficient history")
    true_ranges = []
    previous_close = candles[0]["close"]
    for candle in candles[1:]:
        true_ranges.append(max(candle["high"]-candle["low"], abs(candle["high"]-previous_close), abs(candle["low"]-previous_close)))
        previous_close = candle["close"]
    value = sum(true_ranges[:period]) / period
    for tr in true_ranges[period:]:
        value = (value*(period-1)+tr) / period
    return value


def interval_metrics(candles: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    e20, e50 = ema(closes,20), ema(closes,50)
    e20_prev = ema(closes[:-1],20)
    rsi, atr_value = rsi_wilder(closes,14), atr(candles,14)
    avg_volume = sum(volumes[-21:-1]) / 20
    rel_volume = volumes[-1] / avg_volume if avg_volume > 0 else 0.0
    previous = candles[-21:-1]
    previous_high = max(x["high"] for x in previous)
    previous_low = min(x["low"] for x in previous)
    last = candles[-1]
    bias = "BULLISH" if last["close"] > e20 > e50 and e20 > e20_prev else "BEARISH" if last["close"] < e20 < e50 and e20 < e20_prev else "NEUTRAL"
    return {
        "close": last["close"], "candle_close_time_ms": last["close_time_ms"],
        "ema20": e20, "ema50": e50, "ema20_slope": e20-e20_prev,
        "rsi14": rsi, "atr14": atr_value, "atr_pct": atr_value/last["close"]*100,
        "relative_volume20": rel_volume, "previous_20_high": previous_high,
        "previous_20_low": previous_low, "bias": bias,
        "breakout_up": last["close"] > previous_high, "breakout_down": last["close"] < previous_low,
    }


def _book_metrics(depth: Dict[str, Any]) -> Dict[str, float]:
    bids, asks = depth.get("bids"), depth.get("asks")
    if not isinstance(bids,list) or not isinstance(asks,list) or not bids or not asks:
        raise DataIntegrityError("order book missing bids/asks")
    bid, ask = _finite(bids[0][0],"best_bid"), _finite(asks[0][0],"best_ask")
    if bid <= 0 or ask <= 0 or bid >= ask:
        raise DataIntegrityError("crossed or invalid order book")
    bid_notional = sum(_finite(x[0],"bid_price")*_finite(x[1],"bid_qty") for x in bids[:10])
    ask_notional = sum(_finite(x[0],"ask_price")*_finite(x[1],"ask_qty") for x in asks[:10])
    total = bid_notional + ask_notional
    if total <= 0:
        raise DataIntegrityError("empty order book depth")
    mid = (bid+ask)/2
    return {"best_bid":bid,"best_ask":ask,"mid":mid,"spread_bps":(ask-bid)/mid*10_000,
            "bid_notional_top10":bid_notional,"ask_notional_top10":ask_notional,
            "bid_depth_share_pct":bid_notional/total*100,"book_imbalance":(bid_notional-ask_notional)/total}


def _latest_funding(snapshot: Dict[str, Any]) -> Optional[float]:
    premium = snapshot.get("premium_index")
    if not isinstance(premium,dict) or premium.get("lastFundingRate") is None:
        return None
    return _finite(premium["lastFundingRate"],"lastFundingRate")


def _funding_interval_hours(snapshot: Dict[str, Any]) -> Optional[float]:
    premium = snapshot.get("premium_index") or {}
    history = (snapshot.get("optional") or {}).get("funding_history") or []
    next_ms = premium.get("nextFundingTime")
    if not next_ms or not isinstance(history,list) or not history:
        return None
    valid_times=[]
    for item in history:
        if isinstance(item,dict) and item.get("fundingTime") is not None:
            try: valid_times.append(int(item["fundingTime"]))
            except (TypeError,ValueError): continue
    if not valid_times: return None
    interval = (int(next_ms)-max(valid_times))/3_600_000.0
    return interval if 0 < interval <= 24 else None


def _cost_model(snapshot: Dict[str, Any], cfg: Dict[str, Any], book: Dict[str, float],
                funding_rate: Optional[float], last_price: float, atr_5m: Optional[float]) -> Dict[str, Any]:
    fee_known = bool(cfg.get("fees_verified")) and cfg.get("fee_taker_bps") is not None
    slip_known = bool(cfg.get("slippage_verified")) and cfg.get("slippage_bps_per_side") is not None
    interval_hours = _funding_interval_hours(snapshot)
    now_ms = int(snapshot.get("collected_end_ms") or _now_ms())
    next_ms = (snapshot.get("premium_index") or {}).get("nextFundingTime")
    holding_hours = max(0.0,float(cfg.get("holding_hours_for_cost_model",4.0)))
    funding_events=0
    if next_ms is not None and interval_hours and holding_hours > 0:
        first=(int(next_ms)-now_ms)/3_600_000.0
        if first <= holding_hours:
            funding_events=1+int(max(0.0,holding_hours-max(0.0,first))//interval_hours)
    fee_rt=2.0*float(cfg["fee_taker_bps"]) if fee_known else None
    slip_rt=2.0*float(cfg["slippage_bps_per_side"]) if slip_known else None
    funding_bps=abs(funding_rate or 0.0)*10_000.0*funding_events if funding_rate is not None else None
    spread_bps=float(book["spread_bps"])
    total=fee_rt+slip_rt+spread_bps+(funding_bps or 0.0) if fee_known and slip_known and funding_bps is not None else None
    atr_bps=(atr_5m/last_price*10_000.0) if atr_5m is not None and last_price > 0 else None
    return {
        "fee_taker_bps_per_side":float(cfg["fee_taker_bps"]) if fee_known else None,
        "fee_round_trip_bps":fee_rt,"measured_spread_bps":spread_bps,
        "slippage_bps_per_side":float(cfg["slippage_bps_per_side"]) if slip_known else None,
        "slippage_round_trip_bps":slip_rt,"funding_rate_per_event":funding_rate,
        "funding_interval_hours":interval_hours,"estimated_funding_events":funding_events,
        "funding_cost_bps_abs_sensitivity":funding_bps,"holding_hours_assumption":holding_hours,
        "estimated_round_trip_cost_bps":total,"net_edge_buffer_bps":float(cfg["net_edge_buffer_bps"]),
        "break_even_gross_move_bps_with_buffer":total+float(cfg["net_edge_buffer_bps"]) if total is not None else None,
        "atr_5m_bps_reference_only":atr_bps,
        "cost_model_status":"VERIFIED_INPUTS" if fee_known and slip_known and funding_bps is not None else "BLOCKED_UNVERIFIED_FEE_OR_SLIPPAGE_OR_FUNDING_INTERVAL",
        "note":"ATR is a movement yardstick, not expectancy; absolute funding is conservative and actual payer depends on direction/rate.",
    }


def analyze_snapshot(snapshot: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg=dict(DEFAULT_CONFIG)
    if config: cfg.update(config)
    symbol=str(cfg["symbol"])
    reasons: List[str]=[]
    metrics: Dict[str,Any]={}
    quality={"valid":True,"reasons":[]}
    try:
        if snapshot.get("symbol") != symbol:
            raise DataIntegrityError("snapshot symbol/config mismatch")
        rules=SymbolRules(**(snapshot.get("rules") or {}))
        if rules.symbol != symbol or rules.status != "TRADING" or rules.contract_type != "PERPETUAL" or rules.quote_asset != "USDT":
            raise DataIntegrityError("contract is not an active USDT perpetual")
        last=_finite((snapshot.get("ticker") or {}).get("price"),"ticker.price")
        premium=snapshot.get("premium_index") or {}
        mark=_finite(premium.get("markPrice"),"premium_index.markPrice")
        if last <= 0 or mark <= 0: raise DataIntegrityError("non-positive last/mark price")
        basis=(mark-last)/last*10_000
        book=_book_metrics(snapshot.get("depth") or {})
        if abs(basis)>50: reasons.append("MARK_LAST_BASIS_EXCEEDS_50_BPS")
        if book["spread_bps"]>float(cfg["max_spread_bps"]): reasons.append("SPREAD_ABOVE_POLICY")
        local_end=int(snapshot.get("collected_end_ms") or _now_ms())
        exchange_now=snapshot.get("exchange_server_time_ms")
        if exchange_now is None:
            reasons.append("EXCHANGE_SERVER_TIME_MISSING"); now=local_end
        else:
            now=int(exchange_now)
            if abs(now-local_end)>10_000: reasons.append("LOCAL_CLOCK_OFFSET_EXCEEDS_10_SECONDS")
        tf={}
        for interval in cfg["required_intervals"]:
            bars=parse_candles((snapshot.get("candles") or {}).get(interval),interval,now_ms=now,min_bars=60)
            tf[interval]=interval_metrics(bars)
            age=now-bars[-1]["close_time_ms"]
            if age<0 or age>INTERVAL_MS[interval]*2: reasons.append(f"{interval.upper()}_LATEST_CANDLE_STALE")
        duration=max(0.0,(local_end-int(snapshot.get("collected_start_ms") or 0))/1000)
        if duration>90: reasons.append("COLLECTION_WINDOW_TOO_SLOW")
        receipt=snapshot.get("quote_received_ms") or {}
        if not receipt: reasons.append("QUOTE_RECEIPT_TIMESTAMPS_MISSING")
        for name,stamp in receipt.items():
            age=(local_end-int(stamp))/1000.0
            if age<0 or age>float(cfg["max_quote_age_seconds"]): reasons.append(f"{name.upper()}_QUOTE_STALE")
        funding=_latest_funding(snapshot)
        if funding is None: reasons.append("FUNDING_RATE_MISSING")
        elif abs(funding)>float(cfg["max_funding_rate_abs"]): reasons.append("FUNDING_RATE_EXTREME")
        h4,h1,m15,m5=tf["4h"],tf["1h"],tf["15m"],tf["5m"]
        long_checks={"h4_trend":h4["bias"]=="BULLISH","h1_trend":h1["bias"]=="BULLISH",
            "m15_alignment":m15["bias"]=="BULLISH","m5_breakout":m5["breakout_up"],
            "m5_relative_volume":m5["relative_volume20"]>=1.2,"m5_rsi_not_overheated":50<=m5["rsi14"]<=72}
        short_checks={"h4_trend":h4["bias"]=="BEARISH","h1_trend":h1["bias"]=="BEARISH",
            "m15_alignment":m15["bias"]=="BEARISH","m5_breakdown":m5["breakout_down"],
            "m5_relative_volume":m5["relative_volume20"]>=1.2,"m5_rsi_not_exhausted":28<=m5["rsi14"]<=50}
        weights={"h4_trend":15,"h1_trend":25,"m15_alignment":20,"m5_breakout":20,"m5_breakdown":20,
            "m5_relative_volume":10,"m5_rsi_not_overheated":10,"m5_rsi_not_exhausted":10}
        score=lambda checks: sum(weights.get(k,0) for k,v in checks.items() if v)
        long_score,short_score=score(long_checks),score(short_checks)
        candidate="LONG" if long_score>=90 and all(long_checks.values()) else "SHORT" if short_score>=90 and all(short_checks.values()) else "NONE"
        size=size_order(float(cfg["entry_notional_usdt"]),last,rules)
        costs=_cost_model(snapshot,cfg,book,funding,last,m5["atr14"])
        stop_bps=1.5*m5["atr14"]/last*10_000.0
        equity=cfg.get("manual_equity_snapshot_usdt")
        risk_budget=float(equity)*float(cfg["risk_per_trade_fraction"]) if equity is not None else None
        stop_loss=float(cfg["entry_notional_usdt"])*stop_bps/10_000.0
        risk={"manual_equity_snapshot_usdt":float(equity) if equity is not None else None,
            "equity_source":cfg.get("equity_source","NOT_CONNECTED"),"risk_budget_usdt":risk_budget,
            "stop_distance_bps_at_1_5_atr":stop_bps,"gross_stop_loss_usdt_per_entry":stop_loss,
            "risk_budget_pass":stop_loss<=risk_budget if risk_budget is not None else None,
            "leverage_scenarios":[{"leverage":float(lev),"initial_margin_usdt":float(cfg["entry_notional_usdt"])/float(lev),
                "approx_margin_loss_pct_at_stop":stop_bps*float(lev)/100.0,
                "theoretical_margin_exhaustion_move_bps":10_000.0/float(lev),
                "conservative_stop_buffer_pass":stop_bps<=(10_000.0/float(lev))*0.40} for lev in cfg["leverage_scenarios"]]}
        if risk["risk_budget_pass"] is False: reasons.append("STOP_LOSS_ESTIMATE_EXCEEDS_RISK_BUDGET")
        if any(not x["conservative_stop_buffer_pass"] for x in risk["leverage_scenarios"]):
            reasons.append("ATR_STOP_TOO_WIDE_FOR_ONE_OR_MORE_LEVERAGE_SCENARIOS")
        if not size["valid"]: reasons.extend(size["reasons"])
        if not cfg.get("fees_verified") or cfg.get("fee_taker_bps") is None: reasons.append("ACCOUNT_TAKER_FEE_UNVERIFIED")
        if not cfg.get("slippage_verified") or cfg.get("slippage_bps_per_side") is None: reasons.append("SLIPPAGE_MODEL_UNCALIBRATED")
        if candidate=="NONE": reasons.append("NO_MULTI_TIMEFRAME_ENTRY_TRIGGER")
        metrics={"last_price":last,"mark_price":mark,"mark_last_basis_bps":basis,"funding_rate":funding,
            "funding_interval_hours":costs["funding_interval_hours"],"book":book,"timeframes":tf,
            "long_checks":long_checks,"short_checks":short_checks,"long_score":long_score,"short_score":short_score,
            "candidate_direction":candidate,"size_check":size,"cost_model":costs,"risk_check":risk,
            "ticker_24h":snapshot.get("ticker_24h"),
            "levels_1h":{"previous_20h_high":h1["previous_20_high"],"previous_20h_low":h1["previous_20_low"]},
            "entry_notional_usdt":float(cfg["entry_notional_usdt"])}
    except (DataIntegrityError,KeyError,TypeError,ValueError) as exc:
        quality["valid"]=False
        quality["reasons"].append(str(exc))
        reasons.append("MARKET_DATA_QUALITY_FAIL")
    if quality["valid"]: quality["reasons"]=list(dict.fromkeys(reasons))
    reasons.extend(["RESEARCH_ONLY_NO_ORDER_ROUTE","LIVE_TRADING_DISABLED"])
    return {"report_version":1,"engine_version":ENGINE_VERSION,"created_at_utc":_utc_now(),
        "symbol":symbol,"status":"NO_TRADE","mode":"READ_ONLY_RESEARCH","market_data_quality":quality,
        "decision_reasons":list(dict.fromkeys(reasons)),"metrics":metrics,
        "policy":{"entry_notional_usdt":cfg["entry_notional_usdt"],
            "interpretation":"USDT notional per entry; not initial margin",
            "analysis_margin_mode":"ISOLATED target only; account settings are not changed",
            "live_trading_enabled":False,"paper_execution_enabled":False}}


class EvidenceLedger:
    """Append-only analysis evidence store with semantic replay conflict detection."""
    def __init__(self, db_path: str):
        self.db_path=db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)),exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("""CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,payload_sha256 TEXT NOT NULL,payload_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,symbol TEXT NOT NULL,status TEXT NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_created ON analysis_runs(created_at_utc)")

    def _connect(self) -> sqlite3.Connection:
        conn=sqlite3.connect(self.db_path,timeout=10.0)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def record(self, report: Dict[str,Any]) -> Dict[str,Any]:
        body=_canonical_json(report); digest=_sha256(body); run_id=digest[:32]
        created=str(report.get("created_at_utc") or _utc_now())
        symbol=str(report.get("symbol") or "UNKNOWN"); status=str(report.get("status") or "UNKNOWN")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row=conn.execute("SELECT payload_sha256,payload_json FROM analysis_runs WHERE run_id=?",(run_id,)).fetchone()
            if row:
                if row[0]!=digest or row[1]!=body: raise LedgerConflict("run identity exists with different report payload")
                return {"run_id":run_id,"payload_sha256":digest,"inserted":False,"replay":True}
            conn.execute("INSERT INTO analysis_runs VALUES(?,?,?,?,?,?)",(run_id,digest,body,created,symbol,status))
        return {"run_id":run_id,"payload_sha256":digest,"inserted":True,"replay":False}

    def status(self) -> Dict[str,Any]:
        with self._connect() as conn:
            integrity=conn.execute("PRAGMA integrity_check").fetchone()[0]
            count=conn.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0]
            latest=conn.execute("SELECT created_at_utc,symbol,status,run_id FROM analysis_runs ORDER BY created_at_utc DESC LIMIT 1").fetchone()
        return {"integrity":integrity,"analysis_runs":count,"latest":latest}


def make_report(snapshot: Dict[str,Any], config: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    report=analyze_snapshot(snapshot,config)
    text=_canonical_json(snapshot)
    report["evidence"]={"source":snapshot.get("source"),"collected_start_ms":snapshot.get("collected_start_ms"),
        "collected_end_ms":snapshot.get("collected_end_ms"),"snapshot_sha256":_sha256(text),
        "optional_data_errors":snapshot.get("optional_errors",{})}
    return report


def main(argv: Optional[Sequence[str]]=None) -> int:
    parser=argparse.ArgumentParser(description="Hustler Capital Engine MAGICUSDT read-only research")
    parser.add_argument("command",choices=["scan","analyze","status","self-test"])
    parser.add_argument("--symbol",default=DEFAULT_CONFIG["symbol"])
    parser.add_argument("--notional",type=float,default=DEFAULT_CONFIG["entry_notional_usdt"])
    parser.add_argument("--config",help="optional JSON config; must not contain credentials")
    parser.add_argument("--snapshot",help="saved Binance public snapshot JSON for offline analysis")
    parser.add_argument("--db",default=os.path.join(os.path.dirname(__file__),"state","magic-research.sqlite3"))
    parser.add_argument("--json",action="store_true")
    args=parser.parse_args(argv)
    cfg=dict(DEFAULT_CONFIG); cfg["symbol"]=args.symbol.upper(); cfg["entry_notional_usdt"]=args.notional
    if args.config:
        with open(args.config,"r",encoding="utf-8") as handle: user_cfg=json.load(handle)
        forbidden=("secret","api_key","apikey","password","token")
        if any(any(word in str(k).lower() for word in forbidden) for k in user_cfg):
            parser.error("credential-bearing config keys are forbidden")
        cfg.update(user_cfg)
    if args.command=="self-test":
        import unittest
        from test_magic_engine import MagicEngineTests
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MagicEngineTests))
        return 0 if result.wasSuccessful() else 1
    if args.command=="status":
        result=EvidenceLedger(args.db).status()
    else:
        if args.snapshot:
            with open(args.snapshot,"r",encoding="utf-8") as handle: snapshot=json.load(handle)
        elif args.command=="scan":
            snapshot=BinanceUSDMClient().collect(cfg["symbol"],int(cfg["history_limit"]))
            snapshot_path=os.path.join(os.path.dirname(os.path.abspath(args.db)),"latest_magic_snapshot.json")
            os.makedirs(os.path.dirname(snapshot_path),exist_ok=True)
            with open(snapshot_path,"w",encoding="utf-8") as handle:
                json.dump(snapshot,handle,sort_keys=True,indent=2,allow_nan=False)
        else:
            parser.error("analyze requires --snapshot PATH")
        result=make_report(snapshot,cfg)
        result["ledger"]=EvidenceLedger(args.db).record(result)
    print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
