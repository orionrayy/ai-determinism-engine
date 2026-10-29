#!/usr/bin/env python3
"""SOL Sentinel: read-only Reku market scanner and paper trader.

No authenticated API, live order, wallet signing, or transfer path is implemented.
Python standard library only; designed for Termux.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

API_BASE = "https://api.reku.id/v2"
USER_AGENT = "SOL-Sentinel/0.1 (+read-only market research)"
MAX_RESPONSE_BYTES = 2_000_000
DEFAULT_CONFIG = {
    "initial_capital_idr": 300_000.0,
    "max_position_pct": 0.50,
    "risk_per_trade_pct": 0.005,
    "max_daily_loss_pct": 0.03,
    "max_spread_bps": 35.0,
    "min_relative_volume": 1.2,
    "min_net_edge_pct": 0.004,
    "round_trip_slippage_pct": 0.002,
    "buy_fee_pct": 0.0010555,
    "sell_fee_pct": 0.0031555,
    "stop_atr_multiple": 1.5,
    "minimum_stop_pct": 0.008,
    "reward_risk_ratio": 2.0,
    "poll_interval_seconds": 30,
    "max_candle_age_seconds": 600,
}


class DataError(RuntimeError):
    """Market data is missing, stale, or invalid."""


def now_ts() -> int:
    return int(time.time())


def utc_iso(ts: int | None = None) -> str:
    return datetime.fromtimestamp(ts if ts is not None else now_ts(), timezone.utc).isoformat()


def fetch_json(path: str, timeout: float = 8.0) -> Any:
    """Fetch fixed-host public JSON. Never accepts an arbitrary host/path."""
    if not path.startswith("/") or "://" in path or path.startswith("//"):
        raise ValueError("Only a relative path on the configured HTTPS host is allowed")
    url = API_BASE + path
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise DataError("API response exceeded size limit")
            if response.status != 200:
                raise DataError(f"HTTP status {response.status}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataError(f"request failed: {type(exc).__name__}: {exc}") from exc
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataError("API returned invalid JSON") from exc


def as_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        nested = value.get("result")
        if isinstance(nested, list):
            value = nested
        else:
            return [value]
    if not isinstance(value, list):
        raise DataError("Expected JSON object/list")
    return [x for x in value if isinstance(x, dict)]


def get_number(obj: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = obj.get(name)
        try:
            if value is not None:
                n = float(value)
                if math.isfinite(n) and n >= 0:
                    return n
        except (TypeError, ValueError):
            continue
    return None


def get_signed_number(obj: dict[str, Any], *names: str) -> float | None:
    """Parse a finite signed number such as a negative 24h change percentage."""
    for name in names:
        value = obj.get(name)
        try:
            if value is not None:
                n = float(value)
                if math.isfinite(n):
                    return n
        except (TypeError, ValueError):
            continue
    return None


@dataclass(frozen=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class MarketSnapshot:
    sampled_at: int
    source: str
    symbol: str
    coin_id: int | None
    last_idr: float | None
    bid_idr: float | None
    ask_idr: float | None
    spread_bps: float | None
    change_pct: float | None
    volume_idr: float | None
    candles: list[Candle]
    depth_bid_idr: float | None = None
    depth_ask_idr: float | None = None
    chart_age_seconds: int | None = None
    errors: list[str] | None = None

    def public_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["candles"] = [asdict(c) for c in self.candles]
        return out


def find_matching(rows: Iterable[dict[str, Any]], keys: tuple[str, ...], expected: str) -> dict[str, Any] | None:
    for row in rows:
        if any(str(row.get(key, "")).strip().upper() == expected for key in keys):
            return row
    return None


def parse_candles(data: Any) -> list[Candle]:
    """Parse legacy Reku candle rows: [ts, open, close, low, high, coinVolume, idrVolume]."""
    if isinstance(data, dict) and isinstance(data.get("result"), list):
        data = data["result"]
    if not isinstance(data, list):
        raise DataError("Chart is not a list")
    candles: list[Candle] = []
    for row in data:
        try:
            if isinstance(row, dict):
                ts = int(row.get("timestamp", row.get("time", row.get("t"))))
                op = float(row.get("open", row.get("o")))
                close = float(row.get("close", row.get("c")))
                low = float(row.get("low", row.get("l")))
                high = float(row.get("high", row.get("h")))
                volume = float(row.get("volume", row.get("v", 0)))
            elif isinstance(row, (list, tuple)) and len(row) >= 6:
                ts, op, close, low, high, volume = int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
            else:
                continue
            values = (op, close, low, high, volume)
            if ts > 10_000_000_000:
                ts //= 1000
            if ts <= 0 or not all(math.isfinite(x) for x in values) or min(op, close, low, high) <= 0 or volume < 0:
                continue
            if low > min(op, close) or high < max(op, close) or low > high:
                continue
            candles.append(Candle(ts, op, high, low, close, volume))
        except (TypeError, ValueError, OverflowError):
            continue
    candles.sort(key=lambda x: x.ts)
    dedup: dict[int, Candle] = {c.ts: c for c in candles}
    return list(dedup.values())


class RekuPublicClient:
    """Public, read-only adapter. Legacy API schema is guarded defensively."""

    def fetch(self) -> MarketSnapshot:
        sampled = now_ts()
        errors: list[str] = []
        try:
            coins = as_rows(fetch_json("/coins"))
            coin = find_matching(coins, ("accountcode", "code", "cd"), "SOL")
            price_rows = as_rows(fetch_json("/price"))
            price = find_matching(price_rows, ("cd", "code", "accountcode"), "SOL")
            if price is None:
                raise DataError("SOL not found in /price response")
            coin_id_value = (coin or {}).get("id", price.get("id"))
            coin_id = int(coin_id_value) if coin_id_value is not None else None
            bidask_rows = as_rows(fetch_json("/bidask"))
            bidask = find_matching(bidask_rows, ("code", "cd", "accountcode"), "SOL")
            bid = get_number(bidask or {}, "bid")
            ask = get_number(bidask or {}, "ask")
            last = get_number(price, "c", "last", "price")
            if last is None or bid is None or ask is None or bid <= 0 or ask < bid:
                raise DataError("SOL last/bid/ask missing or invalid")
            mid = (bid + ask) / 2.0
            spread_bps = ((ask - bid) / mid) * 10_000 if mid else None
            candles: list[Candle] = []
            chart_age: int | None = None
            depth_bid = depth_ask = None
            if coin_id is not None:
                try:
                    raw_chart = fetch_json(f"/chart?t=2&f=5&id={coin_id}")
                    candles = parse_candles(raw_chart)
                    if candles:
                        chart_age = max(0, sampled - candles[-1].ts)
                except (DataError, ValueError) as exc:
                    errors.append(f"chart: {exc}")
                try:
                    book = fetch_json(f"/orderbook?id={coin_id}")
                    if isinstance(book, dict):
                        depth_bid = sum(max(0.0, float(row[0])) for row in book.get("b", [])[:5] if isinstance(row, list) and row)
                        depth_ask = sum(max(0.0, float(row[0])) for row in book.get("s", [])[:5] if isinstance(row, list) and row)
                except (DataError, TypeError, ValueError, IndexError) as exc:
                    errors.append(f"orderbook: {exc}")
            else:
                errors.append("coin identifier unavailable; chart/depth skipped")
            return MarketSnapshot(
                sampled_at=sampled, source="reku_public_legacy", symbol="SOL/IDR",
                coin_id=coin_id, last_idr=last, bid_idr=bid, ask_idr=ask,
                spread_bps=spread_bps, change_pct=get_signed_number(price, "cp", "change_pct"),
                volume_idr=get_number(price, "v", "volume_idr"), candles=candles,
                depth_bid_idr=depth_bid, depth_ask_idr=depth_ask,
                chart_age_seconds=chart_age, errors=errors,
            )
        except (DataError, ValueError, TypeError) as exc:
            raise DataError(str(exc)) from exc


def ema(values: list[float], period: int) -> list[float]:
    if period <= 0:
        raise ValueError("period must be positive")
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * float(value) + (1.0 - alpha) * result[-1])
    return result


def rsi(values: list[float], period: int = 14) -> float | None:
    if period <= 0 or len(values) < period + 1:
        return None
    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    window = changes[-period:]
    gains = sum(max(0.0, x) for x in window) / period
    losses = sum(max(0.0, -x) for x in window) / period
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def atr(candles: list[Candle], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    trs: list[float] = []
    for previous, current in zip(candles[-period-1:-1], candles[-period:]):
        trs.append(max(current.high-current.low, abs(current.high-previous.close), abs(current.low-previous.close)))
    return sum(trs) / len(trs) if trs else None


def completed_candles(candles: list[Candle], sampled_at: int, interval_seconds: int = 300) -> list[Candle]:
    # Legacy chart timestamps are treated as candle-open timestamps conservatively.
    return [c for c in candles if c.ts + interval_seconds <= sampled_at]


def features(candles: list[Candle]) -> dict[str, float | bool | None]:
    closes = [c.close for c in candles]
    if len(candles) < 30:
        return {"ready": False, "count": float(len(candles))}
    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    curr_rsi = rsi(closes, 14)
    curr_atr = atr(candles, 14)
    previous_volume = [c.volume for c in candles[-21:-1]]
    avg_volume = sum(previous_volume) / len(previous_volume) if previous_volume else 0.0
    rel_volume = candles[-1].volume / avg_volume if avg_volume > 0 else 0.0
    prior_highs = [c.high for c in candles[-13:-1]]
    breakout = bool(prior_highs and closes[-1] > max(prior_highs))
    return {
        "ready": True, "count": float(len(candles)), "close": closes[-1],
        "ema9": e9[-1], "ema21": e21[-1], "rsi14": curr_rsi,
        "atr14": curr_atr, "atr_pct": (curr_atr / closes[-1]) if curr_atr and closes[-1] else None,
        "relative_volume": rel_volume, "breakout": breakout,
        "trend_up": e9[-1] > e21[-1], "price_above_ema9": closes[-1] > e9[-1],
    }


def analyze(snapshot: MarketSnapshot, config: dict[str, Any], sampled_at: int | None = None) -> dict[str, Any]:
    sampled = sampled_at if sampled_at is not None else snapshot.sampled_at
    candles = completed_candles(snapshot.candles, sampled)
    f = features(candles)
    blockers: list[str] = []
    if snapshot.last_idr is None or snapshot.bid_idr is None or snapshot.ask_idr is None:
        blockers.append("missing current price/bid/ask")
    if not candles:
        blockers.append("no completed chart candles")
    elif sampled - candles[-1].ts > int(config["max_candle_age_seconds"]) + 300:
        blockers.append("chart candle is stale")
    if not f.get("ready"):
        blockers.append("insufficient candles (need 30 completed)")
    if snapshot.spread_bps is None or snapshot.spread_bps > float(config["max_spread_bps"]):
        blockers.append("spread guard")
    if snapshot.candles and snapshot.chart_age_seconds is None:
        blockers.append("chart timestamp unavailable")
    if not blockers:
        if not f["trend_up"]:
            blockers.append("EMA trend not bullish")
        if not f["price_above_ema9"]:
            blockers.append("close below EMA9")
        if f["rsi14"] is None or not 50 <= float(f["rsi14"]) <= 68:
            blockers.append("RSI outside 50-68 window")
        if float(f["relative_volume"] or 0) < float(config["min_relative_volume"]):
            blockers.append("relative volume below threshold")
        if not f["breakout"]:
            blockers.append("no confirmed 12-candle breakout")
    fee_round_trip = float(config["buy_fee_pct"]) + float(config["sell_fee_pct"]) + float(config["round_trip_slippage_pct"])
    stop_pct = max(float(f.get("atr_pct") or 0) * float(config["stop_atr_multiple"]), float(config["minimum_stop_pct"]))
    target_pct = stop_pct * float(config["reward_risk_ratio"])
    expected_net_edge = target_pct - fee_round_trip
    if expected_net_edge < float(config["min_net_edge_pct"]):
        blockers.append("expected net edge does not cover modeled cost plus buffer")
    return {
        "status": "BUY_SETUP" if not blockers else ("NO_DATA" if any(x in " ".join(blockers) for x in ("missing", "no completed", "stale", "insufficient", "timestamp")) else "WAIT"),
        "source": snapshot.source, "symbol": snapshot.symbol, "sampled_at": sampled,
        "price_idr": snapshot.last_idr, "bid_idr": snapshot.bid_idr, "ask_idr": snapshot.ask_idr,
        "spread_bps": snapshot.spread_bps, "chart_age_seconds": snapshot.chart_age_seconds,
        "features": f, "signal_candle_ts": candles[-1].ts if candles else None,
        "stop_pct": stop_pct, "target_pct": target_pct,
        "modeled_round_trip_cost_pct": fee_round_trip, "expected_net_edge_pct": expected_net_edge,
        "blockers": blockers,
    }


class Journal:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path)
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=3000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots(
          id INTEGER PRIMARY KEY, sampled_at INTEGER NOT NULL, source TEXT NOT NULL,
          status TEXT NOT NULL, price_idr REAL, bid_idr REAL, ask_idr REAL,
          spread_bps REAL, payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS paper_state(
          id INTEGER PRIMARY KEY CHECK(id=1), state_json TEXT NOT NULL, updated_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trades(
          id INTEGER PRIMARY KEY, opened_at INTEGER NOT NULL, closed_at INTEGER,
          status TEXT NOT NULL, entry_price REAL NOT NULL, exit_price REAL,
          quantity REAL NOT NULL, notional_idr REAL NOT NULL, entry_fee REAL NOT NULL,
          exit_fee REAL, stop_price REAL NOT NULL, target_price REAL NOT NULL,
          pnl_net_idr REAL, exit_reason TEXT, entry_candle_ts INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events(
          id INTEGER PRIMARY KEY, ts INTEGER NOT NULL, kind TEXT NOT NULL, message TEXT NOT NULL, payload_json TEXT NOT NULL
        );
        """)
        self.db.commit()

    def event(self, kind: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self.db.execute("INSERT INTO events(ts,kind,message,payload_json) VALUES(?,?,?,?)",
                        (now_ts(), kind, message, json.dumps(payload or {}, sort_keys=True)))
        self.db.commit()

    def save_snapshot(self, snapshot: MarketSnapshot, report: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO snapshots(sampled_at,source,status,price_idr,bid_idr,ask_idr,spread_bps,payload_json) VALUES(?,?,?,?,?,?,?,?)",
            (snapshot.sampled_at, snapshot.source, report["status"], snapshot.last_idr,
             snapshot.bid_idr, snapshot.ask_idr, snapshot.spread_bps,
             json.dumps({"snapshot": snapshot.public_dict(), "analysis": report}, sort_keys=True)))
        self.db.commit()

    def get_state(self, config: dict[str, Any]) -> dict[str, Any]:
        row = self.db.execute("SELECT state_json FROM paper_state WHERE id=1").fetchone()
        if row:
            state = json.loads(row["state_json"])
        else:
            capital = float(config["initial_capital_idr"])
            state = {"cash_idr": capital, "position": None, "last_entry_candle_ts": None,
                     "daily_date": datetime.now().date().isoformat(),
                     "day_start_equity_idr": capital, "daily_locked": False, "daily_realized_pnl_idr": 0.0}
            self.save_state(state)
        return state

    def save_state(self, state: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO paper_state(id,state_json,updated_at) VALUES(1,?,?) "
            "ON CONFLICT(id) DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at",
            (json.dumps(state, sort_keys=True), now_ts()))
        self.db.commit()

    def status(self, config: dict[str, Any]) -> dict[str, Any]:
        state = self.get_state(config)
        latest = self.db.execute("SELECT sampled_at,source,status,price_idr,spread_bps FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
        recent = self.db.execute("SELECT id,opened_at,closed_at,status,pnl_net_idr,exit_reason FROM trades ORDER BY id DESC LIMIT 20").fetchall()
        return {"database": str(self.path), "state": state, "latest_snapshot": dict(latest) if latest else None,
                "recent_trades": [dict(x) for x in recent]}

    def close(self) -> None:
        self.db.close()


class PaperTrader:
    def __init__(self, journal: Journal, config: dict[str, Any]):
        self.journal = journal
        self.config = config

    def equity(self, state: dict[str, Any], mark_bid: float | None = None) -> float:
        pos = state.get("position")
        return float(state["cash_idr"]) + (float(pos["quantity"]) * float(mark_bid or pos["entry_fill"]) if pos else 0.0)

    def update_day(self, state: dict[str, Any], equity: float) -> None:
        today = datetime.now().date().isoformat()
        if state.get("daily_date") != today:
            state["daily_date"] = today
            state["day_start_equity_idr"] = equity
            state["daily_realized_pnl_idr"] = 0.0
            state["daily_locked"] = False
        day_start = max(1.0, float(state.get("day_start_equity_idr", equity)))
        daily_pnl = float(state.get("daily_realized_pnl_idr", 0.0))
        pos = state.get("position")
        if pos and pos.get("mark_bid"):
            daily_pnl += float(pos["quantity"]) * (float(pos["mark_bid"]) - float(pos["entry_fill"]))
        if daily_pnl <= -day_start * float(self.config["max_daily_loss_pct"]):
            state["daily_locked"] = True

    def process(self, snapshot: MarketSnapshot, report: dict[str, Any]) -> dict[str, Any]:
        state = self.journal.get_state(self.config)
        bid = float(snapshot.bid_idr or 0.0)
        ask = float(snapshot.ask_idr or 0.0)
        equity = self.equity(state, bid if bid > 0 else None)
        self.update_day(state, equity)
        pos = state.get("position")
        if pos and bid > 0:
            pos["mark_bid"] = bid
            reason = None
            if bid <= float(pos["stop_price"]):
                reason = "stop_observed"
            elif bid >= float(pos["target_price"]):
                reason = "target_observed"
            if reason:
                sell_fee_rate = float(self.config["sell_fee_pct"])
                slippage = float(self.config["round_trip_slippage_pct"]) / 2.0
                exit_fill = bid * (1.0 - slippage)
                exit_fee = exit_fill * float(pos["quantity"]) * sell_fee_rate
                proceeds = exit_fill * float(pos["quantity"]) - exit_fee
                pnl = proceeds - (float(pos["entry_fill"]) * float(pos["quantity"]) + float(pos["entry_fee"]))
                state["cash_idr"] += proceeds
                state["daily_realized_pnl_idr"] = float(state.get("daily_realized_pnl_idr", 0.0)) + pnl
                self.journal.db.execute(
                    "UPDATE trades SET closed_at=?,status='closed',exit_price=?,exit_fee=?,pnl_net_idr=?,exit_reason=? WHERE id=?",
                    (now_ts(), exit_fill, exit_fee, pnl, reason, pos["trade_id"]))
                self.journal.event("paper_exit", reason, {"trade_id": pos["trade_id"], "pnl_net_idr": round(pnl, 2)})
                state["position"] = None
                self.update_day(state, self.equity(state))
                pos = None
        if pos is None and not state.get("daily_locked") and report.get("status") == "BUY_SETUP":
            f = report["features"]
            candle_ts = int(report.get("signal_candle_ts") or 0)
            if candle_ts and candle_ts != state.get("last_entry_candle_ts") and ask > bid > 0:
                equity = self.equity(state)
                risk_budget = equity * float(self.config["risk_per_trade_pct"])
                stop_pct = float(report["stop_pct"])
                round_trip_cost_pct = (float(self.config["buy_fee_pct"]) + float(self.config["sell_fee_pct"])
                                       + float(self.config["round_trip_slippage_pct"]))
                risk_per_idr = ask * (stop_pct + round_trip_cost_pct)
                qty_by_risk = risk_budget / risk_per_idr if risk_per_idr > 0 else 0.0
                max_notional = equity * float(self.config["max_position_pct"])
                qty_by_notional = max_notional / ask
                qty = min(qty_by_risk, qty_by_notional)
                slippage = float(self.config["round_trip_slippage_pct"]) / 2.0
                entry_fill = ask * (1.0 + slippage)
                entry_fee = entry_fill * qty * float(self.config["buy_fee_pct"])
                required_cash = entry_fill * qty + entry_fee
                if required_cash > float(state["cash_idr"]) and required_cash > 0:
                    qty *= max(0.0, float(state["cash_idr"]) / required_cash)
                    entry_fee = entry_fill * qty * float(self.config["buy_fee_pct"])
                    required_cash = entry_fill * qty + entry_fee
                if qty > 0 and required_cash <= float(state["cash_idr"]):
                    stop = entry_fill * (1.0 - stop_pct)
                    target = entry_fill * (1.0 + float(report["target_pct"]))
                    state["cash_idr"] -= required_cash
                    cursor = self.journal.db.execute(
                        "INSERT INTO trades(opened_at,status,entry_price,quantity,notional_idr,entry_fee,stop_price,target_price,entry_candle_ts) "
                        "VALUES(?,'open',?,?,?,?,?,?,?)",
                        (now_ts(), entry_fill, qty, entry_fill*qty, entry_fee, stop, target, candle_ts))
                    trade_id = cursor.lastrowid
                    state["position"] = {"trade_id": trade_id, "entry_fill": entry_fill, "quantity": qty,
                                         "entry_fee": entry_fee, "stop_price": stop, "target_price": target,
                                         "entry_candle_ts": candle_ts, "mark_bid": bid}
                    state["last_entry_candle_ts"] = candle_ts
                    self.journal.event("paper_entry", "BUY_SETUP accepted by paper risk sizing",
                                       {"trade_id": trade_id, "notional_idr": round(entry_fill*qty, 2),
                                        "stop_price": round(stop, 2), "target_price": round(target, 2)})
        state["position"] = state.get("position")
        self.update_day(state, self.equity(state, bid if bid > 0 else None))
        self.journal.save_state(state)
        pos = state.get("position")
        return {
            "mode": "paper_only", "cash_idr": round(float(state["cash_idr"]), 2),
            "equity_idr": round(self.equity(state, bid if bid > 0 else None), 2),
            "position_open": bool(pos), "position": pos,
            "daily_locked": bool(state.get("daily_locked")),
            "daily_realized_pnl_idr": round(float(state.get("daily_realized_pnl_idr", 0.0)), 2),
        }


def load_config(path: str | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path:
        user_config = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(user_config, dict):
            raise ValueError("config must be a JSON object")
        config.update(user_config)
    for key in ("max_position_pct", "risk_per_trade_pct", "max_daily_loss_pct",
                "min_relative_volume", "min_net_edge_pct", "round_trip_slippage_pct",
                "buy_fee_pct", "sell_fee_pct", "stop_atr_multiple", "minimum_stop_pct", "reward_risk_ratio"):
        if float(config[key]) < 0:
            raise ValueError(f"{key} must not be negative")
    if not 0 < float(config["max_position_pct"]) <= 1:
        raise ValueError("max_position_pct must be in (0, 1]")
    if not 0 < float(config["risk_per_trade_pct"]) <= 0.02:
        raise ValueError("risk_per_trade_pct must be in (0, 0.02]")
    if float(config["initial_capital_idr"]) <= 0:
        raise ValueError("initial_capital_idr must be positive")
    return config


def run_cycle(client: RekuPublicClient, journal: Journal, config: dict[str, Any]) -> dict[str, Any]:
    sampled = now_ts()
    try:
        snapshot = client.fetch()
        report = analyze(snapshot, config, sampled)
        paper = PaperTrader(journal, config).process(snapshot, report)
        journal.save_snapshot(snapshot, report)
        return {"ok": True, "timestamp": utc_iso(sampled), "analysis": report, "paper": paper,
                "warnings": snapshot.errors or []}
    except DataError as exc:
        journal.event("data_error", str(exc))
        return {"ok": False, "timestamp": utc_iso(sampled), "mode": "paper_only",
                "status": "NO_DATA", "error": str(exc), "action": "no new paper entry; retry later"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SOL Sentinel read-only scanner and paper trader")
    parser.add_argument("--config", help="optional JSON config")
    parser.add_argument("--db", default="data/sol_sentinel.sqlite3", help="local SQLite path")
    parser.add_argument("--once", action="store_true", help="run one market-data cycle")
    parser.add_argument("--doctor", action="store_true", help="run one read-only API/shape diagnostic")
    parser.add_argument("--status", action="store_true", help="print local journal status without network")
    parser.add_argument("--export-report", metavar="PATH", help="write a safe aggregate report JSON")
    parser.add_argument("--interval", type=int, default=None, help="poll seconds (minimum 10)")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        journal = Journal(args.db)
        if args.status:
            print(json.dumps(journal.status(config), indent=2, sort_keys=True))
            journal.close()
            return 0
        if args.export_report:
            status = journal.status(config)
            trades = journal.db.execute("SELECT COUNT(*) AS n, SUM(CASE WHEN status='closed' THEN 1 ELSE 0 END) AS closed, "
                                        "SUM(CASE WHEN pnl_net_idr > 0 THEN 1 ELSE 0 END) AS wins, "
                                        "SUM(CASE WHEN pnl_net_idr <= 0 THEN 1 ELSE 0 END) AS losses, "
                                        "COALESCE(SUM(pnl_net_idr),0) AS net_pnl FROM trades").fetchone()
            exported = {"generated_at": utc_iso(), "mode": "paper_only", "status": status,
                        "trade_stats": dict(trades) if trades else {}}
            target = Path(args.export_report)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(exported, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps({"ok": True, "report_path": str(target)}))
            journal.close()
            return 0
        interval = args.interval if args.interval is not None else int(config["poll_interval_seconds"])
        if interval < 10:
            parser.error("--interval must be at least 10 seconds")
        client = RekuPublicClient()
        if args.doctor:
            try:
                snap = client.fetch()
                completed = completed_candles(snap.candles, snap.sampled_at)
                report = {"ok": True, "source": snap.source, "symbol": snap.symbol,
                          "coin_id": snap.coin_id, "has_price": snap.last_idr is not None,
                          "has_bid_ask": snap.bid_idr is not None and snap.ask_idr is not None,
                          "raw_candle_count": len(snap.candles),
                          "completed_candle_count": len(completed),
                          "signal_data_ready": len(completed) >= 30 and snap.chart_age_seconds is not None,
                          "chart_age_seconds": snap.chart_age_seconds,
                          "spread_bps": snap.spread_bps, "warnings": snap.errors or [],
                          "sampled_at": utc_iso(snap.sampled_at),
                          "note": "Public market-data check only; never an order or profitability guarantee."}
                print(json.dumps(report, sort_keys=True))
                journal.close()
                return 0 if report["has_price"] and report["has_bid_ask"] else 2
            except DataError as exc:
                print(json.dumps({"ok": False, "status": "NO_DATA", "error": str(exc)}, sort_keys=True))
                journal.close()
                return 2
        while True:
            result = run_cycle(client, journal, config)
            print(json.dumps(result, sort_keys=True), flush=True)
            if args.once:
                break
            time.sleep(interval)
        journal.close()
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
