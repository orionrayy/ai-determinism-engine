#!/usr/bin/env python3
"""Persistent, read-only-market-data SOL/IDR forward-paper orchestrator.

This module is intentionally paper-only. It reads public INDODAX market data,
runs the frozen strategy, supervises simulated risk, and records all state in
one durable SQLite database shared with the existing depth execution ledger
and microstructure collector. It has no authenticated order, wallet, signer,
deposit, withdrawal, or transfer client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Callable, Sequence

from depth_execution import (
    DepthSnapshot,
    ExecutionError,
    OrderRequest,
    OrderType,
    Side as ExecutionSide,
    SQLitePaperExecutionLedger,
    TimeInForce,
)
from execution_adapters import (
    ExecutionMode,
    ExecutionPolicy,
    IndodaxPublicMarketDataAdapter,
    MarketCandle,
    MarketSpec,
)
from indodax_microstructure import (
    DepthClient,
    DepthStore,
    MicrostructureError,
    Snapshot as IndodaxDepthSnapshot,
)
from paper_engine import (
    BacktestConfig,
    Bar,
    aggregate_bars,
    atr_at,
    ema_series,
    rsi_at,
    signal_at,
)

SCHEMA_VERSION = 1
CONFIG_VERSION = "sol-sentinel-forward-paper-v1.0.0"
STRATEGY_VERSION = "sol-sentinel-mtf-breakout-v1"
UTC = timezone.utc
ONE_MINUTE_MS = 60_000


class ForwardPaperError(RuntimeError):
    """A fail-closed forward-paper pipeline error."""


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ForwardPaperError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def decimal(value: Any, name: str, *, allow_zero: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ForwardPaperError(f"{name} must be a finite Decimal")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception:
        raise ForwardPaperError(f"{name} must be a finite Decimal") from None
    if not result.is_finite() or (result < 0 if allow_zero else result <= 0):
        raise ForwardPaperError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return result


def floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if value <= 0 or step <= 0:
        return Decimal("0")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ForwardPaperConfig:
    initial_cash_idr: Decimal = Decimal("300000")
    history_minutes: int = 600
    fee_buy_bps: Decimal = Decimal("20.555")
    fee_sell_bps: Decimal = Decimal("41.555")
    mode_minimum_notional_idr: Decimal = Decimal("25000")
    minimum_net_edge_bps: Decimal = Decimal("25")
    slippage_assumption_bps_per_side: Decimal = Decimal("10")
    risk_fraction: Decimal = Decimal("0.005")
    max_exposure_fraction: Decimal = Decimal("0.50")
    daily_loss_limit_fraction: Decimal = Decimal("0.03")
    max_drawdown_fraction: Decimal = Decimal("0.10")
    warning_drawdown_fraction: Decimal = Decimal("0.02")
    max_spread_bps: Decimal = Decimal("50")
    max_depth_age_seconds: Decimal = Decimal("10")
    max_candle_age_seconds: Decimal = Decimal("120")
    max_exchange_timestamp_age_seconds: Decimal = Decimal("30")
    max_open_orders: int = 1
    max_daily_entry_orders: int = 10
    stop_atr_multiple: Decimal = Decimal("1.0")
    target_atr_multiple: Decimal = Decimal("1.5")
    maximum_hold_minutes: int = 30
    order_ttl_minutes: int = 15
    minimum_liquidity_coverage: Decimal = Decimal("1.25")

    def validate(self) -> None:
        if self.initial_cash_idr <= 0 or not self.initial_cash_idr.is_finite():
            raise ForwardPaperError("initial_cash_idr must be finite and positive")
        if self.history_minutes < 450 or self.history_minutes > 1440 or self.history_minutes % 15:
            raise ForwardPaperError("history_minutes must be 450..1440 and divisible by 15")
        for name in (
            "fee_buy_bps", "fee_sell_bps", "mode_minimum_notional_idr",
            "minimum_net_edge_bps", "slippage_assumption_bps_per_side",
            "risk_fraction", "max_exposure_fraction", "daily_loss_limit_fraction",
            "max_drawdown_fraction", "warning_drawdown_fraction", "max_spread_bps",
            "max_depth_age_seconds", "max_candle_age_seconds",
            "max_exchange_timestamp_age_seconds", "stop_atr_multiple",
            "target_atr_multiple", "minimum_liquidity_coverage",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ForwardPaperError(f"{name} must be a finite non-negative Decimal")
        if self.risk_fraction <= 0 or self.risk_fraction >= 1:
            raise ForwardPaperError("risk_fraction must be between zero and one")
        if self.max_exposure_fraction <= 0 or self.max_exposure_fraction > 1:
            raise ForwardPaperError("max_exposure_fraction must be in (0,1]")
        if self.daily_loss_limit_fraction <= 0 or self.daily_loss_limit_fraction >= 1:
            raise ForwardPaperError("daily_loss_limit_fraction must be in (0,1)")
        if self.max_drawdown_fraction <= 0 or self.max_drawdown_fraction >= 1:
            raise ForwardPaperError("max_drawdown_fraction must be in (0,1)")
        if self.max_open_orders < 1 or self.max_daily_entry_orders < 1:
            raise ForwardPaperError("order limits must be positive integers")
        if self.maximum_hold_minutes < 1 or self.order_ttl_minutes < 1:
            raise ForwardPaperError("hold and order TTL must be positive")
        if self.max_depth_age_seconds <= 0 or self.max_candle_age_seconds <= 0:
            raise ForwardPaperError("freshness limits must be positive")

    def as_json(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in asdict(self).items()
        }

    @property
    def config_hash(self) -> str:
        return digest({"config_version": CONFIG_VERSION, "config": self.as_json(), "strategy": STRATEGY_VERSION})


class ForwardState:
    """Durable cycle, signal-intent, daily-risk, position, and activation state."""

    def __init__(self, database_path: str | Path, config: ForwardPaperConfig):
        self.path = str(Path(database_path).expanduser())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=15000")
        return db

    def _initialize(self) -> None:
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS fp_metadata(
              key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS fp_control(
              control_key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS fp_cycles(
              cycle_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, completed_at TEXT,
              candle_close_ms INTEGER NOT NULL, source_snapshot_id TEXT NOT NULL,
              collector_observation_id TEXT NOT NULL, status TEXT NOT NULL,
              result_json TEXT NOT NULL, error TEXT);
            CREATE INDEX IF NOT EXISTS fp_cycles_time ON fp_cycles(started_at);
            CREATE TABLE IF NOT EXISTS fp_signals(
              signal_id TEXT PRIMARY KEY, candle_close_ms INTEGER NOT NULL,
              side TEXT NOT NULL, decision TEXT NOT NULL, reason_json TEXT NOT NULL,
              intent_json TEXT, order_id TEXT, created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS fp_signals_time ON fp_signals(created_at);
            CREATE TABLE IF NOT EXISTS fp_daily_risk(
              trading_date TEXT PRIMARY KEY, opening_equity_idr TEXT NOT NULL,
              peak_equity_idr TEXT NOT NULL, risk_state TEXT NOT NULL,
              daily_loss_fraction TEXT NOT NULL, drawdown_fraction TEXT NOT NULL,
              updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS fp_position(
              position_key INTEGER PRIMARY KEY CHECK(position_key=1),
              entry_signal_id TEXT NOT NULL, entry_order_id TEXT NOT NULL,
              entry_candle_close_ms INTEGER NOT NULL, atr_idr TEXT NOT NULL,
              entry_price_idr TEXT, stop_price_idr TEXT, target_price_idr TEXT,
              status TEXT NOT NULL, updated_at TEXT NOT NULL);
            """)
            row = db.execute("SELECT value FROM fp_metadata WHERE key='schema_version'").fetchone()
            if row and int(row["value"]) != SCHEMA_VERSION:
                raise ForwardPaperError("unsupported forward-paper state schema")
            db.execute("INSERT OR IGNORE INTO fp_metadata VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
            config_row = db.execute("SELECT value FROM fp_metadata WHERE key='config_hash'").fetchone()
            if config_row and config_row["value"] != self.config.config_hash:
                raise ForwardPaperError(
                    "persisted forward-paper configuration differs from current configuration; "
                    "refusing an implicit portfolio/risk reset"
                )
            db.execute("INSERT OR IGNORE INTO fp_metadata VALUES('config_hash',?)", (self.config.config_hash,))
            db.execute("INSERT OR IGNORE INTO fp_metadata VALUES('config_version',?)", (CONFIG_VERSION,))
            db.execute(
                "INSERT OR IGNORE INTO fp_control(control_key,value,updated_at) VALUES('operating_state','ACTIVE',?)",
                (iso_utc(utc_now()),),
            )
            db.execute(
                "INSERT OR IGNORE INTO fp_control(control_key,value,updated_at) VALUES('halt_reason','',?)",
                (iso_utc(utc_now()),),
            )

    def control(self) -> dict[str, str]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT control_key,value FROM fp_control").fetchall()
        return {row["control_key"]: row["value"] for row in rows}

    def set_control(self, state: str, reason: str = "") -> None:
        if state not in ("ACTIVE", "HALTED", "MANUAL_REVIEW_REQUIRED"):
            raise ForwardPaperError("invalid operating state")
        now = iso_utc(utc_now())
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO fp_control VALUES('operating_state',?,?) "
                "ON CONFLICT(control_key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                (state, now),
            )
            db.execute(
                "INSERT INTO fp_control VALUES('halt_reason',?,?) "
                "ON CONFLICT(control_key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                (reason[:500], now),
            )

    def begin_cycle(
        self, cycle_id: str, candle_close_ms: int, snapshot_id: str, collector_id: str
    ) -> bool:
        now = iso_utc(utc_now())
        with closing(self._connect()) as db, db:
            cur = db.execute(
                "INSERT OR IGNORE INTO fp_cycles VALUES(?,?,?,?,?,?,?,?,?)",
                (cycle_id, now, None, candle_close_ms, snapshot_id, collector_id, "STARTED", "{}", None),
            )
            return cur.rowcount == 1

    def finish_cycle(self, cycle_id: str, status: str, result: dict[str, Any], error: str | None = None) -> None:
        now = iso_utc(utc_now())
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE fp_cycles SET completed_at=?,status=?,result_json=?,error=? WHERE cycle_id=?",
                (now, status, canonical_json(result), error[:1000] if error else None, cycle_id),
            )

    def recover_started_cycles(self) -> int:
        now = iso_utc(utc_now())
        with closing(self._connect()) as db, db:
            cur = db.execute(
                "UPDATE fp_cycles SET completed_at=?,status='RECOVERED',error='incomplete cycle recovered after ledger reconciliation' "
                "WHERE status='STARTED' AND completed_at IS NULL",
                (now,),
            )
            return cur.rowcount

    def record_signal(
        self, signal_id: str, candle_close_ms: int, side: str, decision: str,
        reasons: Sequence[str], intent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = iso_utc(utc_now())
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "INSERT OR IGNORE INTO fp_signals VALUES(?,?,?,?,?,?,?,?,?)",
                    (signal_id, candle_close_ms, side, decision, canonical_json(list(reasons)),
                     canonical_json(intent) if intent is not None else None, None, now, now),
                )
                row = db.execute("SELECT * FROM fp_signals WHERE signal_id=?", (signal_id,)).fetchone()
                if row is None:
                    raise ForwardPaperError("signal record failed to persist")
                db.execute("COMMIT")
                return dict(row)
            except Exception:
                db.execute("ROLLBACK")
                raise

    def get_signal(self, signal_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM fp_signals WHERE signal_id=?", (signal_id,)).fetchone()
            return dict(row) if row else None

    def pending_intents(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT * FROM fp_signals WHERE decision='ORDER_INTENT' AND order_id IS NULL "
                "ORDER BY created_at,signal_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def attach_order(self, signal_id: str, order_id: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE fp_signals SET order_id=?,updated_at=? WHERE signal_id=?",
                (order_id, iso_utc(utc_now()), signal_id),
            )

    def count_daily_entry_intents(self, trading_date: str) -> int:
        with closing(self._connect()) as db:
            return int(db.execute(
                "SELECT COUNT(*) FROM fp_signals WHERE side='BUY' AND decision='ORDER_INTENT' "
                "AND substr(created_at,1,10)=?",
                (trading_date,),
            ).fetchone()[0])

    def daily_risk(self, trading_date: str, marked_equity: Decimal) -> dict[str, Decimal | str]:
        now = iso_utc(utc_now())
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT * FROM fp_daily_risk WHERE trading_date=?", (trading_date,)
                ).fetchone()
                if row is None:
                    db.execute(
                        "INSERT INTO fp_daily_risk VALUES(?,?,?,?,?,?,?)",
                        (trading_date, str(marked_equity), str(marked_equity), "NORMAL", "0", "0", now),
                    )
                    result = {
                        "opening_equity_idr": marked_equity,
                        "peak_equity_idr": marked_equity,
                        "risk_state": "NORMAL",
                        "daily_loss_fraction": Decimal("0"),
                        "drawdown_fraction": Decimal("0"),
                    }
                else:
                    opening = decimal(row["opening_equity_idr"], "daily opening equity")
                    peak = max(decimal(row["peak_equity_idr"], "daily peak equity"), marked_equity)
                    daily_loss = max(Decimal("0"), (opening - marked_equity) / opening)
                    drawdown = max(Decimal("0"), (peak - marked_equity) / peak)
                    state = row["risk_state"]
                    db.execute(
                        "UPDATE fp_daily_risk SET peak_equity_idr=?,daily_loss_fraction=?,drawdown_fraction=?,updated_at=? "
                        "WHERE trading_date=?",
                        (str(peak), str(daily_loss), str(drawdown), now, trading_date),
                    )
                    result = {
                        "opening_equity_idr": opening,
                        "peak_equity_idr": peak,
                        "risk_state": state,
                        "daily_loss_fraction": daily_loss,
                        "drawdown_fraction": drawdown,
                    }
                db.execute("COMMIT")
                return result
            except Exception:
                db.execute("ROLLBACK")
                raise

    def position(self) -> dict[str, Any] | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM fp_position WHERE position_key=1").fetchone()
            return dict(row) if row else None

    def save_position(
        self, *, signal_id: str, order_id: str, candle_close_ms: int, atr_idr: Decimal,
        entry_price: Decimal | None, stop_price: Decimal | None, target_price: Decimal | None,
        status: str,
    ) -> None:
        now = iso_utc(utc_now())
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO fp_position VALUES(1,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(position_key) DO UPDATE SET entry_signal_id=excluded.entry_signal_id,"
                "entry_order_id=excluded.entry_order_id,entry_candle_close_ms=excluded.entry_candle_close_ms,"
                "atr_idr=excluded.atr_idr,entry_price_idr=excluded.entry_price_idr,"
                "stop_price_idr=excluded.stop_price_idr,target_price_idr=excluded.target_price_idr,"
                "status=excluded.status,updated_at=excluded.updated_at",
                (signal_id, order_id, candle_close_ms, str(atr_idr),
                 str(entry_price) if entry_price is not None else None,
                 str(stop_price) if stop_price is not None else None,
                 str(target_price) if target_price is not None else None, status, now),
            )

    def close_position(self) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE fp_position SET status='CLOSED',updated_at=? WHERE position_key=1",
                (iso_utc(utc_now()),),
            )

    def activation(self, *, commit_sha: str, first_cycle_id: str, portfolio: dict[str, Any]) -> None:
        now = iso_utc(utc_now())
        payload = {
            "activated_at": now,
            "commit_sha": commit_sha,
            "config_version": CONFIG_VERSION,
            "config_hash": self.config.config_hash,
            "first_cycle_id": first_cycle_id,
            "mode": "PAPER",
            "portfolio": portfolio,
            "live_execution_enabled": False,
        }
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO fp_metadata(key,value) VALUES('activation_record',?) "
                "ON CONFLICT(key) DO NOTHING",
                (canonical_json(payload),),
            )

    def summary(self) -> dict[str, Any]:
        with closing(self._connect()) as db:
            cycles = db.execute(
                "SELECT COUNT(*) total,SUM(status='COMPLETED') completed,SUM(status='ERROR') errors,"
                "SUM(status='RECOVERED') recovered FROM fp_cycles"
            ).fetchone()
            signals = db.execute(
                "SELECT COUNT(*) total,SUM(decision='NO_TRADE') no_trade,"
                "SUM(decision='RISK_REJECTED') risk_rejected,SUM(decision='ORDER_INTENT') order_intents "
                "FROM fp_signals"
            ).fetchone()
            activation = db.execute(
                "SELECT value FROM fp_metadata WHERE key='activation_record'"
            ).fetchone()
        return {
            "cycles": dict(cycles),
            "signals": dict(signals),
            "activation": json.loads(activation["value"]) if activation else None,
            "control": self.control(),
        }

    def checkpoint(self) -> None:
        with closing(self._connect()) as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")


@dataclass(frozen=True)
class RiskDecision:
    state: str
    allow_new_entries: bool
    reasons: tuple[str, ...]
    equity_idr: Decimal
    exposure_fraction: Decimal
    daily_loss_fraction: Decimal
    drawdown_fraction: Decimal


class RiskSupervisor:
    """Persistent paper-only portfolio risk supervisor and circuit breaker."""

    def __init__(self, state: ForwardState, config: ForwardPaperConfig):
        self.state = state
        self.config = config

    def evaluate(
        self, portfolio: dict[str, Decimal], snapshot: DepthSnapshot, open_orders: Sequence[dict[str, Any]],
        *, now: datetime,
    ) -> RiskDecision:
        bid = snapshot.bids[0].price
        equity = portfolio["cash_idr"] + portfolio["sol_quantity"] * bid
        if not equity.is_finite() or equity <= 0:
            self.state.set_control("MANUAL_REVIEW_REQUIRED", "non-positive or invalid marked portfolio equity")
            return RiskDecision("MANUAL_REVIEW_REQUIRED", False, ("invalid_marked_equity",), equity,
                                Decimal("1"), Decimal("1"), Decimal("1"))
        date_key = now.astimezone(UTC).date().isoformat()
        daily = self.state.daily_risk(date_key, equity)
        opening = decimal(daily["opening_equity_idr"], "daily opening equity")
        peak = decimal(daily["peak_equity_idr"], "daily peak equity")
        daily_loss = decimal(daily["daily_loss_fraction"], "daily loss", allow_zero=True)
        drawdown = decimal(daily["drawdown_fraction"], "drawdown", allow_zero=True)
        exposure = (portfolio["sol_quantity"] * bid / equity) if equity > 0 else Decimal("1")
        control = self.state.control()
        reasons: list[str] = []
        if daily_loss >= self.config.daily_loss_limit_fraction:
            self.state.set_control("HALTED", "daily_loss_limit_reached")
            reasons.append("daily_loss_limit_reached")
        if drawdown >= self.config.max_drawdown_fraction:
            self.state.set_control("HALTED", "maximum_drawdown_reached")
            reasons.append("maximum_drawdown_reached")
        if exposure > self.config.max_exposure_fraction:
            reasons.append("portfolio_exposure_above_limit")
        control = self.state.control()
        operating = control.get("operating_state", "ACTIVE")
        if operating == "MANUAL_REVIEW_REQUIRED":
            reasons.append(control.get("halt_reason") or "manual_review_required")
        elif operating == "HALTED":
            reasons.append(control.get("halt_reason") or "paper_halted")
        elif len(open_orders) >= self.config.max_open_orders:
            reasons.append("open_order_limit_reached")
        spread = snapshot.spread_bps
        if spread > self.config.max_spread_bps:
            reasons.append("spread_above_limit")
        age = Decimal(str((now.astimezone(UTC) - snapshot.observed_at).total_seconds()))
        if age < 0 or age > self.config.max_depth_age_seconds:
            reasons.append("depth_snapshot_stale")
        if drawdown >= self.config.warning_drawdown_fraction and operating == "ACTIVE":
            state = "WARNING"
        else:
            state = operating if operating != "ACTIVE" else "NORMAL"
        allow = (
            operating == "ACTIVE"
            and daily_loss < self.config.daily_loss_limit_fraction
            and drawdown < self.config.max_drawdown_fraction
            and exposure < self.config.max_exposure_fraction
            and len(open_orders) < self.config.max_open_orders
            and spread <= self.config.max_spread_bps
            and Decimal("0") <= age <= self.config.max_depth_age_seconds
        )
        return RiskDecision(state, allow, tuple(dict.fromkeys(reasons)), equity, exposure, daily_loss, drawdown)

    def cancel_entry_residuals(self, ledger: SQLitePaperExecutionLedger, *, at: datetime) -> list[str]:
        cancelled: list[str] = []
        for order in ledger.list_open_orders():
            if order["side"] == ExecutionSide.BUY.value:
                ledger.cancel_order(order["order_id"], at=at, reason="risk_circuit_breaker")
                cancelled.append(order["order_id"])
        return cancelled

    def size_buy(
        self, *, portfolio: dict[str, Decimal], market: MarketSpec, snapshot: DepthSnapshot,
        atr_idr: Decimal,
    ) -> tuple[Decimal, tuple[str, ...]]:
        reasons: list[str] = []
        equity = portfolio["cash_idr"] + portfolio["sol_quantity"] * snapshot.bids[0].price
        ask = snapshot.asks[0].price
        stop_distance = atr_idr * self.config.stop_atr_multiple
        if stop_distance <= 0:
            return Decimal("0"), ("invalid_atr_stop_distance",)
        total_cost_bps = (
            self.config.fee_buy_bps + self.config.fee_sell_bps
            + snapshot.spread_bps + self.config.slippage_assumption_bps_per_side * 2
        )
        loss_per_sol = stop_distance + ask * total_cost_bps / Decimal("10000")
        risk_budget = equity * self.config.risk_fraction
        risk_qty = risk_budget / loss_per_sol
        exposure_qty = equity * self.config.max_exposure_fraction / ask
        cash_qty = portfolio["cash_idr"] / (ask * (Decimal("1") + self.config.fee_buy_bps / Decimal("10000")))
        visible_ask_qty = sum((x.quantity for x in snapshot.asks), Decimal("0"))
        liquidity_qty = visible_ask_qty / self.config.minimum_liquidity_coverage
        quantity = floor_to_step(min(risk_qty, exposure_qty, cash_qty, liquidity_qty), market.quantity_step)
        notional = quantity * ask
        if quantity < market.min_quantity_base:
            reasons.append("minimum_base_quantity_not_met")
        if notional < max(market.min_notional_quote, self.config.mode_minimum_notional_idr):
            reasons.append("minimum_order_notional_not_met")
        if portfolio["sol_quantity"] > 0:
            reasons.append("existing_position_open")
        if not snapshot.asks or visible_ask_qty <= 0:
            reasons.append("ask_liquidity_unavailable")
        if reasons:
            return Decimal("0"), tuple(reasons)
        return quantity, ()


def _build_features(bars: Sequence[Bar]) -> dict[str, list[float | None]]:
    closes = [bar.close for bar in bars]
    ema9, ema21 = ema_series(closes, 9), ema_series(closes, 21)
    rsi_values: list[float | None] = []
    atr_values: list[float | None] = []
    rv_values: list[float | None] = []
    prior_highs: list[float | None] = []
    for index, bar in enumerate(bars):
        rsi_values.append(rsi_at(closes, index))
        atr_values.append(atr_at(bars, index))
        if index >= 20:
            baseline = sum(x.volume for x in bars[index - 20:index]) / 20.0
            rv_values.append(bar.volume / baseline if baseline > 0 else None)
            prior_highs.append(max(x.high for x in bars[index - 20:index]))
        else:
            rv_values.append(None)
            prior_highs.append(None)
    return {
        "ema9": list(ema9), "ema21": list(ema21), "rsi14": rsi_values,
        "atr14": atr_values, "relative_volume20": rv_values, "prior_high20": prior_highs,
    }


def _market_candle_to_bar(candle: MarketCandle) -> Bar:
    if candle.interval_minutes != 1 or candle.quote_currency.upper() != "IDR":
        raise ForwardPaperError("forward strategy requires native 1m IDR candles")
    return Bar(
        open_time_ms=candle.open_time_ms,
        open=float(candle.open), high=float(candle.high), low=float(candle.low),
        close=float(candle.close), volume=float(candle.volume), close_time_ms=candle.close_time_ms,
    )


def evaluate_latest_signal(
    candles: Sequence[MarketCandle], config: ForwardPaperConfig, spread_bps: Decimal,
) -> dict[str, Any]:
    bars1 = [_market_candle_to_bar(x) for x in candles]
    if len(bars1) < 450:
        raise ForwardPaperError("insufficient native history for 15m indicator warmup")
    for previous, current in zip(bars1, bars1[1:]):
        if current.open_time_ms - previous.open_time_ms != ONE_MINUTE_MS:
            raise ForwardPaperError("native strategy input contains a candle gap")
    bars5 = aggregate_bars(bars1, 5)
    bars15 = aggregate_bars(bars1, 15)
    features1, features5, features15 = (
        _build_features(bars1), _build_features(bars5), _build_features(bars15)
    )
    backtest_config = BacktestConfig(
        initial_capital_idr=float(config.initial_cash_idr),
        quote_to_idr_rate=1.0,
        quote_to_idr_source="native INDODAX SOL/IDR; no FX conversion",
        input_venue="INDODAX",
        input_symbol="SOL/IDR",
        input_price_currency="IDR",
        fee_model_name="INDODAX_CUSTOM",
        fee_model_source="INDODAX official Pro all-in fees checked 2026-10-30; account-specific fee tier not verified",
        entry_fee_bps=float(config.fee_buy_bps),
        exit_fee_bps=float(config.fee_sell_bps),
        spread_bps_round_trip=float(spread_bps),
        slippage_bps_per_side=float(config.slippage_assumption_bps_per_side),
        minimum_net_edge_bps=float(config.minimum_net_edge_bps),
        minimum_relative_volume=1.5,
        minimum_order_notional_idr=float(config.mode_minimum_notional_idr),
        stop_atr_multiple=float(config.stop_atr_multiple),
        target_atr_multiple=float(config.target_atr_multiple),
        maximum_hold_bars=config.maximum_hold_minutes,
    )
    backtest_config.validate()
    last_index = len(bars1) - 1
    accepted, reason, atr5 = signal_at(
        bars1, features1, bars5, features5, bars15, features15,
        last_index, backtest_config,
        [x.close_time_ms for x in bars5], [x.close_time_ms for x in bars15],
    )
    latest_1 = {
        key: features1[key][last_index]
        for key in ("ema9", "ema21", "rsi14", "atr14", "relative_volume20", "prior_high20")
    }
    last5_index = max(0, len(bars5) - 1)
    last15_index = max(0, len(bars15) - 1)
    metrics = {
        "1m": latest_1,
        "5m": {key: features5[key][last5_index] for key in ("ema9", "ema21", "rsi14", "atr14")},
        "15m": {key: features15[key][last15_index] for key in ("ema9", "ema21")},
    }
    reasons = [] if accepted else [x for x in str(reason).split(",") if x]
    return {
        "accepted": bool(accepted),
        "reason": str(reason),
        "reasons": reasons,
        "atr5_idr": str(Decimal(str(atr5))) if atr5 is not None else None,
        "candle_close_ms": bars1[-1].close_time_ms,
        "strategy_version": STRATEGY_VERSION,
        "config_hash": config.config_hash,
        "metrics": metrics,
    }


def _intent_to_request(payload: dict[str, Any]) -> OrderRequest:
    return OrderRequest(
        client_order_id=payload["client_order_id"],
        venue=payload["venue"],
        symbol=payload["symbol"],
        side=ExecutionSide(payload["side"]),
        quantity=Decimal(payload["quantity"]),
        order_type=OrderType(payload["order_type"]),
        limit_price=Decimal(payload["limit_price"]) if payload.get("limit_price") is not None else None,
        time_in_force=TimeInForce(payload["time_in_force"]),
        created_at=datetime.fromisoformat(payload["created_at"]),
        expires_at=datetime.fromisoformat(payload["expires_at"]) if payload.get("expires_at") else None,
        quantity_step=Decimal(payload["quantity_step"]),
        price_tick=Decimal(payload["price_tick"]),
        fee_bps=Decimal(payload["fee_bps"]),
        fee_quantum=Decimal(payload["fee_quantum"]),
        max_quote_age_seconds=Decimal(payload["max_quote_age_seconds"]),
    )


def _request_to_intent(request: OrderRequest, *, purpose: str, candle_close_ms: int, atr_idr: Decimal | None, reason: str) -> dict[str, Any]:
    return {
        "client_order_id": request.client_order_id,
        "venue": request.venue,
        "symbol": request.symbol,
        "side": request.side.value,
        "quantity": str(request.quantity),
        "order_type": request.order_type.value,
        "limit_price": str(request.limit_price) if request.limit_price is not None else None,
        "time_in_force": request.time_in_force.value,
        "created_at": iso_utc(request.created_at),
        "expires_at": iso_utc(request.expires_at) if request.expires_at else None,
        "quantity_step": str(request.quantity_step),
        "price_tick": str(request.price_tick),
        "fee_bps": str(request.fee_bps),
        "fee_quantum": str(request.fee_quantum),
        "max_quote_age_seconds": str(request.max_quote_age_seconds),
        "purpose": purpose,
        "candle_close_ms": candle_close_ms,
        "atr_idr": str(atr_idr) if atr_idr is not None else None,
        "reason": reason,
    }


class ForwardPaperOrchestrator:
    """Runs one bounded, deterministic forward-paper cycle per invocation."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        config: ForwardPaperConfig | None = None,
        market_data: Any = None,
        depth_client: Any = None,
        now: Callable[[], datetime] = utc_now,
    ):
        self.path = str(Path(database_path).expanduser())
        self.config = config or ForwardPaperConfig()
        self.config.validate()
        self.now = now
        # All three subsystems deliberately share one durable SQLite file.
        self.ledger = SQLitePaperExecutionLedger(self.path, initial_cash_idr=self.config.initial_cash_idr)
        self.state = ForwardState(self.path, self.config)
        self.depth_store = DepthStore(self.path)
        self.market_data = market_data or IndodaxPublicMarketDataAdapter(timeout=5.0, attempts=2)
        self.depth_client = depth_client or DepthClient(timeout=5.0, attempts=2, interval_seconds=1.0)
        self.risk = RiskSupervisor(self.state, self.config)
        ExecutionPolicy(mode=ExecutionMode.PAPER).validate()

    def _sample_depth(self) -> tuple[IndodaxDepthSnapshot, str, bool, int]:
        observed = self.now().astimezone(UTC)
        started = time.monotonic()
        try:
            snap = self.depth_client.snapshot()
            elapsed = max(0, int((time.monotonic() - started) * 1000))
            observation_id, inserted = self.depth_store.append(snap)
            self.depth_store.attempt(observed, elapsed)
            return snap, observation_id, inserted, elapsed
        except (MicrostructureError, OSError, TimeoutError, urllib.error.URLError) as exc:
            elapsed = max(0, int((time.monotonic() - started) * 1000))
            self.depth_store.attempt(observed, elapsed, exc)
            raise ForwardPaperError(f"INDODAX depth collection failed: {type(exc).__name__}: {str(exc)[:240]}") from None

    def _validate_depth(self, raw: IndodaxDepthSnapshot, now: datetime) -> DepthSnapshot:
        snapshot = DepthSnapshot.from_indodax(raw)
        age = Decimal(str((now.astimezone(UTC) - snapshot.observed_at).total_seconds()))
        if age < 0 or age > self.config.max_depth_age_seconds:
            raise ForwardPaperError(f"depth snapshot stale or from future: age_seconds={age}")
        if snapshot.exchange_timestamp_ms is not None:
            if snapshot.exchange_timestamp_ms < 1_000_000_000_000:
                raise ForwardPaperError("exchange order-book timestamp is not expressed in milliseconds")
            exchange_time = datetime.fromtimestamp(snapshot.exchange_timestamp_ms / 1000, UTC)
            exchange_age = Decimal(str((now.astimezone(UTC) - exchange_time).total_seconds()))
            if exchange_age < Decimal("-5") or exchange_age > self.config.max_exchange_timestamp_age_seconds:
                raise ForwardPaperError(f"exchange order-book timestamp outside freshness window: {exchange_age}")
        if snapshot.venue != "INDODAX" or snapshot.symbol != "SOL/IDR":
            raise ForwardPaperError("depth snapshot venue/pair identity mismatch")
        if snapshot.spread_bps > self.config.max_spread_bps:
            raise ForwardPaperError(f"quoted spread {snapshot.spread_bps} bps exceeds paper entry safety limit")
        return snapshot

    def _fetch_history(self, market: MarketSpec, now: datetime) -> list[MarketCandle]:
        now_ms = int(now.timestamp() * 1000)
        end_ms = (now_ms // ONE_MINUTE_MS) * ONE_MINUTE_MS
        start_ms = end_ms - self.config.history_minutes * ONE_MINUTE_MS
        candles = self.market_data.get_history(market, start_ms, end_ms, interval_minutes=1, now_ms=now_ms)
        if len(candles) != self.config.history_minutes:
            raise ForwardPaperError(
                f"native history count mismatch: expected {self.config.history_minutes}, got {len(candles)}"
            )
        latest = candles[-1]
        age = Decimal(str((now_ms - latest.close_time_ms) / 1000))
        if latest.quote_currency.upper() != "IDR" or age < 0 or age > self.config.max_candle_age_seconds:
            raise ForwardPaperError(f"latest native candle invalid or stale: age_seconds={age}")
        previous = None
        for candle in candles:
            if candle.interval_minutes != 1 or candle.quote_currency.upper() != "IDR":
                raise ForwardPaperError("history contains non-native or non-1m candles")
            if previous is not None and candle.open_time_ms - previous != ONE_MINUTE_MS:
                raise ForwardPaperError("history contains a gap or out-of-order candle")
            previous = candle.open_time_ms
        return candles

    def _resolve_pending_intents(self) -> list[str]:
        actions: list[str] = []
        now = self.now().astimezone(UTC)
        for row in self.state.pending_intents():
            payload = json.loads(row["intent_json"]) if row["intent_json"] else None
            if not payload:
                self.state.set_control("MANUAL_REVIEW_REQUIRED", "pending order intent has no serialized request")
                raise ForwardPaperError("pending order intent has no serialized request")
            existing = self.ledger.find_order_by_client_order_id(payload["client_order_id"])
            if existing is None:
                expires = datetime.fromisoformat(payload["expires_at"]) if payload.get("expires_at") else None
                if expires is not None and now >= expires:
                    actions.append(f"expired_unsubmitted_intent:{row['signal_id']}")
                    continue
                order_id = self.ledger.create_order(_intent_to_request(payload))
                actions.append(f"recovered_order_intent:{row['signal_id']}")
            else:
                order_id = existing["order_id"]
                actions.append(f"reconciled_existing_order:{row['signal_id']}")
            self.state.attach_order(row["signal_id"], order_id)
            if payload["purpose"] == "ENTRY" and payload.get("atr_idr"):
                atr_value = Decimal(payload["atr_idr"])
                self.state.save_position(
                    signal_id=row["signal_id"], order_id=order_id,
                    candle_close_ms=int(payload["candle_close_ms"]), atr_idr=atr_value,
                    entry_price=None, stop_price=None, target_price=None, status="PENDING",
                )
        return actions

    def _process_open_orders(self, snapshot: DepthSnapshot) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for order in self.ledger.list_open_orders():
            if order["state"].value == "RECOVERY_REQUIRED":
                self.state.set_control("MANUAL_REVIEW_REQUIRED", "execution ledger contains RECOVERY_REQUIRED order")
                raise ForwardPaperError(f"order {order['order_id']} requires manual review")
            result = self.ledger.process_snapshot(order["order_id"], snapshot, as_of=self.now().astimezone(UTC))
            results.append(result.as_dict())
        return results

    def _active_open_orders(self) -> list[dict[str, Any]]:
        return self.ledger.list_open_orders()

    def _sync_position(self, candle_close_ms: int) -> dict[str, Any] | None:
        portfolio = self.ledger.portfolio()
        qty = portfolio["sol_quantity"]
        position = self.state.position()
        if qty <= 0:
            if position and position["status"] != "CLOSED":
                self.state.close_position()
            return self.state.position()
        if position is None or position["status"] == "CLOSED":
            self.state.set_control("MANUAL_REVIEW_REQUIRED", "SOL inventory exists without durable strategy position metadata")
            raise ForwardPaperError("unattributed simulated SOL inventory; manual review required")
        atr_value = decimal(position["atr_idr"], "position ATR")
        entry = portfolio["avg_entry_cost_idr"]
        stop = entry - atr_value * self.config.stop_atr_multiple
        target = entry + atr_value * self.config.target_atr_multiple
        self.state.save_position(
            signal_id=position["entry_signal_id"], order_id=position["entry_order_id"],
            candle_close_ms=int(position["entry_candle_close_ms"]), atr_idr=atr_value,
            entry_price=entry, stop_price=stop, target_price=target, status="OPEN",
        )
        return self.state.position()

    def _make_request(
        self, *, client_order_id: str, side: ExecutionSide, quantity: Decimal,
        market: MarketSpec, now: datetime,
    ) -> OrderRequest:
        return OrderRequest(
            client_order_id=client_order_id, venue="INDODAX", symbol="SOL/IDR",
            side=side, quantity=quantity, order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC, created_at=now,
            expires_at=now + timedelta(minutes=self.config.order_ttl_minutes),
            quantity_step=market.quantity_step, price_tick=market.price_tick,
            fee_bps=self.config.fee_buy_bps if side is ExecutionSide.BUY else self.config.fee_sell_bps,
            fee_quantum=Decimal("0.01"), max_quote_age_seconds=self.config.max_depth_age_seconds,
        )

    def _persist_intent(
        self, *, signal_id: str, candle_close_ms: int, purpose: str,
        side: ExecutionSide, quantity: Decimal, market: MarketSpec,
        atr_idr: Decimal | None, reason: str,
    ) -> dict[str, Any]:
        now = self.now().astimezone(UTC)
        request = self._make_request(
            client_order_id=signal_id, side=side, quantity=quantity, market=market, now=now
        )
        intent = _request_to_intent(
            request, purpose=purpose, candle_close_ms=candle_close_ms, atr_idr=atr_idr, reason=reason
        )
        self.state.record_signal(
            signal_id, candle_close_ms, side.value, "ORDER_INTENT", (reason,), intent
        )
        existing = self.ledger.find_order_by_client_order_id(signal_id)
        order_id = existing["order_id"] if existing else self.ledger.create_order(request)
        self.state.attach_order(signal_id, order_id)
        if purpose == "ENTRY" and atr_idr is not None:
            self.state.save_position(
                signal_id=signal_id, order_id=order_id, candle_close_ms=candle_close_ms,
                atr_idr=atr_idr, entry_price=None, stop_price=None, target_price=None, status="PENDING",
            )
        return {"signal_id": signal_id, "client_order_id": signal_id, "order_id": order_id, "intent": intent}

    def _execute_new_order(self, order_id: str) -> dict[str, Any]:
        raw, observation_id, inserted, latency = self._sample_depth()
        snapshot = self._validate_depth(raw, self.now())
        order = self.ledger.get_order(order_id)
        if snapshot.observed_at <= datetime.fromisoformat(order["created_at"]):
            raise ForwardPaperError("execution snapshot must be observed strictly after order creation")
        result = self.ledger.process_snapshot(order_id, snapshot, as_of=self.now().astimezone(UTC))
        return {
            "collector_observation_id": observation_id,
            "snapshot_id": snapshot.snapshot_id,
            "new_observation": inserted,
            "request_latency_ms": latency,
            "execution": result.as_dict(),
        }

    def _manage_exit(
        self, *, market: MarketSpec, snapshot: DepthSnapshot, candle_close_ms: int,
        open_orders: Sequence[dict[str, Any]],
    ) -> dict[str, Any] | None:
        portfolio = self.ledger.portfolio()
        if portfolio["sol_quantity"] <= 0:
            return None
        if any(x["side"] == ExecutionSide.SELL.value for x in open_orders):
            return None
        position = self.state.position()
        if not position or position["status"] not in ("OPEN", "PENDING"):
            self.state.set_control("MANUAL_REVIEW_REQUIRED", "position lacks active stop/target metadata")
            raise ForwardPaperError("cannot manage unattributed position")
        if not position.get("stop_price_idr") or not position.get("target_price_idr"):
            return None
        bid = snapshot.bids[0].price
        stop = Decimal(position["stop_price_idr"])
        target = Decimal(position["target_price_idr"])
        held_minutes = max(0, (candle_close_ms - int(position["entry_candle_close_ms"])) // ONE_MINUTE_MS)
        reason = None
        if bid <= stop:
            reason = "stop_loss_triggered"
        elif bid >= target:
            reason = "take_profit_triggered"
        elif held_minutes >= self.config.maximum_hold_minutes:
            reason = "maximum_holding_time_reached"
        if reason is None:
            return None
        qty = floor_to_step(portfolio["sol_quantity"], market.quantity_step)
        if qty < market.min_quantity_base or qty * bid < market.min_notional_quote:
            self.state.set_control("MANUAL_REVIEW_REQUIRED", "position residual is below venue minimum; cannot simulate a standard exit")
            raise ForwardPaperError("position residual below venue minimum; manual review required")
        signal_id = digest({
            "purpose": "EXIT", "entry_signal_id": position["entry_signal_id"],
            "candle_close_ms": candle_close_ms, "reason": reason,
        })
        return self._persist_intent(
            signal_id=signal_id, candle_close_ms=candle_close_ms, purpose="EXIT",
            side=ExecutionSide.SELL, quantity=qty, market=market, atr_idr=None, reason=reason,
        )

    def run_once(self) -> dict[str, Any]:
        """Run one complete paper cycle; all external requests are public read-only GETs."""
        started = self.now().astimezone(UTC)
        recovery = self.ledger.recover()
        recovered_cycles = self.state.recover_started_cycles()
        if self.state.control().get("operating_state") == "MANUAL_REVIEW_REQUIRED":
            raise ForwardPaperError(self.state.control().get("halt_reason") or "manual review required")
        market = self.market_data.get_market_spec(
            "SOL/IDR", mode_minimum_notional_quote=self.config.mode_minimum_notional_idr
        )
        if market.venue.upper() != "INDODAX" or market.symbol.upper() != "SOL/IDR" or market.quote_asset.upper() != "IDR":
            raise ForwardPaperError("market metadata is not native INDODAX SOL/IDR")
        candles = self._fetch_history(market, started)
        raw, collector_id, inserted, latency = self._sample_depth()
        snapshot = self._validate_depth(raw, self.now())
        latest = candles[-1]
        if latest.close_time_ms > int(snapshot.observed_at.timestamp() * 1000):
            raise ForwardPaperError("candle timestamp is later than the depth observation")
        cycle_id = digest({
            "config": self.config.config_hash, "candle_close_ms": latest.close_time_ms,
            "snapshot_id": snapshot.snapshot_id,
        })[:40]
        created = self.state.begin_cycle(
            cycle_id, latest.close_time_ms, snapshot.snapshot_id, collector_id
        )
        if not created:
            return {"status": "DUPLICATE_CYCLE", "cycle_id": cycle_id, "recovery": recovery}
        actions: list[str] = []
        try:
            actions.extend(self._resolve_pending_intents())
            order_results = self._process_open_orders(snapshot)
            portfolio = self.ledger.portfolio()
            risk_decision = self.risk.evaluate(
                portfolio, snapshot, self._active_open_orders(), now=self.now().astimezone(UTC)
            )
            if risk_decision.state in ("HALTED", "MANUAL_REVIEW_REQUIRED"):
                actions.extend(
                    "cancelled_entry:" + order_id
                    for order_id in self.risk.cancel_entry_residuals(self.ledger, at=self.now().astimezone(UTC))
                )
            position = self._sync_position(latest.close_time_ms)
            exit_intent = self._manage_exit(
                market=market, snapshot=snapshot, candle_close_ms=latest.close_time_ms,
                open_orders=self._active_open_orders(),
            )
            new_execution: dict[str, Any] | None = None
            if exit_intent is not None:
                new_execution = self._execute_new_order(exit_intent["order_id"])
                order_results.append(new_execution["execution"])
                actions.append("protective_exit:" + exit_intent["signal_id"])
            elif not self._active_open_orders() and self.ledger.portfolio()["sol_quantity"] == 0:
                signal_result = evaluate_latest_signal(candles, self.config, snapshot.spread_bps)
                signal_id = digest({
                    "strategy": STRATEGY_VERSION, "config": self.config.config_hash,
                    "candle_close_ms": latest.close_time_ms,
                })[:40]
                previous = self.state.get_signal(signal_id)
                if previous is None:
                    if not signal_result["accepted"]:
                        self.state.record_signal(
                            signal_id, latest.close_time_ms, "BUY", "NO_TRADE",
                            signal_result["reasons"] or [signal_result["reason"]],
                        )
                    else:
                        current_portfolio = self.ledger.portfolio()
                        refreshed_risk = self.risk.evaluate(
                            current_portfolio, snapshot, self._active_open_orders(),
                            now=self.now().astimezone(UTC),
                        )
                        if not refreshed_risk.allow_new_entries:
                            self.state.record_signal(
                                signal_id, latest.close_time_ms, "BUY", "RISK_REJECTED",
                                refreshed_risk.reasons or ("risk_gate_blocked",),
                            )
                        else:
                            atr_value = decimal(signal_result["atr5_idr"], "ATR")
                            quantity, size_reasons = self.risk.size_buy(
                                portfolio=current_portfolio, market=market,
                                snapshot=snapshot, atr_idr=atr_value,
                            )
                            if size_reasons:
                                self.state.record_signal(
                                    signal_id, latest.close_time_ms, "BUY", "RISK_REJECTED", size_reasons,
                                )
                            elif self.state.count_daily_entry_intents(started.date().isoformat()) >= self.config.max_daily_entry_orders:
                                self.state.record_signal(
                                    signal_id, latest.close_time_ms, "BUY", "RISK_REJECTED",
                                    ("maximum_daily_entry_order_count_reached",),
                                )
                            else:
                                ask_depth_value = sum((x.price * x.quantity for x in snapshot.asks), Decimal("0"))
                                if ask_depth_value < quantity * snapshot.asks[0].price * self.config.minimum_liquidity_coverage:
                                    self.state.record_signal(
                                        signal_id, latest.close_time_ms, "BUY", "RISK_REJECTED",
                                        ("visible_ask_depth_below_order_coverage",),
                                    )
                                else:
                                    self._persist_intent(
                                        signal_id=signal_id, candle_close_ms=latest.close_time_ms,
                                        purpose="ENTRY", side=ExecutionSide.BUY, quantity=quantity,
                                        market=market, atr_idr=atr_value, reason="multi_timeframe_breakout",
                                    )
                                    actions.append("entry_intent:" + signal_id)
                                    try:
                                        new_execution = self._execute_new_order(signal_id_to_order(self.ledger, signal_id))
                                        order_results.append(new_execution["execution"])
                                    except ForwardPaperError as exc:
                                        actions.append("entry_pending_fresh_depth_unavailable:" + str(exc))
                else:
                    actions.append("signal_slot_already_processed:" + signal_id)
            position = self._sync_position(latest.close_time_ms)
            portfolio = self.ledger.portfolio()
            reconciliation = self.ledger.recover()
            risk_after = self.risk.evaluate(
                portfolio, snapshot, self._active_open_orders(), now=self.now().astimezone(UTC)
            )
            result = {
                "schema_version": "sol-sentinel-forward-cycle-v1",
                "cycle_id": cycle_id,
                "started_at": iso_utc(started),
                "completed_at": iso_utc(self.now().astimezone(UTC)),
                "mode": "PAPER",
                "venue": "INDODAX",
                "symbol": "SOL/IDR",
                "quote_currency": "IDR",
                "strategy_version": STRATEGY_VERSION,
                "config_version": CONFIG_VERSION,
                "config_hash": self.config.config_hash,
                "candle_count": len(candles),
                "candle_close_ms": latest.close_time_ms,
                "collector_observation_id": collector_id,
                "collector_observation_inserted": inserted,
                "depth_request_latency_ms": latency,
                "execution_snapshot_id": snapshot.snapshot_id,
                "spread_bps": str(snapshot.spread_bps),
                "signal": json.loads(self.state.get_signal(digest({
                    "strategy": STRATEGY_VERSION, "config": self.config.config_hash,
                    "candle_close_ms": latest.close_time_ms,
                })[:40])["reason_json"]) if self.state.get_signal(digest({
                    "strategy": STRATEGY_VERSION, "config": self.config.config_hash,
                    "candle_close_ms": latest.close_time_ms,
                })[:40]) else [],
                "actions": actions,
                "orders_processed": order_results,
                "new_order_execution": new_execution,
                "position": position,
                "portfolio": {key: str(value) for key, value in portfolio.items()},
                "risk": {
                    "state": risk_after.state,
                    "allow_new_entries": risk_after.allow_new_entries,
                    "reasons": list(risk_after.reasons),
                    "equity_idr": str(risk_after.equity_idr),
                    "exposure_fraction": str(risk_after.exposure_fraction),
                    "daily_loss_fraction": str(risk_after.daily_loss_fraction),
                    "drawdown_fraction": str(risk_after.drawdown_fraction),
                },
                "reconciliation": reconciliation,
                "recovery": recovery,
                "recovered_incomplete_cycles": recovered_cycles,
                "read_only_market_data": True,
                "live_orders_enabled": False,
            }
            self.state.finish_cycle(cycle_id, "COMPLETED", result)
            if reconciliation["consistent"]:
                self.state.activation(
                    commit_sha=os.environ.get("GITHUB_SHA", "local-unpinned"),
                    first_cycle_id=cycle_id,
                    portfolio={key: str(value) for key, value in portfolio.items()},
                )
            else:
                self.state.set_control("MANUAL_REVIEW_REQUIRED", "portfolio reconciliation failed")
                raise ForwardPaperError("portfolio reconciliation failed")
            self.state.checkpoint()
            return result
        except Exception as exc:
            self.state.finish_cycle(cycle_id, "ERROR", {"cycle_id": cycle_id}, f"{type(exc).__name__}: {str(exc)}")
            self.state.checkpoint()
            raise

    def operator_halt(self, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise ForwardPaperError("a non-empty halt reason is required")
        self.state.set_control("HALTED", reason)
        cancelled = self.risk.cancel_entry_residuals(self.ledger, at=self.now().astimezone(UTC))
        self.ledger.recover()
        self.state.checkpoint()
        return {"operating_state": "HALTED", "reason": reason, "cancelled_buy_orders": cancelled}

    def operator_resume(self, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise ForwardPaperError("manual resume requires an explicit operator reason")
        if not self.ledger.recover()["consistent"]:
            raise ForwardPaperError("cannot resume: ledger reconciliation is inconsistent")
        self.state.set_control("ACTIVE", "operator_resume: " + reason)
        self.state.checkpoint()
        return {"operating_state": "ACTIVE", "reason": reason, "mode": "PAPER"}


def signal_id_to_order(ledger: SQLitePaperExecutionLedger, client_order_id: str) -> str:
    row = ledger.find_order_by_client_order_id(client_order_id)
    if row is None:
        raise ForwardPaperError("durable order intent exists without a corresponding execution-ledger order")
    return row["order_id"]


def run_cycles(
    orchestrator: ForwardPaperOrchestrator, *, cycles: int, interval_seconds: int,
) -> list[dict[str, Any]]:
    if isinstance(cycles, bool) or not isinstance(cycles, int) or not 1 <= cycles <= 100:
        raise ForwardPaperError("cycles must be in 1..100")
    if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, int) or interval_seconds < 0:
        raise ForwardPaperError("interval_seconds must be a non-negative integer")
    results = []
    for index in range(cycles):
        result = orchestrator.run_once()
        results.append(result)
        if index + 1 < cycles and interval_seconds:
            time.sleep(interval_seconds)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="~/.sol-sentinel/forward-paper.sqlite3")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--interval-seconds", type=int, default=0)
    parser.add_argument("--history-minutes", type=int, default=600)
    parser.add_argument("--initial-cash-idr", default="300000")
    parser.add_argument("--halt", action="store_true", help="persistently halt new paper entries")
    parser.add_argument("--halt-reason", default="operator_requested_halt")
    parser.add_argument("--resume", action="store_true", help="explicitly resume paper mode after reconciliation")
    parser.add_argument("--resume-reason", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = ForwardPaperConfig(
            initial_cash_idr=Decimal(args.initial_cash_idr),
            history_minutes=args.history_minutes,
        )
        orchestrator = ForwardPaperOrchestrator(args.db_path, config=config)
        if args.halt:
            output: Any = orchestrator.operator_halt(args.halt_reason)
        elif args.resume:
            output = orchestrator.operator_resume(args.resume_reason)
        else:
            output = run_cycles(
                orchestrator, cycles=args.cycles, interval_seconds=args.interval_seconds
            )
        if args.json:
            print(canonical_json(output))
        else:
            print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
        return 0
    except (ForwardPaperError, ExecutionError, MicrostructureError, OSError, ValueError, sqlite3.Error) as exc:
        error = {
            "status": "BLOCKED",
            "mode": "PAPER",
            "error_class": type(exc).__name__,
            "error": str(exc)[:1000],
            "live_orders_enabled": False,
        }
        print(json.dumps(error, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
