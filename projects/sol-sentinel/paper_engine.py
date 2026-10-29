#!/usr/bin/env python3
"""Offline, cost-aware SOL strategy replay. Never places orders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ONE_MINUTE_MS = 60_000
TIMEFRAME_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000}
SCHEMA_VERSION = "sol-sentinel-paper-v1"


@dataclass(frozen=True)
class Bar:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital_idr: float = 300_000.0
    risk_fraction: float = 0.005
    max_exposure_fraction: float = 0.50
    daily_loss_fraction: float = 0.03
    entry_fee_bps: float = 10.555
    exit_fee_bps: float = 31.555
    spread_bps_round_trip: float = 10.0
    slippage_bps_per_side: float = 10.0
    minimum_net_edge_bps: float = 25.0
    minimum_relative_volume: float = 1.5
    stop_atr_multiple: float = 1.0
    target_atr_multiple: float = 1.5
    maximum_hold_bars: int = 30

    def validate(self) -> None:
        positive = ("initial_capital_idr", "stop_atr_multiple", "target_atr_multiple", "maximum_hold_bars")
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        fractions = ("risk_fraction", "max_exposure_fraction", "daily_loss_fraction")
        for name in fractions:
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("entry_fee_bps", "exit_fee_bps", "spread_bps_round_trip", "slippage_bps_per_side", "minimum_net_edge_bps", "minimum_relative_volume"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.maximum_hold_bars != int(self.maximum_hold_bars):
            raise ValueError("maximum_hold_bars must be an integer")


@dataclass
class Position:
    signal_time_ms: int
    entry_time_ms: int
    entry_reference_price: float
    entry_fill_price: float
    quantity: float
    entry_fee_idr: float
    stop_price: float
    target_price: float
    entry_atr: float
    bars_held: int = 0


def iso_utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).isoformat()


def parse_timestamp_ms(raw: str) -> int:
    value = raw.strip()
    if not value:
        raise ValueError("empty timestamp")
    try:
        numeric = float(value)
        if math.isfinite(numeric):
            return int(numeric * 1000 if abs(numeric) < 100_000_000_000 else numeric)
    except ValueError:
        pass
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        raise ValueError("ISO timestamps must include timezone")
    return int(dt.timestamp() * 1000)


def read_bars_csv(path: str | Path) -> list[Bar]:
    file_path = Path(path)
    bars: list[Bar] = []
    with file_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        names = {name.strip().lower(): name for name in reader.fieldnames if name}
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(names)
        if missing:
            raise ValueError("CSV missing columns: " + ", ".join(sorted(missing)))
        timestamp_key = names.get("open_time_ms") or names.get("timestamp") or names.get("time")
        if not timestamp_key:
            raise ValueError("CSV requires open_time_ms, timestamp, or time")
        close_time_key = names.get("close_time_ms")
        for line_number, row in enumerate(reader, start=2):
            try:
                ts = parse_timestamp_ms(row[timestamp_key])
                close_time = int(float(row[close_time_key])) if close_time_key and row.get(close_time_key) else ts + ONE_MINUTE_MS - 1
                bar = Bar(
                    open_time_ms=ts,
                    open=float(row[names["open"]]),
                    high=float(row[names["high"]]),
                    low=float(row[names["low"]]),
                    close=float(row[names["close"]]),
                    volume=float(row[names["volume"]]),
                    close_time_ms=close_time,
                )
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"invalid CSV row {line_number}: {exc}") from exc
            values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"non-finite OHLCV at CSV row {line_number}")
            if min(bar.open, bar.high, bar.low, bar.close) <= 0 or bar.volume < 0:
                raise ValueError(f"non-positive price or negative volume at CSV row {line_number}")
            if bar.high < max(bar.open, bar.close, bar.low) or bar.low > min(bar.open, bar.close, bar.high):
                raise ValueError(f"impossible high/low at CSV row {line_number}")
            if bar.close_time_ms < bar.open_time_ms:
                raise ValueError(f"close timestamp precedes open at CSV row {line_number}")
            bars.append(bar)
    if len(bars) < 100:
        raise ValueError(f"need at least 100 one-minute bars; got {len(bars)}")
    for previous, current in zip(bars, bars[1:]):
        if current.open_time_ms <= previous.open_time_ms:
            raise ValueError("CSV timestamps must be strictly increasing and unique")
        if current.open_time_ms - previous.open_time_ms != ONE_MINUTE_MS:
            raise ValueError(f"missing/non-1m candle between {iso_utc(previous.open_time_ms)} and {iso_utc(current.open_time_ms)}")
    return bars


def ema_series(values: Sequence[float], period: int) -> list[float]:
    if period < 1:
        raise ValueError("EMA period must be >= 1")
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * float(value) + (1.0 - alpha) * result[-1])
    return result


def rsi_at(closes: Sequence[float], index: int, period: int = 14) -> float | None:
    if period < 1 or index < period:
        return None
    deltas = [closes[j] - closes[j - 1] for j in range(index - period + 1, index + 1)]
    gains = sum(max(v, 0.0) for v in deltas) / period
    losses = sum(max(-v, 0.0) for v in deltas) / period
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    strength = gains / losses
    return 100.0 - 100.0 / (1.0 + strength)


def atr_at(bars: Sequence[Bar], index: int, period: int = 14) -> float | None:
    if period < 1 or index < period:
        return None
    ranges = []
    for j in range(index - period + 1, index + 1):
        current = bars[j]
        previous_close = bars[j - 1].close
        ranges.append(max(
            current.high - current.low,
            abs(current.high - previous_close),
            abs(current.low - previous_close),
        ))
    return sum(ranges) / len(ranges)


def aggregate_bars(bars: Sequence[Bar], factor: int) -> list[Bar]:
    """Aggregate contiguous 1m bars; reject partial/missing buckets instead of inventing data."""
    if factor < 1:
        raise ValueError("aggregation factor must be >= 1")
    groups: dict[int, list[Bar]] = {}
    bucket_ms = factor * ONE_MINUTE_MS
    for bar in bars:
        bucket = (bar.open_time_ms // bucket_ms) * bucket_ms
        groups.setdefault(bucket, []).append(bar)
    result: list[Bar] = []
    for bucket, rows in sorted(groups.items()):
        rows.sort(key=lambda item: item.open_time_ms)
        if len(rows) != factor:
            continue
        if rows[0].open_time_ms != bucket:
            continue
        if any(rows[i].open_time_ms - rows[i - 1].open_time_ms != ONE_MINUTE_MS for i in range(1, len(rows))):
            continue
        if rows[-1].open_time_ms != bucket + (factor - 1) * ONE_MINUTE_MS:
            continue
        result.append(Bar(
            open_time_ms=rows[0].open_time_ms,
            open=rows[0].open,
            high=max(item.high for item in rows),
            low=min(item.low for item in rows),
            close=rows[-1].close,
            volume=sum(item.volume for item in rows),
            close_time_ms=rows[-1].close_time_ms,
        ))
    return result


def signal_at(
    bars1: Sequence[Bar],
    features1: dict[str, list[float | None]],
    bars5: Sequence[Bar],
    features5: dict[str, list[float | None]],
    bars15: Sequence[Bar],
    features15: dict[str, list[float | None]],
    index1: int,
    config: BacktestConfig,
    close_times5: Sequence[int] | None = None,
    close_times15: Sequence[int] | None = None,
) -> tuple[bool, str, float | None]:
    decision_bar = bars1[index1]
    times5 = close_times5 if close_times5 is not None else [b.close_time_ms for b in bars5]
    times15 = close_times15 if close_times15 is not None else [b.close_time_ms for b in bars15]
    idx5 = bisect_right(times5, decision_bar.close_time_ms) - 1
    idx15 = bisect_right(times15, decision_bar.close_time_ms) - 1
    if idx5 < 25 or idx15 < 25 or index1 < 25:
        return False, "indicator_warmup", None

    close15 = bars15[idx15].close
    ema9_15, ema21_15 = features15["ema9"][idx15], features15["ema21"][idx15]
    close5 = bars5[idx5].close
    ema9_5, ema21_5 = features5["ema9"][idx5], features5["ema21"][idx5]
    rsi5, atr5 = features5["rsi14"][idx5], features5["atr14"][idx5]
    rsi1, rv1, prior_high = features1["rsi14"][index1], features1["relative_volume20"][index1], features1["prior_high20"][index1]

    checks = [
        (ema9_15 is not None and ema21_15 is not None and ema9_15 > ema21_15 and close15 > ema21_15, "15m_trend_not_confirmed"),
        (ema9_5 is not None and ema21_5 is not None and ema9_5 > ema21_5 and close5 > ema9_5, "5m_momentum_not_confirmed"),
        (rsi5 is not None and 48.0 <= rsi5 <= 70.0, "5m_rsi_outside_entry_range"),
        (prior_high is not None and decision_bar.close > prior_high, "1m_breakout_not_confirmed"),
        (rv1 is not None and rv1 >= config.minimum_relative_volume, "1m_volume_confirmation_missing"),
        (rsi1 is not None and rsi1 < 78.0, "1m_momentum_exhaustion_or_missing"),
        (atr5 is not None and atr5 > 0, "5m_atr_missing"),
    ]
    failures = [reason for passed, reason in checks if not passed]
    if failures:
        return False, ",".join(failures), atr5

    total_cost_bps = config.entry_fee_bps + config.exit_fee_bps + config.spread_bps_round_trip + 2 * config.slippage_bps_per_side
    gross_target_bps = (float(atr5) * config.target_atr_multiple / close5) * 10_000.0
    if gross_target_bps - total_cost_bps < config.minimum_net_edge_bps:
        return False, "estimated_net_edge_below_threshold", atr5
    return True, "multi_timeframe_breakout", atr5


def mark_equity(
    cash: float,
    position: Position | None,
    price: float,
    config: BacktestConfig | None = None,
) -> float:
    """Mark equity at an estimated liquidation value, including exit friction when configured."""
    if position is None:
        return cash
    liquidation_price = price
    exit_fee = 0.0
    if config is not None:
        adverse_bps_per_side = config.spread_bps_round_trip / 2.0 + config.slippage_bps_per_side
        liquidation_price = price * (1.0 - adverse_bps_per_side / 10_000.0)
        exit_fee = position.quantity * liquidation_price * config.exit_fee_bps / 10_000.0
    return cash + position.quantity * liquidation_price - exit_fee


def resolve_exit(
    bar: Bar,
    stop_price: float,
    target_price: float,
    bars_held: int,
    maximum_hold_bars: int,
) -> tuple[str | None, float | None]:
    """Resolve OHLC exits conservatively; a gap through a stop fills at the worse open."""
    hit_stop = bar.low <= stop_price
    hit_target = bar.high >= target_price
    if hit_stop:
        reason = "stop_loss" if not hit_target else "stop_and_target_same_bar_stop_first"
        return reason, min(bar.open, stop_price)
    if hit_target:
        return "take_profit", target_price
    if bars_held >= maximum_hold_bars:
        return "maximum_hold", bar.close
    return None, None


def backtest(bars: Sequence[Bar], config: BacktestConfig, dataset_sha256: str = "") -> dict[str, Any]:
    config.validate()
    if len(bars) < 100:
        raise ValueError("at least 100 contiguous 1m bars are required")
    for previous, current in zip(bars, bars[1:]):
        if current.open_time_ms - previous.open_time_ms != ONE_MINUTE_MS:
            raise ValueError("backtest bars must be contiguous 1m candles")

    bars5 = aggregate_bars(bars, 5)
    bars15 = aggregate_bars(bars, 15)
    if len(bars5) < 30 or len(bars15) < 30:
        raise ValueError("not enough complete 5m/15m candles for multi-timeframe warmup")

    def build_features(items: Sequence[Bar]) -> dict[str, list[float | None]]:
        closes = [b.close for b in items]
        e9, e21 = ema_series(closes, 9), ema_series(closes, 21)
        rsi_values: list[float | None] = []
        atr_values: list[float | None] = []
        rv_values: list[float | None] = []
        prior_highs: list[float | None] = []
        for index, bar in enumerate(items):
            rsi_values.append(rsi_at(closes, index))
            atr_values.append(atr_at(items, index))
            if index >= 20:
                baseline = sum(items[j].volume for j in range(index - 20, index)) / 20.0
                rv_values.append(bar.volume / baseline if baseline > 0 else None)
                prior_highs.append(max(items[j].high for j in range(index - 20, index)))
            else:
                rv_values.append(None)
                prior_highs.append(None)
        return {
            "ema9": list(e9), "ema21": list(e21), "rsi14": rsi_values,
            "atr14": atr_values, "relative_volume20": rv_values, "prior_high20": prior_highs,
        }

    features1, features5, features15 = build_features(bars), build_features(bars5), build_features(bars15)
    close_times5 = [b.close_time_ms for b in bars5]
    close_times15 = [b.close_time_ms for b in bars15]
    cash = config.initial_capital_idr
    position: Position | None = None
    pending: tuple[int, float] | None = None  # (signal candle index, ATR at signal)
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    peak_equity = cash
    max_drawdown_fraction = 0.0
    total_entry_fees = 0.0
    total_exit_fees = 0.0
    daily_start_equity = cash
    active_day = datetime.fromtimestamp(bars[0].open_time_ms / 1000, timezone.utc).date().isoformat()
    daily_locked = False
    lock_events = 0

    for index, bar in enumerate(bars):
        day = datetime.fromtimestamp(bar.open_time_ms / 1000, timezone.utc).date().isoformat()
        if day != active_day:
            active_day = day
            daily_start_equity = mark_equity(cash, position, bar.open, config)
            daily_locked = False

        # Execute a candidate only on the next candle's open; never fill using the signal candle close.
        if pending is not None and position is None:
            signal_index, signal_atr = pending
            pending = None
            reference = bar.open
            adverse_fill_bps_per_side = config.spread_bps_round_trip / 2.0 + config.slippage_bps_per_side
            entry_fill = reference * (1.0 + adverse_fill_bps_per_side / 10_000.0)
            equity_at_open = mark_equity(cash, None, reference)
            risk_cash = equity_at_open * config.risk_fraction
            stop_distance = signal_atr * config.stop_atr_multiple
            estimated_round_trip_fraction = (
                config.entry_fee_bps + config.exit_fee_bps + config.spread_bps_round_trip + 2 * config.slippage_bps_per_side
            ) / 10_000.0
            risk_per_unit = stop_distance + entry_fill * estimated_round_trip_fraction
            qty_by_risk = risk_cash / risk_per_unit if risk_per_unit > 0 else 0.0
            qty_by_exposure = equity_at_open * config.max_exposure_fraction / entry_fill
            qty_by_cash = cash / (entry_fill * (1.0 + config.entry_fee_bps / 10_000.0))
            quantity = min(qty_by_risk, qty_by_exposure, qty_by_cash)
            if quantity > 0 and math.isfinite(quantity):
                entry_fee = quantity * entry_fill * config.entry_fee_bps / 10_000.0
                cash -= quantity * entry_fill + entry_fee
                total_entry_fees += entry_fee
                position = Position(
                    signal_time_ms=bars[signal_index].close_time_ms,
                    entry_time_ms=bar.open_time_ms,
                    entry_reference_price=reference,
                    entry_fill_price=entry_fill,
                    quantity=quantity,
                    entry_fee_idr=entry_fee,
                    stop_price=entry_fill - stop_distance,
                    target_price=entry_fill + signal_atr * config.target_atr_multiple,
                    entry_atr=signal_atr,
                )

        # Conservative OHLC fill model: if stop and target both touch in one candle, assume stop first.
        exit_reason: str | None = None
        exit_reference: float | None = None
        if position is not None:
            position.bars_held += 1
            exit_reason, exit_reference = resolve_exit(
                bar, position.stop_price, position.target_price,
                position.bars_held, config.maximum_hold_bars,
            )
            if exit_reason and exit_reference is not None:
                adverse_fill_bps_per_side = config.spread_bps_round_trip / 2.0 + config.slippage_bps_per_side
                exit_fill = exit_reference * (1.0 - adverse_fill_bps_per_side / 10_000.0)
                exit_fee = position.quantity * exit_fill * config.exit_fee_bps / 10_000.0
                cash += position.quantity * exit_fill - exit_fee
                total_exit_fees += exit_fee
                gross_pnl = (exit_fill - position.entry_fill_price) * position.quantity
                net_pnl = gross_pnl - position.entry_fee_idr - exit_fee
                trades.append({
                    "trade_id": hashlib.sha256(
                        f"{dataset_sha256}:{position.signal_time_ms}:LONG".encode()
                    ).hexdigest()[:24],
                    "side": "LONG",
                    "signal_time": iso_utc(position.signal_time_ms),
                    "entry_time": iso_utc(position.entry_time_ms),
                    "exit_time": iso_utc(bar.close_time_ms),
                    "entry_reference_price": position.entry_reference_price,
                    "entry_fill_price": position.entry_fill_price,
                    "exit_reference_price": exit_reference,
                    "exit_fill_price": exit_fill,
                    "quantity": position.quantity,
                    "entry_fee_idr": position.entry_fee_idr,
                    "exit_fee_idr": exit_fee,
                    "pnl_before_exchange_fees_idr": gross_pnl,
                    "entry_execution_friction_idr": max(0.0, (position.entry_fill_price - position.entry_reference_price) * position.quantity),
                    "exit_execution_friction_idr": max(0.0, (exit_reference - exit_fill) * position.quantity),
                    "execution_friction_cost_idr": max(0.0, (position.entry_fill_price - position.entry_reference_price) * position.quantity) + max(0.0, (exit_reference - exit_fill) * position.quantity),
                    "net_pnl_idr": net_pnl,
                    "bars_held": position.bars_held,
                    "exit_reason": exit_reason,
                })
                position = None

        equity = mark_equity(cash, position, bar.close, config)
        if day == active_day and equity <= daily_start_equity * (1.0 - config.daily_loss_fraction):
            if not daily_locked:
                lock_events += 1
            daily_locked = True

        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown_fraction = max(max_drawdown_fraction, drawdown)
        curve.append({"time": iso_utc(bar.close_time_ms), "equity_idr": equity, "cash_idr": cash, "drawdown_pct": drawdown * 100.0})

        if index < len(bars) - 1 and position is None and not daily_locked:
            candidate, _reason, current_atr = signal_at(
                bars, features1, bars5, features5, bars15, features15, index, config,
                close_times5=close_times5, close_times15=close_times15,
            )
            if candidate and current_atr is not None and current_atr > 0:
                pending = (index, current_atr)

    # Force-close an open simulated position at the last observed close; this is an explicit end-of-data assumption.
    if position is not None:
        last = bars[-1]
        exit_reference = last.close
        exit_fill = exit_reference * (1.0 - config.slippage_bps_per_side / 10_000.0)
        exit_fee = position.quantity * exit_fill * config.exit_fee_bps / 10_000.0
        cash += position.quantity * exit_fill - exit_fee
        total_exit_fees += exit_fee
        gross_pnl = (exit_fill - position.entry_fill_price) * position.quantity
        net_pnl = gross_pnl - position.entry_fee_idr - exit_fee
        trades.append({
            "trade_id": hashlib.sha256(f"{dataset_sha256}:{position.signal_time_ms}:LONG".encode()).hexdigest()[:24],
            "side": "LONG", "signal_time": iso_utc(position.signal_time_ms),
            "entry_time": iso_utc(position.entry_time_ms), "exit_time": iso_utc(last.close_time_ms),
            "entry_reference_price": position.entry_reference_price, "entry_fill_price": position.entry_fill_price,
            "exit_reference_price": exit_reference, "exit_fill_price": exit_fill,
            "quantity": position.quantity, "entry_fee_idr": position.entry_fee_idr, "exit_fee_idr": exit_fee,
            "pnl_before_exchange_fees_idr": gross_pnl,
            "entry_execution_friction_idr": max(0.0, (position.entry_fill_price - position.entry_reference_price) * position.quantity),
            "exit_execution_friction_idr": max(0.0, (exit_reference - exit_fill) * position.quantity),
            "execution_friction_cost_idr": max(0.0, (position.entry_fill_price - position.entry_reference_price) * position.quantity) + max(0.0, (exit_reference - exit_fill) * position.quantity),
            "net_pnl_idr": net_pnl,
            "bars_held": position.bars_held, "exit_reason": "end_of_data_forced_close",
        })
        position = None
        curve[-1]["equity_idr"] = cash
        curve[-1]["cash_idr"] = cash

    final_equity = cash
    net_pnl = final_equity - config.initial_capital_idr
    trade_pnls = [float(trade["net_pnl_idr"]) for trade in trades]
    wins = [value for value in trade_pnls if value > 0]
    losses = [value for value in trade_pnls if value < 0]
    gross_wins, gross_losses = sum(wins), abs(sum(losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (None if not wins else "infinite")
    max_consecutive_losses = 0
    current_losing_streak = 0
    for value in trade_pnls:
        current_losing_streak = current_losing_streak + 1 if value < 0 else 0
        max_consecutive_losses = max(max_consecutive_losses, current_losing_streak)

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "OFFLINE_BACKTEST_ONLY",
        "execution_enabled": False,
        "symbol": "SOLUSDT",
        "venue_reference": "Binance public spot candles; not Reku SOL/IDR executable prices",
        "dataset_sha256": dataset_sha256,
        "period_start": iso_utc(bars[0].open_time_ms),
        "period_end": iso_utc(bars[-1].close_time_ms),
        "bars_1m": len(bars),
        "bars_5m_complete": len(bars5),
        "bars_15m_complete": len(bars15),
        "config": asdict(config),
        "initial_equity_idr": config.initial_capital_idr,
        "final_equity_idr": final_equity,
        "net_pnl_idr": net_pnl,
        "net_return_pct": net_pnl / config.initial_capital_idr * 100.0,
        "closed_trades": len(trades),
        "win_rate_pct": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_fraction * 100.0,
        "max_consecutive_losses": max_consecutive_losses,
        "entry_fees_idr": total_entry_fees,
        "exit_fees_idr": total_exit_fees,
        "total_fees_idr": total_entry_fees + total_exit_fees,
        "total_execution_friction_idr": sum(float(trade.get("execution_friction_cost_idr", 0.0)) for trade in trades),
        "modeled_spread_bps_round_trip": config.spread_bps_round_trip,
        "modeled_slippage_bps_per_side": config.slippage_bps_per_side,
        "daily_loss_lock_events": lock_events,
        "trades": trades,
        "equity_curve": curve,
        "warnings": [
            "Historical simulation is not a profit forecast.",
            "Fees and slippage are configurable assumptions; replace with verified venue-specific values.",
            "Intrabar stop/target ordering is ambiguous in OHLC data; stop-first is assumed conservatively.",
            "This backtest does not model exchange queue priority, latency, partial fills, outages, tax, or Reku-specific price differences.",
            "The parameter set is a hypothesis and must be evaluated out-of-sample; results are not evidence of future returns.",
        ],
    }


def persist_ledger(db_path: str | Path, report: dict[str, Any], run_id: str) -> None:
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL,
                initial_equity_idr REAL NOT NULL,
                final_equity_idr REAL NOT NULL,
                net_pnl_idr REAL NOT NULL,
                report_json TEXT NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                run_id TEXT NOT NULL,
                trade_id TEXT NOT NULL,
                trade_index INTEGER NOT NULL,
                net_pnl_idr REAL NOT NULL,
                trade_json TEXT NOT NULL,
                PRIMARY KEY (run_id, trade_id),
                FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            )
        """)
        db.execute(
            "INSERT OR IGNORE INTO backtest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, report["schema_version"], datetime.now(timezone.utc).isoformat(),
                report.get("dataset_sha256", ""), report["initial_equity_idr"],
                report["final_equity_idr"], report["net_pnl_idr"],
                json.dumps(report, ensure_ascii=False, sort_keys=True),
            ),
        )
        for idx, trade in enumerate(report["trades"]):
            db.execute(
                "INSERT OR IGNORE INTO paper_trades VALUES (?, ?, ?, ?, ?)",
                (run_id, trade["trade_id"], idx, trade["net_pnl_idr"], json.dumps(trade, sort_keys=True)),
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="local 1m OHLCV CSV; see capture_history.py")
    parser.add_argument("--initial-capital-idr", type=float, default=300_000.0)
    parser.add_argument("--risk-fraction", type=float, default=0.005)
    parser.add_argument("--max-exposure-fraction", type=float, default=0.50)
    parser.add_argument("--daily-loss-fraction", type=float, default=0.03)
    parser.add_argument("--entry-fee-bps", type=float, default=10.555, help="assumption; verify current venue fee")
    parser.add_argument("--exit-fee-bps", type=float, default=31.555, help="assumption; verify current venue fee")
    parser.add_argument("--spread-bps-round-trip", type=float, default=10.0, help="assumed entry/exit spread cost")
    parser.add_argument("--slippage-bps-per-side", type=float, default=10.0, help="assumption per side")
    parser.add_argument("--minimum-net-edge-bps", type=float, default=25.0)
    parser.add_argument("--minimum-relative-volume", type=float, default=1.5)
    parser.add_argument("--stop-atr-multiple", type=float, default=1.0)
    parser.add_argument("--target-atr-multiple", type=float, default=1.5)
    parser.add_argument("--maximum-hold-bars", type=int, default=30)
    parser.add_argument("--report-path", default="", help="optional private local JSON report path")
    parser.add_argument("--db-path", default="~/.sol-sentinel/paper_ledger.sqlite3", help="local SQLite ledger; set empty string to disable")
    args = parser.parse_args(argv)
    config = BacktestConfig(
        initial_capital_idr=args.initial_capital_idr, risk_fraction=args.risk_fraction,
        max_exposure_fraction=args.max_exposure_fraction, daily_loss_fraction=args.daily_loss_fraction,
        entry_fee_bps=args.entry_fee_bps, exit_fee_bps=args.exit_fee_bps, spread_bps_round_trip=args.spread_bps_round_trip,
        slippage_bps_per_side=args.slippage_bps_per_side, minimum_net_edge_bps=args.minimum_net_edge_bps,
        minimum_relative_volume=args.minimum_relative_volume, stop_atr_multiple=args.stop_atr_multiple,
        target_atr_multiple=args.target_atr_multiple, maximum_hold_bars=args.maximum_hold_bars,
    )
    try:
        config.validate()
        data_path = Path(args.csv).expanduser()
        data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
        bars = read_bars_csv(data_path)
        report = backtest(bars, config, data_sha)
        params = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
        run_id = hashlib.sha256(f"{data_sha}:{params}:{SCHEMA_VERSION}".encode()).hexdigest()[:32]
        report["run_id"] = run_id
        if args.db_path:
            persist_ledger(args.db_path, report, run_id)
            report["ledger_path"] = str(Path(args.db_path).expanduser())
        if args.report_path:
            report_path = Path(args.report_path).expanduser()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        summary = {key: report[key] for key in (
            "mode", "execution_enabled", "run_id", "period_start", "period_end", "bars_1m",
            "initial_equity_idr", "final_equity_idr", "net_pnl_idr", "net_return_pct",
            "closed_trades", "win_rate_pct", "profit_factor", "max_drawdown_pct",
            "total_fees_idr", "total_execution_friction_idr", "daily_loss_lock_events", "ledger_path",
        ) if key in report}
        summary["warning"] = "OFFLINE BACKTEST ONLY — no orders placed; no future return implied."
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, csv.Error, sqlite3.Error) as exc:
        print(json.dumps({"mode": "OFFLINE_BACKTEST_ONLY", "execution_enabled": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
