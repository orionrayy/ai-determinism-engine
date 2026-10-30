#!/usr/bin/env python3
"""Read-only, native INDODAX SOL/IDR strategy-family research.

Signals are evaluated at completed 1m closes and orders are simulated no earlier
than the next candle open. This module never connects to an account or writes a
paper/live ledger.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evaluate_splits import split_chronologically
from paper_engine import (
    BacktestConfig,
    Bar,
    aggregate_bars,
    atr_at,
    backtest,
    ema_series,
    read_bars_csv,
    rsi_at,
)

SCHEMA = "sol-sentinel-strategy-research-v1"
SEED = 20261030
STRATEGY_NAMES = (
    "trend_breakout",
    "trend_pullback",
    "range_mean_reversion",
    "volatility_expansion",
    "regime_adaptive",
)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    stop_atr: float
    target_atr: float
    max_hold: int
    target_mode: str = "atr"


SPECS = {
    "trend_breakout": StrategySpec("trend_breakout", 1.0, 1.6, 30),
    "trend_pullback": StrategySpec("trend_pullback", 1.1, 1.8, 45),
    "range_mean_reversion": StrategySpec("range_mean_reversion", 1.0, 1.0, 30, "mean"),
    "volatility_expansion": StrategySpec("volatility_expansion", 1.2, 2.0, 20),
    "regime_adaptive": StrategySpec("regime_adaptive", 1.1, 1.7, 35),
}


def _utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, timezone.utc).isoformat()


def audit_dataset(csv_path: str | Path, metadata_path: str | Path) -> tuple[list[Bar], dict[str, Any]]:
    path = Path(csv_path)
    meta_path = Path(metadata_path)
    if not path.is_file() or not meta_path.is_file():
        raise ValueError("native CSV and capture metadata sidecar are both required")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("data_status") != "OK":
        raise ValueError("capture sidecar data_status is not OK")
    if (meta.get("venue"), meta.get("pair"), meta.get("input_price_currency")) != ("INDODAX", "SOL/IDR", "IDR"):
        raise ValueError("dataset must be native INDODAX SOL/IDR in IDR")
    if meta.get("fx_conversion_applied") is not False:
        raise ValueError("native research refuses converted or ambiguous price data")
    if meta.get("dataset_sha256") != digest:
        raise ValueError("CSV SHA-256 differs from the capture sidecar")
    if meta.get("source_endpoint") != "https://indodax.com/tradingview/history_v2":
        raise ValueError("dataset source endpoint is not the declared INDODAX native-history endpoint")

    bars = read_bars_csv(path)
    timestamps = [bar.open_time_ms for bar in bars]
    duplicates = len(timestamps) - len(set(timestamps))
    gaps = sum(1 for a, b in zip(timestamps, timestamps[1:]) if b - a != 60_000)
    if not bars or duplicates or gaps:
        raise ValueError(f"native 1m series is empty or discontinuous: duplicates={duplicates}, gaps={gaps}")
    if meta.get("count") != len(bars):
        raise ValueError("CSV row count differs from sidecar count")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    volumes = [float(row["volume"]) for row in rows]
    positive = sorted(x for x in volumes if x > 0)
    p99 = positive[min(len(positive) - 1, math.ceil(0.99 * len(positive)) - 1)] if positive else None
    outliers = sum(1 for x in volumes if p99 is not None and x > p99 * 20.0)
    filters = meta.get("market_filters")
    if not isinstance(filters, dict):
        raise ValueError("native pair filters are required in the metadata sidecar")
    for field in ("price_tick", "quantity_step", "minimum_base_quantity"):
        try:
            value = Decimal(str(filters[field]))
        except (KeyError, InvalidOperation, TypeError):
            raise ValueError(f"missing or invalid native market filter: {field}")
        if not value.is_finite() or value <= 0:
            raise ValueError(f"native market filter {field} must be finite and positive")

    quality = {
        "venue": "INDODAX",
        "pair": "SOL/IDR",
        "quote_currency": "IDR",
        "source_endpoint": meta["source_endpoint"],
        "interval": "1m",
        "timezone": "UTC",
        "timestamp_unit": "milliseconds",
        "first_open_utc": _utc(bars[0].open_time_ms),
        "last_open_utc": _utc(bars[-1].open_time_ms),
        "last_close_utc": _utc(bars[-1].close_time_ms),
        "captured_at_utc": meta.get("captured_at_utc"),
        "records": len(bars),
        "file_size_bytes": len(payload),
        "sha256": digest,
        "missing_intervals": gaps,
        "duplicate_timestamps": duplicates,
        "invalid_ohlc_rows": 0,
        "zero_volume_rows": sum(1 for x in volumes if x == 0),
        "negative_volume_rows": sum(1 for x in volumes if x < 0),
        "volume_p99": p99,
        "volume_gt_20x_p99": outliers,
        "fx_conversion_applied": False,
        "sidecar_market_filters": meta.get("market_filters", {}),
        "same_run_quote": meta.get("market_quote"),
        "same_run_quote_status": meta.get("market_quote_status"),
    }
    return bars, quality


def _series_features(bars: Sequence[Bar], fast_period: int, slow_period: int) -> dict[str, list[float | None]]:
    closes = [bar.close for bar in bars]
    fast = ema_series(closes, fast_period)
    slow = ema_series(closes, slow_period)
    rsi_values = [rsi_at(closes, i, 14) for i in range(len(bars))]
    atr_values = [atr_at(bars, i, 14) for i in range(len(bars))]
    rv_values: list[float | None] = []
    high20: list[float | None] = []
    high10: list[float | None] = []
    mean20: list[float | None] = []
    std20: list[float | None] = []
    atr_ratio: list[float | None] = []
    for i, bar in enumerate(bars):
        prior = bars[max(0, i - 20):i]
        if len(prior) >= 20:
            baseline = statistics.fmean(x.volume for x in prior)
            rv_values.append(bar.volume / baseline if baseline > 0 else None)
            high20.append(max(x.high for x in prior))
            mean = statistics.fmean(x.close for x in prior)
            mean20.append(mean)
            std20.append(statistics.pstdev(x.close for x in prior))
        else:
            rv_values.append(None)
            high20.append(None)
            mean20.append(None)
            std20.append(None)
        prior10 = bars[max(0, i - 10):i]
        high10.append(max(x.high for x in prior10) if len(prior10) == 10 else None)
        atr_prior = [float(x) for x in atr_values[max(0, i - 30):i] if x is not None and x > 0]
        current_atr = atr_values[i]
        reference = statistics.fmean(atr_prior) if len(atr_prior) >= 10 else None
        atr_ratio.append(float(current_atr) / reference if current_atr and reference else None)
    return {
        "ema_fast": list(fast),
        "ema_slow": list(slow),
        "rsi": rsi_values,
        "atr": atr_values,
        "relative_volume": rv_values,
        "prior_high20": high20,
        "prior_high10": high10,
        "mean20": mean20,
        "std20": std20,
        "atr_ratio": atr_ratio,
    }


def prepare_features(bars: Sequence[Bar]) -> dict[str, Any]:
    bars5 = aggregate_bars(bars, 5)
    bars15 = aggregate_bars(bars, 15)
    return {
        "bars1": bars,
        "bars5": bars5,
        "bars15": bars15,
        "f1": _series_features(bars, 9, 20),
        "f5": _series_features(bars5, 8, 20),
        "f15": _series_features(bars15, 20, 50),
        "times5": [x.close_time_ms for x in bars5],
        "times15": [x.close_time_ms for x in bars15],
    }


def signal_at(name: str, features: dict[str, Any], index: int) -> dict[str, Any]:
    bars1, bars5, bars15 = features["bars1"], features["bars5"], features["bars15"]
    f1, f5, f15 = features["f1"], features["f5"], features["f15"]
    bar = bars1[index]
    i5 = bisect_right(features["times5"], bar.close_time_ms) - 1
    i15 = bisect_right(features["times15"], bar.close_time_ms) - 1
    if index < 50 or i5 < 50 or i15 < 50:
        return {"accepted": False, "reasons": ["indicator_warmup"], "regime": "UNKNOWN"}

    close15 = bars15[i15].close
    atr15 = f15["atr"][i15]
    ema20_15, ema50_15 = f15["ema_fast"][i15], f15["ema_slow"][i15]
    strength = ((ema20_15 - ema50_15) / atr15) if atr15 and atr15 > 0 else 0.0
    trend = ema20_15 > ema50_15 and close15 > ema20_15 and strength >= 0.15
    ranging = abs(strength) <= 0.35
    regime = "UPTREND" if trend else ("RANGE" if ranging else "OTHER")

    close5 = bars5[i5].close
    m5_up = f5["ema_fast"][i5] > f5["ema_slow"][i5] and close5 >= f5["ema_fast"][i5]
    rsi5 = f5["rsi"][i5]
    rsi1 = f1["rsi"][index]
    atr1 = f1["atr"][index]
    rv1 = f1["relative_volume"][index]
    prior_high20 = f1["prior_high20"][index]
    prior_high10 = f1["prior_high10"][index]
    mean20 = f1["mean20"][index]
    std20 = f1["std20"][index]
    atr_ratio = f1["atr_ratio"][index]
    zscore = ((bar.close - mean20) / std20) if mean20 is not None and std20 and std20 > 0 else None

    selected = name
    if name == "regime_adaptive":
        selected = "trend_pullback" if trend else ("range_mean_reversion" if ranging else "none")
    reasons: list[str] = []
    target_price = None
    if atr1 is None or atr1 <= 0:
        reasons.append("atr_unavailable")

    if selected == "trend_breakout":
        checks = (
            (trend, "15m_uptrend_missing"),
            (m5_up, "5m_momentum_missing"),
            (prior_high20 is not None and bar.close > prior_high20, "1m_breakout_missing"),
            (rv1 is not None and rv1 >= 1.35, "relative_volume_below_1_35"),
            (rsi1 is not None and 48.0 <= rsi1 <= 76.0, "rsi_outside_48_76"),
        )
        reasons.extend(reason for ok, reason in checks if not ok)
    elif selected == "trend_pullback":
        previous_close = bars1[index - 1].close
        previous_ema = f1["ema_fast"][index - 1]
        reclaimed = previous_close <= previous_ema and bar.close > f1["ema_fast"][index]
        checks = (
            (trend, "15m_uptrend_missing"),
            (m5_up, "5m_momentum_missing"),
            (reclaimed, "1m_pullback_reclaim_missing"),
            (rv1 is not None and rv1 >= 0.80, "relative_volume_below_0_80"),
            (rsi1 is not None and 43.0 <= rsi1 <= 69.0, "rsi_outside_43_69"),
        )
        reasons.extend(reason for ok, reason in checks if not ok)
    elif selected == "range_mean_reversion":
        checks = (
            (ranging, "range_regime_missing"),
            (zscore is not None and zscore <= -1.50, "lower_band_deviation_missing"),
            (rsi1 is not None and rsi1 <= 40.0, "rsi_not_oversold"),
            (atr_ratio is None or atr_ratio < 1.60, "volatility_expansion_blocks_reversion"),
            (bar.close >= bar.open or bar.close > bars1[index - 1].close, "reversal_confirmation_missing"),
        )
        reasons.extend(reason for ok, reason in checks if not ok)
        if mean20 is not None:
            target_price = mean20
    elif selected == "volatility_expansion":
        checks = (
            (atr_ratio is not None and atr_ratio >= 1.25, "atr_expansion_below_1_25"),
            (rv1 is not None and rv1 >= 1.70, "relative_volume_below_1_70"),
            (prior_high10 is not None and bar.close > prior_high10, "10_bar_breakout_missing"),
            (bar.close > bar.open, "positive_candle_missing"),
            (rsi1 is not None and rsi1 < 82.0, "momentum_exhaustion"),
            (trend or m5_up, "directional_context_missing"),
        )
        reasons.extend(reason for ok, reason in checks if not ok)
    elif selected == "none":
        reasons.append("no_eligible_market_regime")
    else:
        raise ValueError(f"unknown strategy {name}")

    if reasons:
        return {"accepted": False, "reasons": reasons, "regime": regime}
    spec = SPECS[name]
    return {
        "accepted": True,
        "reasons": [],
        "regime": regime,
        "atr": float(atr1),
        "target_price": target_price,
        "stop_atr": spec.stop_atr,
        "target_atr": spec.target_atr,
        "max_hold": spec.max_hold,
        "target_mode": spec.target_mode,
        "selected_substrategy": selected,
    }


def _mark_equity(cash: float, position: dict[str, Any] | None, close: float, adverse_bps: float, exit_fee_bps: float) -> float:
    if position is None:
        return cash
    liquidation = close * (1.0 - adverse_bps / 10_000.0)
    fee = position["quantity"] * liquidation * exit_fee_bps / 10_000.0
    return cash + position["quantity"] * liquidation - fee


def simulate_segment(
    name: str,
    bars: Sequence[Bar],
    features: dict[str, Any],
    start: int,
    end: int,
    *,
    entry_fee_bps: float,
    exit_fee_bps: float,
    spread_bps_round_trip: float,
    slippage_bps_per_side: float,
    min_notional_idr: float,
    initial_capital: float = 300_000.0,
    risk_fraction: float = 0.005,
    max_exposure_fraction: float = 0.50,
    daily_loss_fraction: float = 0.03,
    quantity_step: str = "0.00000001",
    minimum_base_quantity: str = "0.00000001",
    maximum_drawdown_fraction: float = 0.10,
) -> dict[str, Any]:
    if not (0 <= start < end <= len(bars)):
        raise ValueError("invalid segment boundaries")
    if not 0 < risk_fraction < 1:
        raise ValueError("risk_fraction must be in (0, 1)")
    if not 0 < maximum_drawdown_fraction < 1:
        raise ValueError("maximum_drawdown_fraction must be in (0, 1)")
    try:
        qty_step_dec = Decimal(str(quantity_step))
        min_qty_dec = Decimal(str(minimum_base_quantity))
    except InvalidOperation as exc:
        raise ValueError("invalid quantity precision from native market metadata") from exc
    if not qty_step_dec.is_finite() or qty_step_dec <= 0 or not min_qty_dec.is_finite() or min_qty_dec <= 0:
        raise ValueError("quantity step and minimum base quantity must be positive")
    spec = SPECS[name]
    adverse = spread_bps_round_trip / 2.0 + slippage_bps_per_side
    friction_rate = (entry_fee_bps + exit_fee_bps + spread_bps_round_trip + 2 * slippage_bps_per_side) / 10_000.0
    cash = initial_capital
    position: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    rejects: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    entry_attempts = 0
    size_rejections = 0
    signal_evaluations = 0
    accepted_signals = 0
    rejection_by_utc_hour: dict[str, Counter[str]] = defaultdict(Counter)
    daily_locked = False
    portfolio_locked = False
    daily_start = initial_capital
    current_day = datetime.fromtimestamp(bars[start].open_time_ms / 1000, timezone.utc).date()
    peak = initial_capital
    max_drawdown = 0.0
    max_drawdown_idr = 0.0
    turnover = 0.0
    equity_curve: list[float] = [initial_capital]

    def close_position(pos: dict[str, Any], reference: float, bar: Bar, reason: str) -> None:
        nonlocal cash, turnover
        exit_fill = reference * (1.0 - adverse / 10_000.0)
        exit_fee = pos["quantity"] * exit_fill * exit_fee_bps / 10_000.0
        cash += pos["quantity"] * exit_fill - exit_fee
        gross = (reference - pos["entry_reference"]) * pos["quantity"]
        spread_cost = pos["quantity"] * (pos["entry_reference"] + reference) * spread_bps_round_trip / 2.0 / 10_000.0
        slippage_cost = pos["quantity"] * (pos["entry_reference"] + reference) * slippage_bps_per_side / 10_000.0
        net = gross - pos["entry_fee"] - exit_fee - spread_cost - slippage_cost
        turnover += pos["quantity"] * reference
        trades.append({
            "entry_time": _utc(pos["entry_time"]),
            "exit_time": _utc(bar.close_time_ms),
            "entry_regime": pos["regime"],
            "selected_substrategy": pos["selected_substrategy"],
            "entry_notional_idr": pos["entry_notional"],
            "risk_cash_idr": pos["risk_cash"],
            "r_multiple": net / pos["risk_cash"] if pos["risk_cash"] > 0 else None,
            "quantity_sol": pos["quantity"],
            "bars_held": pos["bars_held"],
            "exit_reason": reason,
            "gross_pnl_idr": gross,
            "fees_idr": pos["entry_fee"] + exit_fee,
            "spread_cost_idr": spread_cost,
            "slippage_cost_idr": slippage_cost,
            "execution_friction_idr": spread_cost + slippage_cost,
            "net_pnl_idr": net,
            "net_return_on_entry_notional_bps": net / pos["entry_notional"] * 10_000.0,
        })

    for i in range(start, end):
        bar = bars[i]
        day = datetime.fromtimestamp(bar.open_time_ms / 1000, timezone.utc).date()
        if day != current_day:
            current_day = day
            daily_start = _mark_equity(cash, position, bar.open, adverse, exit_fee_bps)
            daily_locked = False

        if pending is not None and position is None:
            entry_attempts += 1
            reference = bar.open
            fill = reference * (1.0 + adverse / 10_000.0)
            risk_cash = cash * risk_fraction
            stop_distance = pending["atr"] * pending["stop_atr"]
            risk_per_unit = stop_distance + fill * friction_rate
            qty_risk = risk_cash / risk_per_unit if risk_per_unit > 0 else 0.0
            qty_exposure = cash * max_exposure_fraction / fill
            qty_cash = cash / (fill * (1.0 + entry_fee_bps / 10_000.0))
            raw_quantity = min(qty_risk, qty_exposure, qty_cash)
            quantity_steps = (Decimal(str(raw_quantity)) / qty_step_dec).to_integral_value(rounding=ROUND_DOWN)
            quantity = float(quantity_steps * qty_step_dec)
            notional = quantity * fill
            if (math.isfinite(notional) and quantity >= float(min_qty_dec)
                    and notional >= min_notional_idr):
                entry_fee = notional * entry_fee_bps / 10_000.0
                cash -= notional + entry_fee
                if pending["target_mode"] == "mean" and pending.get("target_price") and pending["target_price"] > fill:
                    target = float(pending["target_price"])
                else:
                    target = fill + pending["atr"] * pending["target_atr"]
                position = {
                    "entry_reference": reference,
                    "entry_fill": fill,
                    "entry_time": bar.open_time_ms,
                    "entry_fee": entry_fee,
                    "entry_notional": notional,
                    "quantity": quantity,
                    "stop": fill - stop_distance,
                    "target": target,
                    "target_atr": pending["target_atr"],
                    "max_hold": pending["max_hold"],
                    "bars_held": 0,
                    "risk_cash": risk_cash,
                    "regime": pending["regime"],
                    "selected_substrategy": pending["selected_substrategy"],
                }
                regime_counts[pending["regime"]] += 1
                turnover += notional
            else:
                size_rejections += 1
                rejects["minimum_order_notional_or_sizing"] += 1
            pending = None

        if position is not None:
            position["bars_held"] += 1
            stop_hit = bar.low <= position["stop"]
            target_hit = bar.high >= position["target"]
            if stop_hit:
                close_position(position, min(bar.open, position["stop"]), bar, "stop_loss" if not target_hit else "stop_and_target_same_bar_stop_first")
                position = None
            elif target_hit:
                close_position(position, position["target"], bar, "take_profit")
                position = None
            elif position["bars_held"] >= position["max_hold"]:
                close_position(position, bar.close, bar, "time_stop")
                position = None

        equity = _mark_equity(cash, position, bar.close, adverse, exit_fee_bps)
        if equity <= daily_start * (1.0 - daily_loss_fraction):
            daily_locked = True
        peak = max(peak, equity)
        dd_idr = peak - equity
        dd = dd_idr / peak if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)
        max_drawdown_idr = max(max_drawdown_idr, dd_idr)
        if dd >= maximum_drawdown_fraction:
            portfolio_locked = True
        equity_curve.append(equity)

        if i < end - 1 and position is None and pending is None and not daily_locked and not portfolio_locked:
            signal = signal_at(name, features, i)
            signal_evaluations += 1
            if signal["accepted"]:
                accepted_signals += 1
                pending = signal
            else:
                hour = datetime.fromtimestamp(bar.close_time_ms / 1000, timezone.utc).strftime("%H")
                for reason in signal["reasons"]:
                    rejects[reason] += 1
                    rejection_by_utc_hour[hour][reason] += 1

    if position is not None:
        close_position(position, bars[end - 1].close, bars[end - 1], "segment_end_forced_close")
        position = None
        equity_curve[-1] = cash
        final_dd_idr = max(0.0, peak - cash)
        max_drawdown_idr = max(max_drawdown_idr, final_dd_idr)
        if peak > 0:
            max_drawdown = max(max_drawdown, final_dd_idr / peak)

    pnl = [float(x["net_pnl_idr"]) for x in trades]
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor: float | str | None = (
        gross_profit / gross_loss if gross_loss > 0 else ("infinite" if gross_profit > 0 else None)
    )
    streak = 0
    max_streak = 0
    for value in pnl:
        streak = streak + 1 if value < 0 else 0
        max_streak = max(max_streak, streak)
    dd_pct = max_drawdown * 100.0
    result = {
        "bars": end - start,
        "period_start": _utc(bars[start].open_time_ms),
        "period_end": _utc(bars[end - 1].close_time_ms),
        "trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_pct": (len(wins) / len(trades) * 100.0) if trades else None,
        "average_win_idr": statistics.fmean(wins) if wins else None,
        "average_loss_idr": statistics.fmean(losses) if losses else None,
        "reward_risk_ratio": (
            statistics.fmean(wins) / abs(statistics.fmean(losses))
            if wins and losses and statistics.fmean(losses) != 0 else None
        ),
        "gross_pnl_idr": sum(float(x["gross_pnl_idr"]) for x in trades),
        "net_pnl_idr": sum(pnl),
        "total_fees_idr": sum(float(x["fees_idr"]) for x in trades),
        "spread_cost_idr": sum(float(x["spread_cost_idr"]) for x in trades),
        "slippage_cost_idr": sum(float(x["slippage_cost_idr"]) for x in trades),
        "execution_friction_idr": sum(float(x["execution_friction_idr"]) for x in trades),
        "net_expectancy_per_trade_idr": statistics.fmean(pnl) if pnl else None,
        "net_expectancy_bps": (
            statistics.fmean(float(x["net_return_on_entry_notional_bps"]) for x in trades) if trades else None
        ),
        "profit_factor": profit_factor,
        "max_drawdown_pct": dd_pct,
        "max_drawdown_idr": max_drawdown_idr,
        "recovery_factor": sum(pnl) / max_drawdown_idr if max_drawdown_idr > 0 else None,
        "max_consecutive_losses": max_streak,
        "average_holding_minutes": statistics.fmean(x["bars_held"] for x in trades) if trades else None,
        "exposure_minutes": sum(x["bars_held"] for x in trades),
        "turnover_idr": turnover,
        "signal_evaluations": signal_evaluations,
        "accepted_signals": accepted_signals,
        "entry_order_attempts": entry_attempts,
        "size_rejections": size_rejections,
        "signal_rejections": dict(sorted(rejects.items())),
        "signal_rejections_by_utc_hour": {
            hour: dict(sorted(counts.items())) for hour, counts in sorted(rejection_by_utc_hour.items())
        },
        "daily_loss_locked": daily_locked,
        "portfolio_drawdown_locked": portfolio_locked,
        "trades_by_regime": {},
        "closed_trade_details": trades,
        "final_equity_idr": cash,
        "net_return_pct": (cash / initial_capital - 1.0) * 100.0,
        "sample_status": "INSUFFICIENT_SAMPLE" if len(trades) < 30 else "SAMPLE_30_PLUS",
        "risk_fraction": risk_fraction,
    }
    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        by_regime[trade["entry_regime"]].append(trade)
    for regime, regime_trades in sorted(by_regime.items()):
        vals = [float(x["net_pnl_idr"]) for x in regime_trades]
        regime_wins = sum(x for x in vals if x > 0)
        regime_losses = abs(sum(x for x in vals if x < 0))
        result["trades_by_regime"][regime] = {
            "trades": len(vals),
            "net_pnl_idr": sum(vals),
            "net_expectancy_idr": statistics.fmean(vals),
            "profit_factor": regime_wins / regime_losses if regime_losses else ("infinite" if regime_wins else None),
        }
    return result


def block_bootstrap_ci(values: Sequence[float], *, seed: int = SEED, iterations: int = 2000, block_size: int = 3) -> dict[str, Any]:
    if len(values) < 30:
        return {"status": "INSUFFICIENT_SAMPLE", "sample_size": len(values), "ci95": None}
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(iterations):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample.extend(values[start:min(n, start + block_size)])
        means.append(statistics.fmean(sample[:n]))
    means.sort()
    return {
        "status": "ESTIMATED_BLOCK_BOOTSTRAP_CI",
        "sample_size": n,
        "block_size_trades": block_size,
        "iterations": iterations,
        "ci95": [means[int(0.025 * (iterations - 1))], means[int(0.975 * (iterations - 1))]],
        "seed": seed,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _mc_target(oos: dict[str, Any], *, initial: float = 300_000.0, target: float = 3_500_000.0,
               days: float, simulations: int = 5000) -> dict[str, Any]:
    trades = oos["closed_trade_details"]
    if len(trades) < 30 or days <= 0:
        return {"status": "INSUFFICIENT_SAMPLE", "oos_trades": len(trades), "probability_target": None}
    r_values = [float(x["r_multiple"]) for x in trades if x.get("r_multiple") is not None and math.isfinite(float(x["r_multiple"]))]
    if len(r_values) < 30:
        return {"status": "INSUFFICIENT_SAMPLE", "oos_trades": len(r_values), "probability_target": None}
    trades_per_day = len(r_values) / days
    horizon_trades = max(1, int(round(trades_per_day)))
    risk_grid = (0.0025, 0.005, 0.01, 0.02, 0.05)
    rows = []
    for grid_index, risk in enumerate(risk_grid):
        rng = random.Random(SEED + grid_index)
        terminal: list[float] = []
        drawdowns: list[float] = []
        hit_target = below_initial = near_ruin = below_minimum = 0
        for _ in range(simulations):
            equity = initial
            peak = initial
            day_start = initial
            max_dd = 0.0
            sample: list[float] = []
            while len(sample) < horizon_trades:
                start = rng.randrange(len(r_values))
                sample.extend(r_values[start:min(len(r_values), start + 3)])
            for r_multiple in sample[:horizon_trades]:
                equity *= max(0.0, 1.0 + risk * r_multiple)
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 1.0)
                if equity <= day_start * 0.97 or max_dd >= 0.10:
                    break
            terminal.append(equity)
            drawdowns.append(max_dd)
            hit_target += equity >= target
            below_initial += equity < initial
            near_ruin += equity < initial * 0.20
            below_minimum += equity < 25_000.0
        ordered = sorted(terminal)
        dd_sorted = sorted(drawdowns)
        rows.append({
            "risk_fraction_per_trade": risk,
            "trades_per_24h_estimate": horizon_trades,
            "probability_target": hit_target / simulations,
            "probability_target_ci95_wilson": wilson_interval(hit_target, simulations),
            "probability_below_initial": below_initial / simulations,
            "probability_near_ruin_below_20pct_initial": near_ruin / simulations,
            "probability_below_pro_minimum_order_capital": below_minimum / simulations,
            "expected_terminal_equity_idr": statistics.fmean(terminal),
            "median_terminal_equity_idr": statistics.median(terminal),
            "p05_terminal_equity_idr": ordered[int(0.05 * (simulations - 1))],
            "p95_terminal_equity_idr": ordered[int(0.95 * (simulations - 1))],
            "median_max_drawdown_pct": statistics.median(dd_sorted) * 100.0,
            "p95_max_drawdown_pct": dd_sorted[int(0.95 * (simulations - 1))] * 100.0,
            "best_simulated_terminal_equity_idr": ordered[-1],
            "worst_simulated_terminal_equity_idr": ordered[0],
            "simulations": simulations,
            "block_size_trades": 3,
            "daily_loss_stop": 0.03,
            "max_drawdown_stop": 0.10,
            "sampling": "moving_block_bootstrap_of_OOS_net_R_multiples",
        })
    return {
        "status": "MODEL_CONDITIONAL_BOOTSTRAP_ESTIMATE",
        "candidate_oos_trades": len(r_values),
        "oos_days": days,
        "trades_per_day_estimate": trades_per_day,
        "risk_grid": rows,
        "limitations": [
            "This is conditional on one strategy's OOS sample and the specified execution model.",
            "Risk scaling assumes net R multiples scale linearly; order-book market impact is not measured.",
            "One 30-day sample cannot represent every market regime or structural break.",
            "A modeled path is not a forecast or guarantee.",
        ],
    }


def _candidate_gate(results: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    for scenario in ("pro_taker_base", "pro_taker_adverse"):
        segments = results.get(scenario, {})
        dev = segments.get("development", {})
        val = segments.get("validation", {})
        oos = segments.get("out_of_sample", {})
        if oos.get("trades", 0) < 30:
            reasons.append(f"{scenario}:oos_trades_below_30")
        for label, metrics in (("development", dev), ("validation", val), ("out_of_sample", oos)):
            if metrics.get("net_expectancy_per_trade_idr") is None or metrics["net_expectancy_per_trade_idr"] <= 0:
                reasons.append(f"{scenario}:{label}:nonpositive_or_undefined_net_expectancy")
            pf = metrics.get("profit_factor")
            if not isinstance(pf, (int, float)) or pf <= 1.10:
                reasons.append(f"{scenario}:{label}:profit_factor_not_above_1_10")
            if metrics.get("max_drawdown_pct", 100.0) > 10.0:
                reasons.append(f"{scenario}:{label}:drawdown_above_10pct")
        regime_rows = oos.get("trades_by_regime", {})
        robust_regimes = sum(1 for row in regime_rows.values()
                             if row.get("trades", 0) >= 10 and row.get("net_expectancy_idr", 0) > 0)
        if robust_regimes < 2:
            reasons.append(f"{scenario}:fewer_than_two_positive_regimes_with_10_trades")
    return ("PAPER_CANDIDATE" if not reasons else "RESEARCH_ONLY_INSUFFICIENT_EVIDENCE", sorted(set(reasons)))


def _config(entry_fee: float, exit_fee: float, spread: float, slippage: float, min_notional: float) -> BacktestConfig:
    return BacktestConfig(
        initial_capital_idr=300_000.0,
        quote_to_idr_rate=1.0,
        quote_to_idr_source="native INDODAX SOL/IDR; FX disabled",
        input_venue="INDODAX",
        input_symbol="SOL/IDR",
        input_price_currency="IDR",
        fee_model_name="INDODAX_PRO_ALL_IN_TAKER_ASSUMPTION",
        fee_model_source="Official INDODAX all-in Pro table effective 2026-10-01; personal account route not verified",
        risk_fraction=0.005,
        max_exposure_fraction=0.50,
        daily_loss_fraction=0.03,
        entry_fee_bps=entry_fee,
        exit_fee_bps=exit_fee,
        spread_bps_round_trip=spread,
        slippage_bps_per_side=slippage,
        minimum_net_edge_bps=25.0,
        minimum_relative_volume=1.5,
        minimum_order_notional_idr=min_notional,
        stop_atr_multiple=1.0,
        target_atr_multiple=1.5,
        maximum_hold_bars=30,
    )


def run_research(
    bars: Sequence[Bar],
    dataset: dict[str, Any],
    *,
    entry_fee_bps: float = 20.555,
    exit_fee_bps: float = 41.555,
    min_notional_idr: float = 25_000.0,
    initial_capital_idr: float = 300_000.0,
) -> dict[str, Any]:
    n = len(bars)
    if n < 3000:
        raise ValueError("at least 3,000 contiguous 1m bars required for 60/20/20 chronology")
    if initial_capital_idr != 300_000.0:
        raise ValueError("the declared v1 research sizing scenario is fixed at Rp300,000")
    splits = split_chronologically(bars)
    n_dev, n_val = len(splits["development"]), len(splits["validation"])
    bounds = {
        "development": (0, n_dev),
        "validation": (n_dev, n_dev + n_val),
        "out_of_sample": (n_dev + n_val, n),
    }
    days_oos = (n - n_dev - n_val) / 1440.0
    features = prepare_features(bars)
    market_filters = dataset.get("sidecar_market_filters") or {}
    quantity_step = str(market_filters["quantity_step"])
    minimum_base_quantity = str(market_filters["minimum_base_quantity"])
    price_tick = str(market_filters["price_tick"])
    quote = dataset.get("same_run_quote") or {}
    try:
        snapshot_spread = float(quote.get("spread_bps"))
        if not math.isfinite(snapshot_spread) or snapshot_spread <= 0:
            raise ValueError
        snapshot_source = "same_run_INDODAX_public_depth_snapshot_point_not_historical_distribution"
    except (TypeError, ValueError, OverflowError):
        snapshot_spread = 25.6669
        snapshot_source = "fallback_historical_point_not_historical_distribution"

    scenarios = {
        "pro_taker_snapshot": {"spread": snapshot_spread, "slippage": 5.0},
        "pro_taker_base": {"spread": max(snapshot_spread, 25.6669), "slippage": 10.0},
        "pro_taker_adverse": {"spread": max(snapshot_spread * 1.75, 40.0), "slippage": 15.0},
        "pro_taker_severe": {"spread": max(snapshot_spread * 2.50, 60.0), "slippage": 25.0},
    }
    baseline = {}
    candidates: dict[str, Any] = {}
    for scenario_name, cost in scenarios.items():
        config = _config(entry_fee_bps, exit_fee_bps, cost["spread"], cost["slippage"], min_notional_idr)
        base_report = backtest(bars, config, dataset_sha256=dataset["sha256"])
        baseline[scenario_name] = {
            "strategy_id": "baseline_v1_locked",
            "input_parameters_unchanged": True,
            "trades": base_report.get("closed_trades", 0),
            "signal_evaluations": base_report.get("signal_evaluations"),
            "signal_candidates": base_report.get("signal_candidates"),
            "signal_rejection_counts": base_report.get("signal_rejection_counts"),
            "signal_rejection_combinations": base_report.get("signal_rejection_combinations"),
            "entry_order_attempts": base_report.get("entry_order_attempts"),
            "entry_size_rejections": base_report.get("entry_size_rejections"),
            "net_pnl_idr": base_report.get("net_pnl_idr"),
            "net_expectancy_per_trade_idr": base_report.get("expectancy_net_per_trade_idr"),
            "profit_factor": base_report.get("profit_factor"),
            "max_drawdown_pct": base_report.get("max_drawdown_pct"),
            "total_fees_idr": base_report.get("total_fees_idr"),
            "execution_friction_idr": base_report.get("total_execution_friction_idr"),
            "sample_status": "INSUFFICIENT_SAMPLE" if base_report.get("closed_trades", 0) < 30 else "SAMPLE_30_PLUS",
        }
        candidates[scenario_name] = {}
        for name in STRATEGY_NAMES:
            segments = {}
            for segment, (start, end) in bounds.items():
                segments[segment] = simulate_segment(
                    name, bars, features, start, end,
                    entry_fee_bps=entry_fee_bps,
                    exit_fee_bps=exit_fee_bps,
                    spread_bps_round_trip=cost["spread"],
                    slippage_bps_per_side=cost["slippage"],
                    min_notional_idr=min_notional_idr,
                    quantity_step=quantity_step,
                    minimum_base_quantity=minimum_base_quantity,
                )
            candidates[scenario_name][name] = segments

    statuses = {}
    for name in STRATEGY_NAMES:
        by_scenario = {scenario: {segment: candidates[scenario][name][segment] for segment in bounds}
                       for scenario in candidates}
        status, reasons = _candidate_gate(by_scenario)
        oos_trades = by_scenario["pro_taker_base"]["out_of_sample"]["closed_trade_details"]
        returns_bps = [float(x["net_return_on_entry_notional_bps"]) for x in oos_trades]
        statuses[name] = {
            "status": status,
            "rejection_reasons": reasons,
            "base_oos_expectancy_block_bootstrap_95ci_bps": block_bootstrap_ci(returns_bps),
        }
    qualified = [name for name, status in statuses.items() if status["status"] == "PAPER_CANDIDATE"]
    target = (
        _mc_target(candidates["pro_taker_base"][qualified[0]]["out_of_sample"], days=days_oos)
        if qualified else
        {
            "status": "INSUFFICIENT_SAMPLE",
            "probability_estimate": None,
            "reason": "No candidate passed the predeclared validation/OOS net-expectancy, trade-count, regime and drawdown gate.",
        }
    )
    required_hurdle = {
        "starting_equity_idr": 300_000,
        "target_equity_idr": 3_500_000,
        "required_multiple": 3_500_000 / 300_000,
        "required_return_pct": (3_500_000 / 300_000 - 1.0) * 100.0,
        "required_compounded_net_return_per_trade": {
            str(trades): (3_500_000 / 300_000) ** (1.0 / trades) - 1.0
            for trades in (10, 25, 50, 100)
        },
        "probability": "INSUFFICIENT_SAMPLE" if not qualified else target.get("risk_grid"),
    }
    return {
        "schema_version": SCHEMA,
        "mode": "OFFLINE_NATIVE_INDODAX_STRATEGY_RESEARCH",
        "execution_enabled": False,
        "paper_state_accessed": False,
        "dataset": dataset,
        "chronological_splits": {
            "ratios": {"development": 0.60, "validation": 0.20, "out_of_sample": 0.20},
            "counts": {key: end - start for key, (start, end) in bounds.items()},
            "boundaries": {
                key: {"start_utc": _utc(bars[start].open_time_ms), "end_utc": _utc(bars[end - 1].close_time_ms)}
                for key, (start, end) in bounds.items()
            },
            "indicator_warmup_uses_only_prior_observations": True,
            "parameters_selected_before_final_oos": True,
        },
        "execution_model": {
            "fee_model": "INDODAX Pro IDR all-in taker assumption",
            "entry_fee_bps": entry_fee_bps,
            "exit_fee_bps": exit_fee_bps,
            "fee_sum_bps": entry_fee_bps + exit_fee_bps,
            "minimum_order_notional_idr": min_notional_idr,
            "price_tick_idr": price_tick,
            "quantity_step_sol": quantity_step,
            "minimum_base_quantity_sol": minimum_base_quantity,
            "quantity_is_rounded_down_to_exchange_step": True,
            "market_order_price_tick_rounding": "not separately applied; fill is price-impact modeled from OHLC reference, not a limit order",
            "risk_fraction": 0.005,
            "max_exposure_fraction": 0.50,
            "daily_loss_fraction": 0.03,
            "max_drawdown_gate_pct": 10.0,
            "slippage_is_assumed_not_measured": True,
            "maker_queue_or_fill_probability_modeled": False,
            "partial_fill_and_market_impact_modelled": False,
            "spread_snapshot_source": snapshot_source,
            "cost_scenarios": {
                key: {
                    "spread_bps_round_trip": val["spread"],
                    "slippage_bps_per_side": val["slippage"],
                    "total_modeled_round_trip_cost_bps": entry_fee_bps + exit_fee_bps + val["spread"] + 2 * val["slippage"],
                } for key, val in scenarios.items()
            },
        },
        "locked_baseline_v1": baseline,
        "candidate_results": candidates,
        "candidate_selection": statuses,
        "promoted_candidate": qualified[0] if qualified else None,
        "target_feasibility": {
            "hurdle": required_hurdle,
            "monte_carlo": target,
            "simulated_only_when_candidate_passes_oos_gate": True,
        },
        "reproducibility": {
            "seed": SEED,
            "indicator_library": "stdlib + existing paper_engine",
            "no_randomized_parameter_search": True,
            "no_future_data_in_features": True,
            "no_active_state_or_exchange_api_access": True,
        },
        "warnings": [
            "The depth snapshot is a single point estimate, not an historical spread distribution.",
            "Taker fees, spread and slippage are modeled assumptions; no realized fills exist to calibrate them.",
            "OHLC stop/target collisions assume stop-first; queue priority, partial fills, latency and market impact are not modeled.",
            "A 30-day window and one short OOS segment may be insufficient to establish a durable edge.",
            "No strategy is promoted by trade frequency alone.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--metadata", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--entry-fee-bps", type=float, default=20.555)
    parser.add_argument("--exit-fee-bps", type=float, default=41.555)
    parser.add_argument("--minimum-order-notional-idr", type=float, default=25_000.0)
    args = parser.parse_args(argv)
    metadata_path = args.metadata or (args.csv + ".meta.json")
    try:
        bars, quality = audit_dataset(args.csv, metadata_path)
        report = run_research(
            bars,
            quality,
            entry_fee_bps=args.entry_fee_bps,
            exit_fee_bps=args.exit_fee_bps,
            min_notional_idr=args.minimum_order_notional_idr,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        summary = {
            "schema_version": SCHEMA,
            "execution_enabled": False,
            "dataset_sha256": quality["sha256"],
            "bars": quality["records"],
            "splits": report["chronological_splits"]["counts"],
            "locked_baseline_trades": {
                key: value["trades"] for key, value in report["locked_baseline_v1"].items()
            },
            "candidate_status": report["candidate_selection"],
            "promoted_candidate": report["promoted_candidate"],
            "target_monte_carlo_status": report["target_feasibility"]["monte_carlo"]["status"],
            "report_path": str(output),
        }
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, csv.Error, json.JSONDecodeError) as exc:
        print(json.dumps({
            "schema_version": SCHEMA,
            "mode": "OFFLINE_NATIVE_INDODAX_STRATEGY_RESEARCH",
            "execution_enabled": False,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }, ensure_ascii=False), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
