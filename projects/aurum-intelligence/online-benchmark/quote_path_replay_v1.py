"""Execution-side bracket-path replay for Bid/Ask OHLC data, research-only.

Outputs gross quote-price PnL per instrument unit and explicit ambiguity bounds. It does not
infer commissions, slippage, swaps, or net account-currency PnL.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence


def _number(value: object, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(field + "_must_be_finite")
    try:
        out = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(field + "_must_be_finite") from exc
    if not out.is_finite():
        raise ValueError(field + "_must_be_finite")
    return out


def _utc(value: datetime | str, field: str) -> datetime:
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(field + "_must_be_iso8601") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(field + "_must_be_timezone_aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class BidAskBar:
    timestamp: datetime | str
    bid_open: Decimal | int | float | str
    bid_high: Decimal | int | float | str
    bid_low: Decimal | int | float | str
    bid_close: Decimal | int | float | str
    ask_open: Decimal | int | float | str
    ask_high: Decimal | int | float | str
    ask_low: Decimal | int | float | str
    ask_close: Decimal | int | float | str

    def normalized(self) -> dict[str, Any]:
        stamp = _utc(self.timestamp, "bar_timestamp")
        out: dict[str, Any] = {"timestamp": stamp}
        for side in ("bid", "ask"):
            values = {name: _number(getattr(self, f"{side}_{name}"), f"{side}_{name}")
                      for name in ("open", "high", "low", "close")}
            if any(value <= 0 for value in values.values()):
                raise ValueError(side + "_prices_must_be_positive")
            if values["high"] < max(values["open"], values["close"]):
                raise ValueError(side + "_high_below_open_or_close")
            if values["low"] > min(values["open"], values["close"]):
                raise ValueError(side + "_low_above_open_or_close")
            if values["high"] < values["low"]:
                raise ValueError(side + "_high_below_low")
            out.update({f"{side}_{key}": value for key, value in values.items()})
        for field in ("open", "high", "low", "close"):
            if out[f"ask_{field}"] < out[f"bid_{field}"]:
                raise ValueError("crossed_quote_at_" + field)
        return out


def simulate_bracket_trade(
    bars: Sequence[BidAskBar], *, decision_index: int, side: str,
    atr: Decimal | int | float | str, take_profit_atr: Decimal | int | float | str = "1.0",
    stop_loss_atr: Decimal | int | float | str = "1.0", horizon_bars: int = 60,
    expected_interval_seconds: int = 60,
) -> dict[str, Any]:
    """Enter after close[i], then replay executable-side TP/SL using Bid/Ask OHLC.

    Gap exits use the observed executable-side open. A same-bar TP/SL collision returns
    conservative and optimistic bounds, not a guessed path. PnL is gross price units per unit.
    """
    if side not in ("long", "short"):
        raise ValueError("side_must_be_long_or_short")
    if isinstance(decision_index, bool) or not isinstance(decision_index, int):
        raise ValueError("decision_index_must_be_integer")
    if isinstance(horizon_bars, bool) or not isinstance(horizon_bars, int) or horizon_bars < 1:
        raise ValueError("horizon_bars_must_be_positive_integer")
    if isinstance(expected_interval_seconds, bool) or not isinstance(expected_interval_seconds, int) or expected_interval_seconds < 1:
        raise ValueError("expected_interval_seconds_must_be_positive_integer")
    if decision_index < 0 or decision_index + horizon_bars >= len(bars):
        raise ValueError("decision_index_or_horizon_out_of_range")

    # Normalize only this path; dataset-wide quality belongs to the source audit.
    rows = [bar.normalized() for bar in bars[decision_index:decision_index + horizon_bars + 1]]
    if len(rows) != horizon_bars + 1:
        raise ValueError("decision_index_or_horizon_out_of_range")
    if any(b["timestamp"] <= a["timestamp"] for a, b in zip(rows, rows[1:])):
        raise ValueError("timestamps_must_be_strictly_increasing")
    atr_value, tp_mult, sl_mult = (_number(atr, "atr"),
                                   _number(take_profit_atr, "take_profit_atr"),
                                   _number(stop_loss_atr, "stop_loss_atr"))
    if atr_value <= 0 or tp_mult <= 0 or sl_mult <= 0:
        raise ValueError("atr_and_barrier_multipliers_must_be_positive")

    entry_index = 1
    entry_bar = rows[entry_index]
    direction = 1 if side == "long" else -1
    entry = entry_bar["ask_open"] if direction == 1 else entry_bar["bid_open"]
    target = entry + Decimal(direction) * tp_mult * atr_value
    stop = entry - Decimal(direction) * sl_mult * atr_value
    risk_distance = sl_mult * atr_value

    def finish(status: str, exit_index: int | None, exit_price: Decimal | None,
               label: int | None, fill_assumption: str, *,
               lower: Decimal | None = None, upper: Decimal | None = None) -> dict[str, Any]:
        pnl = None if exit_price is None else Decimal(direction) * (exit_price - entry)
        low_pnl, high_pnl = (pnl if lower is None else lower), (pnl if upper is None else upper)
        return {
            "schema_version": "1.0.0", "status": status, "side": side,
            "decision_index": decision_index, "entry_index": decision_index + entry_index,
            "exit_index": decision_index + exit_index if exit_index is not None else None,
            "entry_time_utc": entry_bar["timestamp"].isoformat(),
            "exit_time_utc": rows[exit_index]["timestamp"].isoformat() if exit_index is not None else None,
            "entry_quote_side_price": str(entry),
            "exit_quote_side_price": str(exit_price) if exit_price is not None else None,
            "exit_quote_side": "bid" if direction == 1 else "ask",
            "target_price": str(target), "stop_price": str(stop), "atr": str(atr_value),
            "take_profit_atr": str(tp_mult), "stop_loss_atr": str(sl_mult),
            "horizon_bars": horizon_bars, "outcome_label_tp_before_sl": label,
            "fill_assumption": fill_assumption,
            "gross_quote_pnl_price_units_per_instrument_unit": str(pnl) if pnl is not None else None,
            "gross_r_per_instrument_unit": str(pnl / risk_distance) if pnl is not None else None,
            "pnl_lower_bound_price_units_per_unit": str(low_pnl) if low_pnl is not None else None,
            "pnl_upper_bound_price_units_per_unit": str(high_pnl) if high_pnl is not None else None,
            "net_pnl": None, "net_pnl_status": "BLOCKED_COST_PROFILE_REQUIRED",
            "live_execution_authorized": False,
        }

    for offset in range(1, horizon_bars + 1):
        index = entry_index + offset - 1
        if index > entry_index:
            elapsed = (rows[index]["timestamp"] - rows[index - 1]["timestamp"]).total_seconds()
            if elapsed != expected_interval_seconds:
                return finish("incomplete_path_gap", None, None, None, "coverage_gap_fail_closed")
        row = rows[index]
        opened = row["bid_open"] if direction == 1 else row["ask_open"]
        high = row["bid_high"] if direction == 1 else row["ask_high"]
        low = row["bid_low"] if direction == 1 else row["ask_low"]
        if direction == 1:
            if opened >= target:
                return finish("take_profit_at_open", index, opened, 1, "favorable_open_gap_quote")
            if opened <= stop:
                return finish("stop_loss_at_open", index, opened, 0, "adverse_open_gap_quote")
            hit_tp, hit_sl = high >= target, low <= stop
        else:
            if opened <= target:
                return finish("take_profit_at_open", index, opened, 1, "favorable_open_gap_quote")
            if opened >= stop:
                return finish("stop_loss_at_open", index, opened, 0, "adverse_open_gap_quote")
            hit_tp, hit_sl = low <= target, high >= stop
        if hit_tp and hit_sl:
            target_pnl = Decimal(direction) * (target - entry)
            stop_pnl = Decimal(direction) * (stop - entry)
            return finish("ambiguous_same_bar", index, None, None, "bounds_only_ohlc_path_unknown",
                          lower=min(target_pnl, stop_pnl), upper=max(target_pnl, stop_pnl))
        if hit_tp:
            return finish("take_profit", index, target, 1, "barrier_touch_assumed_filled_at_target")
        if hit_sl:
            return finish("stop_loss", index, stop, 0, "barrier_touch_assumed_filled_at_stop")

    final_index = entry_index + horizon_bars - 1
    final = rows[final_index]
    close_price = final["bid_close"] if direction == 1 else final["ask_close"]
    return finish("horizon_timeout", final_index, close_price, 0, "horizon_close_quote")
