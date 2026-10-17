#!/usr/bin/env python3
"""P0 chronological TP-before-SL research protocol. Stdlib wrapper over v1.3 calibration diagnostics.

This is a research evaluator, not an execution simulator. Same-bar barrier collisions are
ambiguous and excluded from binary scoring. Historical baseline forecasts are strictly
prequential: only outcomes matured by a decision index may affect that decision's prior.
"""
from __future__ import annotations

import heapq
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "temporal-calibration-v1.3"
if str(_CALIBRATION_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIBRATION_DIR))
from calibration_diagnostics import (  # noqa: E402
    apply_platt_scaler,
    fit_platt_scaler,
    moving_block_bootstrap_intervals,
    probability_metrics,
)

PROTOCOL_VERSION = "1.4.0"
PROTOCOL_NAME = "purged-chronological-tp-before-sl"
TARGET = "TP reached before SL within fixed horizon; timeout is non-event; same-bar TP/SL is ambiguous"
WARMUP_BARS = 60
ATR_PERIOD = 14
TP_ATR = 1.5
SL_ATR = 1.0
MAX_HORIZON_BARS = 12
MIN_MATURED_LABELS = 100
CALIBRATION_FRACTION_START = 0.70
HOLDOUT_FRACTION_START = 0.80
COST_SENSITIVITY_R = (0.0, 0.025, 0.05, 0.10)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(name + "_must_be_finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + "_must_be_finite") from exc
    if not math.isfinite(result):
        raise ValueError(name + "_must_be_finite")
    return result


def _true_range(rows: Sequence[Mapping[str, Any]], index: int) -> float:
    row = rows[index]
    high, low = float(row["high"]), float(row["low"])
    if index == 0:
        return high - low
    previous_close = float(rows[index - 1]["close"])
    return max(high - low, abs(high - previous_close), abs(low - previous_close))


def _atr_at_decision(rows: Sequence[Mapping[str, Any]], decision_index: int, period: int) -> float | None:
    if decision_index + 1 < period:
        return None
    start = decision_index - period + 1
    values = [_true_range(rows, i) for i in range(start, decision_index + 1)]
    value = statistics.fmean(values)
    return value if math.isfinite(value) and value > 0 else None


def label_tp_before_sl(
    rows: Sequence[Mapping[str, Any]],
    decision_index: int,
    side: int,
    *,
    atr_period: int = ATR_PERIOD,
    take_profit_atr: float = TP_ATR,
    stop_loss_atr: float = SL_ATR,
    horizon_bars: int = MAX_HORIZON_BARS,
) -> dict[str, Any]:
    """Label one decision made after close[i], entering at open[i+1].

    Target/stop are anchored to the causal ATR at close[i]. OHLC cannot determine the
    intrabar path; a bar that touches both barriers is marked ambiguous, never guessed.
    """
    if isinstance(decision_index, bool) or not isinstance(decision_index, int):
        raise ValueError("decision_index_must_be_integer")
    if decision_index < 0 or decision_index >= len(rows) - 1:
        raise ValueError("decision_index_out_of_range")
    if isinstance(side, bool) or side not in (-1, 1):
        raise ValueError("side_must_be_minus1_or_plus1")
    if isinstance(atr_period, bool) or not isinstance(atr_period, int) or atr_period < 2:
        raise ValueError("atr_period_must_be_integer_at_least_2")
    if isinstance(horizon_bars, bool) or not isinstance(horizon_bars, int) or horizon_bars < 1:
        raise ValueError("horizon_bars_must_be_positive_integer")
    tp_mult, sl_mult = _finite(take_profit_atr, "take_profit_atr"), _finite(stop_loss_atr, "stop_loss_atr")
    if tp_mult <= 0 or sl_mult <= 0:
        raise ValueError("barrier_multipliers_must_be_positive")

    entry_index = decision_index + 1
    atr = _atr_at_decision(rows, decision_index, atr_period)
    base = {
        "decision_index": decision_index,
        "entry_index": entry_index,
        "target_exit_index": None,
        "side": side,
        "atr": atr,
        "entry_price": None,
        "take_profit_price": None,
        "stop_loss_price": None,
        "actual_tp": None,
        "resolution": "unresolved",
        "gross_r": None,
        "horizon_bars": horizon_bars,
    }
    if atr is None:
        return {**base, "resolution": "insufficient_atr_history"}
    final_index = entry_index + horizon_bars - 1
    if final_index >= len(rows):
        return {**base, "resolution": "truncated_horizon"}
    entry = _finite(rows[entry_index]["open"], "entry_price")
    tp = entry + side * tp_mult * atr
    sl = entry - side * sl_mult * atr
    base.update({"entry_price": entry, "take_profit_price": tp, "stop_loss_price": sl})

    for index in range(entry_index, final_index + 1):
        high = _finite(rows[index]["high"], "high")
        low = _finite(rows[index]["low"], "low")
        if side == 1:
            hit_tp, hit_sl = high >= tp, low <= sl
        else:
            hit_tp, hit_sl = low <= tp, high >= sl
        if hit_tp and hit_sl:
            return {**base, "target_exit_index": index, "actual_tp": None,
                    "resolution": "ambiguous_same_bar", "gross_r": None}
        if hit_tp:
            return {**base, "target_exit_index": index, "actual_tp": 1,
                    "resolution": "take_profit_first", "gross_r": tp_mult}
        if hit_sl:
            return {**base, "target_exit_index": index, "actual_tp": 0,
                    "resolution": "stop_loss_first", "gross_r": -sl_mult}

    exit_close = _finite(rows[final_index]["close"], "timeout_close")
    return {**base, "target_exit_index": final_index, "actual_tp": 0,
            "resolution": "horizon_timeout_no_tp",
            "gross_r": side * (exit_close - entry) / atr}


def audit_dataset_quality(rows: Sequence[Mapping[str, Any]], expected_interval_seconds: int = 300) -> dict[str, Any]:
    """Descriptive quality profile for rows already accepted by the strict parser."""
    if not rows:
        raise ValueError("cannot_audit_empty_dataset")
    if isinstance(expected_interval_seconds, bool) or not isinstance(expected_interval_seconds, int) or expected_interval_seconds <= 0:
        raise ValueError("expected_interval_seconds_must_be_positive_integer")
    timestamps = [row["timestamp"] for row in rows]
    intervals = [(timestamps[i] - timestamps[i - 1]).total_seconds() for i in range(1, len(timestamps))]
    ordered = sorted(intervals)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None
    close_returns = [
        abs(float(rows[i]["close"]) / float(rows[i - 1]["close"]) - 1.0)
        for i in range(1, len(rows)) if float(rows[i - 1]["close"]) > 0
    ]
    median_abs_return = statistics.median(close_returns) if close_returns else 0.0
    deviations = [abs(value - median_abs_return) for value in close_returns]
    mad = statistics.median(deviations) if deviations else 0.0
    jump_threshold = median_abs_return + (10.0 * 1.4826 * mad)
    offsets = sorted({str(ts.utcoffset()) if ts.utcoffset() is not None else "naive" for ts in timestamps})
    return {
        "status": "STRUCTURALLY_VALID_AFTER_STRICT_PARSER",
        "bars": len(rows),
        "timestamps_strictly_increasing": all(timestamps[i] > timestamps[i - 1] for i in range(1, len(timestamps))),
        "duplicate_timestamps": 0,
        "duplicate_check_note": "strict parser rejects non-increasing timestamps before this audit",
        "source_index_continuity": "not_available_in_source_schema",
        "timezone_offsets_observed": offsets,
        "provider_timezone_semantics_verified": False,
        "expected_interval_seconds_assumed": expected_interval_seconds,
        "median_interval_seconds": statistics.median(intervals) if intervals else None,
        "p95_interval_seconds": p95,
        "intervals_over_2x_expected": sum(value > 2 * expected_interval_seconds for value in intervals),
        "intervals_over_1h": sum(value > 3600 for value in intervals),
        "zero_volume_bars": sum(float(row["volume"]) == 0 for row in rows),
        "flat_candles": sum(
            float(row["open"]) == float(row["high"]) == float(row["low"]) == float(row["close"])
            for row in rows
        ),
        "median_absolute_close_return_fraction": median_abs_return,
        "robust_price_jump_threshold_fraction": jump_threshold,
        "robust_price_jumps": sum(value > jump_threshold for value in close_returns),
        "hard_failures_after_parse": 0,
        "limitations": [
            "This profile only describes structurally accepted OHLCV rows.",
            "No source index is supplied, so missing provider rows cannot be distinguished from market closures.",
            "A reported UTC offset does not prove the provider's intended timezone.",
            "Price-jump flags are robust statistical diagnostics, not automatic data deletions.",
        ],
    }


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total <= 0:
        return {"lower_95": None, "upper_95": None}
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt((p * (1 - p) / total) + z * z / (4 * total * total)) / denom
    return {"lower_95": max(0.0, center - radius), "upper_95": min(1.0, center + radius)}


def _summary_events(events: Sequence[Mapping[str, Any]], costs: Sequence[float]) -> dict[str, Any]:
    scored = [row for row in events if row.get("actual_tp") in (0, 1) and row.get("gross_r") is not None]
    tp_count = sum(row["actual_tp"] == 1 for row in scored)
    ambiguous = sum(row.get("resolution") == "ambiguous_same_bar" for row in events)
    outcomes = [float(row["gross_r"]) for row in scored]
    return {
        "candidate_signals": len(events),
        "resolved_binary_labels": len(scored),
        "tp_first_count": tp_count,
        "non_tp_first_count": len(scored) - tp_count,
        "ambiguous_same_bar_count": ambiguous,
        "insufficient_atr_count": sum(row.get("resolution") == "insufficient_atr_history" for row in events),
        "truncated_horizon_count": sum(row.get("resolution") == "truncated_horizon" for row in events),
        "tp_first_rate": tp_count / len(scored) if scored else None,
        "tp_first_rate_wilson_95": _wilson(tp_count, len(scored)),
        "mean_gross_r_per_resolved_signal": statistics.fmean(outcomes) if outcomes else None,
        "cost_sensitivity_r_per_signal": [
            {
                "assumed_cost_r": float(cost),
                "mean_net_r_per_resolved_signal": statistics.fmean(outcomes) - float(cost) if outcomes else None,
                "is_broker_observed_cost": False,
            }
            for cost in costs
        ],
        "warning": "Event-level overlapping opportunities; not a capital/exposure-aware portfolio backtest.",
    }


def _score_oos_group(
    events: Sequence[Mapping[str, Any]],
    calibration_start: int,
    holdout_start: int,
    *,
    min_matured_labels: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    candidates = [
        row for row in events
        if calibration_start <= row["decision_index"] < holdout_start
        and row.get("actual_tp") in (0, 1)
        and row.get("target_exit_index") is not None
        and row["target_exit_index"] <= holdout_start
        and row.get("training_labels_available", 0) >= min_matured_labels
        and row.get("p_tp_prequential") is not None
    ]
    holdout = [
        row for row in events
        if row["decision_index"] >= holdout_start
        and row.get("actual_tp") in (0, 1)
        and row.get("target_exit_index") is not None
        and row.get("training_labels_available", 0) >= min_matured_labels
        and row.get("p_tp_prequential") is not None
    ]
    base = {
        "status": "insufficient_matured_real_oos_rows",
        "calibration_examples": len(candidates),
        "holdout_examples": len(holdout),
        "min_matured_labels_before_forecast": min_matured_labels,
        "calibration_boundary_index": calibration_start,
        "holdout_start_index": holdout_start,
        "calibration_labels_purged_after_boundary": sum(
            1 for row in events
            if calibration_start <= row["decision_index"] < holdout_start
            and row.get("actual_tp") in (0, 1)
            and row.get("target_exit_index") is not None
            and row["target_exit_index"] > holdout_start
        ),
        "deployment_calibrated": False,
        "validated": False,
        "live_signal_enabled": False,
    }
    if len(candidates) < 100 or len(holdout) < 100:
        return base
    probabilities = [float(row["p_tp_prequential"]) for row in candidates]
    outcomes = [int(row["actual_tp"]) for row in candidates]
    fit = fit_platt_scaler(probabilities, outcomes, min_samples=100)
    reference = (sum(outcomes) + 1.0) / (len(outcomes) + 2.0)
    if not fit.get("fitted"):
        return {**base, "status": fit.get("status", "calibrator_fit_failed"),
                "platt_fit": fit, "calibration_reference_probability": reference}
    hp = [float(row["p_tp_prequential"]) for row in holdout]
    hy = [int(row["actual_tp"]) for row in holdout]
    calibrated = [apply_platt_scaler(p, slope=fit["slope"], intercept=fit["intercept"]) for p in hp]
    block_size = max(12, min(288, len(holdout) // 20))
    block_size = min(block_size, len(holdout))
    raw_metrics = probability_metrics(hp, hy, bins=10, reference_probability=reference)
    calibrated_metrics = probability_metrics(calibrated, hy, bins=10, reference_probability=reference)
    raw_intervals = moving_block_bootstrap_intervals(
        hp, hy, bins=10, block_size=block_size, replicates=bootstrap_replicates, seed=20261017
    )
    calibrated_intervals = moving_block_bootstrap_intervals(
        calibrated, hy, bins=10, block_size=block_size, replicates=bootstrap_replicates, seed=20261018
    )
    return {
        **base,
        "status": "REAL_HISTORICAL_OOS_DIAGNOSTIC_ONLY",
        "calibration_examples": len(candidates),
        "holdout_examples": len(holdout),
        "calibration_reference_probability": reference,
        "calibration_date_range": {
            "start": candidates[0].get("decision_time"),
            "end": candidates[-1].get("decision_time"),
        },
        "holdout_date_range": {
            "start": holdout[0].get("decision_time"),
            "end": holdout[-1].get("decision_time"),
        },
        "platt_fit": fit,
        "raw_holdout_metrics": raw_metrics,
        "calibrated_holdout_metrics": calibrated_metrics,
        "raw_holdout_block_bootstrap_intervals": raw_intervals,
        "calibrated_holdout_block_bootstrap_intervals": calibrated_intervals,
        "comparison": {
            "brier_improvement_calibrated_vs_raw": raw_metrics["brier_score"] - calibrated_metrics["brier_score"],
            "log_loss_improvement_calibrated_vs_raw": raw_metrics["log_loss"] - calibrated_metrics["log_loss"],
            "ece_reduction_calibrated_vs_raw": raw_metrics["ece_equal_width_descriptive"] - calibrated_metrics["ece_equal_width_descriptive"],
            "interpretation": "Positive values favor Platt calibration on this historical holdout only; this does not validate a tradable edge.",
        },
        "deployment_calibrated": False,
        "validated": False,
        "live_signal_enabled": False,
        "limitations": [
            "The underlying forecast is a matured-label expanding base-rate baseline, not a learned alpha model.",
            "The holdout is chronological within one community OHLCV source, not an independent market feed.",
            "OHLC same-bar path ambiguity, broker spread, commissions, swaps, slippage and actual fills remain unresolved.",
            "Holdout outcomes are scoring-only for the frozen Platt calibrator; the prequential base-rate algorithm updates only from labels matured by each decision time.",
        ],
    }


def run_tp_sl_protocol(
    rows: Sequence[Mapping[str, Any]],
    signal_fn: Callable[[Sequence[Mapping[str, Any]], int, str], int],
    rules: Sequence[str],
    *,
    warmup_bars: int = WARMUP_BARS,
    atr_period: int = ATR_PERIOD,
    take_profit_atr: float = TP_ATR,
    stop_loss_atr: float = SL_ATR,
    horizon_bars: int = MAX_HORIZON_BARS,
    min_matured_labels: int = MIN_MATURED_LABELS,
    bootstrap_replicates: int = 60,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the preregistered chronological barrier evaluation over each eligible real-data decision."""
    if len(rows) < warmup_bars + horizon_bars + 100:
        return ({
            "status": "INSUFFICIENT_REAL_HISTORY",
            "protocol_version": PROTOCOL_VERSION,
            "protocol": PROTOCOL_NAME,
            "target": TARGET,
            "decision_output": "NO_TRADE",
            "deployment_calibrated": False,
            "validated": False,
            "live_signal_enabled": False,
            "limitations": ["Insufficient chronological history for this protocol."],
        }, [])
    if isinstance(warmup_bars, bool) or not isinstance(warmup_bars, int) or warmup_bars < atr_period:
        raise ValueError("warmup_must_cover_atr_period")
    if isinstance(min_matured_labels, bool) or not isinstance(min_matured_labels, int) or min_matured_labels < 1:
        raise ValueError("min_matured_labels_must_be_positive_integer")
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or not 20 <= bootstrap_replicates <= 500:
        raise ValueError("bootstrap_replicates_out_of_range")

    n = len(rows)
    last_decision_exclusive = n - horizon_bars
    eligible = last_decision_exclusive - warmup_bars
    development_end = warmup_bars + int(eligible * CALIBRATION_FRACTION_START)
    calibration_end = warmup_bars + int(eligible * HOLDOUT_FRACTION_START)
    boundaries = {
        "development": (warmup_bars, development_end),
        "calibration": (development_end, calibration_end),
        "final_holdout": (calibration_end, last_decision_exclusive),
    }

    events: list[dict[str, Any]] = []
    for index in range(warmup_bars, last_decision_exclusive):
        for rule in rules:
            side = signal_fn(rows, index, rule)
            if side == 0:
                continue
            if isinstance(side, bool) or side not in (-1, 1):
                raise ValueError("signal_fn_must_return_minus1_zero_or_plus1")
            label = label_tp_before_sl(
                rows, index, side, atr_period=atr_period, take_profit_atr=take_profit_atr,
                stop_loss_atr=stop_loss_atr, horizon_bars=horizon_bars,
            )
            events.append({
                "rule": rule,
                "decision_time": rows[index]["timestamp"].isoformat(),
                **label,
            })

    events.sort(key=lambda row: (row["decision_index"], row["rule"], row["side"]))
    # Per rule and direction, use only earlier event outcomes that matured by decision time.
    queues: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
    counts: dict[tuple[str, int], int] = {}
    positives: dict[tuple[str, int], int] = {}
    sequence = 0
    for row in events:
        key = (row["rule"], int(row["side"]))
        queue = queues.setdefault(key, [])
        counts.setdefault(key, 0)
        positives.setdefault(key, 0)
        now = int(row["decision_index"])
        while queue and queue[0][0] <= now:
            _exit_index, _sequence, outcome = heapq.heappop(queue)
            counts[key] += 1
            positives[key] += outcome
        available = counts[key]
        row["training_labels_available"] = available
        row["p_tp_prequential"] = (
            (positives[key] + 1.0) / (available + 2.0)
            if available >= min_matured_labels else None
        )
        row["forecast_method"] = "expanding_laplace_base_rate_on_same_rule_and_side_matured_labels"
        if row.get("actual_tp") in (0, 1) and row.get("target_exit_index") is not None:
            sequence += 1
            heapq.heappush(queue, (int(row["target_exit_index"]), sequence, int(row["actual_tp"])))

    group_reports: dict[str, Any] = {}
    for rule in rules:
        sides = {}
        for side in (-1, 1):
            subset = [row for row in events if row["rule"] == rule and row["side"] == side]
            if not subset:
                continue
            calibration_start, holdout_start = development_end, calibration_end
            sides["short" if side == -1 else "long"] = {
                "overall_event_summary": _summary_events(subset, COST_SENSITIVITY_R),
                "oos_probability_evaluation": _score_oos_group(
                    subset, calibration_start, holdout_start,
                    min_matured_labels=min_matured_labels,
                    bootstrap_replicates=bootstrap_replicates,
                ),
            }
        group_reports[rule] = sides or {"status": "NO_DIRECTIONAL_SIGNALS"}

    event_segment_counts = {}
    for name, (start, end) in boundaries.items():
        event_segment_counts[name] = {
            "start_index": start,
            "end_index_exclusive": end,
            "start_time": rows[start]["timestamp"].isoformat(),
            "end_time": rows[end - 1]["timestamp"].isoformat(),
            "decision_opportunities": end - start,
            "target_labels_fully_within_dataset": True,
        }
    report = {
        "status": "DIAGNOSTIC_ONLY",
        "protocol_version": PROTOCOL_VERSION,
        "protocol": PROTOCOL_NAME,
        "target": TARGET,
        "decision_contract": "signal after close[i]; enter at open[i+1]; ATR uses bars ending at i; inspect up to 12 bars starting at entry",
        "barrier_contract": {
            "atr_period_bars": atr_period,
            "take_profit_multiple_atr": take_profit_atr,
            "stop_loss_multiple_atr": stop_loss_atr,
            "max_horizon_bars_from_entry": horizon_bars,
            "same_bar_both_barriers": "ambiguous; excluded from binary scoring and calibration",
            "neither_barrier_by_horizon": "actual_tp=0; record mark-to-market gross R separately",
            "decision_on_same_bar_path": "OHLC cannot reveal ordering; never assume TP-first",
        },
        "probability_contract": {
            "baseline": "expanding Laplace event-rate forecast partitioned by fixed rule and direction",
            "maturity_rule": "a previous label enters the baseline only when target_exit_index <= current decision_index",
            "min_matured_labels_before_scoring": min_matured_labels,
            "calibrator": "regularized monotone Platt scaling fitted only on calibration-partition matured outcomes",
            "final_holdout": "last chronological 20%; excluded from calibrator fitting and selection",
            "calibration_labels_with_exit_after_holdout_start_are_purged": True,
        },
        "dataset_rows": n,
        "eligible_decisions": eligible,
        "eligible_decision_start_index": warmup_bars,
        "eligible_decision_end_exclusive": last_decision_exclusive,
        "chronological_partitions": event_segment_counts,
        "candidate_trade_events": len(events),
        "scored_binary_events": sum(row.get("actual_tp") in (0, 1) for row in events),
        "ambiguous_same_bar_events": sum(row.get("resolution") == "ambiguous_same_bar" for row in events),
        "insufficient_history_or_truncated_events": sum(row.get("actual_tp") is None and row.get("resolution") in ("insufficient_atr_history", "truncated_horizon") for row in events),
        "rules": group_reports,
        "decision_output": "NO_TRADE",
        "deployment_calibrated": False,
        "validated": False,
        "live_signal_enabled": False,
        "limitations": [
            "This is one-source historical community OHLCV research, not an independent external validation.",
            "Binary TP-before-SL labels cannot reconstruct intrabar ordering when both barriers are touched in a candle.",
            "Direction-specific OHLC prices are not broker Bid/Ask quotes, tick paths or actual fills.",
            "The expanding rate is a deliberately simple base-rate comparator, not a learned alpha strategy.",
            "Cost sensitivity values are hypothetical R-units, not broker-observed costs.",
            "Event-level results can overlap in time; no simultaneous capital, exposure or margin simulator is implied.",
            "No result enables a live signal, validates profitability, or establishes world-leading accuracy.",
        ],
    }
    return report, events
