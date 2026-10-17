#!/usr/bin/env python3
"""Leakage-aware TP-before-SL out-of-sample diagnostic; never enables trading."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Reuse the project's tested calibration implementation without adding dependencies.
_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "temporal-calibration-v1.3"
if str(_CALIBRATION_PATH) not in sys.path:
    sys.path.insert(0, str(_CALIBRATION_PATH))
from calibration_diagnostics import (  # noqa: E402
    apply_platt_scaler,
    fit_platt_scaler,
    moving_block_bootstrap_intervals,
    probability_metrics,
)

HORIZON_BARS = 12
ATR_WINDOW = 14
TP_ATR_MULTIPLE = 1.5
SL_ATR_MULTIPLE = 1.0
TRAIN_FRACTION = 0.60
CALIBRATION_FRACTION = 0.20
MIN_TRAIN_ROWS = 300
MIN_CALIBRATION_ROWS = 100
MIN_HOLDOUT_ROWS = 100
PAIRED_BOOTSTRAP_REPLICATES = 250
FEATURE_NAMES = (
    "return_1_atr",
    "return_3_atr",
    "return_12_atr",
    "ema5_minus_ema10_atr",
    "body_atr",
    "range_atr",
    "range_volatility_ratio_5_20",
)


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


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, value))))


def _ema_series(values: Sequence[float], span: int) -> list[float]:
    if span < 1:
        raise ValueError("ema_span_must_be_positive")
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def prepare_features(rows: Sequence[Mapping[str, Any]]) -> list[list[float] | None]:
    """Build each feature using bars through that decision close only."""
    if len(rows) < max(ATR_WINDOW, 20) + 1:
        return [None] * len(rows)
    closes = [_finite(row["close"], "close") for row in rows]
    opens = [_finite(row["open"], "open") for row in rows]
    highs = [_finite(row["high"], "high") for row in rows]
    lows = [_finite(row["low"], "low") for row in rows]
    ranges = [highs[i] - lows[i] for i in range(len(rows))]
    true_ranges = []
    for i in range(len(rows)):
        if i == 0:
            true_ranges.append(ranges[i])
        else:
            true_ranges.append(max(
                ranges[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))
    ema5 = _ema_series(closes, 5)
    ema10 = _ema_series(closes, 10)
    prefix = [0.0]
    for value in true_ranges:
        prefix.append(prefix[-1] + value)
    features: list[list[float] | None] = [None] * len(rows)
    for i in range(20, len(rows)):
        start = max(0, i - ATR_WINDOW + 1)
        atr = (prefix[i + 1] - prefix[start]) / (i - start + 1)
        if not math.isfinite(atr) or atr <= 1e-12:
            continue
        mean_range_5 = statistics.fmean(ranges[i - 4:i + 1])
        mean_range_20 = statistics.fmean(ranges[i - 19:i + 1])
        raw = [
            (closes[i] - closes[i - 1]) / atr,
            (closes[i] - closes[i - 3]) / atr,
            (closes[i] - closes[i - 12]) / atr,
            (ema5[i] - ema10[i]) / atr,
            (closes[i] - opens[i]) / atr,
            ranges[i] / atr,
            mean_range_5 / mean_range_20 if mean_range_20 > 1e-12 else 1.0,
        ]
        features[i] = [
            max(-10.0, min(10.0, value)) if index < 6
            else max(0.0, min(10.0, value))
            for index, value in enumerate(raw)
        ]
    return features


def label_barrier(
    rows: Sequence[Mapping[str, Any]],
    decision_index: int,
    atr: float,
    side: str,
    *,
    horizon_bars: int = HORIZON_BARS,
    tp_atr_multiple: float = TP_ATR_MULTIPLE,
    sl_atr_multiple: float = SL_ATR_MULTIPLE,
) -> dict[str, Any]:
    """Label next-open entry using OHLC; ambiguous same-bar hits resolve stop-first."""
    if side not in ("long", "short"):
        raise ValueError("side_must_be_long_or_short")
    if horizon_bars < 1 or decision_index < 0 or decision_index + horizon_bars >= len(rows):
        raise ValueError("invalid_decision_or_horizon")
    atr = _finite(atr, "atr")
    if atr <= 0:
        raise ValueError("atr_must_be_positive")
    entry = _finite(rows[decision_index + 1]["open"], "entry")
    if side == "long":
        target = entry + tp_atr_multiple * atr
        stop = entry - sl_atr_multiple * atr
    else:
        target = entry - tp_atr_multiple * atr
        stop = entry + sl_atr_multiple * atr

    for index in range(decision_index + 1, decision_index + horizon_bars + 1):
        row = rows[index]
        op = _finite(row["open"], "open")
        high = _finite(row["high"], "high")
        low = _finite(row["low"], "low")
        if side == "long":
            # A gap beyond a barrier is resolved at the open before intrabar range logic.
            if op <= stop:
                return {"outcome": "SL_FIRST", "label": 0, "exit_index": index, "bars_elapsed": index - decision_index}
            if op >= target:
                return {"outcome": "TP_FIRST", "label": 1, "exit_index": index, "bars_elapsed": index - decision_index}
            stop_hit, target_hit = low <= stop, high >= target
        else:
            if op >= stop:
                return {"outcome": "SL_FIRST", "label": 0, "exit_index": index, "bars_elapsed": index - decision_index}
            if op <= target:
                return {"outcome": "TP_FIRST", "label": 1, "exit_index": index, "bars_elapsed": index - decision_index}
            stop_hit, target_hit = high >= stop, low <= target

        if stop_hit and target_hit:
            return {"outcome": "SL_FIRST_SAME_BAR_AMBIGUITY", "label": 0, "exit_index": index, "bars_elapsed": index - decision_index}
        if stop_hit:
            return {"outcome": "SL_FIRST", "label": 0, "exit_index": index, "bars_elapsed": index - decision_index}
        if target_hit:
            return {"outcome": "TP_FIRST", "label": 1, "exit_index": index, "bars_elapsed": index - decision_index}
    return {
        "outcome": "TIMEOUT_WITHOUT_TARGET",
        "label": 0,
        "exit_index": decision_index + horizon_bars,
        "bars_elapsed": horizon_bars,
    }


def build_samples(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon_bars: int = HORIZON_BARS,
) -> dict[str, list[dict[str, Any]]]:
    """Create causal features and target labels; long and short outcomes stay separate."""
    if horizon_bars < 1:
        raise ValueError("horizon_must_be_positive")
    features = prepare_features(rows)
    closes = [_finite(row["close"], "close") for row in rows]
    highs = [_finite(row["high"], "high") for row in rows]
    lows = [_finite(row["low"], "low") for row in rows]
    ranges = [highs[i] - lows[i] for i in range(len(rows))]
    tr = []
    for i in range(len(rows)):
        tr.append(ranges[i] if i == 0 else max(
            ranges[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])
        ))
    prefix = [0.0]
    for value in tr:
        prefix.append(prefix[-1] + value)
    samples = {"long": [], "short": []}
    for i in range(20, len(rows) - horizon_bars):
        if features[i] is None:
            continue
        start = max(0, i - ATR_WINDOW + 1)
        atr = (prefix[i + 1] - prefix[start]) / (i - start + 1)
        for side in ("long", "short"):
            label = label_barrier(rows, i, atr, side, horizon_bars=horizon_bars)
            samples[side].append({
                "decision_index": i,
                "target_exit_index": label["exit_index"],
                "outcome": label["outcome"],
                "actual_tp": label["label"],
                "bars_elapsed": label["bars_elapsed"],
                "features": list(features[i]),
            })
    return samples


def split_matured_samples(samples: Sequence[Mapping[str, Any]], row_count: int) -> dict[str, Any]:
    """Chronological 60/20/20 split; purge any outcome that crosses the next boundary."""
    if row_count < 1:
        raise ValueError("row_count_must_be_positive")
    train_end = int(row_count * TRAIN_FRACTION)
    holdout_start = int(row_count * (TRAIN_FRACTION + CALIBRATION_FRACTION))
    train = [
        row for row in samples
        if row["decision_index"] < train_end and row["target_exit_index"] < train_end
    ]
    calibration = [
        row for row in samples
        if train_end <= row["decision_index"] < holdout_start
        and row["target_exit_index"] < holdout_start
    ]
    holdout = [row for row in samples if row["decision_index"] >= holdout_start]
    return {
        "train": train,
        "calibration": calibration,
        "holdout": holdout,
        "train_end_index_exclusive": train_end,
        "holdout_start_index": holdout_start,
        "train_purged_unmatured": sum(
            1 for row in samples
            if row["decision_index"] < train_end and row["target_exit_index"] >= train_end
        ),
        "calibration_purged_unmatured": sum(
            1 for row in samples
            if train_end <= row["decision_index"] < holdout_start
            and row["target_exit_index"] >= holdout_start
        ),
    }


def fit_logistic(
    train_features: Sequence[Sequence[float]],
    train_labels: Sequence[int],
    *,
    iterations: int = 100,
    learning_rate: float = 0.12,
    l2: float = 0.05,
) -> dict[str, Any]:
    """Small dependency-free, fixed-protocol L2 logistic baseline."""
    if not train_features or len(train_features) != len(train_labels):
        raise ValueError("training_data_empty_or_mismatched")
    width = len(FEATURE_NAMES)
    if any(len(row) != width for row in train_features):
        raise ValueError("feature_width_mismatch")
    labels = [int(y) for y in train_labels]
    if any(y not in (0, 1) for y in labels) or len(set(labels)) < 2:
        raise ValueError("training_labels_must_contain_both_classes")
    n = len(train_features)
    means = [statistics.fmean(row[j] for row in train_features) for j in range(width)]
    scales = []
    for j in range(width):
        variance = statistics.fmean((row[j] - means[j]) ** 2 for row in train_features)
        scales.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
    scaled = [[(row[j] - means[j]) / scales[j] for j in range(width)] for row in train_features]
    positives = sum(labels)
    bias = math.log((positives + 0.5) / (n - positives + 0.5))
    weights = [0.0] * width
    for _ in range(iterations):
        grad = [0.0] * width
        grad_bias = 0.0
        for row, target in zip(scaled, labels):
            probability = _sigmoid(bias + sum(w * x for w, x in zip(weights, row)))
            error = probability - target
            grad_bias += error
            for j, x in enumerate(row):
                grad[j] += error * x
        bias -= learning_rate * grad_bias / n
        for j in range(width):
            weights[j] -= learning_rate * (grad[j] / n + l2 * weights[j])
    if not all(math.isfinite(v) for v in weights + [bias]):
        raise ValueError("non_finite_model_rejected")
    return {
        "weights": weights,
        "bias": bias,
        "means": means,
        "scales": scales,
        "training_rows": n,
        "training_positives": positives,
        "training_negatives": n - positives,
        "iterations": iterations,
        "l2": l2,
        "feature_names": list(FEATURE_NAMES),
        "feature_scaler_fitted_on_training_only": True,
    }


def predict_probability(model: Mapping[str, Any], features: Sequence[float]) -> float:
    if len(features) != len(model["weights"]):
        raise ValueError("feature_width_mismatch")
    scaled = [
        (float(value) - mean) / scale
        for value, mean, scale in zip(features, model["means"], model["scales"])
    ]
    return _sigmoid(model["bias"] + sum(w * x for w, x in zip(model["weights"], scaled)))



def paired_block_bootstrap_improvements(
    raw_probabilities: Sequence[Any],
    calibrated_probabilities: Sequence[Any],
    outcomes: Sequence[Any],
    *,
    block_size: int = HORIZON_BARS,
    replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
    seed: int = 20261019,
) -> dict[str, Any]:
    """Paired moving-block bootstrap for raw-minus-calibrated proper-score gains.

    Positive values mean calibration reduced the score. Resampling the same temporal
    indices for both probability sets preserves the paired comparison and short-range
    dependence better than comparing two independent confidence intervals.
    """
    if not raw_probabilities or len(raw_probabilities) != len(calibrated_probabilities) or len(raw_probabilities) != len(outcomes):
        raise ValueError("paired_bootstrap_length_mismatch_or_empty")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size_must_be_positive_integer")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or not 20 <= replicates <= 5000:
        raise ValueError("replicates_must_be_integer_between_20_and_5000")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed_must_be_integer")

    raw = [_finite(value, f"raw_probability_{i}") for i, value in enumerate(raw_probabilities)]
    calibrated = [_finite(value, f"calibrated_probability_{i}") for i, value in enumerate(calibrated_probabilities)]
    labels = []
    for i, value in enumerate(outcomes):
        if isinstance(value, bool) or value not in (0, 1):
            raise ValueError(f"outcome_must_be_binary_at_{i}")
        labels.append(int(value))
    if any(not 0.0 <= value <= 1.0 for value in raw + calibrated):
        raise ValueError("probability_out_of_range")
    n = len(labels)
    size = min(block_size, n)
    rng = random.Random(seed)
    improvements = {"brier_score": [], "log_loss": []}
    raw_sum = {key: 0.0 for key in improvements}
    calibrated_sum = {key: 0.0 for key in improvements}
    eps = 1e-12

    def score(probability: float, label: int, metric: str) -> float:
        clipped = min(1.0 - eps, max(eps, probability))
        if metric == "brier_score":
            return (probability - label) ** 2
        return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))

    for p, c, y in zip(raw, calibrated, labels):
        for key in improvements:
            raw_sum[key] += score(p, y, key)
            calibrated_sum[key] += score(c, y, key)
    point = {key: (raw_sum[key] - calibrated_sum[key]) / n for key in improvements}

    for _ in range(replicates):
        indexes = []
        while len(indexes) < n:
            start = rng.randrange(0, n - size + 1)
            indexes.extend(range(start, start + size))
        indexes = indexes[:n]
        sums = {key: [0.0, 0.0] for key in improvements}
        for index in indexes:
            p, c, y = raw[index], calibrated[index], labels[index]
            for key in improvements:
                sums[key][0] += score(p, y, key)
                sums[key][1] += score(c, y, key)
        for key, (raw_score, calibrated_score) in sums.items():
            improvements[key].append((raw_score - calibrated_score) / n)

    metrics = {}
    for key, values in improvements.items():
        values.sort()
        metrics[key] = {
            "point_estimate_raw_minus_calibrated": point[key],
            "lower_95": values[max(0, int(0.025 * (replicates - 1)))],
            "upper_95": values[min(replicates - 1, int(0.975 * (replicates - 1)))],
            "fraction_positive_bootstrap_improvements": sum(value > 0 for value in values) / replicates,
            "replicates": replicates,
        }
    return {
        "method": "paired_moving_block_bootstrap_raw_minus_calibrated",
        "n": n,
        "block_size": size,
        "seed": seed,
        "replicates": replicates,
        "metrics": metrics,
        "positive_improvement_favors_calibration": True,
        "interpretation": "If an interval includes zero, this run does not establish that calibration improves the corresponding proper score. This is a sensitivity interval, not production validation.",
    }


def evaluate_side(
    rows: Sequence[Mapping[str, Any]],
    side_samples: Sequence[Mapping[str, Any]],
    side: str,
    *,
    minimum_train_rows: int = MIN_TRAIN_ROWS,
    minimum_calibration_rows: int = MIN_CALIBRATION_ROWS,
    minimum_holdout_rows: int = MIN_HOLDOUT_ROWS,
    bootstrap_replicates: int = 40,
) -> dict[str, Any]:
    split = split_matured_samples(side_samples, len(rows))
    train, calibration, holdout = split["train"], split["calibration"], split["holdout"]
    base = {
        "side": side,
        "status": "INSUFFICIENT_HISTORY",
        "target": "tp_before_sl_within_horizon",
        "horizon_bars": HORIZON_BARS,
        "tp_atr_multiple": TP_ATR_MULTIPLE,
        "sl_atr_multiple": SL_ATR_MULTIPLE,
        "collision_policy": "stop_first_when_both_barriers_hit_in_same_ohlc_bar",
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "holdout_rows": len(holdout),
        "train_end_index_exclusive": split["train_end_index_exclusive"],
        "holdout_start_index": split["holdout_start_index"],
        "train_purged_unmatured_rows": split["train_purged_unmatured"],
        "calibration_purged_unmatured_rows": split["calibration_purged_unmatured"],
        "probability_model_validated": False,
        "calibrator_validated": False,
        "deployment_calibrated": False,
        "live_signal_enabled": False,
    }
    if len(train) < minimum_train_rows or len(calibration) < minimum_calibration_rows or len(holdout) < minimum_holdout_rows:
        return base
    try:
        model = fit_logistic(
            [row["features"] for row in train],
            [row["actual_tp"] for row in train],
        )
    except ValueError as exc:
        return {**base, "status": "MODEL_FIT_BLOCKED", "fit_error": str(exc)}

    calibration_probabilities = [predict_probability(model, row["features"]) for row in calibration]
    calibration_outcomes = [int(row["actual_tp"]) for row in calibration]
    holdout_probabilities = [predict_probability(model, row["features"]) for row in holdout]
    holdout_outcomes = [int(row["actual_tp"]) for row in holdout]
    reference_probability = (sum(calibration_outcomes) + 1.0) / (len(calibration_outcomes) + 2.0)
    fit = fit_platt_scaler(
        calibration_probabilities,
        calibration_outcomes,
        min_samples=minimum_calibration_rows,
        iterations=200,
        l2=0.05,
    )
    base.update({
        "status": "DIAGNOSTIC_ONLY",
        "model_kind": "fixed_protocol_l2_logistic_baseline",
        "model_features": list(FEATURE_NAMES),
        "model_training": {
            "training_rows": len(train),
            "model_parameter_sha256": hashlib.sha256(json.dumps({
                "weights": model["weights"], "bias": model["bias"],
                "means": model["means"], "scales": model["scales"],
            }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
            "training_positive_rate": sum(int(row["actual_tp"]) for row in train) / len(train),
            "features_use_data_through_decision_close_only": True,
            "labels_matured_before_calibration_period": all(
                row["target_exit_index"] < split["train_end_index_exclusive"] for row in train
            ),
        },
        "calibration": {
            "rows": len(calibration),
            "positive_count": sum(calibration_outcomes),
            "positive_rate": sum(calibration_outcomes) / len(calibration_outcomes),
            "max_target_exit_index": max(row["target_exit_index"] for row in calibration),
            "first_holdout_decision_index": split["holdout_start_index"],
            "all_calibration_labels_matured_before_holdout": all(
                row["target_exit_index"] < split["holdout_start_index"] for row in calibration
            ),
            "reference_probability_from_calibration_only": reference_probability,
            "calibrator_fit": fit,
        },
        "holdout": {
            "rows": len(holdout),
            "positive_count": sum(holdout_outcomes),
            "positive_rate": sum(holdout_outcomes) / len(holdout_outcomes),
            "first_decision_index": min(row["decision_index"] for row in holdout),
            "last_decision_index": max(row["decision_index"] for row in holdout),
            "max_target_exit_index": max(row["target_exit_index"] for row in holdout),
            "outcome_classes": {
                outcome: sum(row["outcome"] == outcome for row in holdout)
                for outcome in ("TP_FIRST", "SL_FIRST", "SL_FIRST_SAME_BAR_AMBIGUITY", "TIMEOUT_WITHOUT_TARGET")
            },
        },
        "raw_holdout_metrics": probability_metrics(
            holdout_probabilities, holdout_outcomes, reference_probability=reference_probability
        ),
    })
    if fit.get("fitted"):
        calibrated = [
            apply_platt_scaler(p, slope=fit["slope"], intercept=fit["intercept"])
            for p in holdout_probabilities
        ]
        raw_bootstrap = moving_block_bootstrap_intervals(
            holdout_probabilities, holdout_outcomes, bins=10,
            block_size=HORIZON_BARS, replicates=bootstrap_replicates, seed=20261017,
        )
        calibrated_bootstrap = moving_block_bootstrap_intervals(
            calibrated, holdout_outcomes, bins=10,
            block_size=HORIZON_BARS, replicates=bootstrap_replicates, seed=20261018,
        )
        adjusted_metrics = probability_metrics(
            calibrated, holdout_outcomes, reference_probability=reference_probability
        )
        paired_improvement_ci = paired_block_bootstrap_improvements(
            holdout_probabilities, calibrated, holdout_outcomes,
            block_size=HORIZON_BARS,
            replicates=PAIRED_BOOTSTRAP_REPLICATES,
            seed=20261019,
        )
        raw_metrics = base["raw_holdout_metrics"]
        base.update({
            "calibrated_holdout_metrics": adjusted_metrics,
            "raw_holdout_block_bootstrap_ci": raw_bootstrap,
            "calibrated_holdout_block_bootstrap_ci": calibrated_bootstrap,
            "paired_calibration_improvement_block_bootstrap_ci": paired_improvement_ci,
            "comparison_calibrated_minus_raw": {
                "brier_improvement": raw_metrics["brier_score"] - adjusted_metrics["brier_score"],
                "log_loss_improvement": raw_metrics["log_loss"] - adjusted_metrics["log_loss"],
                "ece_reduction": raw_metrics["ece_equal_width_descriptive"] - adjusted_metrics["ece_equal_width_descriptive"],
                "positive_value_favors_calibrated_diagnostics_only": True,
            },
        })
    else:
        base["calibrated_holdout_metrics"] = None
        base["calibration_fit_status"] = fit.get("status")
    base["limitations"] = [
        "The target is a diagnostic OHLC TP-before-SL event, not actual broker execution.",
        "Entry uses next-bar open; OHLC cannot reveal intrabar event order, so same-bar barrier collisions are resolved stop-first.",
        "Spread, commission, swap, slippage, liquidity and real fills are not observed in these datasets.",
        "The model, threshold, features and calibration protocol are fixed in advance; this run does not tune for profit.",
        "Calibration performance on this holdout is research evidence only and does not authorize deployment.",
        "Source licensing, broker provenance, quote side and timezone semantics remain unresolved by this metric.",
    ]
    return base


def run_barrier_oos(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_id: str,
    bootstrap_replicates: int = 40,
) -> dict[str, Any]:
    """Run long/short exact-target OOS metrics without changing any trading gate."""
    base = {
        "status": "INSUFFICIENT_HISTORY",
        "source_id": source_id,
        "target": "tp_before_sl_within_12_bars",
        "row_count": len(rows),
        "split_protocol": "fixed_chronological_60_train_20_calibration_20_holdout_with_matured_label_purging",
        "live_signal_enabled": False,
        "live_order_execution_enabled": False,
        "deployment_calibrated": False,
        "validated": False,
    }
    if len(rows) < 1000:
        return {**base, "reason": "minimum_1000_bars_required_for_research_benchmark"}
    samples = build_samples(rows)
    sides = {
        side: evaluate_side(rows, side_samples, side, bootstrap_replicates=bootstrap_replicates)
        for side, side_samples in samples.items()
    }
    passed = all(item["status"] == "DIAGNOSTIC_ONLY" for item in sides.values())
    return {
        **base,
        "status": "DIAGNOSTIC_ONLY" if passed else "INSUFFICIENT_HISTORY_OR_MODEL_FIT",
        "decision_gate": "NO_TRADE",
        "features": list(FEATURE_NAMES),
        "feature_time_contract": "features through decision close i only; entry at open i+1",
        "label_contract": {
            "entry": "next_bar_open",
            "take_profit": f"{TP_ATR_MULTIPLE}x ATR({ATR_WINDOW})",
            "stop_loss": f"{SL_ATR_MULTIPLE}x ATR({ATR_WINDOW})",
            "maximum_horizon_bars": HORIZON_BARS,
            "same_bar_collision": "stop-first",
            "timeout": "label 0 for event not reached within horizon and report timeout count separately",
        },
        "sides": sides,
        "predictive_model_validated": False,
        "deployment_calibrated": False,
        "live_signal_enabled": False,
        "live_order_execution_enabled": False,
        "production_approval": "BLOCKED",
    }


if __name__ == "__main__":
    print("This module is a library used by the online benchmark runner; live execution is disabled.")
