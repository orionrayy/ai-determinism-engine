"""Deterministic, point-in-time alpha-model training for Serevex stages 1-3.

Standard-library-only logistic regression with chronological fit/selection/calibration/
final-holdout cutoffs. This is a research forecaster, not a trade authorization system.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

MODEL_VERSION = "serevex-alpha-logistic-v4-embargoed-dual-score-ood"
FEATURE_NAMES = (
    "atr_pct",
    "trend_efficiency",
    "momentum_1_atr",
    "momentum_5_atr",
    "candle_body_to_range",
    "close_location_in_range",
    "bar_range_to_atr",
    "volume_log_ratio",
)
L2_GRID = (0.01, 0.1, 1.0)
DEFAULT_L2 = 0.1
MIN_TRAIN_ROWS = 30
MIN_SELECTION_ROWS = 30
MIN_CALIBRATION_ROWS = 30
TRAIN_EPOCHS = 100
CALIBRATION_EPOCHS = 160
MIN_BLEND_SELECTION_ROWS = 30
BLEND_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
BLEND_BLOCK_SIZE = 12  # source-decision bars evaluated per block
BLEND_EMBARGO_BARS = 12  # target horizon gap between scored blocks
BLEND_MIN_BLOCKS = 8
BLEND_CRITICAL_Z = 2.58
MIN_POSITIVE_BLOCK_SHARE = 0.70
OOD_Z_LIMIT = 4.0
OOD_MAX_FEATURE_FRACTION = 0.25
HOLDOUT_BLOCK_BOOTSTRAP_REPLICATES = 300


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _finite_feature(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _labelled_before(
    events: Sequence[Mapping[str, Any]],
    cutoff: int,
    allowed_partitions: set[str],
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for row in events:
        decision = row.get("decision_index")
        maturity = row.get("label_available_index")
        label = row.get("label")
        if (
            row.get("partition") in allowed_partitions
            and isinstance(decision, int) and not isinstance(decision, bool)
            and isinstance(maturity, int) and not isinstance(maturity, bool)
            and decision < cutoff
            and maturity < cutoff
            and row.get("label_mature") is True
            and label in (0, 1) and not isinstance(label, bool)
        ):
            rows.append(row)
    rows.sort(key=lambda r: (int(r["decision_index"]), str(r.get("decision_id", ""))))
    return rows


def _raw_vector(row: Mapping[str, Any]) -> list[float | None]:
    values = row.get("alpha_features")
    if not isinstance(values, Mapping):
        values = {}
    return [_finite_feature(values.get(name)) for name in FEATURE_NAMES]


def _fit_scaler(samples: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    vectors = [_raw_vector(row) for row in samples]
    means: list[float] = []
    scales: list[float] = []
    for column in range(len(FEATURE_NAMES)):
        observed = [x[column] for x in vectors if x[column] is not None]
        if not observed:
            means.append(0.0)
            scales.append(1.0)
            continue
        mean = sum(float(x) for x in observed) / len(observed)
        variance = sum((float(x) - mean) ** 2 for x in observed) / len(observed)
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale > 1e-9 else 1.0)
    return means, scales


def _transform(
    vector: Sequence[float | None], means: Sequence[float], scales: Sequence[float]
) -> list[float]:
    out = []
    for value, mean, scale in zip(vector, means, scales):
        raw = mean if value is None else value
        out.append(max(-6.0, min(6.0, (float(raw) - mean) / scale)))
    return out


def _fit_logistic(samples: Sequence[Mapping[str, Any]], l2: float) -> dict[str, Any]:
    if len(samples) < MIN_TRAIN_ROWS:
        raise ValueError("alpha_training_support_below_minimum")
    means, scales = _fit_scaler(samples)
    xs = [_transform(_raw_vector(row), means, scales) for row in samples]
    ys = [int(row["label"]) for row in samples]
    positives = sum(ys)
    intercept = math.log((positives + 1.0) / (len(ys) - positives + 1.0))
    coefficients = [0.0] * len(FEATURE_NAMES)

    # Full-batch gradient descent is deterministic and keeps the dependency set empty.
    # Features are training-only standardized/clipped; the intercept is not penalized.
    for epoch in range(TRAIN_EPOCHS):
        grad_intercept = 0.0
        grad = [0.0] * len(FEATURE_NAMES)
        for x, y in zip(xs, ys):
            error = _sigmoid(intercept + sum(w * v for w, v in zip(coefficients, x))) - y
            grad_intercept += error
            for j, value in enumerate(x):
                grad[j] += error * value
        inv_n = 1.0 / len(ys)
        rate = 0.16 / math.sqrt(1.0 + epoch / 80.0)
        intercept -= rate * grad_intercept * inv_n
        for j in range(len(coefficients)):
            coefficients[j] -= rate * (grad[j] * inv_n + float(l2) * coefficients[j])
        if not math.isfinite(intercept) or any(not math.isfinite(w) for w in coefficients):
            raise ValueError("alpha_optimizer_nonfinite_state")

    return {
        "model_version": MODEL_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
        "intercept": intercept,
        "l2": float(l2),
        "train_rows": len(samples),
        "train_labels_sha256": _sha([
            [row.get("decision_id", row.get("decision_index")), int(row["label"])]
            for row in samples
        ]),
    }


def _predict_raw(model: Mapping[str, Any] | None, row: Mapping[str, Any]) -> float:
    if model is None:
        return 0.5
    x = _transform(_raw_vector(row), model["means"], model["scales"])
    z = float(model["intercept"]) + sum(
        float(w) * v for w, v in zip(model["coefficients"], x)
    )
    return min(max(_sigmoid(z), 1e-6), 1.0 - 1e-6)


def _brier(predictions: Sequence[float], labels: Sequence[int]) -> float | None:
    if not labels:
        return None
    return sum((float(p) - int(y)) ** 2 for p, y in zip(predictions, labels)) / len(labels)


def _log_loss(predictions: Sequence[float], labels: Sequence[int]) -> float | None:
    if not labels:
        return None
    total = 0.0
    for p, y in zip(predictions, labels):
        q = min(max(float(p), 1e-12), 1.0 - 1e-12)
        total -= int(y) * math.log(q) + (1 - int(y)) * math.log(1.0 - q)
    return total / len(labels)


def _out_of_domain_fraction(model: Mapping[str, Any] | None, row: Mapping[str, Any], *, z_limit: float = OOD_Z_LIMIT) -> float:
    """Training-only scaler based OOD diagnostic; current holdout labels are never involved."""
    if model is None:
        return 1.0
    vector = _raw_vector(row)
    means = model["means"]
    scales = model["scales"]
    flagged = 0
    for value, mean, scale in zip(vector, means, scales):
        if value is None or not math.isfinite(float(value)) or float(scale) <= 0.0:
            flagged += 1
            continue
        z = abs((float(value) - float(mean)) / float(scale))
        if not math.isfinite(z) or z > z_limit:
            flagged += 1
    return flagged / len(FEATURE_NAMES)


def _binary_log_loss(probability: float, label: int) -> float:
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return -(int(label) * math.log(p) + (1 - int(label)) * math.log(1.0 - p))


def _embargoed_time_blocks(
    rows: Sequence[Mapping[str, Any]],
    *,
    block_span_bars: int = BLEND_BLOCK_SIZE,
    embargo_bars: int = BLEND_EMBARGO_BARS,
) -> tuple[list[list[Mapping[str, Any]]], list[Mapping[str, Any]]]:
    """Build deterministic source-index blocks separated by a target-horizon embargo.

    Block phase is anchored to absolute source decision indices, not filtered event rank.
    With a 12-bar label horizon, each scored 12-bar source window is followed by 12
    source bars that are excluded from governor/bootstrap estimation. This prevents
    adjacent scored windows from having overlapping target horizons.
    """
    if block_span_bars < 1 or embargo_bars < 0:
        raise ValueError("temporal_block_and_embargo_parameters_invalid")
    period = block_span_bars + embargo_bars
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    used: list[Mapping[str, Any]] = []
    for row in sorted(rows, key=lambda r: int(r["decision_index"])):
        decision_index = int(row["decision_index"])
        phase = decision_index % period
        if phase >= block_span_bars:
            continue
        block_id = decision_index // period
        grouped.setdefault(block_id, []).append(row)
        used.append(row)
    return [grouped[key] for key in sorted(grouped)], used


def _block_stats(block_gains: Sequence[Sequence[float]], critical_z: float) -> dict[str, Any]:
    """Equal-weight temporal-block statistics, avoiding false precision from dense blocks."""
    block_means = [sum(gains) / len(gains) for gains in block_gains if gains]
    mean_gain = sum(block_means) / len(block_means) if block_means else 0.0
    if len(block_means) >= 2:
        variance = sum((x - mean_gain) ** 2 for x in block_means) / (len(block_means) - 1)
        se = math.sqrt(max(0.0, variance) / len(block_means))
        lower = mean_gain - critical_z * se
        positive_share = sum(x > 0.0 for x in block_means) / len(block_means)
    else:
        se = float("inf")
        lower = float("-inf")
        positive_share = 0.0
    return {
        "mean_gain": mean_gain,
        "block_standard_error": se if math.isfinite(se) else None,
        "conservative_lower_screen": lower if math.isfinite(lower) else None,
        "independent_blocks": len(block_means),
        "positive_block_share": positive_share,
        "block_means": block_means,
    }


def _choose_baseline_shrinkage_weight(
    rows: Sequence[Mapping[str, Any]],
    calibrated_alpha_probabilities: Mapping[int, float],
    *,
    minimum_rows: int = MIN_BLEND_SELECTION_ROWS,
    block_size: int = BLEND_BLOCK_SIZE,
    minimum_blocks: int = BLEND_MIN_BLOCKS,
    critical_z: float = BLEND_CRITICAL_Z,
) -> dict[str, Any]:
    """Use only a later calibration sub-block to govern the fixed alpha/baseline blend.

    The screen is deliberately conservative: candidate gains are paired against the
    row-wise causal baseline, uncertainty is estimated from chronological non-overlap
    block means, and a Bonferroni-style critical multiplier is used across nonzero
    weights. This is a research heuristic, not an exact finite-sample guarantee.
    """
    ordered = sorted(
        [r for r in rows if r.get("label_mature") is True and r.get("label") in (0, 1)
         and isinstance(r.get("decision_index"), int) and not isinstance(r.get("decision_index"), bool)],
        key=lambda r: int(r["decision_index"]),
    )
    blocks, used = _embargoed_time_blocks(
        ordered, block_span_bars=block_size, embargo_bars=BLEND_EMBARGO_BARS
    )
    base = {
        "alpha_weight": 0.0,
        "status": "BASELINE_ONLY_INSUFFICIENT_CALIBRATION_SUPPORT",
        "selection_rows": len(ordered),
        "selection_rows_used": len(used),
        "block_size_source_bars": block_size,
        "embargo_bars": BLEND_EMBARGO_BARS,
        "independent_block_count": len(blocks),
        "critical_multiplier": critical_z,
        "minimum_positive_block_share": MIN_POSITIVE_BLOCK_SHARE,
        "family_size_nonzero_weights": len(BLEND_WEIGHTS) - 1,
        "dual_score_gate_required": ["brier", "log_loss"],
        "candidate_diagnostics": [],
        "selected_by": "paired_brier_and_log_loss_gain_vs_rowwise_causal_baseline_with_embargoed_time_block_stability",
    }
    if len(used) < minimum_rows or len(blocks) < minimum_blocks:
        return base

    candidates: list[dict[str, Any]] = []
    for weight in BLEND_WEIGHTS:
        brier_block_gains: list[list[float]] = []
        logloss_block_gains: list[list[float]] = []
        for block in blocks:
            block_brier: list[float] = []
            block_logloss: list[float] = []
            for row in block:
                index = int(row["decision_index"])
                y = int(row["label"])
                baseline = min(max(float(row.get("baseline_probability", 0.5)), 1e-6), 1.0 - 1e-6)
                alpha_p = min(max(float(calibrated_alpha_probabilities[index]), 1e-6), 1.0 - 1e-6)
                mixed = min(max((1.0 - weight) * baseline + weight * alpha_p, 1e-6), 1.0 - 1e-6)
                block_brier.append((baseline - y) ** 2 - (mixed - y) ** 2)
                block_logloss.append(_binary_log_loss(baseline, y) - _binary_log_loss(mixed, y))
            brier_block_gains.append(block_brier)
            logloss_block_gains.append(block_logloss)
        brier = _block_stats(brier_block_gains, critical_z)
        logloss = _block_stats(logloss_block_gains, critical_z)
        eligible = bool(
            weight > 0.0
            and brier["conservative_lower_screen"] is not None
            and logloss["conservative_lower_screen"] is not None
            and float(brier["conservative_lower_screen"]) > 0.0
            and float(logloss["conservative_lower_screen"]) > 0.0
            and float(brier["positive_block_share"]) >= MIN_POSITIVE_BLOCK_SHARE
            and float(logloss["positive_block_share"]) >= MIN_POSITIVE_BLOCK_SHARE
        )
        candidates.append({
            "alpha_weight": weight,
            "brier_improvement_baseline_minus_blend": brier["mean_gain"],
            "brier_block_standard_error": brier["block_standard_error"],
            "brier_conservative_lower_screen": brier["conservative_lower_screen"],
            "brier_positive_block_share": brier["positive_block_share"],
            "log_loss_improvement_baseline_minus_blend": logloss["mean_gain"],
            "log_loss_block_standard_error": logloss["block_standard_error"],
            "log_loss_conservative_lower_screen": logloss["conservative_lower_screen"],
            "log_loss_positive_block_share": logloss["positive_block_share"],
            "independent_blocks": min(brier["independent_blocks"], logloss["independent_blocks"]),
            "eligible": eligible,
        })
    eligible = [x for x in candidates if x["eligible"]]
    if eligible:
        selected = max(eligible, key=lambda x: (
            float(x["brier_improvement_baseline_minus_blend"]),
            float(x["log_loss_improvement_baseline_minus_blend"]),
            -float(x["alpha_weight"]),
        ))
        base["alpha_weight"] = float(selected["alpha_weight"])
        base["status"] = "CALIBRATION_BLOCK_SELECTED_ALPHA_WEIGHT_PASSES_DUAL_SCORE_STABILITY_SCREEN"
    else:
        base["alpha_weight"] = 0.0
        base["status"] = "BASELINE_ONLY_NO_CONFIRMED_DUAL_SCORE_GAIN"
    base["candidate_diagnostics"] = candidates
    base["selection_labels_sha256"] = _sha([
        [row.get("decision_id", row.get("decision_index")), int(row["label"])]
        for row in used
    ])
    return base

def _paired_block_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    model_probabilities: Sequence[float],
    baseline_probabilities: Sequence[float],
    *,
    replicates: int = HOLDOUT_BLOCK_BOOTSTRAP_REPLICATES,
    block_size: int = BLEND_BLOCK_SIZE,
    seed: int = 20261019,
) -> dict[str, Any]:
    """Bootstrap paired improvement across source-index blocks separated by horizon embargo."""
    if len(rows) != len(model_probabilities) or len(rows) != len(baseline_probabilities):
        raise ValueError("paired_bootstrap_inputs_must_have_equal_lengths")
    if block_size < 1:
        raise ValueError("bootstrap_block_size_must_be_positive")
    per_decision = {
        int(row["decision_index"]): (dict(row), float(model_p), float(baseline_p))
        for row, model_p, baseline_p in zip(rows, model_probabilities, baseline_probabilities)
    }
    blocks, used = _embargoed_time_blocks(
        [item[0] for item in per_decision.values()],
        block_span_bars=block_size,
        embargo_bars=BLEND_EMBARGO_BARS,
    )
    if len(blocks) < BLEND_MIN_BLOCKS:
        return {
            "status": "INSUFFICIENT_EMBARGOED_TEMPORAL_BLOCK_SUPPORT",
            "rows": len(used), "holdout_rows_total": len(rows),
            "independent_blocks": len(blocks), "block_span_bars": block_size,
            "embargo_bars": BLEND_EMBARGO_BARS, "replicates": 0,
            "mean_improvement_baseline_minus_model": None, "ci95": None,
        }
    block_gains: list[float] = []
    for block in blocks:
        gains = []
        for row in block:
            _, model_p, baseline_p = per_decision[int(row["decision_index"])]
            y = int(row["label"])
            gains.append((baseline_p - y) ** 2 - (model_p - y) ** 2)
        if gains:
            block_gains.append(sum(gains) / len(gains))
    if len(block_gains) < BLEND_MIN_BLOCKS:
        return {
            "status": "INSUFFICIENT_EMBARGOED_TEMPORAL_BLOCK_SUPPORT",
            "rows": len(used), "holdout_rows_total": len(rows),
            "independent_blocks": len(block_gains), "block_span_bars": block_size,
            "embargo_bars": BLEND_EMBARGO_BARS, "replicates": 0,
            "mean_improvement_baseline_minus_model": None, "ci95": None,
        }
    rng = random.Random(seed)
    boot_means = sorted(
        sum(block_gains[rng.randrange(len(block_gains))] for _ in range(len(block_gains))) / len(block_gains)
        for _ in range(replicates)
    )
    lower = boot_means[max(0, math.ceil(0.025 * len(boot_means)) - 1)]
    upper = boot_means[min(len(boot_means) - 1, math.ceil(0.975 * len(boot_means)) - 1)]
    return {
        "status": "EMBARGOED_SOURCE_INDEX_BLOCK_BOOTSTRAP_DIAGNOSTIC",
        "rows": len(used), "holdout_rows_total": len(rows),
        "independent_blocks": len(block_gains), "block_span_bars": block_size,
        "embargo_bars": BLEND_EMBARGO_BARS, "replicates": replicates,
        "mean_improvement_baseline_minus_model": sum(block_gains) / len(block_gains),
        "ci95": [lower, upper],
        "sampling_method": "equal-weight paired Brier means from 12-source-bar blocks separated by a 12-source-bar target-horizon embargo",
        "interpretation": "This is an embargoed-block diagnostic, not an exact finite-sample confidence guarantee or selection-bias correction.",
    }


def _reliability_diagnostics(
    probabilities: Sequence[float], labels: Sequence[int], *, bins: int = 10
) -> dict[str, Any]:
    if len(probabilities) != len(labels):
        raise ValueError("reliability_inputs_must_have_equal_lengths")
    if not labels:
        return {"rows": 0, "mean_predicted_probability": None, "observed_positive_rate": None, "ece": None, "bins": []}
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, label in zip(probabilities, labels):
        p = min(max(float(probability), 0.0), 1.0)
        bucket_index = min(bins - 1, int(p * bins))
        buckets[bucket_index].append((p, int(label)))
    bin_reports = []
    ece = 0.0
    n = len(labels)
    for index, bucket in enumerate(buckets):
        if not bucket:
            bin_reports.append({"bin": index, "lower": index / bins, "upper": (index + 1) / bins, "rows": 0, "mean_predicted": None, "observed_rate": None})
            continue
        mean_predicted = sum(x[0] for x in bucket) / len(bucket)
        observed_rate = sum(x[1] for x in bucket) / len(bucket)
        ece += len(bucket) / n * abs(mean_predicted - observed_rate)
        bin_reports.append({"bin": index, "lower": index / bins, "upper": (index + 1) / bins, "rows": len(bucket), "mean_predicted": mean_predicted, "observed_rate": observed_rate})
    return {
        "rows": n,
        "mean_predicted_probability": sum(float(x) for x in probabilities) / n,
        "observed_positive_rate": sum(labels) / n,
        "ece": ece,
        "bins": bin_reports,
        "interpretation": "ECE depends on binning and is descriptive; Brier/log-loss and out-of-sample replication remain primary.",
    }


def _fit_calibrator(samples: Sequence[Mapping[str, Any]], raw_probabilities: Mapping[int, float]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in samples]
    probs = [raw_probabilities[int(row["decision_index"])] for row in samples]
    if len(samples) < MIN_CALIBRATION_ROWS or len(set(labels)) < 2:
        return {
            "status": "IDENTITY_INSUFFICIENT_CALIBRATION_SUPPORT",
            "intercept": 0.0, "slope": 1.0, "logit_mean": 0.0, "logit_scale": 1.0,
            "rows": len(samples),
        }
    logits = [_logit(p) for p in probs]
    mean = sum(logits) / len(logits)
    variance = sum((x - mean) ** 2 for x in logits) / len(logits)
    scale = math.sqrt(variance)
    if scale < 1e-6:
        return {
            "status": "IDENTITY_CONSTANT_RAW_SCORES",
            "intercept": 0.0, "slope": 1.0, "logit_mean": 0.0, "logit_scale": 1.0,
            "rows": len(samples),
        }
    xs = [(x - mean) / scale for x in logits]
    positives = sum(labels)
    intercept = math.log((positives + 1.0) / (len(labels) - positives + 1.0))
    slope = 0.0
    for epoch in range(CALIBRATION_EPOCHS):
        gi = 0.0
        gs = 0.0
        for x, y in zip(xs, labels):
            err = _sigmoid(intercept + slope * x) - y
            gi += err
            gs += err * x
        rate = 0.08 / math.sqrt(1.0 + epoch / 100.0)
        intercept -= rate * gi / len(labels)
        slope -= rate * (gs / len(labels) + 0.02 * slope)
        intercept = max(-8.0, min(8.0, intercept))
        slope = max(-5.0, min(5.0, slope))
    return {
        "status": "PLATT_LOGISTIC_CALIBRATION_RESEARCH_ONLY",
        "intercept": intercept, "slope": slope,
        "logit_mean": mean, "logit_scale": scale, "rows": len(samples),
    }


def _apply_calibrator(probability: float, calibrator: Mapping[str, Any]) -> float:
    if str(calibrator.get("status", "")).startswith("IDENTITY"):
        return min(max(float(probability), 1e-6), 1.0 - 1e-6)
    z = (_logit(probability) - float(calibrator["logit_mean"])) / float(calibrator["logit_scale"])
    return min(max(_sigmoid(float(calibrator["intercept"]) + float(calibrator["slope"]) * z), 1e-6), 1.0 - 1e-6)


def train_alpha_model(
    events: Sequence[Mapping[str, Any]],
    *,
    partition_start_indices: Mapping[str, int],
    as_of_index: int,
) -> dict[str, Any]:
    """Train/select/calibrate alpha without using labels outside each prior cutoff.

    Development fits a scout; model-selection chooses L2 on later matured labels;
    calibration refits using only labels available before calibration starts and fits
    Platt scaling only on calibration labels mature before final holdout; final holdout
    predictions use a frozen model and frozen calibrator. Development emits 0.5 cold-start
    forecasts because no training fit could have existed at the start of that partition.
    """
    required = ("development", "model_selection", "calibration", "final_holdout")
    if not isinstance(as_of_index, int) or isinstance(as_of_index, bool) or as_of_index < 1:
        raise ValueError("alpha_as_of_index_must_be_positive_integer")
    starts: dict[str, int] = {}
    for name in required:
        value = partition_start_indices.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("alpha_partition_start_missing_or_invalid:" + name)
        starts[name] = value
    if list(starts.values()) != sorted(starts.values()) or len(set(starts.values())) != 4:
        raise ValueError("alpha_partition_starts_must_be_strictly_chronological")

    development_start = starts["development"]
    selection_start = starts["model_selection"]
    calibration_start = starts["calibration"]
    holdout_start = starts["final_holdout"]

    development_train = _labelled_before(
        events, selection_start, {"development"}
    )
    selection_rows = [
        row for row in events
        if row.get("partition") == "model_selection"
        and isinstance(row.get("decision_index"), int) and not isinstance(row.get("decision_index"), bool)
        and isinstance(row.get("label_available_index"), int) and not isinstance(row.get("label_available_index"), bool)
        and row["decision_index"] < calibration_start
        and row["label_available_index"] < calibration_start
        and row.get("label_mature") is True and row.get("label") in (0, 1)
    ]
    candidate_metrics: list[dict[str, Any]] = []
    candidate_models: dict[float, dict[str, Any]] = {}
    scout_model = None
    if len(development_train) >= MIN_TRAIN_ROWS:
        if len(selection_rows) >= MIN_SELECTION_ROWS:
            # Fit each preregistered candidate once; reuse the default-L2 fit as the
            # frozen development scout so selection-period forecasts are not refit
            # after observing selection labels.
            candidate_models = {float(l2): _fit_logistic(development_train, float(l2)) for l2 in L2_GRID}
            scout_model = candidate_models.get(float(DEFAULT_L2))
            for l2 in L2_GRID:
                model = candidate_models[float(l2)]
                preds = [_predict_raw(model, row) for row in selection_rows]
                labels = [int(row["label"]) for row in selection_rows]
                candidate_metrics.append({
                    "l2": float(l2), "rows": len(labels), "brier": _brier(preds, labels),
                    "log_loss": _log_loss(preds, labels),
                })
            chosen = min(candidate_metrics, key=lambda x: (float(x["brier"]), float(x["l2"])))
            selected_l2 = float(chosen["l2"])
            selection_status = "SELECTED_BY_MODEL_SELECTION_BRIER_NOT_FINAL_HOLDOUT"
        else:
            scout_model = _fit_logistic(development_train, DEFAULT_L2)
            selected_l2 = DEFAULT_L2
            selection_status = "DEFAULT_L2_INSUFFICIENT_MODEL_SELECTION_SUPPORT"
    else:
        selected_l2 = DEFAULT_L2
        selection_status = "DEFAULT_L2_INSUFFICIENT_DEVELOPMENT_SUPPORT"

    final_train = _labelled_before(
        events, calibration_start, {"development", "model_selection"}
    )
    final_model = _fit_logistic(final_train, selected_l2) if len(final_train) >= MIN_TRAIN_ROWS else None

    calibration_width = max(0, holdout_start - calibration_start)
    calibration_fit_cutoff = calibration_start + max(1, int(calibration_width * 0.60))
    calibration_rows = [
        row for row in events
        if row.get("partition") == "calibration"
        and isinstance(row.get("decision_index"), int) and not isinstance(row.get("decision_index"), bool)
        and isinstance(row.get("label_available_index"), int) and not isinstance(row.get("label_available_index"), bool)
        and row["decision_index"] < holdout_start
        and row["label_available_index"] < holdout_start
        and row.get("label_mature") is True and row.get("label") in (0, 1)
    ]
    calibration_fit_rows = _labelled_before(
        events, calibration_fit_cutoff, {"calibration"}
    )
    raw_calibration = {
        int(row["decision_index"]): _predict_raw(final_model, row)
        for row in calibration_rows
    }
    calibrator = _fit_calibrator(calibration_fit_rows, raw_calibration)
    calibrated_calibration = {
        int(row["decision_index"]): _apply_calibrator(
            raw_calibration[int(row["decision_index"])], calibrator
        )
        for row in calibration_rows
    }
    blend_selection_rows = [
        row for row in calibration_rows
        if int(row["decision_index"]) >= calibration_fit_cutoff
        and int(row["label_available_index"]) < holdout_start
    ]
    probability_governor = _choose_baseline_shrinkage_weight(
        blend_selection_rows, calibrated_calibration
    )
    selected_alpha_weight = float(probability_governor["alpha_weight"])

    raw_predictions: dict[str, float] = {}
    calibrated_alpha_predictions: dict[str, float] = {}
    out_of_domain_fractions: dict[str, float | None] = {}
    forecast_policy_by_index: dict[str, str] = {}
    predictions: dict[str, float] = {}
    forecast_stages: dict[str, str] = {}
    for row in sorted(events, key=lambda r: int(r["decision_index"])):
        di = int(row["decision_index"])
        partition = row["partition"]
        # OOD scoring only applies once the final-holdout model is frozen. Earlier
        # partitions must not receive diagnostics computed with a model trained on later rows.
        ood_fraction = _out_of_domain_fraction(final_model, row) if partition == "final_holdout" else None
        forecast_policy = "ALPHA_ELIGIBLE" if partition == "final_holdout" else "NOT_APPLICABLE_PRE_FINAL_HOLDOUT"
        if partition == "development":
            raw_probability = 0.5
            calibrated_probability = 0.5
            probability = 0.5
            stage = "DEVELOPMENT_COLD_START_BASELINE"
        elif partition == "model_selection":
            raw_probability = _predict_raw(scout_model, row)
            calibrated_probability = raw_probability
            probability = raw_probability
            stage = "DEVELOPMENT_FIT_SCOUT_FIXED_BEFORE_MODEL_SELECTION"
        elif partition == "calibration":
            raw_probability = _predict_raw(final_model, row)
            calibrated_probability = _apply_calibrator(raw_probability, calibrator)
            # Calibration-block forecasts stay prequential/uncalibrated in the ledger;
            # the later calibration sub-block selects the frozen holdout governor only.
            probability = raw_probability
            stage = "REFIT_BEFORE_CALIBRATION_UNCALIBRATED"
        elif partition == "final_holdout":
            raw_probability = _predict_raw(final_model, row)
            calibrated_probability = _apply_calibrator(raw_probability, calibrator)
            baseline = min(max(float(row.get("baseline_probability", 0.5)), 1e-6), 1.0 - 1e-6)
            if ood_fraction is not None and ood_fraction >= OOD_MAX_FEATURE_FRACTION:
                probability = baseline
                forecast_policy = "BASELINE_FALLBACK_OUT_OF_DOMAIN_FEATURES"
                stage = "BASELINE_FALLBACK_OUT_OF_DOMAIN_FEATURES"
            else:
                probability = (1.0 - selected_alpha_weight) * baseline + selected_alpha_weight * calibrated_probability
                stage = "FROZEN_FINAL_MODEL_CALIBRATOR_AND_DUAL_SCORE_GOVERNOR"
        else:
            raise ValueError("alpha_unknown_partition:" + str(partition))
        raw_predictions[str(di)] = float(raw_probability)
        calibrated_alpha_predictions[str(di)] = float(calibrated_probability)
        out_of_domain_fractions[str(di)] = None if ood_fraction is None else float(ood_fraction)
        forecast_policy_by_index[str(di)] = forecast_policy
        predictions[str(di)] = min(max(float(probability), 1e-6), 1.0 - 1e-6)
        forecast_stages[str(di)] = stage

    holdout_rows = [
        row for row in events
        if row.get("partition") == "final_holdout"
        and row.get("label_mature") is True and row.get("label") in (0, 1)
        and isinstance(row.get("label_available_index"), int)
        and row["label_available_index"] < as_of_index
    ]
    holdout_labels = [int(row["label"]) for row in holdout_rows]
    holdout_alpha = [predictions[str(int(row["decision_index"]))] for row in holdout_rows]
    holdout_raw_alpha = [raw_predictions[str(int(row["decision_index"]))] for row in holdout_rows]
    holdout_calibrated_alpha = [calibrated_alpha_predictions[str(int(row["decision_index"]))] for row in holdout_rows]
    holdout_baseline = [
        min(max(float(row.get("baseline_probability", 0.5)), 1e-6), 1.0 - 1e-6)
        for row in holdout_rows
    ]
    alpha_brier = _brier(holdout_alpha, holdout_labels)
    raw_alpha_brier = _brier(holdout_raw_alpha, holdout_labels)
    calibrated_alpha_brier = _brier(holdout_calibrated_alpha, holdout_labels)
    baseline_brier = _brier(holdout_baseline, holdout_labels)
    alpha_logloss = _log_loss(holdout_alpha, holdout_labels)
    calibrated_alpha_logloss = _log_loss(holdout_calibrated_alpha, holdout_labels)
    baseline_logloss = _log_loss(holdout_baseline, holdout_labels)
    holdout_brier_interval = _paired_block_bootstrap_ci(
        holdout_rows, holdout_alpha, holdout_baseline,
    )
    holdout_reliability = _reliability_diagnostics(holdout_alpha, holdout_labels)
    holdout_ood_flags = [
        float(out_of_domain_fractions[str(int(row["decision_index"]))] or 0.0) >= OOD_MAX_FEATURE_FRACTION
        for row in holdout_rows
    ]
    holdout_ood_fraction_rate = (sum(holdout_ood_flags) / len(holdout_ood_flags)) if holdout_ood_flags else None

    calibration_labels_sha = _sha([
        [row.get("decision_id", row.get("decision_index")), int(row["label"])]
        for row in calibration_fit_rows
    ])
    blend_selection_labels_sha = probability_governor.get("selection_labels_sha256")
    model_payload = {
        "protocol": MODEL_VERSION,
        "features": list(FEATURE_NAMES),
        "selection_status": selection_status,
        "selected_l2": selected_l2,
        "development_start_index": development_start,
        "model_selection_start_index": selection_start,
        "calibration_start_index": calibration_start,
        "calibration_fit_cutoff_index": calibration_fit_cutoff,
        "final_holdout_start_index": holdout_start,
        "training_model": final_model,
        "calibrator": calibrator,
        "calibration_fit_labels_sha256": calibration_labels_sha,
        "blend_selection_labels_sha256": blend_selection_labels_sha,
        "selected_alpha_weight": selected_alpha_weight,
        "probability_governor_protocol": "embargoed_source_index_block_dual_score_ood_fallback_v4",
        "probability_governor_policy": {
            "blend_weights": list(BLEND_WEIGHTS),
            "critical_multiplier": BLEND_CRITICAL_Z,
            "block_span_source_bars": BLEND_BLOCK_SIZE,
            "embargo_source_bars": BLEND_EMBARGO_BARS,
            "minimum_positive_block_share": MIN_POSITIVE_BLOCK_SHARE,
            "minimum_calibration_rows": MIN_BLEND_SELECTION_ROWS,
            "minimum_independent_blocks": BLEND_MIN_BLOCKS,
        },
        "out_of_domain_policy": {
            "z_score_limit": OOD_Z_LIMIT,
            "minimum_feature_fraction_to_fallback": OOD_MAX_FEATURE_FRACTION,
            "fallback_probability": "exact_rowwise_causal_baseline",
        },
    }
    model_hash = _sha(model_payload)
    return {
        "model_version": MODEL_VERSION,
        "status": "ALPHA_RESEARCH_TRAINED" if final_model is not None else "ALPHA_RESEARCH_INSUFFICIENT_TRAINING_SUPPORT",
        "selected_l2": selected_l2,
        "selection_status": selection_status,
        "candidate_selection_metrics": candidate_metrics,
        "partition_start_indices": starts,
        "training_policy": {
            "development_train_rows": len(development_train),
            "final_model_train_rows": len(final_train),
            "calibration_rows": len(calibration_rows),
            "calibration_fit_rows": len(calibration_fit_rows),
            "blend_selection_rows": len(blend_selection_rows),
            "calibration_fit_cutoff_index": calibration_fit_cutoff,
            "holdout_rows_scored": len(holdout_rows),
            "max_final_training_label_available_index": max(
                (int(row["label_available_index"]) for row in final_train), default=None
            ),
            "strict_training_maturity_cutoff_index": calibration_start,
            "model_selection_label_cutoff_index": calibration_start,
            "calibration_label_cutoff_index": holdout_start,
            "final_holdout_labels_used_for_fit_or_calibration": False,
            "final_holdout_predictions_are_frozen": True,
        },
        "calibrator": calibrator,
        "final_model_coefficients": None if final_model is None else {
            "intercept": final_model["intercept"],
            "coefficients_by_feature": dict(zip(FEATURE_NAMES, final_model["coefficients"])),
            "means_by_feature": dict(zip(FEATURE_NAMES, final_model["means"])),
            "scales_by_feature": dict(zip(FEATURE_NAMES, final_model["scales"])),
            "l2": final_model["l2"],
            "train_labels_sha256": final_model["train_labels_sha256"],
        },
        "final_holdout_metrics": {
            "rows": len(holdout_rows),
            "raw_alpha_brier_score": raw_alpha_brier,
            "calibrated_alpha_brier_score": calibrated_alpha_brier,
            "alpha_brier_score": alpha_brier,
            "causal_baseline_brier_score": baseline_brier,
            "paired_brier_improvement_ci_vs_causal_baseline": holdout_brier_interval,
            "reliability_diagnostics": holdout_reliability,
            "out_of_domain_fallback_rows": sum(holdout_ood_flags),
            "out_of_domain_fallback_rate": holdout_ood_fraction_rate,
            "out_of_domain_policy": {
                "z_score_limit": OOD_Z_LIMIT,
                "minimum_feature_fraction_to_fallback": OOD_MAX_FEATURE_FRACTION,
                "fallback_probability": "exact_rowwise_causal_baseline",
            },
            "brier_improvement_baseline_minus_alpha": (
                None if alpha_brier is None or baseline_brier is None else baseline_brier - alpha_brier
            ),
            "alpha_log_loss": alpha_logloss,
            "calibrated_alpha_log_loss": calibrated_alpha_logloss,
            "causal_baseline_log_loss": baseline_logloss,
            "log_loss_improvement_baseline_minus_alpha": (
                None if alpha_logloss is None or baseline_logloss is None else baseline_logloss - alpha_logloss
            ),
            "evaluation_is_research_only": True,
            "positive_improvement_is_not_economic_or_independent_validation": True,
        },
        "model_artifact_sha256": model_hash,
        "probability_governor": probability_governor,
        "predictions_by_decision_index": predictions,
        "raw_predictions_by_decision_index": raw_predictions,
        "calibrated_alpha_predictions_by_decision_index": calibrated_alpha_predictions,
        "out_of_domain_fractions_by_decision_index": out_of_domain_fractions,
        "forecast_policy_by_decision_index": forecast_policy_by_index,
        "forecast_stage_by_decision_index": forecast_stages,
        "model_validated": False,
        "deployment_calibrated": False,
        "live_signal_enabled": False,
        "live_order_execution_enabled": False,
        "decision_output": "NO_TRADE",
    }
