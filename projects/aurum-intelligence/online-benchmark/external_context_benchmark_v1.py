"""Same-row, chronological external-context ablation benchmark; research only."""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "temporal-calibration-v1.3"
if str(_CALIBRATION_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIBRATION_DIR))
from calibration_diagnostics import apply_platt_scaler, fit_platt_scaler, probability_metrics

from point_in_time_features_v1 import THEME_WHITELIST

PROTOCOL_VERSION = "aurum-external-context-ablation-v1"
DEV_FRACTION = 0.70
CALIBRATION_FRACTION = 0.80
HOLDOUT_FRACTION = 0.20
L2_LAMBDA = 1.0
LEARNING_RATE = 0.05
MAX_STEPS = 200
MIN_CALIBRATION_SAMPLES = 100
MIN_SCALE_RECORDS = 100_000
MIN_CONTEXT_FAMILY_SUPPORT_ROWS = 30
VALID_CONTEXT_SOURCE_STATES = frozenset({"PASS", "PARTIAL"})
DEFAULT_PRICE_FEATURES = (
    "return_1", "return_3", "return_6", "return_12", "momentum_6",
    "atr_fraction", "atr_pct", "spread_fraction", "spread", "hour_sin",
    "hour_cos", "weekday_sin", "weekday_cos", "volatility_12",
    "close_return_1", "close_return_3", "close_return_12", "rsi_14",
)
CONTEXT_FEATURES = {
    "cot": (
        "feature_cot_net_position", "feature_cot_net_position_change",
        "feature_cot_net_position_oi_fraction", "feature_cot_managed_money_imbalance",
        "feature_cot_net_position_oi_change", "feature_cot_open_interest_change_fraction",
    ),
    "macro": ("feature_macro_level", "feature_macro_change"),
    "news": (
        "feature_news_count_1h", "feature_news_count_24h", "feature_news_tone_mean_24h",
        *(f"feature_news_{theme.lower()}_count_24h" for theme in THEME_WHITELIST),
    ),
}
ABLATIONS = {
    "price_only": (),
    "price_cot": ("cot",),
    "price_macro": ("macro",),
    "price_news": ("news",),
    "price_combined": ("cot", "macro", "news"),
}


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=_json_default, allow_nan=False).encode()).hexdigest()


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _aware_utc(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name}_must_be_timezone_aware")
    return value.astimezone(timezone.utc)


def _safe_probability(value: float) -> float:
    return max(1e-12, min(1.0 - 1e-12, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def _label(row: Mapping[str, Any]) -> int:
    raw = row.get("label", row.get("actual_tp"))
    if isinstance(raw, bool) or raw not in (0, 1):
        raise ValueError("label_must_be_binary_0_or_1")
    return int(raw)


def _row_facts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    last_idx: int | None = None
    last_time: datetime | None = None
    for i, row in enumerate(rows):
        decision_id = str(row.get("decision_id", ""))
        if not decision_id or decision_id in seen:
            raise ValueError(f"missing_or_duplicate_decision_id_at_{i}")
        seen.add(decision_id)
        decision_time = _aware_utc(row.get("decision_time"), "decision_time")
        decision_index = row.get("decision_index")
        if decision_index is None:
            decision_index = i
        exit_index = row.get("target_exit_index")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (decision_index, exit_index)):
            raise ValueError(f"invalid_decision_or_exit_index_at_{i}")
        if exit_index < decision_index:
            raise ValueError(f"exit_precedes_decision_at_{i}")
        if last_idx is not None and decision_index <= last_idx:
            raise ValueError("decision_indices_must_be_strictly_increasing")
        if last_time is not None and decision_time <= last_time:
            raise ValueError("decision_times_must_be_strictly_increasing")
        mature_flag = row.get("label_mature", True)
        if not isinstance(mature_flag, bool):
            raise ValueError(f"label_mature_must_be_boolean_at_{i}")
        fact = {
            "decision_id": decision_id, "decision_time": decision_time, "decision_index": decision_index,
            "target_exit_index": exit_index, "label": _label(row), "label_mature": mature_flag,
            "row": row,
        }
        facts.append(fact)
        last_idx, last_time = decision_index, decision_time
    if len(facts) < 50:
        raise ValueError("at_least_50_chronological_rows_required")
    return facts


def prequential_base_rate_predictions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Laplace-smoothed base rate using only previously decided, already matured labels."""
    facts = _row_facts_allow_small(rows)
    facts.sort(key=lambda x: (x["decision_index"], x["decision_id"]))
    pending: list[tuple[int, int, int, int]] = []
    matured_n = 0
    matured_positive = 0
    result: list[dict[str, Any]] = []
    for position, fact in enumerate(facts):
        decision_index = fact["decision_index"]
        while pending and pending[0][0] <= decision_index:
            _exit, _pos, label, _seq = heapq.heappop(pending)
            matured_n += 1
            matured_positive += label
        result.append({
            "decision_id": fact["decision_id"],
            "decision_index": decision_index,
            "probability": (matured_positive + 1.0) / (matured_n + 2.0),
            "matured_history_count": matured_n,
        })
        if fact["label_mature"]:
            heapq.heappush(pending, (
                fact["target_exit_index"], position, fact["label"], position
            ))
    # Map results back to the caller's given order for predictable row joins.
    by_id = {row["decision_id"]: row for row in result}
    return [by_id[str(row.get("decision_id", ""))] for row in rows]


def _row_facts_allow_small(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Public helper accepts tiny fixtures; benchmark itself enforces the 50-row minimum.
    facts = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        decision_id = str(row.get("decision_id", ""))
        if not decision_id or decision_id in seen:
            raise ValueError(f"missing_or_duplicate_decision_id_at_{i}")
        seen.add(decision_id)
        decision_time = _aware_utc(row.get("decision_time"), "decision_time")
        decision_index = row.get("decision_index", i)
        exit_index = row.get("target_exit_index", decision_index)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (decision_index, exit_index)):
            raise ValueError(f"invalid_decision_or_exit_index_at_{i}")
        if exit_index < decision_index:
            raise ValueError(f"exit_precedes_decision_at_{i}")
        mature_flag = row.get("label_mature", True)
        if not isinstance(mature_flag, bool):
            raise ValueError(f"label_mature_must_be_boolean_at_{i}")
        facts.append({
            "decision_id": decision_id, "decision_time": decision_time, "decision_index": decision_index,
            "target_exit_index": exit_index, "label": _label(row), "label_mature": mature_flag,
            "row": row,
        })
    return facts


def paired_moving_block_brier_ci(
    model_probabilities: Sequence[float], model_outcomes: Sequence[int],
    baseline_probabilities: Sequence[float], baseline_outcomes: Sequence[int], *,
    block_size: int = 12, replicates: int = 250, seed: int = 20261017,
) -> dict[str, Any]:
    if not (len(model_probabilities) == len(model_outcomes) == len(baseline_probabilities) == len(baseline_outcomes)):
        raise ValueError("paired_bootstrap_lengths_must_match")
    if len(model_probabilities) < 2:
        raise ValueError("paired_bootstrap_needs_at_least_two_rows")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size_must_be_positive")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or not 20 <= replicates <= 5000:
        raise ValueError("replicates_must_be_between_20_and_5000")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed_must_be_integer")
    if list(model_outcomes) != list(baseline_outcomes):
        raise ValueError("paired_bootstrap_requires_identical_labels")
    n = len(model_probabilities)
    size = min(block_size, n)
    improvements = []
    for p, y, bp in zip(model_probabilities, model_outcomes, baseline_probabilities):
        if y not in (0, 1) or not 0 <= float(p) <= 1 or not 0 <= float(bp) <= 1:
            raise ValueError("invalid_bootstrap_probability_or_label")
    paired_differences = [
        (_safe_probability(float(bp)) - int(y)) ** 2 - (_safe_probability(float(p)) - int(y)) ** 2
        for p, y, bp in zip(model_probabilities, model_outcomes, baseline_probabilities)
    ]
    rng = random.Random(seed)
    for _ in range(replicates):
        sampled: list[float] = []
        while len(sampled) < n:
            start = rng.randrange(0, n - size + 1)
            sampled.extend(paired_differences[start:start + size])
        improvements.append(sum(sampled[:n]) / n)
    improvements.sort()
    return {
        "method": "paired_moving_block_bootstrap_percentile",
        "block_size": size, "seed": seed, "replicates": replicates,
        "brier_improvement_baseline_minus_model": sum(paired_differences) / n,
        "lower_95": improvements[max(0, int(0.025 * (replicates - 1)))],
        "upper_95": improvements[min(replicates - 1, int(0.975 * (replicates - 1)))],
        "interpretation": "Positive favors model Brier over the same-row causal baseline; sensitivity diagnostic only.",
    }


def _numeric(row: Mapping[str, Any], key: str, family: str | None = None) -> float | None:
    if family:
        if family == "news":
            field = "coverage_news_1h" if key == "feature_news_count_1h" else "coverage_news_24h"
            coverage = str(row.get(field, row.get("coverage_news", "unknown")))
            allowed = {"available", "available_zero_events"}
        else:
            coverage = str(row.get(f"coverage_{family}", "unknown"))
            allowed = {"cot": {"available"}, "macro": {"available"}}[family]
        if coverage not in allowed:
            return None
        availability_map = row.get("context_available_at_by_family")
        if isinstance(availability_map, Mapping) and family in availability_map:
            available_at = _aware_utc(availability_map[family], f"context_available_at_{family}")
            decision_time = _aware_utc(row.get("decision_time"), "decision_time")
            if available_at > decision_time:
                return None
        latest_map = row.get("feature_family_latest_available_at")
        if isinstance(latest_map, Mapping) and family in latest_map:
            latest_at = _aware_utc(latest_map[family], f"feature_family_latest_available_at_{family}")
            decision_time = _aware_utc(row.get("decision_time"), "decision_time")
            if latest_at > decision_time:
                return None
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _family_of_key(key: str) -> str | None:
    for family, keys in CONTEXT_FEATURES.items():
        if key in keys:
            return family
    return None


def _fit_scaler(rows: Sequence[Mapping[str, Any]], feature_keys: Sequence[str]) -> tuple[list[str], dict[str, dict[str, float]]]:
    fitted_keys: list[str] = []
    parameters: dict[str, dict[str, float]] = {}
    for key in feature_keys:
        family = _family_of_key(key)
        values = [v for row in rows if (v := _numeric(row, key, family)) is not None]
        if not values:
            continue
        values.sort()
        middle = len(values) // 2
        median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        scale = math.sqrt(variance)
        if not math.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        fitted_keys.append(key)
        parameters[key] = {"median": median, "mean": mean, "scale": scale}
    return fitted_keys, parameters


def _matrix(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], params: Mapping[str, Mapping[str, float]]) -> list[list[float]]:
    matrix = []
    for row in rows:
        out = []
        for key in keys:
            family = _family_of_key(key)
            value = _numeric(row, key, family)
            p = params[key]
            if value is None:
                value = p["median"]
            standardized = (value - p["mean"]) / p["scale"]
            out.append(max(-10.0, min(10.0, standardized)))
        matrix.append(out)
    return matrix


def _fit_logistic(x: Sequence[Sequence[float]], y: Sequence[int]) -> dict[str, Any]:
    if len(x) != len(y) or not x or not x[0] or any(len(row) != len(x[0]) for row in x):
        raise ValueError("invalid_logistic_training_matrix")
    if any(label not in (0, 1) for label in y):
        raise ValueError("logistic_labels_must_be_binary")
    n, p = len(x), len(x[0])
    weights = [0.0] * p
    bias = 0.0
    for _ in range(MAX_STEPS):
        grad_w = [0.0] * p
        grad_b = 0.0
        for features, label in zip(x, y):
            probability = _sigmoid(sum(w * value for w, value in zip(weights, features)) + bias)
            error = probability - label
            grad_b += error / n
            for j, value in enumerate(features):
                grad_w[j] += error * value / n
        for j in range(p):
            grad_w[j] += L2_LAMBDA * weights[j] / n
            weights[j] -= LEARNING_RATE * grad_w[j]
        bias -= LEARNING_RATE * grad_b
    if not all(math.isfinite(v) for v in [*weights, bias]):
        raise ValueError("nonfinite_logistic_parameters")
    return {"weights": weights, "bias": bias, "steps": MAX_STEPS, "l2_lambda": L2_LAMBDA,
            "learning_rate": LEARNING_RATE, "initialization": "zeros", "optimizer": "full_batch_gradient_descent"}


def _predict(model: Mapping[str, Any], x: Sequence[Sequence[float]]) -> list[float]:
    weights = model["weights"]
    bias = model["bias"]
    return [_safe_probability(_sigmoid(sum(w * value for w, value in zip(weights, row)) + bias)) for row in x]


def _metric_bundle(probabilities: Sequence[float], labels: Sequence[int], baseline: Sequence[float]) -> dict[str, Any]:
    p = probability_metrics(probabilities, labels, bins=10)
    brier = sum((float(pv) - int(y)) ** 2 for pv, y in zip(probabilities, labels)) / len(labels)
    log_loss = -sum(
        int(y) * math.log(_safe_probability(float(pv))) + (1 - int(y)) * math.log(1 - _safe_probability(float(pv)))
        for pv, y in zip(probabilities, labels)
    ) / len(labels)
    base_brier = sum((float(bp) - int(y)) ** 2 for bp, y in zip(baseline, labels)) / len(labels)
    base_ll = -sum(
        int(y) * math.log(_safe_probability(float(bp))) + (1 - int(y)) * math.log(1 - _safe_probability(float(bp)))
        for bp, y in zip(baseline, labels)
    ) / len(labels)
    return {
        "n": len(labels), "brier_score": brier, "log_loss": log_loss,
        "ece_equal_width_descriptive": p.get("ece_equal_width_descriptive"),
        "mce_equal_width_descriptive": p.get("mce_equal_width_descriptive"),
        "reliability_bins": p.get("reliability_bins"),
        "causal_prequential_baseline_brier": base_brier, "causal_prequential_baseline_log_loss": base_ll,
        "brier_skill_vs_prequential": 1 - brier / base_brier if base_brier > 0 else None,
        "log_loss_improvement_vs_prequential": base_ll - log_loss,
        "event_rate": p.get("event_rate"), "mean_probability": p.get("mean_probability"),
    }


def _coverage_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {}
    for family in ("cot", "macro"):
        allowed = {"cot": {"available"}, "macro": {"available"}}[family]
        count = sum(str(row.get(f"coverage_{family}", "unknown")) in allowed for row in rows)
        summary[family] = {"eligible_rows": count, "total_rows": len(rows),
                           "coverage_fraction": count / len(rows) if rows else 0.0}
    for key, field in (
        ("news", "coverage_news"),
        ("news_1h", "coverage_news_1h"),
        ("news_24h", "coverage_news_24h"),
    ):
        allowed = {"available", "available_zero_events"}
        count = sum(str(row.get(field, row.get("coverage_news", "unknown"))) in allowed for row in rows)
        summary[key] = {"eligible_rows": count, "total_rows": len(rows),
                        "coverage_fraction": count / len(rows) if rows else 0.0}
    return summary


def _calibration_status(fit: Mapping[str, Any]) -> str:
    status = str(fit.get("status", "unknown"))
    return {
        "insufficient_calibration_samples": "INSUFFICIENT_CALIBRATION_SAMPLES",
        "single_class_calibration_data": "SINGLE_CLASS_CALIBRATION_DATA",
        "fit_complete": "FIT_COMPLETE",
    }.get(status, status.upper())


def _family_support(rows: Sequence[Mapping[str, Any]], family: str) -> dict[str, Any]:
    """Count as-of-eligible development/calibration/holdout rows by context family."""
    keys = CONTEXT_FEATURES[family]
    per_feature = {
        key: sum(_numeric(row, key, family) is not None for row in rows)
        for key in keys
    }
    eligible_rows = sum(
        any(_numeric(row, key, family) is not None for key in keys)
        for row in rows
    )
    return {
        "eligible_rows": eligible_rows,
        "total_rows": len(rows),
        "eligible_fraction": eligible_rows / len(rows) if rows else 0.0,
        "eligible_values_by_feature": per_feature,
        "minimum_eligible_rows_required": MIN_CONTEXT_FAMILY_SUPPORT_ROWS,
    }


def run_context_ablation_benchmark(
    rows: Sequence[Mapping[str, Any]], *,
    context_observation_count: int,
    target_context_records: int = MIN_SCALE_RECORDS,
    price_feature_keys: Sequence[str] | None = None,
    source_statuses: Mapping[str, str] | None = None,
    bootstrap_replicates: int = 250,
    block_size: int = 12,
    seed: int = 20261017,
) -> dict[str, Any]:
    if isinstance(context_observation_count, bool) or not isinstance(context_observation_count, int) or context_observation_count < 0:
        raise ValueError("context_observation_count_must_be_nonnegative_integer")
    if target_context_records < 1:
        raise ValueError("target_context_records_must_be_positive")
    facts = _row_facts(rows)
    n = len(facts)
    dev_stop = int(n * DEV_FRACTION)
    cal_stop = int(n * CALIBRATION_FRACTION)
    if dev_stop < 30 or cal_stop <= dev_stop or n - cal_stop < 20:
        raise ValueError("insufficient_chronological_slices")
    first_cal_index = facts[dev_stop]["decision_index"]
    first_holdout_index = facts[cal_stop]["decision_index"]
    maximum_observed_index = max(f["decision_index"] for f in facts)

    development = [f for f in facts[:dev_stop] if f["label_mature"] and f["target_exit_index"] <= first_cal_index]
    calibration = [f for f in facts[dev_stop:cal_stop]
                   if f["label_mature"] and f["target_exit_index"] <= first_holdout_index]
    holdout = [f for f in facts[cal_stop:]
               if f["label_mature"] and f["target_exit_index"] <= maximum_observed_index]
    if len(development) < 30 or len(holdout) < 20:
        raise ValueError("insufficient_matured_development_or_holdout_rows")
    dev_rows = [f["row"] for f in development]
    cal_rows = [f["row"] for f in calibration]
    hold_rows = [f["row"] for f in holdout]
    dev_labels = [f["label"] for f in development]
    cal_labels = [f["label"] for f in calibration]
    hold_labels = [f["label"] for f in holdout]
    if len(set(dev_labels)) < 2:
        raise ValueError("development_labels_must_contain_both_classes")

    if price_feature_keys is None:
        candidate_price_keys = [key for key in DEFAULT_PRICE_FEATURES if any(key in row for row in dev_rows)]
    else:
        candidate_price_keys = list(price_feature_keys)
        if not candidate_price_keys or any(not isinstance(key, str) or not key for key in candidate_price_keys):
            raise ValueError("price_feature_keys_must_be_nonempty")
        if any(key in CONTEXT_FEATURES[family] for key in candidate_price_keys for family in CONTEXT_FEATURES):
            raise ValueError("price_features_cannot_include_context_columns")

    baseline_predictions = prequential_base_rate_predictions(rows)
    baseline_by_id = {x["decision_id"]: x for x in baseline_predictions}
    baseline_holdout = [float(baseline_by_id[f["decision_id"]]["probability"]) for f in holdout]
    outcome_hash = _sha(hold_labels)
    row_hash = _sha([f["decision_id"] for f in holdout])
    family_source_statuses = dict(source_statuses or {})
    scale_met = context_observation_count >= target_context_records
    scale_gate = {
        "status": "PASS" if scale_met else "NOT_MET",
        "observations": context_observation_count,
        "target": target_context_records,
        "passed": bool(scale_met),
        "synthetic_or_duplicated_observations_added": 0,
    }
    report: dict[str, Any] = {
        "protocol": {
            "version": PROTOCOL_VERSION,
            "development_fraction_end": DEV_FRACTION,
            "calibration_fraction_end": CALIBRATION_FRACTION,
            "final_holdout_fraction": HOLDOUT_FRACTION,
            "development_row_count": len(development),
            "calibration_row_count_mature_after_purge": len(calibration),
            "holdout_row_count_mature": len(holdout),
            "first_calibration_decision_index": first_cal_index,
            "first_holdout_decision_index": first_holdout_index,
            "maximum_observed_decision_index": maximum_observed_index,
            "boundary_purge": True,
            "holdout_used_for_selection": False,
            "holdout_used_for_calibration": False,
            "feature_scaler_fitted_on": "matured_development_rows_only",
            "model_fitted_on": "matured_development_rows_only",
            "platt_fitted_on": "matured_calibration_rows_only",
            "holdout_role": "final_scoring_only",
            "logistic": {"lambda": L2_LAMBDA, "learning_rate": LEARNING_RATE, "max_steps": MAX_STEPS,
                         "initialization": "zero", "solver": "full_batch_gradient_descent"},
            "price_features": candidate_price_keys,
            "context_features": {k: list(v) for k, v in CONTEXT_FEATURES.items()},
            "min_platt_calibration_samples": MIN_CALIBRATION_SAMPLES,
        },
        "scale_gate": scale_gate,
        "source_statuses": family_source_statuses,
        "coverage": _coverage_summary([f["row"] for f in facts]),
        "cohort": {
            "input_rows": n, "scored_row_count": len(holdout), "scored_rows_sha256": row_hash,
            "labels_sha256": outcome_hash,
        },
        "causal_prequential_baseline": _metric_bundle(baseline_holdout, hold_labels, baseline_holdout),
        "ablations": {},
        "decision_gate": "NO_TRADE", "deployment_calibrated": False,
        "validated": False, "live_signal_enabled": False,
        "research_only": True,
        "limitations": [
            "Pipeline success is not proof of alpha or profitable execution.",
            "Calibration is scored on one fixed final chronological tail and is not deployment validation.",
            "Source rights, upstream independence, broker equivalence, and exact release time require independent evidence.",
            "The source scale gate is independent from software CI and is NOT_MET until real unique observations reach the target.",
        ],
    }

    price_only_holdout_probabilities: list[float] | None = None
    for ablation_name, context_families in ABLATIONS.items():
        requested_context_families = list(context_families)
        requested = list(candidate_price_keys)
        context_support = {
            family: {
                "development": _family_support(dev_rows, family),
                "calibration": _family_support(cal_rows, family),
                "holdout": _family_support(hold_rows, family),
            }
            for family in requested_context_families
        }
        requested_context_features = [
            key for family in requested_context_families for key in CONTEXT_FEATURES[family]
        ]
        context_features_meeting_split_support = [
            key for key in requested_context_features
            if all(
                context_support[_family_of_key(key)][split]["eligible_values_by_feature"].get(key, 0)
                >= MIN_CONTEXT_FAMILY_SUPPORT_ROWS
                for split in ("development", "calibration", "holdout")
            )
        ]
        # Admit a context feature only when that exact key meets the support floor in all splits.
        requested.extend(context_features_meeting_split_support)
        feature_keys, scaler = _fit_scaler(dev_rows, requested)
        effective_context_feature_keys = [
            key for key in feature_keys if _family_of_key(key) in requested_context_families
        ]
        missing_family_reasons: dict[str, list[str]] = {}
        for family in requested_context_families:
            reasons = []
            declared_source_status = family_source_statuses.get(family)
            if declared_source_status is not None and str(declared_source_status).upper() not in VALID_CONTEXT_SOURCE_STATES:
                reasons.append("source_status_not_qualified_for_feature_use")
            if not any(_family_of_key(key) == family for key in context_features_meeting_split_support):
                reasons.append("no_context_feature_meets_minimum_support_across_splits")
            for split in ("development", "calibration", "holdout"):
                if context_support[family][split]["eligible_rows"] < MIN_CONTEXT_FAMILY_SUPPORT_ROWS:
                    reasons.append(f"insufficient_{split}_support")
            if reasons:
                missing_family_reasons[family] = reasons
        effective_context_families = [
            family for family in requested_context_families if family not in missing_family_reasons
        ]
        omitted_context_families = [
            family for family in requested_context_families if family in missing_family_reasons
        ]
        semantic_metadata = {
            "requested_context_families": requested_context_families,
            "effective_context_families": effective_context_families,
            "omitted_context_families": omitted_context_families,
            "requested_context_feature_keys": requested_context_features,
            "effective_context_feature_keys": effective_context_feature_keys,
            "omitted_context_feature_keys": [
                key for key in requested_context_features if key not in effective_context_feature_keys
            ],
            "context_family_support_by_split": context_support,
            "minimum_context_family_support_rows": MIN_CONTEXT_FAMILY_SUPPORT_ROWS,
            "missing_context_family_reasons": missing_family_reasons,
        }
        if not feature_keys:
            report["ablations"][ablation_name] = {
                "status": "NOT_ELIGIBLE_NO_DEVELOPMENT_FEATURES",
                **semantic_metadata,
            }
            continue
        if omitted_context_families:
            report["ablations"][ablation_name] = {
                "status": "NOT_ELIGIBLE_REQUIRED_CONTEXT_FAMILY_MISSING",
                **semantic_metadata,
            }
            continue
        try:
            x_dev = _matrix(dev_rows, feature_keys, scaler)
            x_cal = _matrix(cal_rows, feature_keys, scaler) if cal_rows else []
            x_hold = _matrix(hold_rows, feature_keys, scaler)
            model = _fit_logistic(x_dev, dev_labels)
            raw_cal = _predict(model, x_cal) if x_cal else []
            raw_hold = _predict(model, x_hold)
        except ValueError as exc:
            report["ablations"][ablation_name] = {"status": "FAILED", "error_code": str(exc)}
            continue
        calibration_fit = fit_platt_scaler(raw_cal, cal_labels, min_samples=MIN_CALIBRATION_SAMPLES)
        calibration_status = _calibration_status(calibration_fit)
        if calibration_fit.get("fitted"):
            scored_probabilities = [
                apply_platt_scaler(p, slope=calibration_fit["slope"], intercept=calibration_fit["intercept"])
                for p in raw_hold
            ]
        else:
            # Report raw holdout diagnostics but label them uncalibrated; never claim calibrated quality.
            scored_probabilities = list(raw_hold)
        metrics = _metric_bundle(scored_probabilities, hold_labels, baseline_holdout)
        if ablation_name == "price_only":
            price_only_holdout_probabilities = list(scored_probabilities)
            incremental_ci = None
        elif price_only_holdout_probabilities is not None:
            incremental_ci = paired_moving_block_brier_ci(
                scored_probabilities, hold_labels,
                price_only_holdout_probabilities, hold_labels,
                block_size=block_size, replicates=bootstrap_replicates,
                seed=seed + 100 + list(ABLATIONS).index(ablation_name),
            )
            incremental_ci["comparison_reference"] = "price_only"
            incremental_ci["brier_improvement_price_only_minus_model"] = incremental_ci[
                "brier_improvement_baseline_minus_model"
            ]
            incremental_ci["interpretation"] = (
                "Positive favors the context ablation over price-only on identical holdout rows and labels; "
                "interval crossing zero is inconclusive."
            )
        else:
            incremental_ci = None
        block_ci = paired_moving_block_brier_ci(
            scored_probabilities, hold_labels, baseline_holdout, hold_labels,
            block_size=block_size, replicates=bootstrap_replicates,
            seed=seed + list(ABLATIONS).index(ablation_name),
        )
        per_regime: dict[str, Any] = {}
        for idx, fact in enumerate(holdout):
            regime = str(fact["row"].get("market_regime", "unclassified"))[:80]
            per_regime.setdefault(regime, {"probabilities": [], "labels": []})
            per_regime[regime]["probabilities"].append(scored_probabilities[idx])
            per_regime[regime]["labels"].append(hold_labels[idx])
        regime_summary = {}
        for regime, bucket in sorted(per_regime.items()):
            if len(bucket["labels"]) < 20:
                regime_summary[regime] = {"status": "INSUFFICIENT_ROWS", "n": len(bucket["labels"])}
            else:
                rm = probability_metrics(bucket["probabilities"], bucket["labels"])
                regime_summary[regime] = {"status": "DESCRIPTIVE", "n": len(bucket["labels"]),
                    "brier_score": rm["brier_score"], "log_loss": rm["log_loss"],
                    "ece_equal_width_descriptive": rm["ece_equal_width_descriptive"],
                    "mce_equal_width_descriptive": rm["mce_equal_width_descriptive"]}
        scaler_hash = _sha({"feature_keys": feature_keys, "parameters": scaler})
        model_hash = _sha(model)
        platt_hash = _sha({k: v for k, v in calibration_fit.items()
                           if k in {"fitted", "status", "slope", "intercept", "n", "positive_count", "negative_count"}})
        prediction_hash = _sha([
            {"decision_id": fact["decision_id"], "probability": scored_probabilities[i]}
            for i, fact in enumerate(holdout)
        ])
        report["ablations"][ablation_name] = {
            "status": "SCORED",
            "feature_keys": feature_keys,
            # Compatibility field now reflects actually effective, not merely requested, families.
            "context_families": effective_context_families,
            **semantic_metadata,
            "scored_row_count": len(holdout),
            "scored_rows_sha256": row_hash,
            "labels_sha256": outcome_hash,
            "development_row_ids_sha256": _sha([f["decision_id"] for f in development]),
            "calibration_row_ids_sha256": _sha([f["decision_id"] for f in calibration]),
            "training_label_count": len(dev_labels),
            "calibration_label_count": len(cal_labels),
            "scaler_sha256": scaler_hash,
            "model_sha256": model_hash,
            "platt_parameters_sha256": platt_hash,
            "prediction_sha256": prediction_hash,
            "calibration_status": calibration_status,
            "calibration_applied": bool(calibration_fit.get("fitted")),
            "calibration_fit_status": calibration_fit.get("status"),
            "calibration_sample_count": calibration_fit.get("n", len(cal_rows)),
            "metrics": metrics,
            "paired_moving_block_ci_vs_causal_baseline": block_ci,
            "paired_moving_block_ci_vs_price_only": incremental_ci,
            "regimes": regime_summary,
            "holdout_selection_or_tuning": False,
        }

    report["summary"] = {
        "scored_ablations": [name for name, item in report["ablations"].items() if item.get("status") == "SCORED"],
        "not_eligible_or_failed_ablations": {name: item.get("status") for name, item in report["ablations"].items()
                                             if item.get("status") != "SCORED"},
        "scale_target_met": scale_met,
        "best_model_claimed": False,
        "alpha_claimed": False,
    }
    return report
