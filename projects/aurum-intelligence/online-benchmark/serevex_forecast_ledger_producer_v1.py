"""Serevex point-in-time forecast-ledger producer v1.

Adapts the existing chronological TP-before-SL evaluation events into the
integrated Serevex ledger contract. Uses standard-library, causal probability
estimators only; this is a research/reference-data adapter, not a trading model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import ci_benchmark
import evaluation_protocol_v1_4 as eval_protocol
import serevex_stage123_integrated_v1 as stage123
import serevex_alpha_model_v1 as alpha_model

PRODUCER_VERSION = "serevex-forecast-ledger-producer-v2-alpha"
WARMUP_BARS = 60
ATR_PERIOD = 14
TAKE_PROFIT_ATR = 1.5
STOP_LOSS_ATR = 1.0
HORIZON_BARS = 12
ROLLING_MATURED_EVENTS = 288
MIN_CONDITIONAL_SUPPORT = 20
REGIMES = ("trend_up", "trend_down", "range")
SESSION_ID = "session_unverified"
SPREAD_BUCKET = "unknown_no_bidask_quotes"
EVENT_RISK_BUCKET = "unknown_no_event_calendar"
EXPERT_SPECS: dict[str, dict[str, Any]] = {
    "alpha_logistic_v1": {"kind": "trained_logistic_alpha", "model_version": alpha_model.MODEL_VERSION,
        "features": list(alpha_model.FEATURE_NAMES), "fit": "development_then_model_selection_refit",
        "calibration": "calibration_partition_only", "final_holdout": "frozen_model_and_calibrator"},
    "constant_50pct": {"kind": "fixed_probability", "probability": 0.5},
    "rolling_matured_rate_288": {"kind": "laplace_rate", "cohort": "same_rule_side", "window_matured_labels": ROLLING_MATURED_EVENTS},
    "regime_matured_rate": {"kind": "laplace_rate", "cohort": "same_rule_side_and_regime", "min_support": MIN_CONDITIONAL_SUPPORT, "fallback": "expanding_baseline"},
    "volatility_matured_rate": {"kind": "laplace_rate", "cohort": "same_rule_side_and_volatility_bucket", "min_support": MIN_CONDITIONAL_SUPPORT, "fallback": "expanding_baseline"},
}
CATEGORY_REGISTRY = {
    "regime_id": list(REGIMES),
    "session_id": [SESSION_ID],
    "volatility_bucket": ["low", "high", "UNKNOWN"],
    "spread_bucket": [SPREAD_BUCKET],
    "event_risk_bucket": [EVENT_RISK_BUCKET],
}


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probability(labels: Sequence[int]) -> float:
    """Laplace-smoothed binary rate."""
    positives = sum(int(x) for x in labels)
    return (positives + 1.0) / (len(labels) + 2.0)


def _normalize_rows(
    rows: Sequence[Mapping[str, Any]], *, naive_timestamp_policy: str
) -> list[dict[str, Any]]:
    if naive_timestamp_policy not in ("reject", "assume_utc_for_diagnostic_only"):
        raise ValueError("naive_timestamp_policy_invalid")
    if not rows:
        raise ValueError("ohlcv_rows_must_be_nonempty")
    normalized: list[dict[str, Any]] = []
    seen_awareness: set[bool] = set()
    previous_time: datetime | None = None
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise ValueError("ohlcv_row_must_be_mapping_at_" + str(index))
        row = dict(source)
        timestamp = row.get("timestamp")
        if not isinstance(timestamp, datetime):
            raise ValueError("ohlcv_timestamp_must_be_datetime_at_" + str(index))
        aware = timestamp.tzinfo is not None and timestamp.utcoffset() is not None
        seen_awareness.add(aware)
        if len(seen_awareness) > 1:
            raise ValueError("mixed_timezone_aware_and_naive_timestamps")
        if not aware:
            if naive_timestamp_policy != "assume_utc_for_diagnostic_only":
                raise ValueError("naive_timestamp_requires_explicit_diagnostic_utc_assumption")
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc)
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError("timestamps_must_be_strictly_increasing")
        previous_time = timestamp
        row["timestamp"] = timestamp
        for field in ("open", "high", "low", "close", "volume"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("ohlcv_" + field + "_must_be_finite_at_" + str(index))
            row[field] = float(value)
        o, h, l, c, v = (row[k] for k in ("open", "high", "low", "close", "volume"))
        if min(o, h, l, c) <= 0 or h < max(o, c) or l > min(o, c) or h < l or v < 0:
            raise ValueError("invalid_ohlcv_geometry_at_" + str(index))
        normalized.append(row)
    return normalized


def _atr_pct_by_index(rows: Sequence[Mapping[str, Any]], period: int) -> list[float | None]:
    true_ranges: list[float] = []
    result: list[float | None] = [None] * len(rows)
    for index, row in enumerate(rows):
        high, low = float(row["high"]), float(row["low"])
        if index == 0:
            true_range = high - low
        else:
            previous_close = float(rows[index - 1]["close"])
            true_range = max(high - low, abs(high - previous_close), abs(low - previous_close))
        true_ranges.append(true_range)
        if index + 1 >= period:
            atr = sum(true_ranges[index - period + 1:index + 1]) / period
            close = float(row["close"])
            result[index] = atr / close if close else None
    return result


def _state_by_index(
    rows: Sequence[Mapping[str, Any]], atr_pct: Sequence[float | None],
    *, volatility_lookback: int = ROLLING_MATURED_EVENTS,
) -> list[dict[str, Any]]:
    """Causal feature-state labels using only bars at or before each decision."""
    states: list[dict[str, Any]] = []
    closes = [float(row["close"]) for row in rows]
    for index, _row in enumerate(rows):
        efficiency = 0.0
        if index >= 10:
            net = closes[index] - closes[index - 10]
            path = sum(abs(closes[j] - closes[j - 1]) for j in range(index - 9, index + 1))
            efficiency = net / path if path > 1e-15 else 0.0
        if efficiency >= 0.65:
            regime = "trend_up"
        elif efficiency <= -0.65:
            regime = "trend_down"
        else:
            regime = "range"

        current_vol = atr_pct[index]
        past_vol = [
            float(value) for value in atr_pct[max(0, index - volatility_lookback):index]
            if value is not None and math.isfinite(float(value))
        ]
        if current_vol is None or not past_vol:
            vol_bucket = "UNKNOWN"
        else:
            threshold = statistics.median(past_vol)
            vol_bucket = "high" if float(current_vol) >= threshold else "low"

        states.append({
            "regime_id": regime,
            "volatility_bucket": vol_bucket,
            "trend_efficiency": float(efficiency),
            "atr_pct": None if current_vol is None else float(current_vol),
            "session_id": SESSION_ID,
            "spread_bucket": SPREAD_BUCKET,
            "event_risk_bucket": EVENT_RISK_BUCKET,
        })
    return states


def _alpha_features_by_index(
    rows: Sequence[Mapping[str, Any]],
    atr_pct: Sequence[float | None],
    states: Sequence[Mapping[str, Any]],
) -> list[dict[str, float]]:
    """Causal OHLCV features; all windows end at the current decision bar."""
    result: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        close = float(row["close"])
        open_price = float(row["open"])
        high, low = float(row["high"]), float(row["low"])
        bar_range = max(high - low, 1e-12)
        current_atr_pct = atr_pct[index]
        atr_scale = max(float(current_atr_pct or 0.0), 1e-8)
        previous_close = float(rows[index - 1]["close"]) if index else close
        prior5_close = float(rows[index - 5]["close"]) if index >= 5 else close
        momentum_1 = math.log(close / previous_close) / atr_scale if index and previous_close > 0 else 0.0
        momentum_5 = (
            math.log(close / prior5_close) / (atr_scale * math.sqrt(5.0))
            if index >= 5 and prior5_close > 0 else 0.0
        )
        body_ratio = (close - open_price) / bar_range
        close_location = 2.0 * (close - low) / bar_range - 1.0
        range_to_atr = ((high - low) / close) / atr_scale if close > 0 else 0.0
        prior_volumes = [
            float(rows[j]["volume"])
            for j in range(max(0, index - 20), index)
            if math.isfinite(float(rows[j]["volume"]))
        ]
        prior_volume_median = statistics.median(prior_volumes) if prior_volumes else float(row["volume"])
        volume_log_ratio = math.log((float(row["volume"]) + 1.0) / (prior_volume_median + 1.0))
        result.append({
            "atr_pct": float(current_atr_pct or 0.0),
            "trend_efficiency": float(states[index]["trend_efficiency"]),
            "momentum_1_atr": max(-8.0, min(8.0, momentum_1)),
            "momentum_5_atr": max(-8.0, min(8.0, momentum_5)),
            "candle_body_to_range": max(-1.0, min(1.0, body_ratio)),
            "close_location_in_range": max(-1.0, min(1.0, close_location)),
            "bar_range_to_atr": max(0.0, min(8.0, range_to_atr)),
            "volume_log_ratio": max(-5.0, min(5.0, volume_log_ratio)),
        })
    return result


def build_forecast_ledger(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_sha256: str,
    source_uri: str,
    rule: str,
    side: str,
    naive_timestamp_policy: str = "reject",
    evidence_kind: str = "real_market_research",
    license_reference: str = "UNVERIFIED_REQUIRES_REVIEW",
    warmup_bars: int = WARMUP_BARS,
    atr_period: int = ATR_PERIOD,
    take_profit_atr: float = TAKE_PROFIT_ATR,
    stop_loss_atr: float = STOP_LOSS_ATR,
    horizon_bars: int = HORIZON_BARS,
) -> dict[str, Any]:
    """Produce one rule/side cohort with forecasts and labels aligned by decision time.

    dataset_sha256 should be computed from the exact input bytes. Source and
    rights flags intentionally remain false, irrespective of supplied URI/hash.
    Naive timestamps are refused unless the caller explicitly opts into a
    diagnostic-only UTC assumption; that assumption never verifies source time.
    """
    if not isinstance(dataset_sha256, str) or len(dataset_sha256) != 64 or any(c not in "0123456789abcdef" for c in dataset_sha256.lower()):
        raise ValueError("dataset_sha256_must_be_sha256")
    if not isinstance(source_uri, str) or not source_uri.strip():
        raise ValueError("source_uri_must_be_nonempty")
    if rule not in ci_benchmark.RULES or rule == "flat_no_trade":
        raise ValueError("rule_must_be_registered_directional_rule")
    if side not in ("long", "short"):
        raise ValueError("side_must_be_long_or_short")
    if evidence_kind not in ("real_market_research", "synthetic_contract_test", "replay_fixture"):
        raise ValueError("evidence_kind_invalid")
    if isinstance(atr_period, bool) or not isinstance(atr_period, int) or atr_period < 2:
        raise ValueError("atr_period_must_be_at_least_2")
    if isinstance(warmup_bars, bool) or not isinstance(warmup_bars, int) or warmup_bars < atr_period:
        raise ValueError("warmup_must_cover_atr_period")
    if isinstance(horizon_bars, bool) or not isinstance(horizon_bars, int) or horizon_bars < 1:
        raise ValueError("horizon_bars_must_be_positive_integer")
    if not all(math.isfinite(float(x)) and float(x) > 0 for x in (take_profit_atr, stop_loss_atr)):
        raise ValueError("barrier_multipliers_must_be_finite_positive")

    normalized_rows = _normalize_rows(rows, naive_timestamp_policy=naive_timestamp_policy)
    if len(normalized_rows) < warmup_bars + horizon_bars + 100:
        raise ValueError("insufficient_history_for_forecast_ledger")
    module_path = Path(__file__).resolve()
    producer_sha = _file_sha(module_path)
    input_awareness = rows[0]["timestamp"].tzinfo is not None and rows[0]["timestamp"].utcoffset() is not None
    timezone_policy = (
        "source_timestamp_tzinfo_normalized_to_utc_source_semantics_unverified"
        if input_awareness else "naive_timestamp_assumed_utc_for_diagnostic_only"
    )
    series_id = "XAUUSD:OHLCV:" + rule + ":" + side + ":tp-before-sl"
    target_id = (
        "xauusd:tp-before-sl:"
        + rule + ":" + side
        + ":atr" + format(float(take_profit_atr), ".6g")
        + "-sl" + format(float(stop_loss_atr), ".6g")
        + "-h" + str(horizon_bars)
    )
    target_spec = {
        "target_id": target_id,
        "rule": rule, "side": side, "atr_period": atr_period,
        "take_profit_atr": float(take_profit_atr), "stop_loss_atr": float(stop_loss_atr),
        "horizon_bars_from_next_open": horizon_bars,
        "timeout_label": 0,
        "same_bar_collision": "unresolved_no_binary_label",
        "source_time_semantics_verified": False,
    }
    expert_hashes = {
        name: _canonical_sha({
            "producer_sha256": producer_sha,
            "expert_id": name,
            "spec": spec,
            "producer_version": PRODUCER_VERSION,
        })
        for name, spec in EXPERT_SPECS.items()
    }
    lineage = {
        "source_uri": source_uri,
        "dataset_sha256": dataset_sha256,
        "license_reference_claim": license_reference,
        "source_timezone_policy": timezone_policy,
        "source_semantics_verified": False,
        "rule": rule,
        "side": side,
        "target": target_spec,
        "producer_version": PRODUCER_VERSION,
        "producer_sha256": producer_sha,
    }
    timestamp_sha = _canonical_sha({
        "available_at": "normalized source decision timestamp (reconstructed reference time only)",
        "source_timezone_policy": timezone_policy,
        "bar_close_semantics_verified": False,
    })
    feature_formulas = {
        "atr_pct": {"formula": "simple_mean_true_range_n_over_current_close", "period": atr_period, "version": 1},
        "trend_efficiency": {"formula": "close_change_10_over_sum_abs_close_changes_10", "threshold": 0.65, "version": 1},
        "momentum_1_atr": {"formula": "log_close_return_1_divided_by_causal_atr_fraction_clipped_8", "version": 1},
        "momentum_5_atr": {"formula": "log_close_return_5_divided_by_causal_atr_fraction_sqrt5_clipped_8", "version": 1},
        "candle_body_to_range": {"formula": "close_minus_open_divided_by_current_high_low_range_clipped_1", "version": 1},
        "close_location_in_range": {"formula": "twice_close_position_in_current_high_low_range_minus_1", "version": 1},
        "bar_range_to_atr": {"formula": "current_high_low_range_fraction_divided_by_causal_atr_fraction_clipped_8", "version": 1},
        "volume_log_ratio": {"formula": "log_current_volume_plus1_over_prior20_volume_median_plus1_clipped_5", "version": 1},
    }
    feature_transforms = {name: _canonical_sha(spec) for name, spec in feature_formulas.items()}
    feature_semantics = {
        "atr_pct": "normalized_average_true_range_fraction",
        "trend_efficiency": "bounded_directional_efficiency",
        "momentum_1_atr": "one_bar_log_return_scaled_by_atr_fraction",
        "momentum_5_atr": "five_bar_log_return_scaled_by_atr_and_sqrt_horizon",
        "candle_body_to_range": "bounded_current_candle_body_fraction",
        "close_location_in_range": "bounded_current_close_location_in_range",
        "bar_range_to_atr": "current_bar_range_scaled_by_atr_fraction",
        "volume_log_ratio": "log_current_volume_to_prior_median_volume_ratio",
    }

    atr_pct = _atr_pct_by_index(normalized_rows, atr_period)
    states = _state_by_index(normalized_rows, atr_pct)
    alpha_features_by_index = _alpha_features_by_index(normalized_rows, atr_pct, states)
    eligible_count = len(normalized_rows) - horizon_bars - warmup_bars
    development_end = warmup_bars + int(eligible_count * 0.50)
    model_selection_end = warmup_bars + int(eligible_count * 0.70)
    calibration_end = warmup_bars + int(eligible_count * 0.80)

    def partition_for(index: int) -> str:
        if index < development_end:
            return "development"
        if index < model_selection_end:
            return "model_selection"
        if index < calibration_end:
            return "calibration"
        return "final_holdout"

    raw_events: list[dict[str, Any]] = []
    decision_end_exclusive = len(normalized_rows) - horizon_bars
    wanted_side = 1 if side == "long" else -1
    for index in range(warmup_bars, decision_end_exclusive):
        signal = ci_benchmark.get_signal(normalized_rows, index, rule)
        if signal != wanted_side:
            continue
        label = eval_protocol.label_tp_before_sl(
            normalized_rows, index, wanted_side, atr_period=atr_period,
            take_profit_atr=take_profit_atr, stop_loss_atr=stop_loss_atr,
            horizon_bars=horizon_bars,
        )
        exit_index = label.get("target_exit_index")
        if not isinstance(exit_index, int) or exit_index <= index or exit_index >= len(normalized_rows):
            raise ValueError("label_exit_index_invalid_for_decision_" + str(index))
        actual_label = label.get("actual_tp")
        label_mature = actual_label in (0, 1)
        if not label_mature and label.get("resolution") != "ambiguous_same_bar":
            raise ValueError("unresolved_non_ambiguous_target_at_" + str(index))
        state = states[index]
        raw_events.append({
            "decision_index": index,
            "partition": partition_for(index),
            "decision_time": normalized_rows[index]["timestamp"].isoformat().replace("+00:00", "Z"),
            "target_exit_index": exit_index,
            "label_available_index": exit_index,
            "label_available_time": normalized_rows[exit_index]["timestamp"].isoformat().replace("+00:00", "Z"),
            "label": None if not label_mature else int(actual_label),
            "label_mature": bool(label_mature),
            "session_id": state["session_id"],
            "regime_id": state["regime_id"],
            "state_tags": {
                "volatility_bucket": state["volatility_bucket"],
                "spread_bucket": state["spread_bucket"],
                "event_risk_bucket": state["event_risk_bucket"],
            },
            "state_tags_available_time": normalized_rows[index]["timestamp"].isoformat().replace("+00:00", "Z"),
            "atr_pct": state["atr_pct"],
            "trend_efficiency": state["trend_efficiency"],
            "resolution": label["resolution"],
            "gross_r": label.get("gross_r"),
            "alpha_features": alpha_features_by_index[index],
        })
    if not raw_events:
        raise ValueError("no_events_for_registered_rule_side_cohort")

    rows_so_far: list[dict[str, Any]] = []
    for event in raw_events:
        di = int(event["decision_index"])
        matured_prior = [
            previous for previous in rows_so_far
            if previous["label_mature"] is True
            and previous["label"] in (0, 1)
            and int(previous["label_available_index"]) < di
        ]
        labels_all = [int(previous["label"]) for previous in matured_prior]
        baseline = _probability(labels_all)
        rolling_labels = labels_all[-ROLLING_MATURED_EVENTS:]
        regime_labels = [
            int(previous["label"]) for previous in matured_prior
            if previous["regime_id"] == event["regime_id"]
        ]
        volatility_labels = [
            int(previous["label"]) for previous in matured_prior
            if previous["state_tags"]["volatility_bucket"] == event["state_tags"]["volatility_bucket"]
        ]
        event["baseline_probability"] = baseline
        event["pre_alpha_expert_probabilities"] = {
            "constant_50pct": 0.5,
            "rolling_matured_rate_288": _probability(rolling_labels),
            "regime_matured_rate": (
                _probability(regime_labels)
                if len(regime_labels) >= MIN_CONDITIONAL_SUPPORT else baseline
            ),
            "volatility_matured_rate": (
                _probability(volatility_labels)
                if len(volatility_labels) >= MIN_CONDITIONAL_SUPPORT else baseline
            ),
        }
        event["matured_label_count_before_decision"] = len(matured_prior)
        event["conditional_regime_support_before_decision"] = len(regime_labels)
        event["conditional_volatility_support_before_decision"] = len(volatility_labels)
        rows_so_far.append(event)

    partition_start_indices = {
        "development": warmup_bars,
        "model_selection": development_end,
        "calibration": model_selection_end,
        "final_holdout": calibration_end,
    }
    alpha_training = alpha_model.train_alpha_model(
        raw_events,
        partition_start_indices=partition_start_indices,
        as_of_index=len(normalized_rows),
    )
    expert_hashes["alpha_logistic_v1"] = alpha_training["model_artifact_sha256"]
    alpha_probabilities = alpha_training["predictions_by_decision_index"]
    alpha_forecast_stages = alpha_training["forecast_stage_by_decision_index"]
    alpha_summary = {
        key: value for key, value in alpha_training.items()
        if key not in ("predictions_by_decision_index", "raw_predictions_by_decision_index", "calibrated_alpha_predictions_by_decision_index", "forecast_stage_by_decision_index")
    }

    forecast_records: list[dict[str, Any]] = []
    for event in raw_events:
        index = int(event["decision_index"])
        decision_time = event["decision_time"]
        feature_values = tuple(
            (
                feature_id,
                event["alpha_features"].get(feature_id),
                feature_transforms[feature_id],
                feature_semantics[feature_id],
            )
            for feature_id in alpha_model.FEATURE_NAMES
        )
        fact_rows = []
        for feature_id, value, transform_sha, semantic_type in feature_values:
            fact_rows.append({
                "feature_id": feature_id,
                "available_at": decision_time,
                "observed_at": decision_time,
                "source_revision": "dataset_sha256:" + dataset_sha256,
                "source_sha256": dataset_sha256,
                "transform_id": feature_id + "_v1",
                "transform_sha256": transform_sha,
                "semantic_type": semantic_type,
                "quote_basis": "ohlcv_close_not_executable_bid_ask",
                "quality_status": "REFERENCE_ONLY_SOURCE_TIME_AND_AVAILABILITY_UNVERIFIED",
                "value": float(value),
                "timestamp_semantics_sha256": timestamp_sha,
            })
        probabilities = dict(event["pre_alpha_expert_probabilities"])
        probabilities["alpha_logistic_v1"] = float(alpha_probabilities[str(index)])
        record = {
            "decision_id": _canonical_sha({
                "series_id": series_id, "decision_index": index,
                "rule": rule, "side": side, "target_id": target_id,
            }),
            "decision_index": index,
            "decision_time": decision_time,
            "partition": partition_for(index),
            "label_available_index": event["label_available_index"],
            "label_available_time": event["label_available_time"],
            "label": event["label"],
            "label_mature": event["label_mature"],
            "regime_id": event["regime_id"],
            "regime_available_index": index,
            "regime_available_time": decision_time,
            "session_id": event["session_id"],
            "baseline_probability": event["baseline_probability"],
            "expert_probabilities": probabilities,
            "state_tags": event["state_tags"],
            "state_tags_available_index": index,
            "state_tags_available_time": event["state_tags_available_time"],
            "feature_facts": fact_rows,
            "target_exit_index": event["target_exit_index"],
            "target_resolution": event["resolution"],
            "target_gross_r_reference_only": event["gross_r"],
            "matured_label_count_before_decision": event["matured_label_count_before_decision"],
            "conditional_regime_support_before_decision": event["conditional_regime_support_before_decision"],
            "conditional_volatility_support_before_decision": event["conditional_volatility_support_before_decision"],
            "forecast_method": "prequential_laplace_estimators_plus_partition_frozen_alpha_logistic_v1",
            "alpha_model_forecast_stage": alpha_forecast_stages[str(index)],
            "alpha_model_artifact_sha256": alpha_training["model_artifact_sha256"],
        }
        forecast_records.append(record)

    ledger = {
        "schema_version": "serevex.forecast-ledger.v1",
        "metadata": {
            "series_id": series_id,
            "target_id": target_id,
            "target_definition_sha256": _canonical_sha(target_spec),
            "dataset_sha256": dataset_sha256.lower(),
            "source_lineage_sha256": _canonical_sha(lineage),
            "model_artifact_sha256_by_expert": expert_hashes,
            "registered_experts": list(EXPERT_SPECS),
            "registered_regimes": list(REGIMES),
            "atlas_category_registry": CATEGORY_REGISTRY,
            "evidence_kind": evidence_kind,
            "source_verification": {
                "provider_provenance_independently_verified": False,
                "data_rights_verified": False,
                "broker_feed_match_verified": False,
            },
            "decision_time_basis": (
                "normalized dataset timestamps used as historical decision markers; "
                "bar-close convention and historical feature availability are unverified; reference-only"
            ),
            "label_definition": (
                "TP before SL after next-bar-open entry; TP=1; SL or timeout=0; "
                "same-bar collision unresolved and excluded; OHLC source only"
            ),
            "source_uri": source_uri,
            "license_reference_claim": license_reference,
            "source_timezone_policy": timezone_policy,
            "source_timestamp_semantics_verified": False,
            "historical_feature_availability_verified": False,
            "broker_quote_basis_verified": False,
            "forecast_producer_version": PRODUCER_VERSION,
            "forecast_producer_sha256": producer_sha,
            "rule": rule,
            "side": side,
            "target_spec": target_spec,
            "expert_specs": EXPERT_SPECS,
            "alpha_model_training": alpha_summary,
            "alpha_feature_schema": {
                "feature_names": list(alpha_model.FEATURE_NAMES),
                "feature_formulas": feature_formulas,
                "feature_transforms_sha256": feature_transforms,
                "point_in_time_note": "OHLCV source timestamp and bar-close availability remain unverified; all computations are causally restricted to bars at or before decision index",
            },
            "partition_policy": {
                "development": "first 50 percent of eligible decision bars",
                "model_selection": "50-70 percent",
                "calibration": "70-80 percent",
                "final_holdout": "last 20 percent",
                "split_basis": "chronological decision-bar index before filtering to signal cohort",
            },
        },
        "records": forecast_records,
    }
    stage123.validate_forecast_ledger(ledger)
    return ledger


def run_end_to_end(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_sha256: str,
    source_uri: str,
    rule: str,
    side: str,
    as_of_index: int,
    naive_timestamp_policy: str = "reject",
    evidence_kind: str = "real_market_research",
    license_reference: str = "UNVERIFIED_REQUIRES_REVIEW",
    atlas_bootstrap_replicates: int = 1000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = build_forecast_ledger(
        rows, dataset_sha256=dataset_sha256, source_uri=source_uri,
        rule=rule, side=side, naive_timestamp_policy=naive_timestamp_policy,
        evidence_kind=evidence_kind, license_reference=license_reference,
    )
    report = stage123.run_serevex_stage123(
        ledger,
        as_of_index=as_of_index,
        score_partition="final_holdout",
        atlas_bootstrap_replicates=atlas_bootstrap_replicates,
    )
    return ledger, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Produce an auditable Serevex forecast ledger and execute stages 1-3 (research only)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Historical OHLCV CSV")
    parser.add_argument("--ledger-output", type=Path, required=True, help="Canonical forecast-ledger JSON output")
    parser.add_argument("--report-output", type=Path, required=True, help="Integrated stages 1-3 diagnostic JSON output")
    parser.add_argument("--source-uri", required=True, help="Original data/source URL or declared local-file URI")
    parser.add_argument("--license-reference", default="UNVERIFIED_REQUIRES_REVIEW")
    parser.add_argument("--rule", choices=[x for x in ci_benchmark.RULES if x != "flat_no_trade"], required=True)
    parser.add_argument("--side", choices=("long", "short"), required=True)
    parser.add_argument("--as-of-index", type=int, required=True, help="Exclusive underlying bar index, not event count")
    parser.add_argument(
        "--assume-naive-utc-for-diagnostic-only", action="store_true",
        help="Explicitly normalize naive source timestamps to UTC for diagnostics; does not verify source timezone.",
    )
    parser.add_argument("--atlas-bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        if args.atlas_bootstrap_replicates < 200 or args.atlas_bootstrap_replicates > 10000:
            raise ValueError("atlas_bootstrap_replicates_must_be_200_to_10000")
        payload = args.input.read_bytes()
        parsed_rows = ci_benchmark.parse_csv(payload)
        policy = "assume_utc_for_diagnostic_only" if args.assume_naive_utc_for_diagnostic_only else "reject"
        ledger, report = run_end_to_end(
            parsed_rows,
            dataset_sha256=hashlib.sha256(payload).hexdigest(),
            source_uri=args.source_uri,
            rule=args.rule, side=args.side, as_of_index=args.as_of_index,
            naive_timestamp_policy=policy, license_reference=args.license_reference,
            atlas_bootstrap_replicates=args.atlas_bootstrap_replicates,
        )
        args.ledger_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.ledger_output.write_text(
            json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        args.report_output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "producer_status": "PASS_LEDGER_AND_CONTRACT_DIAGNOSTICS",
            "ledger_record_count": len(ledger["records"]),
            "ledger_sha256": stage123.validate_forecast_ledger(ledger)["ledger_sha256"],
            "report_sha256": report["report_sha256"],
            "report_status": report["status"],
            "decision_output": "NO_TRADE",
            "live_signal_enabled": False,
            "live_order_execution_enabled": False,
            "ledger_output": str(args.ledger_output),
            "report_output": str(args.report_output),
            "evidence_kind": ledger["metadata"]["evidence_kind"],
            "source_verification": ledger["metadata"]["source_verification"],
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
