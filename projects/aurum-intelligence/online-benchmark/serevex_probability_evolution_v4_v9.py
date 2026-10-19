"""Serevex probability evolution stack v4.0-v9.0.

Research-only, standard-library implementation. Consumes pre-generated, point-in-time
candidate probabilities and a row-specific causal baseline. It does not train a market
forecaster, create trade instructions, or authorize live execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

VERSION_ORDER = ("4.0", "4.1", "4.2", "5.0", "6.0", "7.0", "8.0", "9.0")
PROBABILITY_FLOOR = 1e-6
BLEND_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_BLOCK_SPAN = 12
DEFAULT_EMBARGO_BARS = 12
DEFAULT_CRITICAL_Z = 2.58
DEFAULT_POSITIVE_BLOCK_SHARE = 0.70
DEFAULT_MIN_SELECTION_BLOCKS = 8
DEFAULT_MIN_GATE_BLOCKS = 4
DEFAULT_V7_MIN_GATE_BLOCKS = 8
DEFAULT_MIN_SELECTION_ROWS = 96
DEFAULT_MIN_REGIME_ROWS = 288
DEFAULT_DRIFT_RECENT_BARS = 192
DEFAULT_HISTORY_WINDOW = 720
DEFAULT_WEIGHT_UPDATE_INTERVAL_BARS = 12
UNKNOWN_REGIME = "__UNKNOWN__"
SCORE_PARTITIONS = ("all_matured", "development", "model_selection", "calibration", "final_holdout")


class ProbabilityInputError(ValueError):
    """Raised when the ledger cannot safely identify decisions or their baseline."""


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _probability(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    p = float(value)
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        return None
    return min(max(p, PROBABILITY_FLOOR), 1.0 - PROBABILITY_FLOOR)


def _label(row: Mapping[str, Any]) -> int | None:
    y = row.get("label")
    if isinstance(y, bool) or not isinstance(y, int) or y not in (0, 1):
        return None
    return y


def _log_loss(p: float, y: int) -> float:
    q = min(max(float(p), PROBABILITY_FLOOR), 1.0 - PROBABILITY_FLOOR)
    return -(y * math.log(q) + (1 - y) * math.log(1.0 - q))


def _normalize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ProbabilityInputError("forecast_ledger_empty")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ProbabilityInputError("forecast_row_not_mapping")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    identity_fields = ("target_id", "instrument", "horizon_bars")
    identity: dict[str, Any] = {}
    for field in identity_fields:
        values = [row.get(field) for row in rows if field in row]
        if values:
            if len(values) != len(rows):
                raise ProbabilityInputError("mixed_cohort_missing_identity_field:" + field)
            if any(value != values[0] for value in values):
                raise ProbabilityInputError("mixed_forecast_cohorts:" + field)
            identity[field] = values[0]
    for ordinal, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise ProbabilityInputError("forecast_row_not_mapping:" + str(ordinal))
        row = dict(source)
        index = row.get("decision_index")
        decision_id = row.get("decision_id")
        if not _integer(index) or index < 0:
            raise ProbabilityInputError("invalid_decision_index:" + str(ordinal))
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ProbabilityInputError("invalid_decision_id:" + str(ordinal))
        if index in seen_indices:
            raise ProbabilityInputError("duplicate_decision_index:" + str(index))
        if decision_id in seen_ids:
            raise ProbabilityInputError("duplicate_decision_id:" + decision_id)
        seen_indices.add(index)
        seen_ids.add(decision_id)
        baseline = _probability(row.get("baseline_probability"))
        if baseline is None:
            raise ProbabilityInputError("invalid_or_missing_causal_baseline:" + decision_id)
        candidate_value = row.get(
            "candidate_probability",
            row.get("probability", row.get("alpha_probability")),
        )
        candidate = _probability(candidate_value)
        reasons: list[str] = []
        feature_available = row.get("feature_available_index")
        if feature_available is not None:
            if not _integer(feature_available) or feature_available > index:
                candidate = None
                reasons.append("FUTURE_OR_INVALID_FEATURE_AVAILABILITY")
        if candidate is None:
            reasons.append("INVALID_OR_MISSING_CANDIDATE_PROBABILITY")
        regime = row.get("regime")
        regime_available = row.get("regime_available_index")
        regime_ok = isinstance(regime, str) and bool(regime.strip())
        if regime_available is not None:
            regime_ok = regime_ok and _integer(regime_available) and regime_available <= index
        if not regime_ok:
            regime = UNKNOWN_REGIME
            if row.get("regime") is not None:
                reasons.append("REGIME_NOT_AVAILABLE_AT_DECISION")
        available_index = row.get("label_available_index")
        if row.get("label_mature") is True and _label(row) in (0, 1) and _integer(available_index) and available_index <= index:
            raise ProbabilityInputError("label_maturity_not_after_decision:" + decision_id)
        row["_candidate"] = candidate
        row["_baseline"] = baseline
        row["_input_reasons"] = reasons
        row["_regime"] = regime if regime_ok else UNKNOWN_REGIME
        row["_original_ordinal"] = ordinal
        normalized.append(row)
    normalized.sort(key=lambda r: int(r["decision_index"]))
    return normalized


def _matured_history(history: Sequence[Mapping[str, Any]], current_index: int) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for row in history:
        di = row["decision_index"]
        ai = row.get("label_available_index")
        y = _label(row)
        if (
            _integer(di) and di < current_index
            and _integer(ai) and ai < current_index
            and row.get("label_mature") is True
            and y in (0, 1)
        ):
            item = dict(row)
            item["_y"] = y
            eligible.append(item)
    return eligible


def _scoring_probability(row: Mapping[str, Any], key: str = "_candidate") -> float | None:
    p = row.get(key)
    return p if isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(float(p)) else None


def _metrics(rows: Sequence[Mapping[str, Any]], probability_key: str) -> dict[str, Any]:
    pairs: list[tuple[float, int]] = []
    for row in rows:
        p = _scoring_probability(row, probability_key)
        y = _label(row)
        if p is not None and y in (0, 1):
            pairs.append((min(max(float(p), PROBABILITY_FLOOR), 1.0 - PROBABILITY_FLOOR), y))
    n = len(pairs)
    if not n:
        return {"rows": 0, "brier": None, "log_loss": None, "ece_10": None, "base_rate": None}
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = sum(_log_loss(p, y) for p, y in pairs) / n
    bins: dict[int, list[tuple[float, int]]] = {}
    for p, y in pairs:
        bins.setdefault(min(9, int(p * 10)), []).append((p, y))
    ece = sum(
        len(values) / n * abs(
            sum(p for p, _ in values) / len(values)
            - sum(y for _, y in values) / len(values)
        )
        for values in bins.values()
    )
    return {
        "rows": n,
        "brier": brier,
        "log_loss": ll,
        "ece_10": ece,
        "base_rate": sum(y for _, y in pairs) / n,
        "mean_probability": sum(p for p, _ in pairs) / n,
    }


def _temporal_blocks(
    rows: Sequence[Mapping[str, Any]], *, block_span: int, embargo: int
) -> list[list[Mapping[str, Any]]]:
    if block_span < 1 or embargo < 0:
        raise ProbabilityInputError("invalid_temporal_block_parameters")
    period = block_span + embargo
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        index = int(row["decision_index"])
        phase = index % period
        if phase < block_span:
            groups.setdefault(index // period, []).append(row)
    return [groups[key] for key in sorted(groups) if groups[key]]


def _block_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    alpha_weight: float,
    block_span: int,
    embargo: int,
    probability_key: str = "_candidate",
    critical_z: float = DEFAULT_CRITICAL_Z,
) -> dict[str, Any]:
    block_brier: list[float] = []
    block_log: list[float] = []
    for block in _temporal_blocks(rows, block_span=block_span, embargo=embargo):
        brier_gain: list[float] = []
        log_gain: list[float] = []
        for row in block:
            p = _scoring_probability(row, probability_key)
            y = _label(row)
            baseline = row.get("_baseline")
            if p is None or y not in (0, 1) or not isinstance(baseline, (int, float)):
                continue
            mixed = min(max((1.0 - alpha_weight) * float(baseline) + alpha_weight * float(p),
                            PROBABILITY_FLOOR), 1.0 - PROBABILITY_FLOOR)
            brier_gain.append((float(baseline) - y) ** 2 - (mixed - y) ** 2)
            log_gain.append(_log_loss(float(baseline), y) - _log_loss(mixed, y))
        if brier_gain:
            block_brier.append(sum(brier_gain) / len(brier_gain))
            block_log.append(sum(log_gain) / len(log_gain))
    def summarize(values: list[float]) -> dict[str, Any]:
        n = len(values)
        if not n:
            return {"blocks": 0, "mean_gain": None, "se": None, "lower": None, "positive_share": None}
        mean = sum(values) / n
        if n < 2:
            se = None
            lower = None
        else:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            se = math.sqrt(max(0.0, variance) / n)
            lower = mean - float(critical_z) * se
        return {
            "blocks": n,
            "mean_gain": mean,
            "se": se,
            "lower": lower,
            "positive_share": sum(x > 0 for x in values) / n,
        }
    brier = summarize(block_brier)
    logloss = summarize(block_log)
    return {"brier": brier, "log_loss": logloss,
            "blocks": min(brier["blocks"], logloss["blocks"])}


def _dual_score_pass(stats: Mapping[str, Any], min_blocks: int, min_positive_share: float) -> bool:
    for metric in ("brier", "log_loss"):
        item = stats[metric]
        if (
            item["blocks"] < min_blocks
            or item["lower"] is None or item["lower"] <= 0.0
            or item["positive_share"] is None or item["positive_share"] < min_positive_share
        ):
            return False
    return True


def _choose_weight(
    history: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    min_gate_blocks: int,
) -> dict[str, Any]:
    candidates = [r for r in history if r.get("_candidate") is not None and _label(r) in (0, 1)]
    candidates.sort(key=lambda r: int(r["decision_index"]))
    if not candidates:
        return {
            "weight": 0.0,
            "status": "BASELINE_ONLY_INSUFFICIENT_EVIDENCE",
            "selection_rows": 0,
            "gate_rows": 0,
            "selection_cutoff_index": None,
            "selection_blocks": 0,
            "gate_blocks": 0,
            "selection_source_block_ids": [],
            "gate_source_block_ids": [],
            "candidate_count": len(BLEND_GRID) - 1,
        }
    # Partition on an absolute source-index block boundary, never by filtered-row
    # rank: otherwise a single 12-bar scoring block could leak into both selection
    # and confirmation sets when signals are sparse.
    split_position = min(max(1, int(len(candidates) * 2 / 3)), max(1, len(candidates) - 1))
    period = int(config["block_span"]) + int(config["embargo_bars"])
    anchor = int(candidates[split_position]["decision_index"]) if len(candidates) > 1 else int(candidates[0]["decision_index"])
    cutoff = (anchor // period) * period
    selection = [row for row in candidates if int(row["decision_index"]) < cutoff]
    gate = [row for row in candidates if int(row["decision_index"]) >= cutoff]
    selection_block_ids = sorted({int(block[0]["decision_index"]) // period for block in _temporal_blocks(selection, block_span=int(config["block_span"]), embargo=int(config["embargo_bars"]))})
    gate_block_ids = sorted({int(block[0]["decision_index"]) // period for block in _temporal_blocks(gate, block_span=int(config["block_span"]), embargo=int(config["embargo_bars"]))})
    base = {
        "weight": 0.0,
        "status": "BASELINE_ONLY_INSUFFICIENT_EVIDENCE",
        "selection_rows": len(selection),
        "gate_rows": len(gate),
        "selection_cutoff_index": cutoff,
        "selection_blocks": len(selection_block_ids),
        "gate_blocks": len(gate_block_ids),
        "selection_source_block_ids": selection_block_ids,
        "gate_source_block_ids": gate_block_ids,
        "candidate_count": len(BLEND_GRID) - 1,
    }
    if len(selection) < int(config["min_selection_rows"]):
        return base
    selection_block_counts = _temporal_blocks(
        selection, block_span=int(config["block_span"]), embargo=int(config["embargo_bars"])
    )
    base["selection_blocks"] = len(selection_block_counts)
    if set(base["selection_source_block_ids"]) & set(base["gate_source_block_ids"]):
        raise ProbabilityInputError("selection_and_gate_share_source_time_block")
    if len(selection_block_counts) < int(config["min_selection_blocks"]):
        return base
    ranked: list[tuple[float, float, float, dict[str, Any]]] = []
    for weight in BLEND_GRID[1:]:
        stats = _block_stats(
            selection, alpha_weight=weight,
            block_span=int(config["block_span"]), embargo=int(config["embargo_bars"]), critical_z=float(config["critical_z"])
        )
        if _dual_score_pass(stats, int(config["min_selection_blocks"]), float(config["positive_block_share"])):
            b, l = stats["brier"]["mean_gain"], stats["log_loss"]["mean_gain"]
            ranked.append((float(b), float(l), -weight, stats))
    if not ranked:
        base["status"] = "BASELINE_ONLY_SELECTION_DUAL_SCORE_FAILED"
        return base
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    weight = -ranked[0][2]
    gate_stats = _block_stats(
        gate, alpha_weight=weight,
        block_span=int(config["block_span"]), embargo=int(config["embargo_bars"]),
        critical_z=float(config["critical_z"])
    )
    base["gate_blocks"] = gate_stats["blocks"]
    base["gate_diagnostics"] = gate_stats
    base["selection_diagnostics"] = ranked[0][3]
    if len(gate) < 1 or not _dual_score_pass(
        gate_stats, min_gate_blocks, float(config["positive_block_share"])
    ):
        base["status"] = "BASELINE_ONLY_CONFIRMATION_GATE_FAILED"
        return base
    base["weight"] = float(weight)
    base["status"] = "ALPHA_BLEND_PASSES_CHRONOLOGICAL_DUAL_SCORE_GATE"
    return base


def _regime_weight(
    history: Sequence[Mapping[str, Any]],
    current_regime: str,
    global_result: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    min_gate_blocks: int,
) -> dict[str, Any]:
    local = [
        row for row in history
        if row.get("_regime") == current_regime and row.get("_candidate") is not None
        and _label(row) in (0, 1)
    ]
    global_weight = float(global_result["weight"])
    if current_regime == UNKNOWN_REGIME or len(local) < int(config["min_regime_rows"]):
        return {
            "weight": global_weight,
            "status": "GLOBAL_FALLBACK_REGIME_SUPPORT_LOW",
            "regime_rows": len(local),
            "global_weight": global_weight,
            "local_weight": None,
            "shrinkage": 0.0,
        }
    local_result = _choose_weight(local, config=config, min_gate_blocks=min_gate_blocks)
    if local_result["weight"] <= 0.0:
        return {
            "weight": global_weight,
            "status": "GLOBAL_FALLBACK_LOCAL_GATE_FAILED",
            "regime_rows": len(local),
            "global_weight": global_weight,
            "local_weight": local_result["weight"],
            "local_status": local_result["status"],
            "shrinkage": 0.0,
        }
    shrinkage = len(local) / (len(local) + int(config["min_regime_rows"]))
    weight = global_weight + shrinkage * (float(local_result["weight"]) - global_weight)
    return {
        "weight": min(max(weight, 0.0), 1.0),
        "status": "REGIME_WEIGHT_SHRUNK_TO_GLOBAL",
        "regime_rows": len(local),
        "global_weight": global_weight,
        "local_weight": float(local_result["weight"]),
        "local_status": local_result["status"],
        "local_diagnostics": local_result,
        "shrinkage": shrinkage,
    }


def _drift_diagnostic(
    history: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    if len(history) < int(config["drift_min_rows"]):
        return {"alarm": False, "status": "DRIFT_UNKNOWN_INSUFFICIENT_HISTORY",
                "recent_rows": len(history), "older_rows": 0}
    history = sorted(history, key=lambda row: int(row["decision_index"]))
    latest = int(history[-1]["decision_index"])
    cutoff = latest - int(config["drift_recent_bars"])
    recent = [r for r in history if int(r["decision_index"]) > cutoff]
    older = [r for r in history if int(r["decision_index"]) <= cutoff]
    # Monitor the forecast actually emitted by this version; never inspect the current label.
    recent_stats = _block_stats(
        recent, alpha_weight=1.0, block_span=int(config["block_span"]),
        embargo=int(config["embargo_bars"]), probability_key="_issued_probability",
        critical_z=float(config["critical_z"])
    )
    older_stats = _block_stats(
        older, alpha_weight=1.0, block_span=int(config["block_span"]),
        embargo=int(config["embargo_bars"]), probability_key="_issued_probability",
        critical_z=float(config["critical_z"])
    )
    rb, ob = recent_stats["brier"], older_stats["brier"]
    enough = rb["blocks"] >= 4 and ob["blocks"] >= 4
    alarm = bool(
        enough
        and rb["mean_gain"] is not None and ob["mean_gain"] is not None
        and rb["mean_gain"] < 0.0 and ob["mean_gain"] > 0.0
        and rb["lower"] is not None and rb["lower"] < 0.0
        and rb["mean_gain"] < ob["mean_gain"]
    )
    return {
        "alarm": alarm,
        "status": "DRIFT_ALARM_BASELINE_FALLBACK" if alarm else (
            "DRIFT_DIAGNOSTIC_AVAILABLE_NO_ALARM" if enough else "DRIFT_UNKNOWN_INSUFFICIENT_BLOCKS"
        ),
        "recent_rows": len(recent),
        "older_rows": len(older),
        "recent_brier_gain": rb["mean_gain"],
        "older_brier_gain": ob["mean_gain"],
        "recent_blocks": rb["blocks"],
        "older_blocks": ob["blocks"],
        "method": "heuristic_blocked_prequential_brier_gain_deterioration",
    }


def _default_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    out = {
        "block_span": DEFAULT_BLOCK_SPAN,
        "embargo_bars": DEFAULT_EMBARGO_BARS,
        "critical_z": DEFAULT_CRITICAL_Z,
        "positive_block_share": DEFAULT_POSITIVE_BLOCK_SHARE,
        "min_selection_blocks": DEFAULT_MIN_SELECTION_BLOCKS,
        "min_gate_blocks": DEFAULT_MIN_GATE_BLOCKS,
        "v7_min_gate_blocks": DEFAULT_V7_MIN_GATE_BLOCKS,
        "min_selection_rows": DEFAULT_MIN_SELECTION_ROWS,
        "min_regime_rows": DEFAULT_MIN_REGIME_ROWS,
        "drift_recent_bars": DEFAULT_DRIFT_RECENT_BARS,
        "drift_min_rows": 2 * DEFAULT_DRIFT_RECENT_BARS,
        "history_window": DEFAULT_HISTORY_WINDOW,
        "weight_update_interval_bars": DEFAULT_WEIGHT_UPDATE_INTERVAL_BARS,
    }
    if config:
        unknown = set(config) - set(out)
        if unknown:
            raise ProbabilityInputError("unknown_policy_config:" + ",".join(sorted(unknown)))
        out.update(config)
    for key in ("block_span", "min_selection_blocks", "min_gate_blocks",
                "v7_min_gate_blocks", "min_selection_rows", "min_regime_rows",
                "drift_recent_bars", "drift_min_rows", "history_window",
                "weight_update_interval_bars"):
        if not _integer(out[key]) or out[key] < 1:
            raise ProbabilityInputError("invalid_policy_config:" + key)
    if not _integer(out["embargo_bars"]) or out["embargo_bars"] < 0:
        raise ProbabilityInputError("invalid_policy_config:embargo_bars")
    if not isinstance(out["critical_z"], (int, float)) or isinstance(out["critical_z"], bool) or not math.isfinite(float(out["critical_z"])) or out["critical_z"] <= 0:
        raise ProbabilityInputError("invalid_policy_config:critical_z")
    if not isinstance(out["positive_block_share"], (int, float)) or isinstance(out["positive_block_share"], bool) or not 0 < out["positive_block_share"] <= 1:
        raise ProbabilityInputError("invalid_policy_config:positive_block_share")
    return out


def _selection_summary(choice: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: choice.get(key)
        for key in (
            "weight", "status", "selection_rows", "gate_rows",
            "selection_blocks", "gate_blocks", "candidate_count",
        )
    }


def _trace_choice(
    trace: list[dict[str, Any]], *, version: str, decision_index: int,
    pool: str, choice: Mapping[str, Any], regime: str | None = None,
) -> None:
    selection = choice.get("selection_diagnostics")
    gate = choice.get("gate_diagnostics")
    trace.append({
        "version": version,
        "decision_index": decision_index,
        "pool": pool,
        "regime": regime,
        **_selection_summary(choice),
        "selection_brier_lower": selection.get("brier", {}).get("lower") if isinstance(selection, Mapping) else None,
        "selection_log_loss_lower": selection.get("log_loss", {}).get("lower") if isinstance(selection, Mapping) else None,
        "gate_brier_lower": gate.get("brier", {}).get("lower") if isinstance(gate, Mapping) else None,
        "gate_log_loss_lower": gate.get("log_loss", {}).get("lower") if isinstance(gate, Mapping) else None,
    })


def _forecast_rows(
    rows: Sequence[Mapping[str, Any]], *, version: str, config: Mapping[str, Any],
    policy_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    maturity_queue = sorted(
        [
            row for row in rows
            if row.get("label_mature") is True and _label(row) in (0, 1)
            and _integer(row.get("label_available_index"))
            and int(row["label_available_index"]) > int(row["decision_index"])
        ],
        key=lambda row: (int(row["label_available_index"]), int(row["decision_index"])),
    )
    maturity_pointer = 0
    matured_pool: list[dict[str, Any]] = []
    choice_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for position, row in enumerate(rows):
        index = int(row["decision_index"])
        while (
            maturity_pointer < len(maturity_queue)
            and int(maturity_queue[maturity_pointer]["label_available_index"]) < index
        ):
            matured_pool.append(maturity_queue[maturity_pointer])
            maturity_pointer += 1
        history = matured_pool[-int(config["history_window"]):]
        baseline = float(row["_baseline"])
        candidate = row["_candidate"]
        reasons = list(row["_input_reasons"])
        weight = 1.0 if candidate is not None else 0.0
        status = "V4_REFERENCE_CANDIDATE" if candidate is not None else "BASELINE_FALLBACK_INVALID_CANDIDATE"
        diagnostics: dict[str, Any] = {"matured_history_rows": len(history)}
        if candidate is None:
            probability = baseline
            weight = 0.0
        elif version == "4.0":
            probability = float(candidate)
        elif version == "4.1":
            probability = float(candidate)
            status = "POINT_IN_TIME_INPUT_CONTRACT_PASSED"
        elif version == "4.2":
            stats = _block_stats(
                history, alpha_weight=1.0, block_span=int(config["block_span"]),
                embargo=int(config["embargo_bars"]), critical_z=float(config["critical_z"])
            )
            diagnostics["dual_score_evidence"] = stats
            enough_rows = len(history) >= int(config["min_selection_rows"])
            passed = enough_rows and _dual_score_pass(
                stats, int(config["min_gate_blocks"]), float(config["positive_block_share"])
            )
            if passed:
                probability = float(candidate)
                status = "DUAL_SCORE_EVIDENCE_SUPPORTED"
            else:
                probability = baseline
                weight = 0.0
                status = "BASELINE_FALLBACK_DUAL_SCORE_NOT_ESTABLISHED"
                reasons.append("DUAL_SCORE_EVIDENCE_INSUFFICIENT_OR_NEGATIVE")
        elif version in ("5.0", "6.0", "7.0", "8.0", "9.0"):
            gate_min = int(config["v7_min_gate_blocks"]) if version in ("7.0", "8.0", "9.0") else int(config["min_gate_blocks"])
            bucket = index // int(config["weight_update_interval_bars"])
            global_key = ("global", bucket, gate_min)
            if global_key not in choice_cache:
                choice_cache[global_key] = _choose_weight(history, config=config, min_gate_blocks=gate_min)
                _trace_choice(policy_trace, version=version, decision_index=index,
                              pool="global", choice=choice_cache[global_key])
            global_choice = choice_cache[global_key]
            diagnostics["global_weight_selection"] = _selection_summary(global_choice)
            weight = float(global_choice["weight"])
            status = str(global_choice["status"])
            if version in ("6.0", "7.0", "8.0", "9.0"):
                regime = str(row["_regime"])
                local_key = ("regime", regime, bucket, gate_min)
                if local_key not in choice_cache:
                    choice_cache[local_key] = _regime_weight(
                        history, regime, global_choice, config=config, min_gate_blocks=gate_min
                    )
                    _trace_choice(policy_trace, version=version, decision_index=index,
                                  pool="regime", regime=regime, choice=choice_cache[local_key])
                local_choice = choice_cache[local_key]
                diagnostics["regime_weight_selection"] = {
                    key: local_choice.get(key) for key in (
                        "weight", "status", "regime_rows", "global_weight", "local_weight",
                        "local_status", "shrinkage"
                    )
                }
                weight = float(local_choice["weight"])
                status = str(local_choice["status"])
            if version in ("7.0", "8.0", "9.0"):
                # The weight selector reserves a later chronological gate disjoint from
                # the selection segment. v7+ requires more independent blocks than v5/v6.
                uncertainty_evidence = global_choice.get("gate_diagnostics")
                if "regime_weight_selection" in diagnostics:
                    local = choice_cache.get(("regime", str(row["_regime"]), bucket, gate_min), {})
                    local_details = local.get("local_diagnostics") if isinstance(local, Mapping) else None
                    if isinstance(local_details, Mapping):
                        uncertainty_evidence = local_details.get("gate_diagnostics", uncertainty_evidence)
                diagnostics["uncertainty_gate"] = {
                    "method": "held_out_chronological_block_lower_screen",
                    "minimum_independent_blocks": int(config["v7_min_gate_blocks"]),
                    "evidence": _selection_summary({"gate_blocks": uncertainty_evidence.get("blocks", 0) if isinstance(uncertainty_evidence, Mapping) else 0,
                        "status": "AVAILABLE" if isinstance(uncertainty_evidence, Mapping) else "INSUFFICIENT",
                        "weight": weight}) if isinstance(uncertainty_evidence, Mapping) else None,
                }
                if weight > 0.0 and (
                    not isinstance(uncertainty_evidence, dict)
                    or not _dual_score_pass(
                        uncertainty_evidence, int(config["v7_min_gate_blocks"]),
                        float(config["positive_block_share"])
                    )
                ):
                    weight = 0.0
                    status = "BASELINE_FALLBACK_UNCERTAINTY_GATE_FAILED"
                    reasons.append("UNTOUCHED_BLOCK_EVIDENCE_NOT_POSITIVE")
            if version in ("8.0", "9.0"):
                drift = _drift_diagnostic(history, config=config)
                diagnostics["drift_guard"] = drift
                if drift["alarm"]:
                    weight = 0.0
                    status = "BASELINE_FALLBACK_DRIFT_ALARM"
                    reasons.append("PREQUENTIAL_DRIFT_ALARM")
            probability = baseline if weight <= 0.0 else (
                (1.0 - weight) * baseline + weight * float(candidate)
            )
            if weight <= 0.0:
                reasons.append("ALPHA_WEIGHT_ZERO_BASELINE_FALLBACK")
        else:
            raise ProbabilityInputError("unsupported_probability_version:" + version)
        probability = min(max(float(probability), PROBABILITY_FLOOR), 1.0 - PROBABILITY_FLOOR)
        emitted = {
            "decision_id": row["decision_id"],
            "decision_index": index,
            "version": version,
            "candidate_probability": candidate,
            "baseline_probability": baseline,
            "probability": probability,
            "alpha_weight": float(weight),
            "status": status,
            "reason_codes": sorted(set(reasons)),
            "matured_history_rows": len(history),
            "regime": row["_regime"],
            "decision_output": "NO_TRADE",
            "model_validated": False,
            "deployment_calibrated": False,
            "live_signal_enabled": False,
            "live_order_execution_enabled": False,
        }
        if diagnostics:
            emitted["diagnostics"] = diagnostics
        output.append(emitted)
        # These are generated forecasts, not observed outcomes.
        row["_issued_probability"] = probability
        row["_issued_weight"] = float(weight)
        row["_issued_status"] = status
    return output

def evaluate_probability_evolution(
    rows: Sequence[Mapping[str, Any]],
    *,
    version: str = "9.0",
    as_of_index: int | None = None,
    config: Mapping[str, Any] | None = None,
    score_partition: str = "all_matured",
) -> dict[str, Any]:
    """Run one version sequentially; every forecast sees only strictly matured prior labels."""
    if version not in VERSION_ORDER:
        raise ProbabilityInputError("unsupported_probability_version:" + str(version))
    if not isinstance(score_partition, str) or score_partition not in SCORE_PARTITIONS:
        raise ProbabilityInputError("unsupported_score_partition:" + str(score_partition))
    settings = _default_config(config)
    normalized = _normalize_rows(rows)
    max_index = max(int(r["decision_index"]) for r in normalized)
    as_of = max_index + 1 if as_of_index is None else as_of_index
    if not _integer(as_of) or as_of < 0:
        raise ProbabilityInputError("invalid_as_of_index")
    # The exclusive as-of boundary controls both emitted forecasts and scored outcomes.
    # Post-cutoff rows are not passed into the prequential runner at all.
    point_in_time_rows = [
        row for row in normalized if int(row["decision_index"]) < as_of
    ]
    policy_trace: list[dict[str, Any]] = []
    forecasts = _forecast_rows(
        point_in_time_rows, version=version, config=settings, policy_trace=policy_trace
    )
    forecast_by_id = {row["decision_id"]: row for row in forecasts}
    scored_rows: list[dict[str, Any]] = []
    for source in normalized:
        ai = source.get("label_available_index")
        di = int(source["decision_index"])
        y = _label(source)
        in_scoring_partition = (
            score_partition == "all_matured" or source.get("partition") == score_partition
        )
        if (
            in_scoring_partition and source.get("label_mature") is True
            and _integer(ai) and ai < as_of and di < as_of and y in (0, 1)
        ):
            forecast = forecast_by_id.get(source["decision_id"])
            if forecast is None:
                raise ProbabilityInputError("as_of_scored_row_missing_emitted_forecast:" + str(source["decision_id"]))
            merged = dict(source)
            merged["_governed_probability"] = forecast["probability"]
            scored_rows.append(merged)
    model_metrics = _metrics(scored_rows, "_governed_probability")
    baseline_metrics = _metrics(scored_rows, "_baseline")
    statuses = Counter(row["status"] for row in forecasts)
    fallbacks = sum(row["alpha_weight"] <= 0.0 for row in forecasts)
    manifest = {
        "protocol": "serevex-probability-evolution",
        "version": version,
        "as_of_index": as_of,
        "config": settings,
        "forecast_identity": [
            {
                "decision_id": row["decision_id"],
                "decision_index": row["decision_index"],
                "candidate_probability": row["candidate_probability"],
                "baseline_probability": row["baseline_probability"],
                "probability": row["probability"],
                "alpha_weight": row["alpha_weight"],
                "status": row["status"],
            }
            for row in forecasts
        ],
    }
    brier_gain = None
    log_gain = None
    if model_metrics["brier"] is not None and baseline_metrics["brier"] is not None:
        brier_gain = baseline_metrics["brier"] - model_metrics["brier"]
        log_gain = baseline_metrics["log_loss"] - model_metrics["log_loss"]
    evidence = {
        "protocol": "serevex-probability-evolution-report",
        "version": version,
        "as_of_index": as_of,
        "score_partition": score_partition,
        "scored_rows": len(scored_rows),
        "forecast_rows_emitted": len(forecasts),
        "rows_at_or_after_as_of_excluded": len(normalized) - len(point_in_time_rows),
        "last_emitted_decision_index": forecasts[-1]["decision_index"] if forecasts else None,
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "baseline_minus_model_brier_gain": brier_gain,
        "baseline_minus_model_log_loss_gain": log_gain,
        "status_counts": dict(sorted(statuses.items())),
        "fallback_rows": fallbacks,
        "fallback_rate": fallbacks / len(forecasts) if forecasts else 1.0,
        "forecast_manifest_sha256": _sha(manifest),
        "model_validated": False,
        "deployment_calibrated": False,
        "decision_output": "NO_TRADE",
    }
    evidence["evaluation_report_sha256"] = _sha(evidence)
    return {
        "protocol_version": version,
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "config": settings,
        "forecasts": forecasts,
        "policy_trace": policy_trace,
        "evidence": evidence,
    }


def evaluate_all_probability_versions(
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of_index: int | None = None,
    config: Mapping[str, Any] | None = None,
    score_partition: str = "all_matured",
) -> dict[str, Any]:
    """Evaluate all preregistered versions on the same row cohort; no version is auto-promoted."""
    if not isinstance(score_partition, str) or score_partition not in SCORE_PARTITIONS:
        raise ProbabilityInputError("unsupported_score_partition:" + str(score_partition))
    results = {
        version: evaluate_probability_evolution(
            rows, version=version, as_of_index=as_of_index, config=config,
            score_partition=score_partition,
        )
        for version in VERSION_ORDER
    }
    comparison = {
        version: {
            "status": report["status"],
            "scored_rows": report["evidence"]["scored_rows"],
            "brier": report["evidence"]["model_metrics"]["brier"],
            "log_loss": report["evidence"]["model_metrics"]["log_loss"],
            "ece_10": report["evidence"]["model_metrics"]["ece_10"],
            "baseline_minus_model_brier_gain": report["evidence"]["baseline_minus_model_brier_gain"],
            "baseline_minus_model_log_loss_gain": report["evidence"]["baseline_minus_model_log_loss_gain"],
            "fallback_rate": report["evidence"]["fallback_rate"],
            "forecast_manifest_sha256": report["evidence"]["forecast_manifest_sha256"],
        }
        for version, report in results.items()
    }
    return {
        "protocol_version": "4.0_to_9.0",
        "status": "RESEARCH_COMPARISON_NOT_PROMOTION",
        "score_partition": score_partition,
        "versions": results,
        "comparison": comparison,
        "promotion_rule": (
            "No version is promoted from metrics in this same comparison alone. "
            "Require a preregistered later untouched holdout and independent replication."
        ),
        "model_validated": False,
        "deployment_calibrated": False,
        "decision_output": "NO_TRADE",
    }


def adapt_forecast_ledger(
    ledger: Mapping[str, Any], *, candidate_expert: str = "alpha_logistic_v1"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt the canonical Serevex forecast-ledger.v1 into this versioned probability contract."""
    if not isinstance(ledger, Mapping) or ledger.get("schema_version") != "serevex.forecast-ledger.v1":
        raise ProbabilityInputError("unsupported_forecast_ledger_schema")
    metadata = ledger.get("metadata")
    records = ledger.get("records")
    if not isinstance(metadata, Mapping) or not isinstance(records, list) or not records:
        raise ProbabilityInputError("forecast_ledger_metadata_or_records_invalid")
    series_id = metadata.get("series_id", "")
    instrument = str(series_id).split(":")[0] if isinstance(series_id, str) and series_id else "UNKNOWN"
    target_spec = metadata.get("target_spec")
    if not isinstance(target_spec, Mapping):
        target_spec = {}
    horizon = target_spec.get("horizon_bars_from_next_open")
    if not _integer(horizon) or horizon < 1:
        raise ProbabilityInputError("forecast_ledger_horizon_invalid")
    target_id = metadata.get("target_id")
    if not isinstance(target_id, str) or not target_id:
        raise ProbabilityInputError("forecast_ledger_target_id_missing")
    adapted: list[dict[str, Any]] = []
    for pos, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ProbabilityInputError("forecast_ledger_record_not_mapping:" + str(pos))
        experts = record.get("expert_probabilities")
        if not isinstance(experts, Mapping):
            experts = {}
        index = record.get("decision_index")
        regime_available = record.get("regime_available_index", record.get("state_tags_available_index"))
        # Features are adapted according to the ledger's declared decision-time contract.
        # Source-time semantics/provenance remain separately qualified in source_evidence.
        feature_available = record.get("feature_available_index", record.get("state_tags_available_index", index))
        adapted.append({
            "decision_id": record.get("decision_id"),
            "decision_index": index,
            "target_id": target_id,
            "instrument": instrument,
            "horizon_bars": horizon,
            "candidate_probability": experts.get(candidate_expert),
            "baseline_probability": record.get("baseline_probability"),
            "label": record.get("label"),
            "label_mature": record.get("label_mature"),
            "label_available_index": record.get("label_available_index"),
            "regime": record.get("regime_id", UNKNOWN_REGIME),
            "regime_available_index": regime_available,
            "feature_available_index": feature_available,
            "partition": record.get("partition"),
        })
    source_verification = metadata.get("source_verification")
    if not isinstance(source_verification, Mapping):
        source_verification = {}
    source_evidence = {
        "candidate_expert": candidate_expert,
        "target_id": target_id,
        "series_id": series_id,
        "dataset_sha256": metadata.get("dataset_sha256"),
        "source_lineage_sha256": metadata.get("source_lineage_sha256"),
        "source_verification": dict(source_verification),
        "source_timestamp_semantics_verified": metadata.get("source_timestamp_semantics_verified", False),
        "historical_feature_availability_verified": metadata.get("historical_feature_availability_verified", False),
        "broker_quote_basis_verified": metadata.get("broker_quote_basis_verified", False),
        "provenance_note": "Ledger digest and declared time contracts do not independently verify publisher provenance, data rights, historical availability, or broker quote semantics.",
    }
    return adapted, source_evidence


def evaluate_forecast_ledger_versions(
    ledger: Mapping[str, Any],
    *,
    candidate_expert: str = "alpha_logistic_v1",
    as_of_index: int | None = None,
    config: Mapping[str, Any] | None = None,
    score_partition: str = "final_holdout",
) -> dict[str, Any]:
    rows, source_evidence = adapt_forecast_ledger(ledger, candidate_expert=candidate_expert)
    suite = evaluate_all_probability_versions(
        rows, as_of_index=as_of_index, config=config, score_partition=score_partition
    )
    return {
        **suite,
        "source_evidence": source_evidence,
        "research_boundary": {
            "source_qualified_for_training": False,
            "predictive_skill_proven": False,
            "independent_replication_complete": False,
            "economic_utility_proven": False,
            "model_validated": False,
            "deployment_calibrated": False,
            "live_signal_enabled": False,
            "live_order_execution_enabled": False,
            "decision_output": "NO_TRADE",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Serevex probability governance v4.0-v9.0 on a canonical forecast ledger")
    parser.add_argument("--input", type=Path, required=True, help="JSON file with schema serevex.forecast-ledger.v1")
    parser.add_argument("--output", type=Path, required=True, help="JSON comparison report")
    parser.add_argument("--as-of-index", type=int, required=True, help="Exclusive underlying decision-bar cutoff")
    parser.add_argument("--candidate-expert", default="alpha_logistic_v1")
    parser.add_argument(
        "--score-partition", choices=SCORE_PARTITIONS, default="final_holdout",
        help="Partition for reported metrics; forecasts still run causally over all rows before the exclusive as-of cutoff",
    )
    args = parser.parse_args(argv)
    try:
        ledger = json.loads(args.input.read_text(encoding="utf-8"))
        report = evaluate_forecast_ledger_versions(
            ledger, candidate_expert=args.candidate_expert, as_of_index=args.as_of_index,
            score_partition=args.score_partition,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "protocol_version": report["protocol_version"],
            "candidate_expert": args.candidate_expert,
            "score_partition": report["score_partition"],
            "versions_evaluated": list(report["versions"]),
            "source_qualified_for_training": False,
            "decision_output": "NO_TRADE",
            "report_path": str(args.output),
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
