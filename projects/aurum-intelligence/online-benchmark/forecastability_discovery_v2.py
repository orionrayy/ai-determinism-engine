"""Multi-scale, paired forecast-skill and drift gate (research only).

The method evaluates only pre-generated causal OOS probabilities. It does not fit
forecasters, claim novelty, infer profitability, or authorize trading.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

PROTOCOL_VERSION = "serevex-forecastability-discovery-v2"
AUTHORITY = {
    "market_promotion_allowed": False,
    "model_validated": False,
    "deployment_calibrated": False,
    "live_signal_enabled": False,
    "live_order_execution_enabled": False,
    "decision_output": "NO_TRADE",
}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _prob(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}_must_be_finite_probability")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}_must_be_finite_probability") from exc
    if not math.isfinite(out) or not 0 <= out <= 1:
        raise ValueError(f"{name}_must_be_finite_probability")
    return out


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile_requires_values")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(q * (len(ordered) + 1)) - 1))
    return ordered[idx]


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _normalize_records(
    records: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    regimes: Sequence[str],
) -> list[dict[str, Any]]:
    if not models or len(set(models)) != len(models) or any(not isinstance(x, str) or not x.strip() for x in models):
        raise ValueError("registered_models_must_be_nonempty_unique_strings")
    if not regimes or len(set(regimes)) != len(regimes) or any(not isinstance(x, str) or not x.strip() for x in regimes):
        raise ValueError("registered_regimes_must_be_nonempty_unique_strings")
    model_set, regime_set = set(models), set(regimes)
    grouped: dict[str, dict[str, Any]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    required = {
        "decision_id", "model_id", "decision_index", "label_available_index",
        "regime_id", "regime_available_index", "segment_id", "probability",
        "baseline_probability", "label", "label_mature",
    }
    for i, src in enumerate(records):
        if not isinstance(src, Mapping):
            raise ValueError(f"record_must_be_mapping_at_{i}")
        missing = sorted(required - set(src))
        if missing:
            raise ValueError(f"missing_fields_at_{i}:{','.join(missing)}")
        row = dict(src)
        did, mid, rid, segment = row["decision_id"], row["model_id"], row["regime_id"], row["segment_id"]
        if not isinstance(did, str) or not did.strip():
            raise ValueError(f"invalid_decision_id_at_{i}")
        if mid not in model_set:
            raise ValueError(f"unregistered_model_at_{i}")
        if rid not in regime_set:
            raise ValueError(f"unregistered_regime_at_{i}")
        if not isinstance(segment, str) or not segment.strip():
            raise ValueError(f"invalid_segment_id_at_{i}")
        di, li, ri = row["decision_index"], row["label_available_index"], row["regime_available_index"]
        if not _is_int(di) or di < 0 or not _is_int(li) or li <= di:
            raise ValueError(f"invalid_or_unmatured_index_contract_at_{i}")
        if not _is_int(ri) or not 0 <= ri <= di:
            raise ValueError(f"regime_not_available_as_of_decision_at_{i}")
        if not isinstance(row["label_mature"], bool):
            raise ValueError(f"label_mature_must_be_boolean_at_{i}")
        label = row["label"]
        if label is not None and (isinstance(label, bool) or label not in (0, 1)):
            raise ValueError(f"label_must_be_0_1_or_none_at_{i}")
        if row["label_mature"] and label is None:
            raise ValueError(f"mature_label_missing_at_{i}")
        row["probability"] = _prob(row["probability"], "probability")
        row["baseline_probability"] = _prob(row["baseline_probability"], "baseline_probability")
        key = (did, mid)
        if key in seen_pairs:
            raise ValueError(f"duplicate_decision_model_pair_at_{i}")
        seen_pairs.add(key)
        shared = {
            "decision_index": di, "label_available_index": li, "regime_id": rid,
            "regime_available_index": ri, "segment_id": segment, "label": label,
            "label_mature": row["label_mature"], "baseline_probability": row["baseline_probability"],
        }
        current = grouped.setdefault(did, {"decision_id": did, **shared, "probabilities": {}})
        for field, value in shared.items():
            if current[field] != value:
                raise ValueError(f"cross_model_cohort_mismatch:{did}:{field}")
        current["probabilities"][mid] = row["probability"]
    expected = set(models)
    for did, row in grouped.items():
        if set(row["probabilities"]) != expected:
            raise ValueError(f"unaligned_model_cohort:{did}")
    rows = sorted(grouped.values(), key=lambda x: x["decision_index"])
    seen_indices: set[int] = set()
    prev = -1
    for row in rows:
        if row["decision_index"] <= prev:
            raise ValueError("decision_indices_must_be_strictly_increasing_and_unique")
        seen_indices.add(row["decision_index"])
        prev = row["decision_index"]
    if not rows:
        raise ValueError("records_must_be_nonempty")
    return rows


def _runs(rows: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Index positions for contiguous decision bars; retain regime transitions inside blocks."""
    out: list[list[int]] = []
    cur: list[int] = []
    prev_index = None
    prev_segment = None
    for pos, row in enumerate(rows):
        idx, segment = row["decision_index"], row["segment_id"]
        if cur and (idx != prev_index + 1 or segment != prev_segment):
            out.append(cur)
            cur = []
        cur.append(pos)
        prev_index, prev_segment = idx, segment
    if cur:
        out.append(cur)
    return out


def _independent_blocks(rows: Sequence[Mapping[str, Any]], regime: str, block_size: int) -> int:
    total = 0
    for run in _runs(rows):
        usable = len(run) // block_size
        for j in range(usable):
            block = run[j * block_size:(j + 1) * block_size]
            if any(rows[pos]["regime_id"] == regime for pos in block):
                total += 1
    return total


def _sample_block_positions(
    runs: Sequence[Sequence[int]], n: int, block_size: int, rng: random.Random
) -> list[int]:
    starts: list[tuple[Sequence[int], int]] = []
    for run in runs:
        if len(run) >= block_size:
            starts.extend((run, start) for start in range(len(run) - block_size + 1))
    if not starts:
        raise ValueError("no_complete_contiguous_bootstrap_blocks")
    positions: list[int] = []
    while len(positions) < n:
        run, start = starts[rng.randrange(len(starts))]
        positions.extend(run[start:start + block_size])
    return positions[:n]


def _skill(row: Mapping[str, Any], model_id: str) -> float:
    y = int(row["label"])
    base = (float(row["baseline_probability"]) - y) ** 2
    model = (float(row["probabilities"][model_id]) - y) ** 2
    return base - model


def assess_forecastability_v2(
    records: Sequence[Mapping[str, Any]],
    *,
    as_of_index: int,
    registered_models: Sequence[str],
    registered_regimes: Sequence[str],
    window_bars: Sequence[int] = (240, 960, 3840),
    block_sizes: Sequence[int] = (6, 12, 24),
    min_rows_per_regime_window: int = 60,
    min_independent_blocks: int = 8,
    bootstrap_replicates: int = 2000,
    familywise_alpha: float = 0.05,
    minimum_skill: float = 0.0,
    drift_warning_margin: float = 0.005,
    seed: int = 20261019,
    evidence_kind: str = "real_market_research",
) -> dict[str, Any]:
    """Assess registered probability experts with paired multi-scale evidence.

    Skill is Brier(baseline) - Brier(model). At each window/block-size pair,
    a synchronized max-|t| moving-block bootstrap supplies simultaneous
    approximate lower bounds across all supported model × regime × window
    hypotheses. Eligibility additionally requires every declared scale to pass.
    This is not an exact guarantee under arbitrary market dependence; report it
    as a dependence-aware sensitivity gate, not proof of alpha.
    """
    if not _is_int(as_of_index) or as_of_index < 0:
        raise ValueError("as_of_index_must_be_nonnegative_integer")
    if not _is_int(min_rows_per_regime_window) or min_rows_per_regime_window < 20:
        raise ValueError("min_rows_per_regime_window_must_be_at_least_20")
    if not _is_int(min_independent_blocks) or min_independent_blocks < 2:
        raise ValueError("min_independent_blocks_must_be_at_least_2")
    if not _is_int(bootstrap_replicates) or bootstrap_replicates < 200 or bootstrap_replicates > 20000:
        raise ValueError("bootstrap_replicates_must_be_200_to_20000")
    if isinstance(familywise_alpha, bool) or not isinstance(familywise_alpha, (int, float)) or not math.isfinite(float(familywise_alpha)) or not 0 < float(familywise_alpha) < 0.2:
        raise ValueError("familywise_alpha_must_be_between_0_and_0_2")
    if isinstance(minimum_skill, bool) or not isinstance(minimum_skill, (int, float)) or not math.isfinite(float(minimum_skill)):
        raise ValueError("minimum_skill_must_be_finite")
    if isinstance(drift_warning_margin, bool) or not isinstance(drift_warning_margin, (int, float)) or not math.isfinite(float(drift_warning_margin)) or drift_warning_margin < 0:
        raise ValueError("drift_warning_margin_must_be_nonnegative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed_must_be_integer")
    if evidence_kind not in {"real_market_research", "synthetic_contract_test", "replay_fixture"}:
        raise ValueError("evidence_kind_unrecognized")
    if not window_bars or any(not _is_int(w) or w < 40 for w in window_bars) or len(set(window_bars)) != len(window_bars):
        raise ValueError("window_bars_must_be_unique_integers_at_least_40")
    if not block_sizes or any(not _is_int(b) or b < 2 for b in block_sizes) or len(set(block_sizes)) != len(block_sizes):
        raise ValueError("block_sizes_must_be_unique_integers_at_least_2")

    all_rows = _normalize_records(records, registered_models, registered_regimes)
    mature = [
        row for row in all_rows
        if row["decision_index"] < as_of_index
        and row["label_available_index"] < as_of_index
        and row["label_mature"] is True
        and row["label"] in (0, 1)
    ]
    if not mature:
        raise ValueError("no_matured_as_of_labels")
    latest_index = max(row["decision_index"] for row in mature)
    windows = sorted((int(w) for w in window_bars), reverse=False)
    pairs = [(m, r) for m in registered_models for r in registered_regimes]
    reports: dict[str, dict[str, Any]] = {m: {} for m in registered_models}
    observed: dict[tuple[str, str, int], float] = {}
    distributions: dict[tuple[str, str, int, int], list[float]] = {}
    support: dict[tuple[str, str, int], dict[str, Any]] = {}
    window_rows_by_size: dict[int, list[dict[str, Any]]] = {}
    for window in windows:
        cutoff = latest_index - window + 1
        rows = [row for row in mature if row["decision_index"] >= cutoff]
        window_rows_by_size[window] = rows
        runs = _runs(rows)
        for model_id, regime_id in pairs:
            key = (model_id, regime_id, window)
            chosen = [row for row in rows if row["regime_id"] == regime_id]
            blocks_by_size = {b: _independent_blocks(rows, regime_id, b) for b in block_sizes}
            supported_blocks = all(blocks_by_size[b] >= min_independent_blocks for b in block_sizes)
            support[key] = {
                "decision_window_bars": window,
                "window_start_index": cutoff,
                "latest_matured_decision_index": latest_index,
                "matured_regime_rows": len(chosen),
                "independent_regime_blocks_by_size": {str(b): blocks_by_size[b] for b in block_sizes},
                "min_rows_required": min_rows_per_regime_window,
                "min_independent_blocks_required": min_independent_blocks,
                "status": "SUPPORTED" if len(chosen) >= min_rows_per_regime_window and supported_blocks else "INSUFFICIENT_SUPPORT",
                "decision_ids_sha256": _sha([row["decision_id"] for row in chosen]),
                "labels_sha256": _sha([row["label"] for row in chosen]),
            }
            if support[key]["status"] != "SUPPORTED":
                continue
            observed[key] = sum(_skill(row, model_id) for row in chosen) / len(chosen)

    # Every bootstrap draw is shared by every model and regime for a given
    # window and block size, preserving within-draw cross-model dependence.
    unsupported_keys = {key for key, meta in support.items() if meta["status"] != "SUPPORTED"}
    max_t_by_block_rep: dict[int, list[float]] = {b: [0.0] * bootstrap_replicates for b in block_sizes}
    for window in windows:
        rows = window_rows_by_size[window]
        runs = _runs(rows)
        relevant = [(m, r, window) for m, r in pairs if (m, r, window) in observed]
        if not relevant:
            continue
        for block_size in block_sizes:
            # Use a deterministic independent substream per window/block length.
            rng = random.Random(seed + window * 37 + block_size * 1009)
            dists = {(*key, block_size): [] for key in relevant}
            valid_replicates = 0
            for _ in range(bootstrap_replicates):
                positions = _sample_block_positions(runs, len(rows), block_size, rng)
                sample = [rows[pos] for pos in positions]
                rep_means: dict[tuple[str, str, int, int], float] = {}
                valid = True
                for key in relevant:
                    model_id, regime_id, _window = key
                    chosen = [row for row in sample if row["regime_id"] == regime_id]
                    if not chosen:
                        valid = False
                        break
                    rep_means[(*key, block_size)] = sum(_skill(row, model_id) for row in chosen) / len(chosen)
                if not valid:
                    continue
                valid_replicates += 1
                for key, value in rep_means.items():
                    dists[key].append(value)
            # Require every replicate to represent every tested regime. Allowing a
            # variable number would misalign synchronized max-statistic draws.
            if valid_replicates != bootstrap_replicates:
                for key in relevant:
                    support[key]["status"] = "BOOTSTRAP_REGIME_SUPPORT_UNSTABLE"
                continue
            for key, dist in dists.items():
                distributions[key] = dist
            # A separate max-|t| critical value for each pre-registered block
            # length, simultaneous across all models, regimes and windows.
            # Window samples use deterministic substreams; cross-window dependence
            # is not modeled exactly, so this gate is deliberately conservative.
            se_by_key = {
                key: _std(distributions[(*key, block_size)])
                for key in relevant
                if (*key, block_size) in distributions
            }
            for key, se in se_by_key.items():
                if se <= 1e-12:
                    support[key]["status"] = "BOOTSTRAP_DEGENERATE"
            valid_keys = [key for key in relevant if support[key]["status"] == "SUPPORTED" and se_by_key.get(key, 0.0) > 1e-12]
            if not valid_keys:
                continue
            # Need a max statistic over the whole registered set. Here missing or
            # degenerate hypotheses block their own model/regime, while their absent
            # statistic cannot be used as a winning candidate.
            for rep in range(bootstrap_replicates):
                stats = []
                for key in valid_keys:
                    dist = distributions[(*key, block_size)]
                    stats.append(abs((dist[rep] - observed[key]) / se_by_key[key]))
                max_t_by_block_rep[block_size][rep] = max(
                    max_t_by_block_rep[block_size][rep],
                    max(stats) if stats else 0.0,
                )

    critical_by_block: dict[int, float] = {}
    for block_size in block_sizes:
        vals = max_t_by_block_rep[block_size]
        critical_by_block[block_size] = _quantile(vals, 1.0 - float(familywise_alpha)) if vals else math.inf

    # Summarize each model × regime across all time windows and block lengths.
    for model_id, regime_id in pairs:
        per_window: dict[str, Any] = {}
        all_pass = True
        all_supported = True
        lower_bounds = []
        means_by_window: list[tuple[int, float]] = []
        for window in windows:
            key = (model_id, regime_id, window)
            meta = support[key]
            item = {**meta, "mean_skill": observed.get(key), "block_sensitivity": {}}
            if meta["status"] != "SUPPORTED" or key not in observed:
                all_supported = False
                all_pass = False
            else:
                means_by_window.append((window, observed[key]))
                for block_size in block_sizes:
                    dkey = (*key, block_size)
                    dist = distributions.get(dkey)
                    if dist is None or len(dist) < max(200, int(bootstrap_replicates * 0.99)):
                        item["block_sensitivity"][str(block_size)] = {"status": "UNAVAILABLE"}
                        all_supported = False
                        all_pass = False
                        continue
                    se = _std(dist)
                    critical = critical_by_block[block_size]
                    lower = observed[key] - critical * se
                    upper = observed[key] + critical * se
                    lower_bounds.append(lower)
                    pass_gate = lower > float(minimum_skill)
                    item["block_sensitivity"][str(block_size)] = {
                        "status": "LOWER_BOUND_ABOVE_MINIMUM" if pass_gate else "SKILL_NOT_DEMONSTRATED",
                        "mean_skill": observed[key],
                        "bootstrap_se": se,
                        "simultaneous_critical_value_max_abs_t": critical,
                        "simultaneous_lower_bound": lower,
                        "simultaneous_upper_bound": upper,
                        "minimum_skill": float(minimum_skill),
                        "replicates": len(dist),
                    }
                    if not pass_gate:
                        all_pass = False
            per_window[str(window)] = item
        ordered_means = sorted(means_by_window)
        recent_mean = ordered_means[0][1] if ordered_means else None
        long_mean = ordered_means[-1][1] if ordered_means else None
        delta_recent_long = recent_mean - long_mean if recent_mean is not None and long_mean is not None else None
        drift_warning = bool(
            delta_recent_long is not None
            and (recent_mean <= float(minimum_skill) or delta_recent_long < -float(drift_warning_margin))
        )
        status = (
            "INSUFFICIENT_SUPPORT" if not all_supported
            else "ELIGIBLE_FOR_INDEPENDENT_REPLICATION" if all_pass
            else "SKILL_NOT_STABLE_ACROSS_WINDOWS_OR_BLOCKS"
        )
        # Conservative competence weight uses the worst corrected lower bound.
        weight = max(0.0, min(lower_bounds) - float(minimum_skill)) if status == "ELIGIBLE_FOR_INDEPENDENT_REPLICATION" and lower_bounds else 0.0
        reports[model_id][regime_id] = {
            "status": status,
            "weight": weight,
            "minimum_corrected_lower_bound": min(lower_bounds) if lower_bounds else None,
            "drift_warning": drift_warning,
            "recent_minus_long_window_skill": delta_recent_long,
            "interpretation": "Drift warning is descriptive; only future, frozen out-of-sample evaluation can establish utility.",
            "windows": per_window,
        }

    eligible = [
        {"model_id": m, "regime_id": r, "weight": reports[m][r]["weight"]}
        for m, r in pairs if reports[m][r]["status"] == "ELIGIBLE_FOR_INDEPENDENT_REPLICATION"
    ]
    for regime_id in registered_regimes:
        group = [item for item in eligible if item["regime_id"] == regime_id]
        total = sum(item["weight"] for item in group)
        if total > 0:
            for item in group:
                item["weight"] /= total
                reports[item["model_id"]][regime_id]["weight"] = item["weight"]
    status = "ELIGIBLE_FOR_INDEPENDENT_REPLICATION" if eligible else "NO_STABLE_PREDICTIVE_SKILL_DEMONSTRATED_OR_INSUFFICIENT_SUPPORT"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "as_of_index": as_of_index,
        "input_records": len(records),
        "matured_decisions": len(mature),
        "latest_matured_decision_index": latest_index,
        "registered_models": list(registered_models),
        "registered_regimes": list(registered_regimes),
        "window_bars": list(windows),
        "block_sizes": list(block_sizes),
        "minimum_skill": float(minimum_skill),
        "familywise_alpha": float(familywise_alpha),
        "simultaneous_method": "synchronized-within-window moving-block bootstrap, studentized max-|t| bounds per block-length sensitivity",
        "simultaneous_bound_caveat": "Approximate under block-bootstrap assumptions; cross-window resampling uses deterministic substreams and is not an exact finite-sample guarantee under arbitrary market dependence.",
        "skill_definition": "Brier(baseline) - Brier(model); positive favors model",
        "support": {f"{m}|{r}|{w}": meta for (m, r, w), meta in support.items()},
        "models": reports,
        "eligible_model_regime_pairs": eligible,
        "research_only": True,
        **AUTHORITY,
    }


def combine_frozen_forecasts_v2(
    gate_report: Mapping[str, Any],
    *,
    current_regime: str,
    current_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Blend only stable historical experts; never enables a trade."""
    if current_regime not in gate_report.get("registered_regimes", []):
        return {"forecast_status": "UNKNOWN_REGIME_NO_FORECAST", "probability": None, "weights": {}, "reason": "regime_not_preregistered", "research_only": True, **AUTHORITY}
    registered = gate_report.get("registered_models", [])
    reports = gate_report.get("models", {})
    weights = {
        model: float(reports.get(model, {}).get(current_regime, {}).get("weight", 0.0))
        for model in registered
        if reports.get(model, {}).get(current_regime, {}).get("status") == "ELIGIBLE_FOR_INDEPENDENT_REPLICATION"
    }
    weights = {model: w for model, w in weights.items() if w > 0}
    if not weights:
        return {"forecast_status": "NO_ELIGIBLE_FORECASTER", "probability": None, "weights": {}, "reason": "no_stable_corrected_skill", "research_only": True, **AUTHORITY}
    probs = {}
    for model in weights:
        if model not in current_probabilities:
            return {"forecast_status": "MISSING_CURRENT_MODEL_PROBABILITY", "probability": None, "weights": {}, "reason": f"missing_current_probability:{model}", "research_only": True, **AUTHORITY}
        probs[model] = _prob(current_probabilities[model], f"current_probability:{model}")
    total = sum(weights.values())
    normalized = {model: weight / total for model, weight in weights.items()}
    return {
        "forecast_status": "EXPERIMENTAL_BLEND_FOR_LATER_UNTOUCHED_EVALUATION",
        "probability": sum(normalized[m] * probs[m] for m in normalized),
        "weights": normalized,
        "reason": "only_model_regime_pairs_with_positive_multiscale_corrected_skill",
        "research_only": True,
        **AUTHORITY,
    }
