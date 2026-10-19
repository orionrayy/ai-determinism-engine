"""Causal, regime-conditioned forecast-skill discovery gate (research only).
This module scores pre-generated out-of-sample probabilities against a causal
baseline. It does not fit forecasters, authorize trades, or claim market alpha.
All model/regime hypotheses must be preregistered before outcomes are inspected.
Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

PROTOCOL_VERSION = "serevex-causal-forecastability-discovery-v1"
_FORBIDDEN_AUTHORITY = {
    "market_promotion_allowed": False,
    "model_validated": False,
    "deployment_calibrated": False,
    "live_signal_enabled": False,
    "live_order_execution_enabled": False,
    "decision_output": "NO_TRADE",
}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field}_must_be_finite_probability")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field}_must_be_finite_probability") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field}_must_be_finite_probability")
    return result


def _sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile_requires_values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(1.0, max(0.0, q))
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def _validate_and_index(
    records: Sequence[Mapping[str, Any]],
    *,
    registered_models: Sequence[str],
    registered_regimes: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not registered_models or any(not isinstance(x, str) or not x.strip() for x in registered_models):
        raise ValueError("registered_models_must_be_nonempty_strings")
    if len(set(registered_models)) != len(registered_models):
        raise ValueError("registered_models_must_be_unique")
    if not registered_regimes or any(not isinstance(x, str) or not x.strip() for x in registered_regimes):
        raise ValueError("registered_regimes_must_be_nonempty_strings")
    if len(set(registered_regimes)) != len(registered_regimes):
        raise ValueError("registered_regimes_must_be_unique")

    model_set = set(registered_models)
    regime_set = set(registered_regimes)
    seen: set[tuple[str, str]] = set()
    cohort: dict[str, dict[str, Any]] = {}
    normalized: list[dict[str, Any]] = []
    required = {
        "decision_id", "model_id", "decision_index", "label_available_index",
        "regime_id", "regime_available_index", "segment_id", "probability",
        "baseline_probability", "label", "label_mature",
    }
    for pos, source in enumerate(records):
        if not isinstance(source, Mapping):
            raise ValueError(f"record_must_be_mapping_at_{pos}")
        missing = sorted(required - set(source))
        if missing:
            raise ValueError(f"missing_fields_at_{pos}:{','.join(missing)}")
        row = dict(source)
        decision_id = row["decision_id"]
        model_id = row["model_id"]
        regime_id = row["regime_id"]
        segment_id = row["segment_id"]
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValueError(f"invalid_decision_id_at_{pos}")
        if model_id not in model_set:
            raise ValueError(f"unregistered_model_at_{pos}")
        if regime_id not in regime_set:
            raise ValueError(f"unregistered_regime_at_{pos}")
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise ValueError(f"invalid_segment_id_at_{pos}")
        di = row["decision_index"]
        li = row["label_available_index"]
        ri = row["regime_available_index"]
        if not _is_int(di) or di < 0:
            raise ValueError(f"invalid_decision_index_at_{pos}")
        if not _is_int(li) or li <= di:
            raise ValueError(f"label_must_mature_after_decision_at_{pos}")
        if not _is_int(ri) or ri < 0 or ri > di:
            raise ValueError(f"regime_must_be_available_as_of_decision_at_{pos}")
        mature = row["label_mature"]
        if not isinstance(mature, bool):
            raise ValueError(f"label_mature_must_be_boolean_at_{pos}")
        label = row["label"]
        if label is not None and (isinstance(label, bool) or label not in (0, 1)):
            raise ValueError(f"label_must_be_0_1_or_none_at_{pos}")
        if mature and label is None:
            raise ValueError(f"mature_label_missing_at_{pos}")
        row["probability"] = _probability(row["probability"], "probability")
        row["baseline_probability"] = _probability(row["baseline_probability"], "baseline_probability")

        pair_key = (decision_id, model_id)
        if pair_key in seen:
            raise ValueError(f"duplicate_decision_model_pair_at_{pos}")
        seen.add(pair_key)
        facts = {
            "decision_id": decision_id,
            "decision_index": di,
            "label_available_index": li,
            "regime_id": regime_id,
            "regime_available_index": ri,
            "segment_id": segment_id,
            "label": label,
            "label_mature": mature,
            "baseline_probability": row["baseline_probability"],
        }
        previous = cohort.get(decision_id)
        if previous is None:
            cohort[decision_id] = {**facts, "models": {model_id}}
        else:
            for key in (
                "decision_index", "label_available_index", "regime_id",
                "regime_available_index", "segment_id", "label", "label_mature",
                "baseline_probability",
            ):
                if previous[key] != facts[key]:
                    raise ValueError(f"cross_model_cohort_mismatch:{decision_id}:{key}")
            if model_id in previous["models"]:
                raise ValueError(f"duplicate_model_in_cohort:{decision_id}:{model_id}")
            previous["models"].add(model_id)
        normalized.append(row)

    if not normalized:
        raise ValueError("records_must_be_nonempty")
    expected_models = set(registered_models)
    for decision_id, facts in cohort.items():
        if facts["models"] != expected_models:
            missing = sorted(expected_models - facts["models"])
            extra = sorted(facts["models"] - expected_models)
            raise ValueError(f"unaligned_model_cohort:{decision_id}:missing={missing}:extra={extra}")

    # The first prototype is single-series XAUUSD. Multi-asset use needs a series key.
    by_index: dict[int, str] = {}
    for decision_id, facts in cohort.items():
        di = facts["decision_index"]
        if di in by_index and by_index[di] != decision_id:
            raise ValueError(f"multiple_decisions_at_same_index:{di}")
        by_index[di] = decision_id
    normalized.sort(key=lambda row: (row["decision_index"], row["model_id"]))
    return normalized, cohort


def _contiguous_runs(items: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    """Split paired skill differentials at time gaps and declared session boundaries."""
    ordered = sorted(items, key=lambda r: r["decision_index"])
    runs: list[list[float]] = []
    current: list[float] = []
    previous_index: int | None = None
    previous_segment: str | None = None
    previous_regime: str | None = None
    for row in ordered:
        index = row["decision_index"]
        segment = row["segment_id"]
        regime = row["regime_id"]
        if current and (
            previous_index is None
            or index != previous_index + 1
            or segment != previous_segment
            or regime != previous_regime
        ):
            runs.append(current)
            current = []
        current.append(float(row["skill_difference"]))
        previous_index = index
        previous_segment = segment
        previous_regime = regime
    if current:
        runs.append(current)
    return runs


def _moving_block_bootstrap_means(
    runs: Sequence[Sequence[float]], *, sample_size: int, block_size: int,
    replicates: int, seed: int,
) -> list[float]:
    eligible = [list(run) for run in runs if run]
    if not eligible:
        return []
    rng = random.Random(seed)
    total_weight = sum(len(run) for run in eligible)
    means: list[float] = []
    for _ in range(replicates):
        sample: list[float] = []
        while len(sample) < sample_size:
            ticket = rng.randrange(total_weight)
            selected: Sequence[float] | None = None
            for run in eligible:
                ticket -= len(run)
                if ticket < 0:
                    selected = run
                    break
            assert selected is not None
            start = rng.randrange(len(selected))
            take = min(block_size, len(selected) - start, sample_size - len(sample))
            sample.extend(selected[start:start + take])
        means.append(sum(sample) / len(sample))
    return means


def assess_forecastability(
    records: Sequence[Mapping[str, Any]],
    *,
    as_of_index: int,
    registered_models: Sequence[str],
    registered_regimes: Sequence[str],
    min_matured_rows: int = 240,
    block_size: int = 12,
    min_independent_blocks: int = 10,
    bootstrap_replicates: int = 2500,
    familywise_alpha: float = 0.05,
    minimum_skill: float = 0.0,
    seed: int = 20261019,
    evidence_kind: str = "real_market_research",
) -> dict[str, Any]:
    """Assess only labels strictly matured before as_of_index.

    Skill = Brier(baseline) - Brier(model), so positive values favor the model.
    All preregistered model × regime hypotheses count toward the multiplicity
    correction. Eligibility means only "eligible for independent replication."
    """
    if not _is_int(as_of_index) or as_of_index < 0:
        raise ValueError("as_of_index_must_be_nonnegative_integer")
    if not _is_int(min_matured_rows) or min_matured_rows < 20:
        raise ValueError("min_matured_rows_must_be_integer_at_least_20")
    if not _is_int(block_size) or block_size < 1:
        raise ValueError("block_size_must_be_positive_integer")
    if not _is_int(min_independent_blocks) or min_independent_blocks < 2:
        raise ValueError("min_independent_blocks_must_be_at_least_2")
    if not _is_int(bootstrap_replicates) or not 200 <= bootstrap_replicates <= 10000:
        raise ValueError("bootstrap_replicates_must_be_between_200_and_10000")
    if isinstance(familywise_alpha, bool) or not isinstance(familywise_alpha, (int, float)) or not math.isfinite(float(familywise_alpha)) or not 0 < float(familywise_alpha) < 0.2:
        raise ValueError("familywise_alpha_must_be_between_0_and_0_2")
    if isinstance(minimum_skill, bool) or not isinstance(minimum_skill, (int, float)) or not math.isfinite(float(minimum_skill)):
        raise ValueError("minimum_skill_must_be_finite")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed_must_be_integer")
    if evidence_kind not in {"real_market_research", "synthetic_contract_test", "replay_fixture"}:
        raise ValueError("evidence_kind_unrecognized")

    normalized, cohort = _validate_and_index(
        records, registered_models=registered_models, registered_regimes=registered_regimes
    )
    matured = [
        row for row in normalized
        if row["decision_index"] < as_of_index
        and row["label_available_index"] < as_of_index
        and row["label_mature"] is True
        and row["label"] in (0, 1)
    ]
    excluded_unmatured = len(normalized) - len(matured)
    family_size = len(registered_models) * len(registered_regimes)
    adjusted_alpha = float(familywise_alpha) / family_size
    reports: dict[str, dict[str, Any]] = {}
    eligible_any = False

    for model_index, model_id in enumerate(registered_models):
        reports[model_id] = {}
        for regime_index, regime_id in enumerate(registered_regimes):
            selected = []
            for row in matured:
                if row["model_id"] != model_id or row["regime_id"] != regime_id:
                    continue
                y = int(row["label"])
                base_loss = (float(row["baseline_probability"]) - y) ** 2
                model_loss = (float(row["probability"]) - y) ** 2
                selected.append({
                    "decision_id": row["decision_id"],
                    "decision_index": row["decision_index"],
                    "segment_id": row["segment_id"],
                    "regime_id": row["regime_id"],
                    "skill_difference": base_loss - model_loss,
                    "label": y,
                    "probability": row["probability"],
                    "baseline_probability": row["baseline_probability"],
                })
            runs = _contiguous_runs(selected)
            independent_blocks = sum(len(run) // block_size for run in runs)
            cohort_hash = _sha([
                {"decision_id": row["decision_id"], "decision_index": row["decision_index"], "label": row["label"]}
                for row in selected
            ])
            base = {
                "model_id": model_id,
                "regime_id": regime_id,
                "matured_rows": len(selected),
                "contiguous_runs": len(runs),
                "independent_full_blocks": independent_blocks,
                "decision_label_cohort_sha256": cohort_hash,
                "familywise_alpha": float(familywise_alpha),
                "bonferroni_tests": family_size,
                "one_sided_alpha_after_familywise_correction": adjusted_alpha,
                "block_size": block_size,
                "bootstrap_replicates": bootstrap_replicates,
                "minimum_skill": float(minimum_skill),
                "weight": 0.0,
            }
            if len(selected) < min_matured_rows or independent_blocks < min_independent_blocks:
                reports[model_id][regime_id] = {
                    **base,
                    "status": "INSUFFICIENT_MATURED_SUPPORT",
                    "mean_skill": None,
                    "lower_skill_bound": None,
                    "upper_skill_bound": None,
                }
                continue

            differences = [float(row["skill_difference"]) for row in selected]
            means = _moving_block_bootstrap_means(
                runs,
                sample_size=len(differences),
                block_size=block_size,
                replicates=bootstrap_replicates,
                seed=seed + model_index * 1009 + regime_index * 9176,
            )
            mean_skill = sum(differences) / len(differences)
            lower = _quantile(means, adjusted_alpha)
            upper = _quantile(means, 1.0 - adjusted_alpha)
            eligible = lower > float(minimum_skill)
            if eligible:
                eligible_any = True
            reports[model_id][regime_id] = {
                **base,
                "status": "ELIGIBLE_FOR_INDEPENDENT_REPLICATION" if eligible else "SKILL_NOT_DEMONSTRATED",
                "mean_skill": mean_skill,
                "lower_skill_bound": lower,
                "upper_skill_bound": upper,
                "bootstrap_distribution_sha256": _sha([round(v, 14) for v in means]),
                "interpretation": "Positive baseline-minus-model Brier skill favors the model; this is a research gate, not alpha proof.",
                "weight": max(0.0, lower - float(minimum_skill)) if eligible else 0.0,
            }

    for regime_id in registered_regimes:
        total = sum(reports[m][regime_id]["weight"] for m in registered_models)
        if total > 0:
            for model_id in registered_models:
                reports[model_id][regime_id]["weight"] /= total

    eligible_pairs = [
        {"model_id": m, "regime_id": r, "weight": reports[m][r]["weight"]}
        for m in registered_models for r in registered_regimes
        if reports[m][r]["status"] == "ELIGIBLE_FOR_INDEPENDENT_REPLICATION"
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "ELIGIBLE_FOR_INDEPENDENT_REPLICATION" if eligible_any else "NO_PREDICTIVE_SKILL_DEMONSTRATED_OR_INSUFFICIENT_SUPPORT",
        "as_of_index": as_of_index,
        "input_record_count": len(normalized),
        "input_decision_count": len(cohort),
        "matured_decision_count": len({row["decision_id"] for row in matured}),
        "matured_model_record_count": len(matured),
        "excluded_unmatured_model_record_count": excluded_unmatured,
        "registered_models": list(registered_models),
        "registered_regimes": list(registered_regimes),
        "evidence_kind": evidence_kind,
        "skill_definition": "Brier(baseline) - Brier(model); positive favors model",
        "uncertainty_method": "pre-registered regime/model family with Bonferroni-adjusted percentile moving-block bootstrap; blocks never cross time gaps or segment boundaries",
        "models": reports,
        "eligible_model_regime_pairs": eligible_pairs,
        "research_only": True,
        **_FORBIDDEN_AUTHORITY,
    }


def combine_current_forecasts(
    gate_report: Mapping[str, Any],
    *,
    current_regime: str,
    current_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Combine only models with positive corrected lower skill bounds.

    The result is an experimental forecast for later evaluation, never trading
    authority. decision_output is always NO_TRADE.
    """
    registered_regimes = gate_report.get("registered_regimes", [])
    registered_models = gate_report.get("registered_models", [])
    if current_regime not in registered_regimes:
        return {
            "forecast_status": "UNKNOWN_REGIME_NO_FORECAST",
            "probability": None,
            "weights": {},
            "reason": "current_regime_not_preregistered",
            "research_only": True,
            **_FORBIDDEN_AUTHORITY,
        }
    model_reports = gate_report.get("models", {})
    weights: dict[str, float] = {}
    for model_id in registered_models:
        entry = model_reports.get(model_id, {}).get(current_regime, {})
        if entry.get("status") == "ELIGIBLE_FOR_INDEPENDENT_REPLICATION" and float(entry.get("weight", 0.0)) > 0:
            weights[model_id] = float(entry["weight"])
    if not weights:
        return {
            "forecast_status": "NO_ELIGIBLE_FORECASTER",
            "probability": None,
            "weights": {},
            "reason": "no_positive_familywise_corrected_skill_lower_bound",
            "research_only": True,
            **_FORBIDDEN_AUTHORITY,
        }

    normalized_current: dict[str, float] = {}
    for model_id in weights:
        if model_id not in current_probabilities:
            return {
                "forecast_status": "MISSING_CURRENT_MODEL_PROBABILITY",
                "probability": None,
                "weights": {},
                "reason": f"missing_current_probability:{model_id}",
                "research_only": True,
                **_FORBIDDEN_AUTHORITY,
            }
        normalized_current[model_id] = _probability(
            current_probabilities[model_id], f"current_probability:{model_id}"
        )
    total = sum(weights.values())
    weights = {key: value / total for key, value in weights.items()}
    combined = sum(weights[key] * normalized_current[key] for key in weights)
    return {
        "forecast_status": "EXPERIMENTAL_FORECAST_ELIGIBLE_FOR_REPLICATION",
        "probability": combined,
        "weights": weights,
        "reason": "positive_corrected_skill_lower_bound_for_preregistered_regime",
        "research_only": True,
        **_FORBIDDEN_AUTHORITY,
    }
