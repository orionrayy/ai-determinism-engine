"""Integrated Serevex stages 1-3: point-in-time ledger, predictability atlas, delayed expert mixer.

Research-only reference implementation using the Python standard library.
It consumes pre-generated expert probabilities; it does not fit an underlying market model.
All outputs preserve NO_TRADE and can never authorize live signals or orders.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime
from typing import Any, Mapping, Sequence

PROTOCOL_VERSION = "serevex-stage123-integrated-v1"
BASELINE_ID = "__baseline__"
RESEARCH_AUTHORITY = {
    "research_only": True,
    "market_promotion_allowed": False,
    "model_validated": False,
    "deployment_calibrated": False,
    "live_signal_enabled": False,
    "live_order_execution_enabled": False,
    "decision_output": "NO_TRADE",
}
PARTITIONS = ("development", "model_selection", "calibration", "final_holdout")
REQUIRED_STATE_TAGS = ("volatility_bucket", "spread_bucket", "event_risk_bucket")


def _sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(field + "_must_be_finite_probability")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(field + "_must_be_finite_probability") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(field + "_must_be_finite_probability")
    return result


def _utc(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field + "_must_be_timezone_aware_iso8601")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(field + "_must_be_timezone_aware_iso8601") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(field + "_must_be_timezone_aware_iso8601")
    return dt.astimezone(datetime.now().astimezone().tzinfo).astimezone(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z")


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(__import__("datetime").timezone.utc)


def _validate_fact(fact: object, decision_time: str, position: int) -> dict[str, Any]:
    if not isinstance(fact, Mapping):
        raise ValueError("feature_fact_must_be_mapping_at_" + str(position))
    required = {
        "feature_id", "available_at", "observed_at", "source_revision",
        "source_sha256", "transform_id", "transform_sha256", "semantic_type",
        "quote_basis", "quality_status",
    }
    missing = sorted(required - set(fact))
    if missing:
        raise ValueError("feature_fact_missing_fields_at_" + str(position) + ":" + ",".join(missing))
    out = dict(fact)
    for key in ("feature_id", "source_revision", "transform_id", "semantic_type", "quote_basis", "quality_status"):
        if not isinstance(out.get(key), str) or not out[key].strip():
            raise ValueError("feature_fact_invalid_" + key + "_at_" + str(position))
    for key in ("source_sha256", "transform_sha256"):
        if not _digest(out.get(key)):
            raise ValueError("feature_fact_invalid_" + key + "_at_" + str(position))
    out["available_at"] = _utc(out["available_at"], "feature_available_at")
    out["observed_at"] = _utc(out["observed_at"], "feature_observed_at")
    if _time(out["available_at"]) > _time(out["observed_at"]):
        raise ValueError("feature_observed_before_available_at_" + str(position))
    if _time(out["available_at"]) > _time(decision_time):
        raise ValueError("future_feature_leakage_at_" + str(position))
    if _time(out["observed_at"]) > _time(decision_time):
        raise ValueError("feature_observed_after_decision_at_" + str(position))
    for optional_time in ("published_at", "event_time"):
        if out.get(optional_time) is not None:
            out[optional_time] = _utc(out[optional_time], "feature_" + optional_time)
    if out.get("value") is not None:
        value = out["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("feature_value_must_be_finite_numeric_or_none_at_" + str(position))
        out["value"] = float(value)
    return out


def validate_forecast_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a single-series, point-in-time forecast ledger.

    This checks schema and temporal contracts, not the truth of upstream provenance,
    licensing assertions, model fitting chronology, or broker representativeness.
    """
    if not isinstance(ledger, Mapping):
        raise ValueError("ledger_must_be_mapping")
    if ledger.get("schema_version") != "serevex.forecast-ledger.v1":
        raise ValueError("unsupported_ledger_schema_version")
    metadata = ledger.get("metadata")
    records = ledger.get("records")
    if not isinstance(metadata, Mapping) or not isinstance(records, list) or not records:
        raise ValueError("metadata_and_nonempty_records_required")
    required_meta = {
        "series_id", "target_id", "target_definition_sha256", "dataset_sha256",
        "source_lineage_sha256", "model_artifact_sha256_by_expert",
        "registered_experts", "registered_regimes", "evidence_kind",
        "source_verification", "decision_time_basis", "label_definition",
    }
    missing_meta = sorted(required_meta - set(metadata))
    if missing_meta:
        raise ValueError("metadata_missing_fields:" + ",".join(missing_meta))
    for key in ("series_id", "target_id", "decision_time_basis", "label_definition"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError("metadata_invalid_" + key)
    for key in ("target_definition_sha256", "dataset_sha256", "source_lineage_sha256"):
        if not _digest(metadata.get(key)):
            raise ValueError("metadata_invalid_" + key)
    experts = metadata.get("registered_experts")
    regimes = metadata.get("registered_regimes")
    if not isinstance(experts, list) or not experts or any(not isinstance(x, str) or not x.strip() or x == BASELINE_ID for x in experts):
        raise ValueError("registered_experts_must_be_nonempty_and_valid")
    if len(set(experts)) != len(experts):
        raise ValueError("registered_experts_must_be_unique")
    if not isinstance(regimes, list) or not regimes or any(not isinstance(x, str) or not x.strip() for x in regimes):
        raise ValueError("registered_regimes_must_be_nonempty_and_valid")
    if len(set(regimes)) != len(regimes):
        raise ValueError("registered_regimes_must_be_unique")
    artifact_hashes = metadata.get("model_artifact_sha256_by_expert")
    if not isinstance(artifact_hashes, Mapping) or set(artifact_hashes) != set(experts):
        raise ValueError("model_artifact_hash_keys_must_match_expert_registry")
    if any(not _digest(artifact_hashes.get(x)) for x in experts):
        raise ValueError("invalid_model_artifact_sha256")
    if metadata.get("evidence_kind") not in ("real_market_research", "synthetic_contract_test", "replay_fixture"):
        raise ValueError("unrecognized_evidence_kind")
    verification = metadata.get("source_verification")
    verification_fields = ("provider_provenance_independently_verified", "data_rights_verified", "broker_feed_match_verified")
    if not isinstance(verification, Mapping) or any(not isinstance(verification.get(k), bool) for k in verification_fields):
        raise ValueError("source_verification_requires_explicit_boolean_flags")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_index = -1
    previous_partition_rank = -1
    previous_decision_time: datetime | None = None
    registered_regimes = set(regimes)
    allowed_partitions = set(PARTITIONS)
    required_row = {
        "decision_id", "decision_index", "decision_time", "partition",
        "label_available_index", "label_available_time", "label", "label_mature",
        "regime_id", "regime_available_index", "regime_available_time",
        "session_id", "baseline_probability", "expert_probabilities",
        "state_tags", "state_tags_available_index", "feature_facts",
    }
    for pos, source in enumerate(records):
        if not isinstance(source, Mapping):
            raise ValueError("record_must_be_mapping_at_" + str(pos))
        missing = sorted(required_row - set(source))
        if missing:
            raise ValueError("record_missing_fields_at_" + str(pos) + ":" + ",".join(missing))
        row = dict(source)
        did = row.get("decision_id")
        if not isinstance(did, str) or not did.strip() or did in seen_ids:
            raise ValueError("decision_id_missing_or_duplicate_at_" + str(pos))
        seen_ids.add(did)
        di, li = row.get("decision_index"), row.get("label_available_index")
        ri, ti = row.get("regime_available_index"), row.get("state_tags_available_index")
        if not _is_int(di) or di < 0 or di <= previous_index:
            raise ValueError("decision_indices_must_be_strictly_increasing_at_" + str(pos))
        if not _is_int(li) or li <= di:
            raise ValueError("label_must_be_available_only_after_decision_at_" + str(pos))
        if not _is_int(ri) or not 0 <= ri <= di:
            raise ValueError("regime_not_available_as_of_decision_at_" + str(pos))
        if not _is_int(ti) or not 0 <= ti <= di:
            raise ValueError("state_tags_not_available_as_of_decision_at_" + str(pos))
        partition = row.get("partition")
        if partition not in allowed_partitions:
            raise ValueError("unknown_partition_at_" + str(pos))
        rank = PARTITIONS.index(partition)
        if rank < previous_partition_rank:
            raise ValueError("partitions_must_be_chronologically_ordered")
        previous_partition_rank = rank

        decision_time = _utc(row.get("decision_time"), "decision_time")
        available_time = _utc(row.get("label_available_time"), "label_available_time")
        regime_time = _utc(row.get("regime_available_time"), "regime_available_time")
        if _time(available_time) <= _time(decision_time):
            raise ValueError("label_availability_time_must_follow_decision_at_" + str(pos))
        if _time(regime_time) > _time(decision_time):
            raise ValueError("future_regime_metadata_at_" + str(pos))
        if previous_decision_time is not None and _time(decision_time) <= previous_decision_time:
            raise ValueError("decision_times_must_be_strictly_increasing")
        previous_decision_time = _time(decision_time)
        previous_index = di
        row["decision_time"] = decision_time
        row["label_available_time"] = available_time
        row["regime_available_time"] = regime_time

        regime = row.get("regime_id")
        if not isinstance(regime, str) or regime not in registered_regimes:
            raise ValueError("unregistered_regime_at_" + str(pos))
        if not isinstance(row.get("session_id"), str) or not row["session_id"].strip():
            raise ValueError("session_id_invalid_at_" + str(pos))
        if not isinstance(row.get("label_mature"), bool):
            raise ValueError("label_mature_must_be_boolean_at_" + str(pos))
        label = row.get("label")
        if label is not None and (isinstance(label, bool) or label not in (0, 1)):
            raise ValueError("label_must_be_binary_or_none_at_" + str(pos))
        if row["label_mature"] and label is None:
            raise ValueError("mature_label_missing_at_" + str(pos))
        row["baseline_probability"] = _probability(row.get("baseline_probability"), "baseline_probability")
        expert_probabilities = row.get("expert_probabilities")
        if not isinstance(expert_probabilities, Mapping) or set(expert_probabilities) != set(experts):
            raise ValueError("expert_probability_keys_must_match_registry_at_" + str(pos))
        row["expert_probabilities"] = {
            expert: _probability(expert_probabilities[expert], "expert_probability_" + expert)
            for expert in experts
        }
        tags = row.get("state_tags")
        if not isinstance(tags, Mapping) or any(not isinstance(k, str) or not k.strip() for k in tags):
            raise ValueError("state_tags_must_be_string_mapping_at_" + str(pos))
        missing_tags = sorted(set(REQUIRED_STATE_TAGS) - set(tags))
        if missing_tags:
            raise ValueError("required_state_tags_missing_at_" + str(pos) + ":" + ",".join(missing_tags))
        if any(not isinstance(v, str) or not v.strip() for v in tags.values()):
            raise ValueError("state_tag_values_must_be_explicit_strings_use_UNKNOWN_at_" + str(pos))
        row["state_tags"] = dict(tags)
        facts = row.get("feature_facts")
        if not isinstance(facts, list):
            raise ValueError("feature_facts_must_be_list_at_" + str(pos))
        row["feature_facts"] = [
            _validate_fact(fact, decision_time, pos * 100000 + fact_pos)
            for fact_pos, fact in enumerate(facts)
        ]
        rows.append(row)

    canonical_meta = dict(metadata)
    canonical_meta["registered_experts"] = list(experts)
    canonical_meta["registered_regimes"] = list(regimes)
    canonical_meta["model_artifact_sha256_by_expert"] = {x: artifact_hashes[x] for x in experts}
    canonical_meta["source_verification"] = {k: verification[k] for k in verification_fields}
    ledger_sha = _sha({"schema_version": ledger["schema_version"], "metadata": canonical_meta, "records": rows})
    verification_complete = all(verification[k] for k in verification_fields)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "contract_status": "PASS",
        "ledger_sha256": ledger_sha,
        "record_count": len(rows),
        "registered_experts": list(experts),
        "registered_regimes": list(regimes),
        "partition_counts": {p: sum(1 for r in rows if r["partition"] == p) for p in PARTITIONS},
        "point_in_time_feature_fact_count": sum(len(r["feature_facts"]) for r in rows),
        "source_verification": dict(canonical_meta["source_verification"]),
        "all_source_verification_flags_true": verification_complete,
        "source_authenticity_independently_verified": False,
        "data_rights_independently_verified": False,
        "normalized_ledger": {"schema_version": ledger["schema_version"], "metadata": canonical_meta, "records": rows},
    }


def _eligible(rows: Sequence[Mapping[str, Any]], *, as_of_index: int, partition: str) -> list[dict[str, Any]]:
    return [
        dict(r) for r in rows
        if r["partition"] == partition
        and r["decision_index"] < as_of_index
        and r["label_available_index"] < as_of_index
        and r["label_mature"] is True
        and r["label"] in (0, 1)
    ]


def _loss(probability: float, label: int) -> tuple[float, float]:
    eps = 1e-15
    p = min(1.0 - eps, max(eps, probability))
    return (probability - label) ** 2, -(label * math.log(p) + (1 - label) * math.log(1 - p))


def _runs(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    ordered = sorted(rows, key=lambda r: r["decision_index"])
    out: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    prior_index: int | None = None
    prior_session: str | None = None
    prior_regime: str | None = None
    for row in ordered:
        if current and (
            row["decision_index"] != prior_index + 1
            or row["session_id"] != prior_session
            or row["regime_id"] != prior_regime
        ):
            out.append(current)
            current = []
        current.append(row)
        prior_index = row["decision_index"]
        prior_session = row["session_id"]
        prior_regime = row["regime_id"]
    if current:
        out.append(current)
    return out


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile_requires_nonempty_values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _bootstrap_mean_ci(
    selected: Sequence[Mapping[str, Any]],
    *, block_size: int, replicates: int, seed: int, alpha: float,
) -> tuple[float, float, int]:
    runs = _runs(selected)
    starts: list[tuple[Sequence[Mapping[str, Any]], int, int]] = []
    for run in runs:
        size = min(block_size, len(run))
        if size > 0:
            for start in range(max(1, len(run) - size + 1)):
                starts.append((run, start, size))
    if not starts:
        return 0.0, 0.0, 0
    rng = random.Random(seed)
    n = len(selected)
    samples: list[float] = []
    for _ in range(replicates):
        differences: list[float] = []
        while len(differences) < n:
            run, start, size = starts[rng.randrange(len(starts))]
            block = run[start:start + size]
            for row in block:
                y = int(row["label"])
                base_loss = _loss(float(row["baseline_probability"]), y)[0]
                model_loss = _loss(float(row["_probability"]), y)[0]
                differences.append(base_loss - model_loss)
                if len(differences) >= n:
                    break
        samples.append(sum(differences) / len(differences))
    return _quantile(samples, alpha / 2), _quantile(samples, 1 - alpha / 2), len(starts)


def build_predictability_atlas(
    records: Sequence[Mapping[str, Any]],
    *,
    registered_experts: Sequence[str],
    as_of_index: int,
    score_partition: str = "final_holdout",
    axes: Sequence[str] = ("regime_id", "session_id", "volatility_bucket", "spread_bucket", "event_risk_bucket"),
    min_rows: int = 80,
    block_size: int = 12,
    min_independent_blocks: int = 8,
    bootstrap_replicates: int = 1000,
    familywise_alpha: float = 0.05,
    minimum_skill: float = 0.0,
    seed: int = 20261019,
) -> dict[str, Any]:
    """Map paired Brier skill across preregistered slices; eligibility is replication-only."""
    if not _is_int(as_of_index) or as_of_index < 0:
        raise ValueError("as_of_index_must_be_nonnegative_integer")
    if score_partition not in PARTITIONS:
        raise ValueError("score_partition_invalid")
    if not axes or any(not isinstance(x, str) or not x.strip() for x in axes) or len(set(axes)) != len(axes):
        raise ValueError("atlas_axes_must_be_unique_nonempty_strings")
    if not _is_int(min_rows) or min_rows < 20 or not _is_int(block_size) or block_size < 2:
        raise ValueError("atlas_support_parameters_invalid")
    if not _is_int(min_independent_blocks) or min_independent_blocks < 2:
        raise ValueError("min_independent_blocks_must_be_at_least_2")
    if not _is_int(bootstrap_replicates) or not 200 <= bootstrap_replicates <= 10000:
        raise ValueError("bootstrap_replicates_must_be_200_to_10000")
    if not isinstance(familywise_alpha, (int, float)) or isinstance(familywise_alpha, bool) or not 0 < float(familywise_alpha) < 0.2:
        raise ValueError("familywise_alpha_invalid")
    if not isinstance(minimum_skill, (int, float)) or isinstance(minimum_skill, bool) or not math.isfinite(float(minimum_skill)):
        raise ValueError("minimum_skill_invalid")
    mature = _eligible(records, as_of_index=as_of_index, partition=score_partition)
    experts = list(registered_experts)
    raw: list[tuple[str, str, str, list[dict[str, Any]]]] = []
    for axis in axes:
        def axis_value(row: Mapping[str, Any]) -> str | None:
            if axis in ("regime_id", "session_id"):
                return str(row[axis])
            return row["state_tags"].get(axis)
        categories = sorted({axis_value(row) for row in mature if axis_value(row) is not None})
        for category in categories:
            group = [r for r in mature if axis_value(r) == category]
            for expert in experts:
                prepared = []
                for row in group:
                    item = dict(row)
                    item["_probability"] = float(row["expert_probabilities"][expert])
                    prepared.append(item)
                raw.append((axis, category, expert, prepared))
    family_size = max(1, len(raw))
    corrected_alpha = float(familywise_alpha) / family_size
    reports: list[dict[str, Any]] = []
    for index, (axis, category, expert, chosen) in enumerate(raw):
        runs = _runs(chosen)
        independent_blocks = sum(len(run) // block_size for run in runs)
        cohort_digest = _sha([{"id": r["decision_id"], "index": r["decision_index"], "label": r["label"]} for r in chosen])
        n = len(chosen)
        if n:
            baseline_brier = sum(_loss(float(r["baseline_probability"]), int(r["label"]))[0] for r in chosen) / n
            expert_brier = sum(_loss(float(r["expert_probabilities"][expert]), int(r["label"]))[0] for r in chosen) / n
            skill = baseline_brier - expert_brier
            logloss_improvement = sum(
                _loss(float(r["baseline_probability"]), int(r["label"]))[1]
                - _loss(float(r["expert_probabilities"][expert]), int(r["label"]))[1]
                for r in chosen
            ) / n
        else:
            baseline_brier = expert_brier = skill = logloss_improvement = None
        status = "INSUFFICIENT_SUPPORT"
        lower = upper = None
        blocks_available = 0
        if n >= min_rows and independent_blocks >= min_independent_blocks:
            lower, upper, starts = _bootstrap_mean_ci(
                chosen, block_size=block_size, replicates=bootstrap_replicates,
                seed=seed + index * 104729, alpha=corrected_alpha,
            )
            blocks_available = starts
            if starts and skill is not None and skill > float(minimum_skill) and lower > float(minimum_skill):
                status = "ELIGIBLE_FOR_INDEPENDENT_REPLICATION"
            else:
                status = "NO_CONFIRMED_INCREMENTAL_SKILL"
        reports.append({
            "axis": axis, "category": category, "expert_id": expert,
            "status": status, "scored_rows": n,
            "independent_full_blocks": independent_blocks,
            "bootstrap_block_starts": blocks_available,
            "cohort_sha256": cohort_digest,
            "baseline_brier": baseline_brier, "expert_brier": expert_brier,
            "mean_brier_skill_baseline_minus_expert": skill,
            "mean_log_loss_improvement_baseline_minus_expert": logloss_improvement,
            "simultaneous_family_size": family_size,
            "nominal_familywise_alpha": float(familywise_alpha),
            "bonferroni_alpha_per_registered_cell": corrected_alpha,
            "simultaneous_approx_interval": [lower, upper] if lower is not None else None,
            "independent_replication_only": True,
        })
    return {
        "protocol_version": "serevex-predictability-atlas-v1",
        "status": "DIAGNOSTIC_ONLY",
        "score_partition": score_partition,
        "as_of_index_exclusive": as_of_index,
        "matured_scored_rows": len(mature),
        "axes": list(axes), "registered_experts": experts,
        "hypothesis_family_size": family_size,
        "results": reports,
        "selection_warning": "Do not choose thresholds or experts on final_holdout; eligibility is replication-only.",
        **RESEARCH_AUTHORITY,
    }


def audit_expert_complementarity(
    records: Sequence[Mapping[str, Any]],
    *, registered_experts: Sequence[str], as_of_index: int,
    score_partition: str = "final_holdout", min_rows: int = 80,
) -> dict[str, Any]:
    """Measure same-cohort residual diversity and simple-blend gain; diagnostic, not a selector."""
    rows = _eligible(records, as_of_index=as_of_index, partition=score_partition)
    experts = list(registered_experts)
    residuals = {
        expert: [float(r["expert_probabilities"][expert]) - int(r["label"]) for r in rows]
        for expert in experts
    }
    results = []
    for i, left in enumerate(experts):
        for right in experts[i + 1:]:
            n = len(rows)
            if n:
                a, b = residuals[left], residuals[right]
                ma, mb = sum(a) / n, sum(b) / n
                va = sum((x - ma) ** 2 for x in a)
                vb = sum((x - mb) ** 2 for x in b)
                corr = None if va <= 1e-15 or vb <= 1e-15 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)
                brier_left = sum(_loss(float(r["expert_probabilities"][left]), int(r["label"]))[0] for r in rows) / n
                brier_right = sum(_loss(float(r["expert_probabilities"][right]), int(r["label"]))[0] for r in rows) / n
                brier_blend = sum(
                    _loss((float(r["expert_probabilities"][left]) + float(r["expert_probabilities"][right])) / 2.0, int(r["label"]))[0]
                    for r in rows
                ) / n
                improvement = min(brier_left, brier_right) - brier_blend
            else:
                corr = brier_left = brier_right = brier_blend = improvement = None
            results.append({
                "expert_a": left, "expert_b": right, "scored_rows": n,
                "residual_correlation": corr,
                "brier_a": brier_left, "brier_b": brier_right,
                "equal_weight_blend_brier": brier_blend,
                "blend_improvement_vs_best_single": improvement,
                "status": "DIAGNOSTIC_ONLY" if n >= min_rows else "INSUFFICIENT_SUPPORT",
                "may_select_using_this_holdout": False,
            })
    return {
        "protocol_version": "serevex-residual-complementarity-audit-v1",
        "status": "DIAGNOSTIC_ONLY",
        "score_partition": score_partition, "matured_scored_rows": len(rows),
        "results": results,
        "interpretation": "Low residual correlation alone is not proof of useful diversity; same-cohort blend improvement is descriptive and cannot be used to choose experts on the final holdout.",
        **RESEARCH_AUTHORITY,
    }


def _update_weights(
    weights: Mapping[str, float], rows: Sequence[Mapping[str, Any]],
    option_ids: Sequence[str], eta: float, fixed_share_rate: float,
) -> dict[str, float]:
    if not rows:
        return dict(weights)
    deltas: dict[str, float] = {}
    for option in option_ids:
        differences = []
        for row in rows:
            y = int(row["label"])
            baseline_loss = _loss(float(row["baseline_probability"]), y)[0]
            p = float(row["baseline_probability"]) if option == BASELINE_ID else float(row["expert_probabilities"][option])
            differences.append(_loss(p, y)[0] - baseline_loss)
        deltas[option] = sum(differences) / len(differences)
    logs = {x: math.log(max(float(weights[x]), 1e-300)) - eta * deltas[x] for x in option_ids}
    peak = max(logs.values())
    expw = {x: math.exp(max(-700.0, logs[x] - peak)) for x in option_ids}
    total = sum(expw.values())
    updated = {x: expw[x] / total for x in option_ids}
    if fixed_share_rate:
        k = len(option_ids)
        updated = {x: (1.0 - fixed_share_rate) * updated[x] + fixed_share_rate / k for x in option_ids}
    normalizer = sum(updated.values())
    return {x: updated[x] / normalizer for x in option_ids}


def run_delayed_feedback_mixer(
    records: Sequence[Mapping[str, Any]],
    *, registered_experts: Sequence[str], as_of_index: int,
    min_global_matured: int = 30, min_regime_matured: int = 40,
    eta: float = 2.0, fixed_share_rate: float = 0.02,
    score_partition: str = "final_holdout",
) -> dict[str, Any]:
    """Run a causal online expert mixer. Updates only with label_available_index < next decision."""
    if not _is_int(as_of_index) or as_of_index < 0:
        raise ValueError("as_of_index_must_be_nonnegative_integer")
    if not _is_int(min_global_matured) or min_global_matured < 1 or not _is_int(min_regime_matured) or min_regime_matured < 1:
        raise ValueError("matured_support_thresholds_must_be_positive_integers")
    if not isinstance(eta, (int, float)) or isinstance(eta, bool) or not math.isfinite(float(eta)) or eta < 0:
        raise ValueError("eta_must_be_finite_nonnegative")
    if not isinstance(fixed_share_rate, (int, float)) or isinstance(fixed_share_rate, bool) or not math.isfinite(float(fixed_share_rate)) or not 0 <= fixed_share_rate < 1:
        raise ValueError("fixed_share_rate_must_be_in_0_inclusive_1_exclusive")
    if score_partition not in PARTITIONS:
        raise ValueError("score_partition_invalid")
    rows = list(records)
    if not rows:
        raise ValueError("records_must_be_nonempty")
    experts = list(registered_experts)
    if not experts or len(set(experts)) != len(experts) or any(not isinstance(x, str) or not x.strip() or x == BASELINE_ID for x in experts):
        raise ValueError("registered_experts_invalid")
    option_ids = experts + [BASELINE_ID]
    regimes = sorted({str(r["regime_id"]) for r in rows})
    global_weights = {x: 1.0 / len(option_ids) for x in option_ids}
    regime_weights = {regime: dict(global_weights) for regime in regimes}
    global_matured = 0
    regime_matured = {regime: 0 for regime in regimes}
    processed: set[str] = set()
    history: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []

    for current in rows:
        di = int(current["decision_index"])
        newly_matured = [
            old for old in history
            if old["decision_id"] not in processed
            and old["label_mature"] is True and old["label"] in (0, 1)
            and old["label_available_index"] < di
        ]
        by_maturity: dict[int, list[dict[str, Any]]] = {}
        for old in newly_matured:
            by_maturity.setdefault(int(old["label_available_index"]), []).append(old)
        for maturity_index in sorted(by_maturity):
            batch = sorted(by_maturity[maturity_index], key=lambda r: r["decision_id"])
            before_hash = _sha(global_weights)
            global_weights = _update_weights(global_weights, batch, option_ids, float(eta), float(fixed_share_rate))
            global_matured += len(batch)
            for regime in regimes:
                regime_batch = [r for r in batch if r["regime_id"] == regime]
                if regime_batch:
                    regime_weights[regime] = _update_weights(
                        regime_weights[regime], regime_batch, option_ids, float(eta), float(fixed_share_rate)
                    )
                    regime_matured[regime] += len(regime_batch)
            for old in batch:
                processed.add(old["decision_id"])
            updates.append({
                "label_available_index": maturity_index,
                "applied_before_decision_index": di,
                "batch_rows": len(batch),
                "global_matured_support_after_update": global_matured,
                "global_weights_sha256_before": before_hash,
                "global_weights_sha256_after": _sha(global_weights),
            })

        regime = str(current["regime_id"])
        if global_matured < min_global_matured:
            weights_used = {BASELINE_ID: 1.0}
            forecast_source = "BASELINE_FALLBACK_INSUFFICIENT_MATURED_HISTORY"
        elif regime_matured.get(regime, 0) >= min_regime_matured:
            weights_used = dict(regime_weights[regime])
            forecast_source = "REGIME_FIXED_SHARE"
        else:
            weights_used = dict(global_weights)
            forecast_source = "GLOBAL_FIXED_SHARE"
        probability_map = {x: float(current["expert_probabilities"][x]) for x in experts}
        probability_map[BASELINE_ID] = float(current["baseline_probability"])
        probability = sum(weights_used.get(x, 0.0) * probability_map[x] for x in weights_used)
        output = {
            "decision_id": current["decision_id"],
            "decision_index": di,
            "partition": current["partition"],
            "regime_id": regime,
            "label_available_index": current["label_available_index"],
            "label_mature": current["label_mature"],
            "label": current["label"],
            "probability": probability,
            "baseline_probability": current["baseline_probability"],
            "weights_used": weights_used,
            "forecast_source": forecast_source,
            "matured_labels_seen_globally_before_decision": global_matured,
            "matured_labels_seen_in_regime_before_decision": regime_matured.get(regime, 0),
        }
        forecast_rows.append(output)
        history.append(dict(current))

    scored_predictions = [
        r for r in forecast_rows
        if r["partition"] == score_partition
        and r["decision_index"] < as_of_index
        and r["label_available_index"] < as_of_index
        and r["label_mature"] is True and r["label"] in (0, 1)
    ]
    n = len(scored_predictions)
    if n:
        candidate_brier = sum(_loss(float(r["probability"]), int(r["label"]))[0] for r in scored_predictions) / n
        baseline_brier = sum(_loss(float(r["baseline_probability"]), int(r["label"]))[0] for r in scored_predictions) / n
        candidate_log = sum(_loss(float(r["probability"]), int(r["label"]))[1] for r in scored_predictions) / n
        baseline_log = sum(_loss(float(r["baseline_probability"]), int(r["label"]))[1] for r in scored_predictions) / n
    else:
        candidate_brier = baseline_brier = candidate_log = baseline_log = None
    return {
        "protocol_version": "serevex-delayed-feedback-mixer-v2-integrated",
        "status": "DIAGNOSTIC_ONLY",
        "as_of_index_exclusive": as_of_index,
        "score_partition": score_partition,
        "scored_rows": n,
        "candidate_brier": candidate_brier,
        "baseline_brier": baseline_brier,
        "brier_skill_baseline_minus_candidate": None if n == 0 else baseline_brier - candidate_brier,
        "candidate_log_loss": candidate_log,
        "baseline_log_loss": baseline_log,
        "log_loss_improvement_baseline_minus_candidate": None if n == 0 else baseline_log - candidate_log,
        "global_matured_labels_processed": global_matured,
        "regime_matured_labels_processed": regime_matured,
        "update_event_count": len(updates),
        "update_events": updates,
        "forecast_ledger": forecast_rows,
        "forecast_ledger_sha256": _sha(forecast_rows),
        "final_global_weights": global_weights,
        "final_global_weights_sha256": _sha(global_weights),
        "updates_only_after_strict_label_maturity": True,
        "mixer_is_model_training": False,
        "mixer_is_market_validated": False,
        **RESEARCH_AUTHORITY,
    }


def run_serevex_stage123(
    ledger: Mapping[str, Any], *,
    as_of_index: int,
    score_partition: str = "final_holdout",
    atlas_axes: Sequence[str] = ("regime_id", "session_id", "volatility_bucket", "spread_bucket", "event_risk_bucket"),
    atlas_min_rows: int = 80,
    atlas_block_size: int = 12,
    atlas_min_independent_blocks: int = 8,
    atlas_bootstrap_replicates: int = 1000,
    min_global_matured: int = 30,
    min_regime_matured: int = 40,
    eta: float = 2.0,
    fixed_share_rate: float = 0.02,
) -> dict[str, Any]:
    """Execute ledger validation, atlas/complementarity and delayed feedback as one contract."""
    validated = validate_forecast_ledger(ledger)
    normalized = validated["normalized_ledger"]
    metadata = normalized["metadata"]
    records = normalized["records"]
    experts = metadata["registered_experts"]
    atlas = build_predictability_atlas(
        records, registered_experts=experts, as_of_index=as_of_index,
        score_partition=score_partition, axes=atlas_axes, min_rows=atlas_min_rows,
        block_size=atlas_block_size, min_independent_blocks=atlas_min_independent_blocks,
        bootstrap_replicates=atlas_bootstrap_replicates,
    )
    complementarity = audit_expert_complementarity(
        records, registered_experts=experts, as_of_index=as_of_index,
        score_partition=score_partition, min_rows=atlas_min_rows,
    )
    mixer = run_delayed_feedback_mixer(
        records, registered_experts=experts, as_of_index=as_of_index,
        min_global_matured=min_global_matured, min_regime_matured=min_regime_matured,
        eta=eta, fixed_share_rate=fixed_share_rate, score_partition=score_partition,
    )
    report_core = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS_CONTRACT_DIAGNOSTIC_ONLY",
        "ledger": {k: v for k, v in validated.items() if k != "normalized_ledger"},
        "stage_1_ledger_contract": "PASS",
        "stage_2_predictability_atlas": atlas,
        "stage_2_expert_complementarity": complementarity,
        "stage_3_delayed_feedback_mixer": mixer,
        "research_boundary": {
            "predictive_skill_proven": False,
            "mathematical_originality_proven": False,
            "source_authenticity_proven": False,
            "net_profitability_proven": False,
            "deployment_ready": False,
            "decision_output": "NO_TRADE",
        },
        **RESEARCH_AUTHORITY,
    }
    report_core["report_sha256"] = _sha(report_core)
    return report_core
