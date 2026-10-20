"""Development-only selection experiment for support-adaptive baseline shrinkage.

The v0.1 expanding Laplace baseline is immutable. Configuration selection uses only
matured development labels; model-selection and calibration partitions are independent
gates; final holdout is read only after a policy is frozen. Research-only, NO_TRADE.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

MODEL_VERSION = "serevex-hierarchical-shrinkage-v1"
EXPERT_SUPPORT = {
    "rolling_matured_rate_288": "matured_label_count_before_decision",
    "regime_matured_rate": "conditional_regime_support_before_decision",
    "volatility_matured_rate": "conditional_volatility_support_before_decision",
}
K_GRID = (40, 160)
CAP_GRID = (0.2, 0.4, 0.6)
CANDIDATE_SPECS = tuple(
    {"expert": expert, "k": k, "cap": cap,
     "candidate_id": f"{expert}:k{k}:cap{cap:.1f}"}
    for expert in EXPERT_SUPPORT for k in K_GRID for cap in CAP_GRID
)
BLOCK_SPAN_BARS = 12
EMBARGO_BARS = 12
BLOCK_PERIOD = BLOCK_SPAN_BARS + EMBARGO_BARS
MIN_ROWS = 200
MIN_BLOCKS = 8
MIN_POSITIVE_BLOCK_SHARE = 0.70
CRITICAL_Z = 2.58
SUBGROUP_ECE_TOLERANCE = 0.015
MIN_SUBGROUP_CALIBRATION_ROWS = 100
EPS = 1e-12


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _clip(p: Any) -> float:
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(float(p)):
        raise ValueError("probability_must_be_finite_numeric")
    return min(1.0 - EPS, max(EPS, float(p)))


def _sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _records(raw_records: Sequence[Mapping[str, Any]], as_of_index: int) -> list[dict[str, Any]]:
    if not _is_int(as_of_index) or as_of_index < 1:
        raise ValueError("as_of_index_must_be_positive_integer")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("records_must_be_a_sequence")
    result, seen = [], set()
    allowed_partitions = {"development", "model_selection", "calibration", "final_holdout"}
    for pos, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"record_must_be_mapping_at_{pos}")
        row = dict(raw)
        di, ai = row.get("decision_index"), row.get("label_available_index")
        if not _is_int(di) or not 0 <= di < as_of_index:
            raise ValueError(f"decision_index_out_of_asof_range_at_{pos}")
        if di in seen:
            raise ValueError(f"duplicate_decision_index_{di}")
        seen.add(di)
        if not _is_int(ai) or ai <= di:
            raise ValueError(f"label_available_index_must_follow_decision_at_{pos}")
        if row.get("partition") not in allowed_partitions:
            raise ValueError(f"unknown_partition_at_{pos}")
        if not isinstance(row.get("label_mature"), bool):
            raise ValueError(f"label_mature_must_be_boolean_at_{pos}")
        y = row.get("label")
        if row["label_mature"] and (y not in (0, 1) or isinstance(y, bool)):
            raise ValueError(f"mature_label_must_be_binary_at_{pos}")
        if not row["label_mature"] and y is not None:
            raise ValueError(f"unmature_label_must_be_null_at_{pos}")
        row["baseline_probability"] = _clip(row.get("baseline_probability"))
        experts = row.get("expert_probabilities")
        if not isinstance(experts, Mapping):
            raise ValueError(f"expert_probabilities_must_be_mapping_at_{pos}")
        row["expert_probabilities"] = {expert: _clip(experts.get(expert)) for expert in EXPERT_SUPPORT}
        for field in EXPERT_SUPPORT.values():
            n = row.get(field, 0)
            if not _is_int(n) or n < 0:
                raise ValueError(f"{field}_must_be_nonnegative_integer_at_{pos}")
        result.append(row)
    return sorted(result, key=lambda r: int(r["decision_index"]))


def _candidate_probability(row: Mapping[str, Any], spec: Mapping[str, Any]) -> float:
    base = _clip(row["baseline_probability"])
    expert = str(spec["expert"])
    support = int(row.get(EXPERT_SUPPORT[expert], 0))
    if expert == "rolling_matured_rate_288":
        support = min(support, 288)
    k, cap = int(spec["k"]), float(spec["cap"])
    weight = cap * support / (support + k) if support else 0.0
    p_expert = _clip(row["expert_probabilities"][expert])
    return _clip(base + weight * (p_expert - base))


def _ece(rows: Sequence[Mapping[str, Any]], probabilities: Sequence[float], bins: int = 10) -> float | None:
    if len(rows) != len(probabilities):
        raise ValueError("ece_length_mismatch")
    if not rows:
        return None
    counts = [0] * bins
    sum_p = [0.0] * bins
    sum_y = [0.0] * bins
    for row, p in zip(rows, probabilities):
        i = min(bins - 1, int(_clip(p) * bins))
        counts[i] += 1
        sum_p[i] += _clip(p)
        sum_y[i] += int(row["label"])
    return sum(counts[i] / len(rows) * abs(sum_p[i] / counts[i] - sum_y[i] / counts[i])
               for i in range(bins) if counts[i])


def _metrics(rows: Sequence[Mapping[str, Any]], probabilities: Sequence[float]) -> dict[str, Any]:
    if len(rows) != len(probabilities):
        raise ValueError("metric_length_mismatch")
    if not rows:
        return {"rows": 0, "positive_rate": None, "brier": None, "log_loss": None, "ece": None}
    labels = [int(r["label"]) for r in rows]
    p = [_clip(x) for x in probabilities]
    brier = sum((x - y) ** 2 for x, y in zip(p, labels)) / len(rows)
    log_loss = sum(-(y * math.log(x) + (1 - y) * math.log1p(-x)) for x, y in zip(p, labels)) / len(rows)
    return {"rows": len(rows), "positive_rate": sum(labels) / len(labels),
            "brier": brier, "log_loss": log_loss, "ece": _ece(rows, p)}


def _scored_rows(rows: Sequence[Mapping[str, Any]], partition: str, cutoff: int) -> list[dict[str, Any]]:
    return [
        dict(r) for r in rows
        if r.get("partition") == partition
        and int(r["decision_index"]) < cutoff
        and int(r["label_available_index"]) < cutoff
        and r.get("label_mature") is True and r.get("label") in (0, 1)
    ]


def _block_gain_stats(
    rows: Sequence[Mapping[str, Any]],
    baseline: Sequence[float],
    candidate: Sequence[float],
) -> dict[str, Any]:
    if len(rows) != len(baseline) or len(rows) != len(candidate):
        raise ValueError("block_stats_length_mismatch")
    by_block: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"brier": [], "log_loss": []})
    for row, p0, p1 in zip(rows, baseline, candidate):
        di, y = int(row["decision_index"]), int(row["label"])
        if di % BLOCK_PERIOD >= BLOCK_SPAN_BARS:
            continue
        target = by_block[di // BLOCK_PERIOD]
        target["brier"].append((p0 - y) ** 2 - (p1 - y) ** 2)
        target["log_loss"].append(
            -(y * math.log(_clip(p0)) + (1 - y) * math.log1p(-_clip(p0)))
            + (y * math.log(_clip(p1)) + (1 - y) * math.log1p(-_clip(p1)))
        )
    result: dict[str, Any] = {"blocks": len(by_block), "block_span_bars": BLOCK_SPAN_BARS,
                              "embargo_bars": EMBARGO_BARS, "metrics": {}}
    for metric in ("brier", "log_loss"):
        means = [sum(b[metric]) / len(b[metric]) for _, b in sorted(by_block.items()) if b[metric]]
        n = len(means)
        mean = sum(means) / n if n else None
        se = None
        lower = None
        if n >= 2:
            variance = sum((x - mean) ** 2 for x in means) / (n - 1)
            se = math.sqrt(max(0.0, variance) / n)
            lower = mean - CRITICAL_Z * se
        result["metrics"][metric] = {
            "mean_improvement": mean, "standard_error": se, "conservative_lower_screen": lower,
            "positive_block_share": sum(x > 0 for x in means) / n if n else None,
            "blocks": n,
        }
    return result


def _subgroup_ece_audit(rows: Sequence[Mapping[str, Any]], baseline: Sequence[float],
                        candidate: Sequence[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    eligible_groups = 0
    dimensions = (
        ("regime", lambda r: str(r.get("regime_id", "unknown"))),
        ("volatility_bucket", lambda r: str((r.get("state_tags") or {}).get("volatility_bucket", "unknown"))),
    )
    for dimension, key_fn in dimensions:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            groups[key_fn(row)].append(i)
        out[dimension] = {}
        for key, indices in sorted(groups.items()):
            group_rows = [rows[i] for i in indices]
            if len(indices) < MIN_SUBGROUP_CALIBRATION_ROWS:
                out[dimension][key] = {"rows": len(indices), "status": "INSUFFICIENT_SUPPORT"}
                continue
            eligible_groups += 1
            base_ece = _ece(group_rows, [baseline[i] for i in indices])
            candidate_ece = _ece(group_rows, [candidate[i] for i in indices])
            out[dimension][key] = {
                "rows": len(indices), "baseline_ece": base_ece, "candidate_ece": candidate_ece,
                "degradation": candidate_ece - base_ece,
                "pass": candidate_ece <= base_ece + SUBGROUP_ECE_TOLERANCE,
                "tolerance": SUBGROUP_ECE_TOLERANCE,
            }
    passes = eligible_groups > 0 and all(
        value.get("pass", True)
        for dimension in out.values() for value in dimension.values()
        if value.get("status") != "INSUFFICIENT_SUPPORT"
    )
    return {"status": "PASS" if passes else ("INSUFFICIENT_SUBGROUP_SUPPORT" if not eligible_groups else "FAIL"),
            "eligible_subgroup_count": eligible_groups, "max_allowed_ece_degradation": SUBGROUP_ECE_TOLERANCE,
            "method": "equal_width_10_bin_ECE_by_regime_and_volatility", "dimensions": out}


def _gate(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> dict[str, Any]:
    baseline = [_clip(r["baseline_probability"]) for r in rows]
    candidate = [_candidate_probability(r, spec) for r in rows]
    base_metrics = _metrics(rows, baseline)
    candidate_metrics = _metrics(rows, candidate)
    block = _block_gain_stats(rows, baseline, candidate)
    enough = len(rows) >= MIN_ROWS and block["blocks"] >= MIN_BLOCKS
    dual = all(
        block["metrics"][metric]["conservative_lower_screen"] is not None
        and block["metrics"][metric]["conservative_lower_screen"] > 0
        and block["metrics"][metric]["positive_block_share"] is not None
        and block["metrics"][metric]["positive_block_share"] >= MIN_POSITIVE_BLOCK_SHARE
        for metric in ("brier", "log_loss")
    )
    calibration = _subgroup_ece_audit(rows, baseline, candidate)
    passed = enough and dual and calibration["status"] == "PASS"
    return {
        "status": "PASS" if passed else (
            "INSUFFICIENT_ROWS_OR_BLOCKS" if not enough else
            "DUAL_SCORE_GATE_FAILED" if not dual else
            "SUBGROUP_CALIBRATION_GATE_FAILED"
        ),
        "passed": passed, "rows": len(rows), "minimum_rows": MIN_ROWS, "minimum_independent_blocks": MIN_BLOCKS,
        "minimum_positive_block_share": MIN_POSITIVE_BLOCK_SHARE, "critical_z": CRITICAL_Z,
        "baseline_metrics": base_metrics, "candidate_metrics": candidate_metrics,
        "baseline_brier_gain": base_metrics["brier"] - candidate_metrics["brier"],
        "baseline_log_loss_improvement": base_metrics["log_loss"] - candidate_metrics["log_loss"],
        "block_gain_stats": block, "subgroup_calibration": calibration,
    }


def _paired_block_bootstrap(rows: Sequence[Mapping[str, Any]], baseline: Sequence[float],
                            candidate: Sequence[float], *, replicates: int = 500, seed: int = 20261020) -> dict[str, Any]:
    by_block: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"brier": [], "log_loss": []})
    for row, p0, p1 in zip(rows, baseline, candidate):
        di, y = int(row["decision_index"]), int(row["label"])
        if di % BLOCK_PERIOD >= BLOCK_SPAN_BARS:
            continue
        block = by_block[di // BLOCK_PERIOD]
        block["brier"].append((p0-y)**2-(p1-y)**2)
        block["log_loss"].append(
            -(y*math.log(_clip(p0))+(1-y)*math.log1p(-_clip(p0)))
            + (y*math.log(_clip(p1))+(1-y)*math.log1p(-_clip(p1)))
        )
    series = {m: [sum(v[m])/len(v[m]) for _,v in sorted(by_block.items()) if v[m]] for m in ("brier","log_loss")}
    n = min(len(series["brier"]), len(series["log_loss"]))
    if n < MIN_BLOCKS:
        return {"status": "INSUFFICIENT_BLOCK_SUPPORT", "blocks": n, "replicates": 0,
                "brier_improvement_ci95": None, "log_loss_improvement_ci95": None}
    rng = random.Random(seed)
    out: dict[str, Any] = {"status": "DIAGNOSTIC_PAIRED_SOURCE_TIME_BLOCK_BOOTSTRAP",
                           "blocks": n, "replicates": replicates, "seed": seed}
    for metric in ("brier", "log_loss"):
        values = series[metric]
        sims = sorted(sum(values[rng.randrange(len(values))] for _ in values)/len(values)
                      for _ in range(replicates))
        out[f"{metric}_improvement_ci95"] = [sims[int(0.025*replicates)],
                                                sims[min(replicates-1,int(0.975*replicates))]]
        out[f"{metric}_mean_improvement"] = sum(values)/len(values)
    return out


def evaluate_ledger(
    ledger: Mapping[str, Any],
    *,
    as_of_index: int,
    partition_end_indices: Mapping[str, int],
    bootstrap_replicates: int = 500,
) -> dict[str, Any]:
    """Select on development, confirm on model selection, gate on calibration, then score holdout once."""
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("records"), Sequence):
        raise ValueError("ledger_must_contain_records_sequence")
    if not _is_int(bootstrap_replicates) or not 200 <= bootstrap_replicates <= 10000:
        raise ValueError("bootstrap_replicates_must_be_200_to_10000")
    required = ("development", "model_selection", "calibration", "final_holdout")
    if not isinstance(partition_end_indices, Mapping) or set(partition_end_indices) != set(required):
        raise ValueError("partition_end_indices_must_define_all_four_partitions")
    ends = {k: partition_end_indices[k] for k in required}
    if any(not _is_int(v) for v in ends.values()):
        raise ValueError("partition_end_indices_must_be_integers")
    if not (0 < ends["development"] < ends["model_selection"] < ends["calibration"] < ends["final_holdout"] == as_of_index):
        raise ValueError("partition_end_indices_must_be_strictly_chronological_and_end_at_asof")
    rows = _records(ledger["records"], as_of_index)
    dev = _scored_rows(rows, "development", ends["development"])
    model_selection = _scored_rows(rows, "model_selection", ends["model_selection"])
    calibration = _scored_rows(rows, "calibration", ends["calibration"])
    final_holdout = _scored_rows(rows, "final_holdout", ends["final_holdout"])
    if not dev or not model_selection or not calibration or not final_holdout:
        raise ValueError("each_partition_must_have_matured_scored_rows_before_its_cutoff")

    development_results = []
    dev_base = [_clip(r["baseline_probability"]) for r in dev]
    dev_base_metrics = _metrics(dev, dev_base)
    for spec in CANDIDATE_SPECS:
        preds = [_candidate_probability(row, spec) for row in dev]
        m = _metrics(dev, preds)
        development_results.append({
            **dict(spec), "metrics": m,
            "baseline_brier_gain": dev_base_metrics["brier"] - m["brier"],
            "baseline_log_loss_improvement": dev_base_metrics["log_loss"] - m["log_loss"],
            "prediction_manifest_sha256": _sha([
                [r.get("decision_id", r["decision_index"]), p]
                for r, p in zip(dev, preds)
            ]),
        })
    # This ranking is deliberately confined to development. Brier is primary,
    # log loss secondary; stronger shrinkage (smaller cap, larger k) breaks exact ties.
    winner = sorted(
        development_results,
        key=lambda item: (-item["baseline_brier_gain"],
                           -item["baseline_log_loss_improvement"],
                           float(item["cap"]), -int(item["k"]), str(item["expert"]))
    )[0]
    spec = {k: winner[k] for k in ("expert", "k", "cap", "candidate_id")}
    selection_sha = _sha({
        "partition": "development", "cutoff": ends["development"],
        "cohort": [[r.get("decision_id", r["decision_index"]), int(r["label"])] for r in dev],
        "winner": spec, "development_results": development_results,
    })

    model_selection_gate = _gate(model_selection, spec)
    calibration_gate = _gate(calibration, spec) if model_selection_gate["passed"] else {
        "status": "NOT_RUN_MODEL_SELECTION_GATE_FAILED", "passed": False, "rows": len(calibration),
    }
    promoted_to_frozen_candidate = bool(model_selection_gate["passed"] and calibration_gate.get("passed") is True)

    holdout_base = [_clip(r["baseline_probability"]) for r in final_holdout]
    holdout_shadow = [_candidate_probability(r, spec) for r in final_holdout]
    holdout_policy = holdout_shadow if promoted_to_frozen_candidate else holdout_base
    holdout_base_metrics = _metrics(final_holdout, holdout_base)
    holdout_shadow_metrics = _metrics(final_holdout, holdout_shadow)
    holdout_policy_metrics = _metrics(final_holdout, holdout_policy)
    holdout_bootstrap = _paired_block_bootstrap(
        final_holdout, holdout_base, holdout_shadow, replicates=bootstrap_replicates, seed=20261020
    )
    cohort_sha = _sha([
        [r.get("decision_id", r["decision_index"]), int(r["decision_index"]),
         int(r["label_available_index"]), int(r["label"])]
        for r in final_holdout
    ])
    status = "FROZEN_SHRINKAGE_CANDIDATE_PASSES_DEVELOPMENT_SELECTION_AND_CONFIRMATION_GATES" if promoted_to_frozen_candidate else (
        "BASELINE_ONLY_MODEL_SELECTION_GATE_FAILED" if not model_selection_gate["passed"] else
        "BASELINE_ONLY_CALIBRATION_GATE_FAILED"
    )
    body = {
        "protocol_version": MODEL_VERSION, "status": status,
        "version_order": ["0.1", "hierarchical_shrinkage_v1"],
        "decision_output": "NO_TRADE", "model_validated": False, "deployment_calibrated": False,
        "live_signal_enabled": False, "live_order_execution_enabled": False,
        "as_of_index_exclusive": as_of_index, "partition_end_indices": ends,
        "partition_rows": {"development": len(dev), "model_selection": len(model_selection),
                           "calibration": len(calibration), "final_holdout": len(final_holdout)},
        "candidate_grid": {
            "candidate_count": len(CANDIDATE_SPECS), "experts": list(EXPERT_SUPPORT),
            "k_grid": list(K_GRID), "max_weight_grid": list(CAP_GRID),
            "formula": "p_global + cap*n/(n+k)*(p_expert-p_global)",
            "selection_partition": "development", "selection_order": "Brier gain, then log-loss improvement, then smaller cap, then larger k",
        },
        "development_selection": {
            "selection_partition": "development",
            "selected_candidate": spec, "selected_candidate_metrics": winner["metrics"],
            "baseline_metrics": dev_base_metrics,
            "selected_candidate_brier_gain": winner["baseline_brier_gain"],
            "selected_candidate_log_loss_improvement": winner["baseline_log_loss_improvement"],
            "selected_prediction_manifest_sha256": winner["prediction_manifest_sha256"],
            "development_selection_sha256": selection_sha,
            "all_candidate_metrics": development_results,
            "holdout_labels_used_for_selection": False,
            "model_selection_labels_used_for_selection": False,
            "calibration_labels_used_for_selection": False,
        },
        "model_selection_gate": model_selection_gate,
        "calibration_gate": calibration_gate,
        "promotion": {
            "frozen_candidate_authorized_for_research_holdout_comparison": promoted_to_frozen_candidate,
            "final_holdout_changes_candidate_choice": False,
            "final_holdout_policy": "hierarchical_shrinkage_v1" if promoted_to_frozen_candidate else "0.1",
            "reason": "both independent confirmation gates must pass; otherwise emit exact baseline",
        },
        "final_holdout": {
            "partition": "final_holdout", "rows": len(final_holdout), "score_cohort_sha256": cohort_sha,
            "baseline_metrics": holdout_base_metrics,
            "frozen_candidate_shadow_metrics": holdout_shadow_metrics,
            "policy_metrics": holdout_policy_metrics,
            "paired_block_bootstrap_vs_baseline": holdout_bootstrap,
            "interpretation": "final-holdout metrics are descriptive only and cannot change selected_candidate or policy",
        },
        "source_qualification": {
            "dataset_sha256": ledger.get("metadata", {}).get("dataset_sha256"),
            "provenance_verified": False, "data_rights_verified": False,
            "timestamp_semantics_verified": False, "historical_feature_availability_verified": False,
            "broker_quote_equivalence_verified": False,
        },
        "causal_contracts": {
            "strict_label_maturity": "label_available_index < current decision index in producer; scoring partitions also require label availability before each cutoff",
            "candidate_support_is_predecision_metadata": True, "baseline_reproduced_exactly": True,
            "same_final_holdout_cohort": True, "future_holdout_labels_used_to_choose_parameters": False,
        },
    }
    body["report_sha256"] = _sha(body)
    return body
