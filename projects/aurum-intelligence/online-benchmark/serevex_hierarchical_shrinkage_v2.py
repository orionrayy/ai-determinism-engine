"""Serevex hierarchical shrinkage v2.0b: causal, block-purged research evaluator.

Research only. The expanding matured-label Laplace baseline is immutable and remains
the fallback unless one development-selected candidate passes two later independent
gates. Final holdout is descriptive and cannot change the candidate choice.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import defaultdict, deque
from typing import Any, Mapping, Sequence

MODEL_VERSION = "serevex-hierarchical-shrinkage-v2.0b-block-purged"
BASELINE = "0.1b"
GOVERNOR = "5.0b"
BLOCK_SPAN = 12
EMBARGO = 12
BLOCK_SIZE = BLOCK_SPAN + EMBARGO
MIN_ROWS = 200
MIN_BLOCKS = 8
MIN_POSITIVE_SHARE = 0.70
CRITICAL_Z = 2.58
ECE_TOLERANCE = 0.015
MIN_SUBGROUP_ROWS = 100
MAX_LABEL_HORIZON_BARS = 12
EPS = 1e-12
PARTITIONS = ("development", "model_selection", "calibration", "final_holdout")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _clip(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("probability_must_be_finite_numeric")
    return min(1.0 - EPS, max(EPS, float(value)))


def _sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _laplace(count: int, positive: int) -> float:
    return (positive + 1.0) / (count + 2.0)


def _group(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and value.lower() not in {"unknown", "none", "null", "na", "n/a"} else None


def _candidate_specs() -> list[dict[str, Any]]:
    result = []
    ordinal = 2
    # 36 single-level policies: rolling, regime and volatility; 12 nested policies.
    for family in ("rolling", "regime", "volatility"):
        for k in (20, 80, 240):
            for cap in (0.20, 0.35, 0.50, 0.65):
                result.append({"version": f"{ordinal // 10}.{ordinal % 10}b", "family": family,
                               "k": k, "cap": cap})
                ordinal += 1
    for parent_k in (20, 80):
        for child_k in (20, 80, 240):
            for cap in (0.25, 0.50):
                result.append({"version": f"{ordinal // 10}.{ordinal % 10}b", "family": "joint",
                               "parent_k": parent_k, "k": child_k, "cap": cap})
                ordinal += 1
    if len(result) != 48 or result[0]["version"] != "0.2b" or result[-1]["version"] != "4.9b":
        raise AssertionError("candidate_grid_contract_failed")
    return result


CANDIDATES = _candidate_specs()


def _validate_records(ledger: Mapping[str, Any], as_of_index: int) -> list[dict[str, Any]]:
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("records"), Sequence):
        raise ValueError("ledger_must_contain_records_sequence")
    if not _is_int(as_of_index) or as_of_index < 1:
        raise ValueError("as_of_index_must_be_positive_integer")
    rows, seen = [], set()
    for position, item in enumerate(ledger["records"]):
        if not isinstance(item, Mapping):
            raise ValueError(f"record_must_be_mapping_at_{position}")
        row = dict(item)
        index, maturity = row.get("decision_index"), row.get("label_available_index")
        if not _is_int(index) or not 0 <= index < as_of_index:
            raise ValueError(f"decision_index_out_of_range_at_{position}")
        if index in seen:
            raise ValueError(f"duplicate_decision_index_{index}")
        seen.add(index)
        if not _is_int(maturity) or maturity <= index:
            raise ValueError(f"label_maturity_must_follow_decision_at_{position}")
        if maturity - index > MAX_LABEL_HORIZON_BARS:
            raise ValueError(f"label_maturity_exceeds_registered_horizon_at_{position}")
        if row.get("partition") not in PARTITIONS:
            raise ValueError(f"invalid_partition_at_{position}")
        if not isinstance(row.get("label_mature"), bool):
            raise ValueError(f"label_mature_must_be_boolean_at_{position}")
        label = row.get("label")
        if row["label_mature"] and (label not in (0, 1) or isinstance(label, bool)):
            raise ValueError(f"mature_label_must_be_binary_at_{position}")
        if not row["label_mature"] and label is not None:
            raise ValueError(f"unmature_label_must_be_null_at_{position}")
        row["baseline_probability"] = _clip(row.get("baseline_probability"))
        row["_regime"] = _group(row.get("regime_id"))
        tags = row.get("state_tags")
        if not isinstance(tags, Mapping):
            tags = {}
        row["_volatility"] = _group(tags.get("volatility_bucket", row.get("volatility_bucket")))
        rows.append(row)
    rows.sort(key=lambda row: int(row["decision_index"]))
    return rows


def _build_causal_features(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Sweep labels by availability time; only maturity index < decision index is visible."""
    pending: list[tuple[int, int, int, str | None, str | None]] = []
    global_count = global_positive = 0
    recent: deque[int] = deque(maxlen=288)
    regime_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    vol_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    joint_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    features: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["decision_index"])
        while pending and pending[0][0] < index:
            _, _, label, regime, volatility = heapq.heappop(pending)
            global_count += 1
            global_positive += label
            recent.append(label)
            if regime is not None:
                regime_counts[regime][0] += 1
                regime_counts[regime][1] += label
            if volatility is not None:
                vol_counts[volatility][0] += 1
                vol_counts[volatility][1] += label
            if regime is not None and volatility is not None:
                joint_counts[(regime, volatility)][0] += 1
                joint_counts[(regime, volatility)][1] += label
        regime, volatility = row["_regime"], row["_volatility"]
        rc = regime_counts[regime] if regime is not None else [0, 0]
        vc = vol_counts[volatility] if volatility is not None else [0, 0]
        jc = joint_counts[(regime, volatility)] if regime is not None and volatility is not None else [0, 0]
        features[index] = {
            "global_count": global_count,
            "p_global": _laplace(global_count, global_positive),
            "rolling_count": len(recent),
            "p_rolling": _laplace(len(recent), sum(recent)),
            "regime_count": rc[0], "p_regime": _laplace(*rc),
            "volatility_count": vc[0], "p_volatility": _laplace(*vc),
            "joint_count": jc[0], "p_joint": _laplace(*jc),
            "regime": regime, "volatility": volatility,
        }
        if row["label_mature"] and row["label"] in (0, 1):
            heapq.heappush(pending, (int(row["label_available_index"]), index, int(row["label"]),
                                     regime, volatility))
    return features


def _verify_baseline(rows: Sequence[Mapping[str, Any]], features: Mapping[int, Mapping[str, Any]],
                     tolerance: float = 1e-10) -> dict[str, Any]:
    mismatch = None
    max_error = 0.0
    for row in rows:
        index = int(row["decision_index"])
        expected = float(features[index]["p_global"])
        actual = float(row["baseline_probability"])
        error = abs(expected - actual)
        max_error = max(max_error, error)
        if error > tolerance and mismatch is None:
            mismatch = {"decision_index": index, "expected": expected, "actual": actual, "error": error}
    return {"verified": mismatch is None, "rows_checked": len(rows),
            "maximum_absolute_error": max_error, "first_mismatch": mismatch,
            "rule": "Laplace expanding prevalence over labels with label_available_index < decision_index"}


def _predict(feature: Mapping[str, Any], spec: Mapping[str, Any]) -> float:
    base, cap, k = float(feature["p_global"]), float(spec["cap"]), int(spec["k"])
    family = spec["family"]
    if family == "rolling":
        support = min(288, int(feature["rolling_count"]))
        weight = cap * support / (support + k) if support else 0.0
        return _clip(base + weight * (float(feature["p_rolling"]) - base))
    if family == "regime":
        support = int(feature["regime_count"])
        weight = cap * support / (support + k) if support else 0.0
        return _clip(base + weight * (float(feature["p_regime"]) - base))
    if family == "volatility":
        support = int(feature["volatility_count"])
        weight = cap * support / (support + k) if support else 0.0
        return _clip(base + weight * (float(feature["p_volatility"]) - base))
    if family == "joint":
        parent_n = int(feature["regime_count"])
        parent_w = cap * parent_n / (parent_n + int(spec["parent_k"])) if parent_n else 0.0
        parent = base + parent_w * (float(feature["p_regime"]) - base)
        child_n = int(feature["joint_count"])
        child_w = cap * child_n / (child_n + k) if child_n else 0.0
        return _clip(parent + child_w * (float(feature["p_joint"]) - parent))
    raise ValueError("unsupported_candidate_family")


def _metrics(rows: Sequence[Mapping[str, Any]], predictions: Sequence[float]) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("metric_length_mismatch")
    if not rows:
        return {"rows": 0, "positive_rate": None, "brier": None, "log_loss": None, "ece": None, "max_bin_gap": None}
    counts, sum_p, sum_y = [0] * 10, [0.0] * 10, [0.0] * 10
    brier = loss = 0.0
    labels = []
    for row, raw_p in zip(rows, predictions):
        p, y = _clip(raw_p), int(row["label"])
        labels.append(y)
        brier += (p - y) ** 2
        loss -= y * math.log(p) + (1 - y) * math.log1p(-p)
        bucket = min(9, int(p * 10))
        counts[bucket] += 1
        sum_p[bucket] += p
        sum_y[bucket] += y
    ece = max_gap = 0.0
    for n, ps, ys in zip(counts, sum_p, sum_y):
        if n:
            gap = abs(ps / n - ys / n)
            ece += n / len(rows) * gap
            max_gap = max(max_gap, gap)
    return {"rows": len(rows), "positive_rate": sum(labels) / len(labels),
            "brier": brier / len(rows), "log_loss": loss / len(rows), "ece": ece, "max_bin_gap": max_gap}


def _complete_block_rows(rows: Sequence[Mapping[str, Any]], partition: str, start: int, end: int):
    """Score sparse event records only inside complete 24-bar source-time blocks.

    The ledger is event-driven: an absolute bar without a signal has no record.
    Therefore a valid source block must not require one ledger row per candle.
    A block is eligible only when its 24-bar time span lies wholly inside one
    frozen partition; only resolved events in the first 12 bars are scored.
    The following 12 bars remain embargoed from scoring.
    """
    by_block: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        index = int(row["decision_index"])
        by_block[index // BLOCK_SIZE].append(row)

    candidate_blocks = sorted({
        int(row["decision_index"]) // BLOCK_SIZE
        for row in rows
        if start <= int(row["decision_index"]) < end
    })
    eligible_blocks: list[int] = []
    cohort: list[dict[str, Any]] = []
    eligible_scored_row_counts: dict[str, int] = {}
    rejected: dict[str, int] = defaultdict(int)

    for block_id in candidate_blocks:
        first = block_id * BLOCK_SIZE
        stop = first + BLOCK_SIZE
        if first < start or stop > end:
            rejected["block_crosses_partition_boundary"] += 1
            continue

        block_events = by_block.get(block_id, [])
        if any(row.get("partition") != partition for row in block_events):
            rejected["partition_tag_mismatch_in_full_block"] += 1
            continue

        source_events = [
            row for row in block_events
            if int(row["decision_index"]) < first + BLOCK_SPAN
        ]
        unresolved = [
            row for row in source_events
            if not row["label_mature"] or row["label"] not in (0, 1)
        ]
        resolved = [
            row for row in source_events
            if row["label_mature"] and row["label"] in (0, 1)
        ]
        late_labels = [
            row for row in resolved
            if int(row["label_available_index"]) >= end
        ]
        if late_labels:
            rejected["label_not_mature_before_partition_cutoff"] += 1
            continue

        score_events = [dict(row) for row in resolved]
        if not score_events:
            rejected["no_resolved_scored_event_in_source_window"] += 1
            continue

        cohort.extend(score_events)
        eligible_blocks.append(block_id)
        eligible_scored_row_counts[str(block_id)] = len(score_events)

    audit = {
        "partition": partition,
        "start_index_inclusive": start,
        "end_index_exclusive": end,
        "block_size": BLOCK_SIZE,
        "scored_bars_per_block": BLOCK_SPAN,
        "embargo_bars_per_block": EMBARGO,
        "candidate_full_blocks": len(candidate_blocks),
        "eligible_full_blocks": eligible_blocks,
        "eligible_block_count": len(eligible_blocks),
        "eligible_scored_rows": len(cohort),
        "eligible_scored_row_counts": eligible_scored_row_counts,
        "rejected_block_reason_counts": dict(sorted(rejected.items())),
        "event_cohort_is_sparse_by_design": True,
        "ledger_is_sparse_event_stream": True,
        "source_row_coverage_required": False,
        "resolved_scored_events_only": True,
        "unresolved_events_excluded_from_metrics": True,
        "partial_or_boundary_crossing_blocks_used": False,
    }
    return cohort, audit

def _block_gains(rows: Sequence[Mapping[str, Any]], baseline: Sequence[float], candidate: Sequence[float]) -> dict[str, Any]:
    if len(rows) != len(baseline) or len(rows) != len(candidate):
        raise ValueError("block_gain_lengths_must_match")
    block_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"brier": [], "log_loss": []})
    for row, p0_raw, p1_raw in zip(rows, baseline, candidate):
        p0, p1, y = _clip(p0_raw), _clip(p1_raw), int(row["label"])
        index = int(row["decision_index"])
        if index % BLOCK_SIZE >= BLOCK_SPAN:
            continue
        values = block_values[index // BLOCK_SIZE]
        values["brier"].append((p0 - y) ** 2 - (p1 - y) ** 2)
        values["log_loss"].append(
            -(y * math.log(p0) + (1 - y) * math.log1p(-p0))
            + y * math.log(p1) + (1 - y) * math.log1p(-p1)
        )
    result = {"blocks": len(block_values), "metrics": {}}
    for metric in ("brier", "log_loss"):
        means = [sum(v[metric]) / len(v[metric]) for _, v in sorted(block_values.items()) if v[metric]]
        n = len(means)
        mean = sum(means) / n if n else None
        se = lower = share = None
        if n:
            share = sum(value > 0 for value in means) / n
        if n >= 2:
            variance = sum((value - mean) ** 2 for value in means) / (n - 1)
            se = math.sqrt(max(0.0, variance) / n)
            lower = mean - CRITICAL_Z * se
        result["metrics"][metric] = {"mean_improvement": mean, "standard_error": se,
                                    "conservative_lower_screen": lower,
                                    "positive_block_share": share, "blocks": n}
    return result


def _subgroup_calibration(rows: Sequence[Mapping[str, Any]], baseline: Sequence[float],
                          candidate: Sequence[float]) -> dict[str, Any]:
    dims = {
        "regime": lambda row: _group(row.get("_regime", row.get("regime_id"))),
        "volatility": lambda row: _group(row.get("_volatility")),
        "joint": lambda row: ((_group(row.get("_regime")), _group(row.get("_volatility")))
                              if _group(row.get("_regime")) and _group(row.get("_volatility")) else None),
    }
    reports, supported = {}, 0
    for name, key_fn in dims.items():
        groups: dict[Any, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            key = key_fn(row)
            if key is not None:
                groups[key].append(i)
        reports[name] = {}
        for key, indices in sorted(groups.items(), key=lambda pair: str(pair[0])):
            if len(indices) < MIN_SUBGROUP_ROWS:
                reports[name][str(key)] = {"rows": len(indices), "status": "INSUFFICIENT_SUPPORT"}
                continue
            supported += 1
            base_ece = _metrics([rows[i] for i in indices], [baseline[i] for i in indices])["ece"]
            cand_ece = _metrics([rows[i] for i in indices], [candidate[i] for i in indices])["ece"]
            reports[name][str(key)] = {"rows": len(indices), "baseline_ece": base_ece, "candidate_ece": cand_ece,
                                       "degradation": cand_ece - base_ece,
                                       "tolerance": ECE_TOLERANCE, "pass": cand_ece <= base_ece + ECE_TOLERANCE}
    supported_items = [item for dim in reports.values() for item in dim.values() if item.get("status") != "INSUFFICIENT_SUPPORT"]
    passed = supported > 0 and all(item.get("pass", False) for item in supported_items)
    return {"status": "PASS" if passed else ("INSUFFICIENT_SUBGROUP_SUPPORT" if not supported else "FAIL"),
            "supported_groups": supported, "minimum_rows_per_group": MIN_SUBGROUP_ROWS,
            "max_allowed_ece_degradation": ECE_TOLERANCE, "dimensions": reports}


def _gate(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any],
          features: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    baseline = [_clip(row["baseline_probability"]) for row in rows]
    predictions = [_predict(features[int(row["decision_index"])], spec) for row in rows]
    m0, m1 = _metrics(rows, baseline), _metrics(rows, predictions)
    block = _block_gains(rows, baseline, predictions)
    enough = len(rows) >= MIN_ROWS and block["blocks"] >= MIN_BLOCKS
    dual = all(
        block["metrics"][metric]["conservative_lower_screen"] is not None
        and block["metrics"][metric]["conservative_lower_screen"] > 0
        and block["metrics"][metric]["positive_block_share"] is not None
        and block["metrics"][metric]["positive_block_share"] >= MIN_POSITIVE_SHARE
        for metric in ("brier", "log_loss")
    )
    calibration = _subgroup_calibration(rows, baseline, predictions)
    passed = enough and dual and calibration["status"] == "PASS"
    return {"status": "PASS" if passed else ("INSUFFICIENT_ROWS_OR_BLOCKS" if not enough else
            "DUAL_SCORE_GATE_FAILED" if not dual else "SUBGROUP_CALIBRATION_GATE_FAILED"),
            "passed": passed, "rows": len(rows), "minimum_rows": MIN_ROWS, "minimum_blocks": MIN_BLOCKS,
            "minimum_positive_block_share": MIN_POSITIVE_SHARE, "critical_z": CRITICAL_Z,
            "baseline_metrics": m0, "candidate_metrics": m1,
            "baseline_brier_gain": m0["brier"] - m1["brier"],
            "baseline_log_loss_improvement": m0["log_loss"] - m1["log_loss"],
            "block_gains": block, "subgroup_calibration": calibration}


def evaluate_ledger(ledger: Mapping[str, Any], *, as_of_index: int,
                    partition_end_indices: Mapping[str, int], bootstrap_replicates: int = 500) -> dict[str, Any]:
    """Select on development, confirm on two later partitions, then inspect holdout once."""
    if not _is_int(bootstrap_replicates) or bootstrap_replicates < 100 or bootstrap_replicates > 10000:
        raise ValueError("bootstrap_replicates_must_be_between_100_and_10000")
    if not isinstance(partition_end_indices, Mapping) or set(partition_end_indices) != set(PARTITIONS):
        raise ValueError("partition_end_indices_must_define_all_four_partitions")
    ends = {name: partition_end_indices[name] for name in PARTITIONS}
    if any(not _is_int(value) for value in ends.values()):
        raise ValueError("partition_end_indices_must_be_integers")
    if not (0 < ends["development"] < ends["model_selection"] < ends["calibration"] < ends["final_holdout"] == as_of_index):
        raise ValueError("partition_end_indices_must_be_chronological_and_end_at_asof")
    rows = _validate_records(ledger, as_of_index)
    for row in rows:
        i = int(row["decision_index"])
        expected = ("development" if i < ends["development"] else
                    "model_selection" if i < ends["model_selection"] else
                    "calibration" if i < ends["calibration"] else "final_holdout")
        if row["partition"] != expected:
            raise ValueError(f"partition_tag_disagrees_with_frozen_index_ranges_at_{i}")
    features = _build_causal_features(rows)
    baseline_audit = _verify_baseline(rows, features)
    if not baseline_audit["verified"]:
        raise ValueError("causal_baseline_reproduction_failed:" + json.dumps(baseline_audit["first_mismatch"], sort_keys=True))

    starts = dict(zip(PARTITIONS, (0, ends["development"], ends["model_selection"], ends["calibration"])))
    block_cohorts, block_audits, ordinary_cohorts = {}, {}, {}
    for name in PARTITIONS:
        block_cohorts[name], block_audits[name] = _complete_block_rows(rows, name, starts[name], ends[name])
        ordinary_cohorts[name] = [row for row in rows if row["partition"] == name
                                  and int(row["decision_index"]) < ends[name]
                                  and int(row["label_available_index"]) < ends[name]
                                  and row["label_mature"] and row["label"] in (0, 1)]
    dev = block_cohorts["development"]
    selection = block_cohorts["model_selection"]
    calibration = block_cohorts["calibration"]
    holdout = ordinary_cohorts["final_holdout"]
    if not dev or not selection or not calibration or not holdout:
        availability = {
            name: {
                "block_purged_rows": len(block_cohorts[name]),
                "eligible_blocks": block_audits[name]["eligible_block_count"],
                "rejected_reasons": block_audits[name]["rejected_block_reason_counts"],
                "ordinary_matured_rows": len(ordinary_cohorts[name]),
            }
            for name in PARTITIONS
        }
        raise ValueError("all_partitions_require_matured_rows_and_final_holdout:" +
                         json.dumps(availability, sort_keys=True))

    baseline_dev = [_clip(row["baseline_probability"]) for row in dev]
    baseline_metrics = _metrics(dev, baseline_dev)
    ranked = []
    for spec in CANDIDATES:
        pred = [_predict(features[int(row["decision_index"])], spec) for row in dev]
        metrics = _metrics(dev, pred)
        ranked.append({**spec, "metrics": metrics, "brier_gain": baseline_metrics["brier"] - metrics["brier"],
                       "log_loss_improvement": baseline_metrics["log_loss"] - metrics["log_loss"],
                       "prediction_sha256": _sha([[row.get("decision_id", row["decision_index"]), p]
                                                  for row, p in zip(dev, pred)])})
    chosen = sorted(ranked, key=lambda item: (-item["brier_gain"], -item["log_loss_improvement"],
                                               float(item["cap"]), int(item.get("k", 0)),
                                               str(item["family"])))[0]
    spec = {key: chosen[key] for key in ("version", "family", "k", "cap", "parent_k") if key in chosen}
    development_hash = _sha({"partition": "development", "cutoff": ends["development"],
                              "cohort": [[row.get("decision_id", row["decision_index"]), int(row["label"])] for row in dev],
                              "winner": spec, "ranking": ranked})
    model_gate = _gate(selection, spec, features)
    calibration_gate = _gate(calibration, spec, features) if model_gate["passed"] else {
        "status": "NOT_RUN_MODEL_SELECTION_GATE_FAILED", "passed": False, "rows": len(calibration)}
    promoted = bool(model_gate["passed"] and calibration_gate.get("passed") is True)

    holdout_base = [_clip(row["baseline_probability"]) for row in holdout]
    holdout_shadow = [_predict(features[int(row["decision_index"])], spec) for row in holdout]
    holdout_policy = holdout_shadow if promoted else holdout_base
    holdout_cohort_hash = _sha([[row.get("decision_id", row["decision_index"]), int(row["decision_index"]),
                                 int(row["label_available_index"]), int(row["label"])] for row in holdout])
    return {
        "protocol_version": MODEL_VERSION,
        "status": "FROZEN_CANDIDATE_PASSES_CONFIRMATION_GATES" if promoted else (
            "BASELINE_ONLY_MODEL_SELECTION_GATE_FAILED" if not model_gate["passed"] else "BASELINE_ONLY_CALIBRATION_GATE_FAILED"),
        "version_order": [BASELINE] + [spec["version"] for spec in CANDIDATES] + [GOVERNOR],
        "candidate_grid": {"count": len(CANDIDATES), "families": ["rolling", "regime", "volatility", "joint"],
                           "k_grid": [20, 80, 240], "cap_grid": [0.20, 0.35, 0.50, 0.65],
                           "joint_parent_k_grid": [20, 80], "selection_partition": "development_only"},
        "partition_end_indices": ends,
        "partition_rows": {name: {"all_matured_rows": len(ordinary_cohorts[name]),
                                  "block_purged_scored_rows": len(block_cohorts[name]),
                                  "block_audit": block_audits[name]} for name in PARTITIONS},
        "baseline_reproduction_audit": baseline_audit,
        "development_selection": {"selected_candidate": spec, "selected_candidate_metrics": chosen["metrics"],
                                  "baseline_metrics": baseline_metrics, "brier_gain": chosen["brier_gain"],
                                  "log_loss_improvement": chosen["log_loss_improvement"],
                                  "selection_sha256": development_hash, "candidate_diagnostics": ranked,
                                  "holdout_labels_used_for_selection": False,
                                  "model_selection_labels_used_for_selection": False,
                                  "calibration_labels_used_for_selection": False},
        "model_selection_gate": model_gate, "calibration_gate": calibration_gate,
        "final_holdout": {"rows": len(holdout), "score_cohort_sha256": holdout_cohort_hash,
                          "baseline_metrics": _metrics(holdout, holdout_base),
                          "candidate_shadow_metrics": _metrics(holdout, holdout_shadow),
                          "policy_metrics": _metrics(holdout, holdout_policy),
                          "candidate_prediction_manifest_sha256": _sha([[row.get("decision_id", row["decision_index"]), p]
                                                                        for row, p in zip(holdout, holdout_shadow)]),
                          "policy": "hierarchical_candidate" if promoted else BASELINE,
                          "final_holdout_can_change_candidate_choice": False},
        "source_qualification": {"dataset_sha256": ledger.get("metadata", {}).get("dataset_sha256"),
                                 "provenance_verified": False, "data_rights_verified": False,
                                 "timestamp_semantics_verified": False, "historical_feature_availability_verified": False,
                                 "broker_quote_equivalence_verified": False},
        "decision_output": "NO_TRADE", "model_validated": False, "deployment_calibrated": False,
        "live_signal_enabled": False, "live_order_execution_enabled": False,
        "research_authority": "diagnostic_only_no_trade",
    }
