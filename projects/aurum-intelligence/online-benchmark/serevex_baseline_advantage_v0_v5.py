"""Serevex baseline-advantage evolution v0.1-v5.0.

Research-only standard-library implementation. The exact row-wise causal
baseline is the champion/fallback. All online selections use only labels that
matured strictly before the current decision. This module never authorizes trades.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

MODEL_VERSION = "serevex-baseline-advantage-v5.0"
VERSION_ORDER = tuple(f"{i / 10:.1f}" for i in range(1, 51))
BASELINE_VERSION = "0.1"
FINAL_GOVERNED_VERSION = "5.0"
CANDIDATE_VERSIONS = VERSION_ORDER[1:-1]
BLOCK_SPAN_BARS = 12
EMBARGO_BARS = 12
BLOCK_PERIOD = BLOCK_SPAN_BARS + EMBARGO_BARS
MIN_HISTORY_ROWS = 200
MIN_BLOCKS = 8
MIN_POSITIVE_BLOCK_SHARE = 0.70
# Sequential Bonferroni alpha-spending across 48 candidates, two score endpoints,
# and all future looks. The normal tail conversion is a screening approximation,
# not a finite-sample confidence-sequence guarantee.
FAMILYWISE_ALPHA = 0.05
ECE_TOLERANCE = 0.015
EPS = 1e-12


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _clip(p: float) -> float:
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(float(p)):
        raise ValueError("probability_must_be_finite_numeric")
    return min(1.0 - EPS, max(EPS, float(p)))


def _sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sequential_critical_z(look_count: int, candidate_count: int) -> tuple[float, float, float]:
    """Return z, per-look alpha and per-test alpha for summable alpha-spending."""
    if not _is_int(look_count) or look_count < 1:
        raise ValueError("look_count_must_be_positive_integer")
    if not _is_int(candidate_count) or candidate_count < 1:
        raise ValueError("candidate_count_must_be_positive_integer")
    alpha_look = FAMILYWISE_ALPHA / (look_count * (look_count + 1.0))
    alpha_per_test = alpha_look / (2.0 * candidate_count)
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        tail = 0.5 * math.erfc(mid / math.sqrt(2.0))
        if tail > alpha_per_test:
            lo = mid
        else:
            hi = mid
    return hi, alpha_look, alpha_per_test


def _brier(p: float, y: int) -> float:
    return (_clip(p) - y) ** 2


def _log_loss(p: float, y: int) -> float:
    p = _clip(p)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def _ece(rows: Sequence[Mapping[str, Any]], predictions: Sequence[float], bins: int = 10) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("ece_input_length_mismatch")
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for row, pred in zip(rows, predictions):
        y = row.get("label")
        if y not in (0, 1) or isinstance(y, bool):
            raise ValueError("ece_requires_binary_labels")
        p = _clip(pred)
        buckets[min(bins - 1, int(p * bins))].append((p, int(y)))
    reliability, ece, max_gap = [], 0.0, 0.0
    for index, values in enumerate(buckets):
        if not values:
            reliability.append({"bin": index, "count": 0, "mean_probability": None, "observed_rate": None, "absolute_gap": None})
            continue
        mean_p = sum(p for p, _ in values) / len(values)
        observed = sum(y for _, y in values) / len(values)
        gap = abs(mean_p - observed)
        ece += len(values) / max(1, len(rows)) * gap
        max_gap = max(max_gap, gap)
        reliability.append({"bin": index, "count": len(values), "mean_probability": mean_p, "observed_rate": observed, "absolute_gap": gap})
    return {"method": "equal_width_10_bin_descriptive", "rows": len(rows),
            "ece": ece if rows else None, "max_bin_gap": max_gap if rows else None,
            "reliability": reliability}


def _valid_records(records: Sequence[Mapping[str, Any]], as_of_index: int) -> list[dict[str, Any]]:
    if not _is_int(as_of_index) or as_of_index < 1:
        raise ValueError("as_of_index_must_be_positive_integer")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records_must_be_a_sequence")
    ordered, seen = [], set()
    for pos, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"record_must_be_mapping_at_{pos}")
        row = dict(raw)
        di, mi = row.get("decision_index"), row.get("label_available_index")
        if not _is_int(di) or di < 0 or di >= as_of_index:
            raise ValueError(f"decision_index_out_of_asof_range_at_{pos}")
        if di in seen:
            raise ValueError(f"duplicate_decision_index_{di}")
        seen.add(di)
        if not _is_int(mi) or mi < di:
            raise ValueError(f"label_available_index_must_be_integer_and_not_precede_decision_at_{pos}")
        if not isinstance(row.get("label_mature"), bool):
            raise ValueError(f"label_mature_must_be_boolean_at_{pos}")
        y = row.get("label")
        if row["label_mature"] and (y not in (0, 1) or isinstance(y, bool)):
            raise ValueError(f"mature_label_must_be_binary_at_{pos}")
        if not row["label_mature"] and y is not None:
            raise ValueError(f"unmature_label_must_be_null_at_{pos}")
        row["baseline_probability"] = _clip(row.get("baseline_probability", 0.5))
        experts = row.get("expert_probabilities", {})
        if not isinstance(experts, Mapping):
            raise ValueError(f"expert_probabilities_must_be_mapping_at_{pos}")
        row["expert_probabilities"] = {
            key: _clip(experts.get(key, row["baseline_probability"]))
            for key in ("rolling_matured_rate_288", "regime_matured_rate", "volatility_matured_rate")
        }
        for key in ("matured_label_count_before_decision", "conditional_regime_support_before_decision", "conditional_volatility_support_before_decision"):
            value = row.get(key, 0)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{key}_must_be_nonnegative_integer_at_{pos}")
        ordered.append(row)
    ordered.sort(key=lambda r: (r["decision_index"], str(r.get("decision_id", ""))))
    return ordered


def _candidates(row: Mapping[str, Any]) -> dict[str, float]:
    """Generate 0.2–4.9 as a monotone research ladder; v0.1 is untouched."""
    base = _clip(row["baseline_probability"])
    exp = row["expert_probabilities"]
    n_roll = min(int(row.get("matured_label_count_before_decision", 0)), 288)
    n_reg = int(row.get("conditional_regime_support_before_decision", 0))
    n_vol = int(row.get("conditional_volatility_support_before_decision", 0))
    # Shrink any specialist's influence when its support is small.
    roll_conf = min(1.0, n_roll / 40.0)
    reg_conf = min(1.0, n_reg / 40.0)
    vol_conf = min(1.0, n_vol / 40.0)
    rolling = _clip(exp["rolling_matured_rate_288"])
    regime = _clip(exp["regime_matured_rate"])
    volatility = _clip(exp["volatility_matured_rate"])
    candidates = {"0.1": base}
    for step in range(2, 50):  # v0.2 ... v4.9, 48 progressively varied hypotheses
        if step <= 20:
            # v0.2–v2.0: progressively increase the capped rolling-rate contribution.
            t = (step - 2) / 18.0
            w_roll, w_reg, w_vol = 0.04 + 0.68 * t, 0.0, 0.0
        elif step <= 30:
            # v2.1–v3.0: add regime evidence while retaining substantial global/rolling mass.
            t = (step - 20) / 10.0
            w_roll, w_reg, w_vol = 0.72 - 0.10 * t, 0.35 * t, 0.0
        elif step <= 40:
            # v3.1–v4.0: add volatility conditioning under an explicit total-weight cap.
            t = (step - 30) / 10.0
            w_roll, w_reg, w_vol = 0.62 - 0.22 * t, 0.35, 0.25 * t
        else:
            # v4.1–v4.9: walk specialist mass back toward baseline to test robust shrinkage.
            t = (step - 40) / 9.0
            specialist_scale = 1.0 - 0.85 * t
            w_roll, w_reg, w_vol = 0.40 * specialist_scale, 0.35 * specialist_scale, 0.25 * specialist_scale
        w_roll *= roll_conf
        w_reg *= reg_conf
        w_vol *= vol_conf
        total = w_roll + w_reg + w_vol
        if total > 1.0 + 1e-12:
            raise ValueError("candidate_specialist_weights_exceed_one")
        p = ((1.0 - total) * base + w_roll * rolling + w_reg * regime + w_vol * volatility)
        candidates[f"{step / 10:.1f}"] = _clip(p)
    return candidates


class _Moments:
    __slots__ = ("n", "total", "squares", "positive")

    def __init__(self) -> None:
        self.n, self.total, self.squares, self.positive = 0, 0.0, 0.0, 0

    def add(self, value: float) -> None:
        self.n += 1
        self.total += float(value)
        self.squares += float(value) ** 2
        self.positive += int(value > 0.0)

    def stats(self, critical_z: float) -> dict[str, Any]:
        if self.n < MIN_BLOCKS:
            return {"blocks": self.n, "mean_gain": None, "standard_error": None, "lower_screen": None, "positive_block_share": None}
        mean = self.total / self.n
        variance = max(0.0, (self.squares - self.n * mean * mean) / (self.n - 1))
        se = math.sqrt(variance / self.n)
        return {"blocks": self.n, "mean_gain": mean, "standard_error": se,
                "lower_screen": mean - critical_z * se, "positive_block_share": self.positive / self.n}


class _CalibrationAccumulator:
    def __init__(self, bins: int = 10) -> None:
        self.bins = bins
        self.count, self.sum_p, self.sum_y = [0] * bins, [0.0] * bins, [0.0] * bins
        self.rows = 0

    def add(self, p: float, y: int) -> None:
        p = _clip(p)
        i = min(self.bins - 1, int(p * self.bins))
        self.count[i] += 1
        self.sum_p[i] += p
        self.sum_y[i] += int(y)
        self.rows += 1

    def ece(self) -> float | None:
        if not self.rows:
            return None
        return sum(n / self.rows * abs(self.sum_p[i] / n - self.sum_y[i] / n)
                   for i, n in enumerate(self.count) if n)


class _Governor:
    """Incremental maturity queue and closed-block stats; linear-time scoring."""
    def __init__(self, expected: Mapping[int, int]) -> None:
        self.expected = dict(expected)
        self.pending: list[tuple[int, int, dict[str, Any]]] = []
        self.open: dict[int, dict[str, list[float]]] = defaultdict(dict)
        self.processed: dict[int, int] = defaultdict(int)
        self.moments = {v: {"brier": _Moments(), "log_loss": _Moments()} for v in CANDIDATE_VERSIONS}
        self.cal = {v: _CalibrationAccumulator() for v in VERSION_ORDER}
        self.matured_scored = 0
        self.cursor = min(self.expected) if self.expected else 0
        self.last_evaluated_blocks = -1
        self.cached_gate: dict[str, Any] | None = None
        self.selection_look_count = 0
        self.last_critical_z: float | None = None
        self.last_alpha_look: float | None = None

    def queue(self, row: Mapping[str, Any]) -> None:
        if row["label_mature"] and row["label"] in (0, 1):
            heapq.heappush(self.pending, (int(row["label_available_index"]), int(row["decision_index"]), dict(row)))

    def _observe(self, row: Mapping[str, Any], forecast: Mapping[str, float]) -> None:
        di, y = int(row["decision_index"]), int(row["label"])
        for version, acc in self.cal.items():
            acc.add(forecast[version], y)
        if di % BLOCK_PERIOD >= BLOCK_SPAN_BARS:
            return
        self.matured_scored += 1
        block = di // BLOCK_PERIOD
        dest = self.open.setdefault(block, {})
        for version in CANDIDATE_VERSIONS:
            p0, p1 = forecast["0.1"], forecast[version]
            val = dest.setdefault(version, [0.0, 0.0, 0.0])
            val[0] += _brier(p0, y) - _brier(p1, y)
            val[1] += _log_loss(p0, y) - _log_loss(p1, y)
            val[2] += 1
        self.processed[block] += 1

    def advance(self, current: int, forecasts: Mapping[int, Mapping[str, float]]) -> None:
        while self.pending and self.pending[0][0] < current:
            maturity, di, row = heapq.heappop(self.pending)
            if di >= current or maturity >= current:
                raise ValueError("prequential_queue_violated_strict_maturity_cutoff")
            self._observe(row, forecasts[di])
        while (self.cursor + 1) * BLOCK_PERIOD <= current:
            block, expected = self.cursor, self.expected.get(self.cursor, 0)
            if self.processed.get(block, 0) < expected:
                break
            data = self.open.get(block, {})
            for version in CANDIDATE_VERSIONS:
                vals = data.get(version)
                if vals and vals[2]:
                    self.moments[version]["brier"].add(vals[0] / vals[2])
                    self.moments[version]["log_loss"].add(vals[1] / vals[2])
            self.cursor += 1

    def select(self) -> dict[str, Any]:
        nblocks = min(self.moments[v]["brier"].n for v in CANDIDATE_VERSIONS)
        # Only re-test after a new closed source-time block adds evidence. This
        # prevents repeated decisions on the same evidence snapshot.
        if self.cached_gate is not None and nblocks == self.last_evaluated_blocks:
            return self.cached_gate
        self.last_evaluated_blocks = nblocks
        base_ece = self.cal[BASELINE_VERSION].ece()
        if self.matured_scored < MIN_HISTORY_ROWS or nblocks < MIN_BLOCKS:
            self.cached_gate = {
                "selected": BASELINE_VERSION, "status": "BASELINE_ONLY_INSUFFICIENT_MATURED_BLOCK_SUPPORT",
                "history_scored_rows": self.matured_scored, "independent_blocks": nblocks, "candidates": {},
            }
            return self.cached_gate

        self.selection_look_count += 1
        critical_z, alpha_look, alpha_per_test = _sequential_critical_z(
            self.selection_look_count, len(CANDIDATE_VERSIONS))
        self.last_critical_z, self.last_alpha_look = critical_z, alpha_look
        diagnostics, accepted = {}, []
        for version in CANDIDATE_VERSIONS:
            brier = self.moments[version]["brier"].stats(critical_z)
            logloss = self.moments[version]["log_loss"].stats(critical_z)
            ece = self.cal[version].ece()
            ok = bool(brier["lower_screen"] is not None and logloss["lower_screen"] is not None
                      and brier["lower_screen"] > 0 and logloss["lower_screen"] > 0
                      and brier["positive_block_share"] >= MIN_POSITIVE_BLOCK_SHARE
                      and logloss["positive_block_share"] >= MIN_POSITIVE_BLOCK_SHARE
                      and ece is not None and base_ece is not None and ece <= base_ece + ECE_TOLERANCE)
            diagnostics[version] = {
                "eligible": ok, "brier": brier, "log_loss": logloss, "ece": ece,
                "baseline_ece": base_ece, "ece_tolerance": ECE_TOLERANCE,
                "look_index": self.selection_look_count, "critical_z": critical_z,
                "alpha_look": alpha_look, "alpha_per_one_sided_test": alpha_per_test,
            }
            if ok:
                accepted.append((min(brier["lower_screen"], logloss["lower_screen"]), version))
        if not accepted:
            self.cached_gate = {
                "selected": BASELINE_VERSION, "status": "BASELINE_ONLY_NO_CONFIRMED_PREQUENTIAL_DUAL_SCORE_GAIN",
                "history_scored_rows": self.matured_scored, "independent_blocks": nblocks,
                "look_index": self.selection_look_count, "critical_z": critical_z,
                "alpha_look": alpha_look, "candidates": diagnostics,
            }
            return self.cached_gate
        margin, selected = sorted(accepted, key=lambda x: (-x[0], int(x[1].replace(".", ""))))[0]
        self.cached_gate = {
            "selected": selected, "status": "SPECIALIST_PASSES_PREQUENTIAL_DUAL_SCORE_AND_CALIBRATION_GATE",
            "history_scored_rows": self.matured_scored, "independent_blocks": nblocks,
            "joint_conservative_margin": margin, "look_index": self.selection_look_count,
            "critical_z": critical_z, "alpha_look": alpha_look, "candidates": diagnostics,
        }
        return self.cached_gate


def _paired_block_bootstrap(rows: Sequence[Mapping[str, Any]], base: Sequence[float], candidate: Sequence[float],
                            *, replicates: int, seed: int) -> dict[str, Any]:
    if len(rows) != len(base) or len(rows) != len(candidate):
        raise ValueError("bootstrap_input_lengths_must_match")
    b, ll = defaultdict(list), defaultdict(list)
    for row, p0, p1 in zip(rows, base, candidate):
        di = int(row["decision_index"])
        if di % BLOCK_PERIOD >= BLOCK_SPAN_BARS:
            continue
        y, block = int(row["label"]), di // BLOCK_PERIOD
        b[block].append(_brier(p0, y) - _brier(p1, y))
        ll[block].append(_log_loss(p0, y) - _log_loss(p1, y))
    bb = [sum(v) / len(v) for _, v in sorted(b.items()) if v]
    lb = [sum(v) / len(v) for _, v in sorted(ll.items()) if v]
    if min(len(bb), len(lb)) < MIN_BLOCKS:
        return {"status": "INSUFFICIENT_EMBARGOED_BLOCK_SUPPORT", "blocks": min(len(bb), len(lb)),
                "replicates": 0, "brier_improvement_ci95": None, "log_loss_improvement_ci95": None}
    rng = random.Random(seed)
    boot_b = sorted(sum(bb[rng.randrange(len(bb))] for _ in bb) / len(bb) for _ in range(replicates))
    boot_l = sorted(sum(lb[rng.randrange(len(lb))] for _ in lb) / len(lb) for _ in range(replicates))
    lo, hi = int(0.025 * replicates), min(replicates - 1, int(0.975 * replicates))
    return {"status": "DIAGNOSTIC_PAIRED_SOURCE_TIME_BLOCK_BOOTSTRAP", "blocks": len(bb), "replicates": replicates,
            "brier_mean_improvement": sum(bb) / len(bb), "brier_improvement_ci95": [boot_b[lo], boot_b[hi]],
            "log_loss_mean_improvement": sum(lb) / len(lb), "log_loss_improvement_ci95": [boot_l[lo], boot_l[hi]],
            "block_span_bars": BLOCK_SPAN_BARS, "embargo_bars": EMBARGO_BARS, "seed": seed}


def _metric(rows: Sequence[Mapping[str, Any]], predictions: Sequence[float]) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("metric_input_length_mismatch")
    if not rows:
        return {"rows": 0, "positive_rate": None, "brier": None, "log_loss": None, "calibration": _ece([], [])}
    labels = [int(r["label"]) for r in rows]
    return {"rows": len(rows), "positive_rate": sum(labels) / len(labels),
            "brier": sum(_brier(p, y) for p, y in zip(predictions, labels)) / len(labels),
            "log_loss": sum(_log_loss(p, y) for p, y in zip(predictions, labels)) / len(labels),
            "calibration": _ece(rows, predictions)}


def evaluate_ledger(ledger: Mapping[str, Any], *, as_of_index: int, score_partition: str = "final_holdout",
                    bootstrap_replicates: int = 1000) -> dict[str, Any]:
    """Evaluate v0.1-v5.0 on identical mature rows with a fixed causal governor."""
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("records"), Sequence):
        raise ValueError("ledger_must_contain_records_sequence")
    if not _is_int(bootstrap_replicates) or not 200 <= bootstrap_replicates <= 10000:
        raise ValueError("bootstrap_replicates_must_be_200_to_10000")
    records = _valid_records(ledger["records"], as_of_index)
    if not isinstance(score_partition, str) or not score_partition:
        raise ValueError("score_partition_must_be_nonempty")
    if not records:
        raise ValueError("empty_forecast_ledger")
    expected = defaultdict(int)
    for row in records:
        di = int(row["decision_index"])
        if row["label_mature"] and row["label"] in (0, 1) and di % BLOCK_PERIOD < BLOCK_SPAN_BARS:
            expected[di // BLOCK_PERIOD] += 1

    governor, forecasts = _Governor(expected), {}
    gate_counts, selected_counts, trace = defaultdict(int), defaultdict(int), []
    trace_by_block = defaultdict(list)
    for row in records:
        di = int(row["decision_index"])
        governor.advance(di, forecasts)
        preds = _candidates(row)
        gate = governor.select()
        selected = gate["selected"]
        preds[FINAL_GOVERNED_VERSION] = preds[selected]
        forecasts[di] = preds
        gate_counts[gate["status"]] += 1
        selected_counts[selected] += 1
        trace.append([row.get("decision_id", di), di, selected, gate["status"], gate.get("independent_blocks", 0)])
        trace_by_block[di // BLOCK_PERIOD].append(trace[-1])
        governor.queue(row)

    scored = [r for r in records if r.get("partition") == score_partition
              and int(r["decision_index"]) < as_of_index
              and int(r["label_available_index"]) < as_of_index
              and r["label_mature"] and r["label"] in (0, 1)]
    if not scored:
        raise ValueError("no_matured_rows_for_requested_score_partition")
    scored.sort(key=lambda r: int(r["decision_index"]))
    indices = [int(r["decision_index"]) for r in scored]
    results, predictions, hashes = {}, {}, {}
    for version in VERSION_ORDER:
        p = [forecasts[i][version] for i in indices]
        predictions[version] = p
        hashes[version] = _sha([[r.get("decision_id", r["decision_index"]), q] for r, q in zip(scored, p)])
        results[version] = {"status": "RESEARCH_DIAGNOSTIC_ONLY", **_metric(scored, p),
                            "prediction_manifest_sha256": hashes[version]}
    base_p, base_m = predictions["0.1"], results["0.1"]
    for version in VERSION_ORDER:
        m = results[version]
        m["baseline_brier_gain"] = base_m["brier"] - m["brier"]
        m["baseline_log_loss_improvement"] = base_m["log_loss"] - m["log_loss"]
        m["ece_delta_vs_baseline"] = base_m["calibration"]["ece"] - m["calibration"]["ece"]
        m["paired_block_interval_vs_baseline"] = (
            {"status": "IDENTITY_BASELINE", "brier_improvement_ci95": [0.0, 0.0],
             "log_loss_improvement_ci95": [0.0, 0.0], "blocks": None, "replicates": 0}
            if version == "0.1" else
            _paired_block_bootstrap(scored, base_p, predictions[version], replicates=bootstrap_replicates,
                                    seed=20261020 + int(version.replace(".", "")))
        )

    slices = {"regime": {}, "volatility_bucket": {}}
    for dimension, key_fn in (
        ("regime", lambda r: str(r.get("regime_id", "unknown"))),
        ("volatility_bucket", lambda r: str((r.get("state_tags") or {}).get("volatility_bucket", "unknown"))),
    ):
        groups = defaultdict(list)
        for i, row in enumerate(scored):
            groups[key_fn(row)].append(i)
        for group, pos in sorted(groups.items()):
            if len(pos) < 30:
                slices[dimension][group] = {"rows": len(pos), "status": "INSUFFICIENT_SUPPORT"}
            else:
                subset = [scored[i] for i in pos]
                slices[dimension][group] = {v: _metric(subset, [predictions[v][i] for i in pos]) for v in VERSION_ORDER}

    baseline_ok = all(predictions["0.1"][i] == _clip(scored[i]["baseline_probability"]) for i in range(len(scored)))
    same_cohort = all(results[v]["rows"] == len(scored) for v in VERSION_ORDER)
    trace_block_hashes = {str(k): _sha(v) for k, v in sorted(trace_by_block.items())}
    score_cohort_sha256 = _sha([
        [r.get("decision_id", r["decision_index"]), int(r["decision_index"]),
         int(r["label_available_index"]), int(r["label"])]
        for r in scored
    ])
    version_metrics_sha256 = _sha(results)
    report_digest = _sha({"model_version": MODEL_VERSION, "as_of_index": as_of_index,
                          "score_partition": score_partition, "score_rows": len(scored),
                          "score_cohort_sha256": score_cohort_sha256,
                          "prediction_hashes": hashes, "version_metrics_sha256": version_metrics_sha256,
                          "selected_counts": dict(sorted(selected_counts.items())),
                          "gate_status_counts": dict(sorted(gate_counts.items())),
                          "selection_trace_block_sha256": trace_block_hashes})
    report = {
        "protocol_version": MODEL_VERSION, "version_order": list(VERSION_ORDER),
        "status": "BASELINE_GOVERNED_RESEARCH_ONLY", "score_partition": score_partition,
        "as_of_index_exclusive": as_of_index, "scored_rows": len(scored), "versions": results,
        "score_cohort_sha256": score_cohort_sha256, "version_metrics_sha256": version_metrics_sha256,
        "prequential_governor": {
            "status": "FIXED_PROTOCOL_PREQUENTIAL_SELECTION",
            "registered_candidate_versions": list(CANDIDATE_VERSIONS),
            "selected_version_counts": dict(sorted(selected_counts.items())),
            "gate_status_counts": dict(sorted(gate_counts.items())),
            "incremental_audit": {"matured_scored_window_rows": governor.matured_scored,
                                  "closed_independent_blocks": min(governor.moments[v]["brier"].n for v in CANDIDATE_VERSIONS),
                                  "selection_look_count": governor.selection_look_count,
                                  "last_critical_z": governor.last_critical_z,
                                  "last_alpha_look": governor.last_alpha_look,
                                  "calibration_rows_by_version": {k: v.rows for k, v in governor.cal.items()}},
            "candidate_gate": {
                "matured_labels_only": True, "strict_maturity_cutoff": "label_available_index < current_decision_index",
                "source_scoring_window_bars": BLOCK_SPAN_BARS, "target_horizon_embargo_bars": EMBARGO_BARS,
                "minimum_matured_scored_rows": MIN_HISTORY_ROWS, "minimum_independent_blocks": MIN_BLOCKS,
                "familywise_alpha": FAMILYWISE_ALPHA,
                "candidate_count": len(CANDIDATE_VERSIONS),
                "one_sided_score_tests_per_look": 2 * len(CANDIDATE_VERSIONS),
                "sequential_look_rule": "alpha_look(k)=0.05/(k*(k+1)); sum_k alpha_look(k)=0.05",
                "selection_look_count": governor.selection_look_count,
                "last_critical_z": governor.last_critical_z,
                "minimum_positive_block_share": MIN_POSITIVE_BLOCK_SHARE,
                "dual_score_required": ["brier", "log_loss"], "max_ece_degradation": ECE_TOLERANCE,
                "fallback": "exact_rowwise_causal_baseline_v0.1",
                "holdout_feedback_policy": "prior labels can update later decisions only after strict maturity; the online policy is fixed before scoring",
            },
            "selection_trace_sha256": _sha(trace), "selection_trace_block_sha256": trace_block_hashes,
            "baseline_reproduction_verified": baseline_ok,
            "same_scored_cohort_verified": same_cohort, "future_label_used_for_own_decision": False,
        },
        "multi_calibration_audit": {
            "metrics": ["brier", "log_loss", "ece", "max_bin_gap"], "reliability_bins": 10,
            "scores_by_regime_and_volatility": slices, "calibration_diagnostics_are_not_used_as_holdout_tuning": True,
            "interpretation": "ECE and subgroup reliability are descriptive; dependence, block size, and sample support remain relevant.",
        },
        "source_time_block_bootstrap": {"block_span_bars": BLOCK_SPAN_BARS, "embargo_bars": EMBARGO_BARS,
                                        "replicates": bootstrap_replicates, "interpretation": "paired descriptive block bootstrap; not a finite-sample guarantee"},
        "validation": {"baseline_reproduction": "PASS" if baseline_ok else "FAIL",
                       "identical_scored_cohort": "PASS" if same_cohort else "FAIL",
                       "unique_monotone_decision_indices": True, "strict_maturity_filter_in_gate": True,
                       "holdout_scores_used_to_choose_static_parameters": False, "live_trade_authority": False,
                       "decision_output": "NO_TRADE", "model_validated": False, "deployment_calibrated": False,
                       "live_signal_enabled": False, "live_order_execution_enabled": False},
        "report_sha256": report_digest,
    }
    if not baseline_ok or not same_cohort:
        raise ValueError("baseline_reproduction_or_same_cohort_contract_failed")
    return report
