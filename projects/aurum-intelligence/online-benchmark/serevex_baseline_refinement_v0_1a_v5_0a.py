"""Serevex baseline refinement v0.1a-v5.0a.

Research-only, standard-library implementation. v0.1a is a naming reset of the
legacy v4.5 blend recipe, not a claim that legacy v4.5 was validated. The exact
row-wise causal baseline remains the independent reference and final fallback.
All online selections use only labels strictly matured before a current decision.
This module never authorizes live trading.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

MODEL_VERSION = "serevex-baseline-refinement-v5.0a.3"
VERSION_ORDER = tuple(f"{i / 10:.1f}a" for i in range(1, 51))
ANCHOR_VERSION = "0.1a"
GOVERNED_VERSION = "5.0a"
CAUSAL_BASELINE = "CAUSAL_BASELINE"
CANDIDATE_VERSIONS = VERSION_ORDER[1:-1]
FIXED_VERSIONS = VERSION_ORDER[:-1]
BLOCK_SPAN_BARS = 12
EMBARGO_BARS = 12
BLOCK_PERIOD = BLOCK_SPAN_BARS + EMBARGO_BARS
MIN_HISTORY_ROWS = 200
MIN_BLOCKS = 8
MIN_POSITIVE_BLOCK_SHARE = 0.70
SELECTION_LOOK_BLOCK_INTERVAL = 8
FAMILYWISE_ALPHA = 0.05
ECE_TOLERANCE = 0.015
EPS = 1e-12

# Exact mass and specialist proportions from the legacy v4.5 recipe.
# Old v4.5 used specialist_scale = 1 - 0.85 * (5 / 9) = 19 / 36.
LEGACY_V45_SCALE = 19.0 / 36.0
SPECIALIST_SHARES = (0.40, 0.35, 0.25)
EXPERT_KEYS = ("rolling_matured_rate_288", "regime_matured_rate", "volatility_matured_rate")
SUPPORT_KEYS = (
    "matured_label_count_before_decision",
    "conditional_regime_support_before_decision",
    "conditional_volatility_support_before_decision",
)
REFERENCES = (CAUSAL_BASELINE, ANCHOR_VERSION)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _clip(probability: Any) -> float:
    if not isinstance(probability, (int, float)) or isinstance(probability, bool) or not math.isfinite(float(probability)):
        raise ValueError("probability_must_be_finite_numeric")
    return min(1.0 - EPS, max(EPS, float(probability)))


def _sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _brier(probability: float, label: int) -> float:
    return (_clip(probability) - label) ** 2


def _log_loss(probability: float, label: int) -> float:
    p = _clip(probability)
    return -(label * math.log(p) + (1 - label) * math.log(1.0 - p))


def _sequential_critical_z(look_count: int, candidate_count: int, endpoint_count: int = 4) -> tuple[float, float, float]:
    """Summable alpha-spending screen; not a universal finite-sample guarantee."""
    for value, name in ((look_count, "look_count"), (candidate_count, "candidate_count"), (endpoint_count, "endpoint_count")):
        if not _is_int(value) or value < 1:
            raise ValueError(f"{name}_must_be_positive_integer")
    alpha_look = FAMILYWISE_ALPHA / (look_count * (look_count + 1.0))
    alpha_per_test = alpha_look / (endpoint_count * candidate_count)
    low, high = 0.0, 12.0
    for _ in range(90):
        mid = (low + high) / 2.0
        tail = 0.5 * math.erfc(mid / math.sqrt(2.0))
        if tail > alpha_per_test:
            low = mid
        else:
            high = mid
    return high, alpha_look, alpha_per_test


def _ece(rows: Sequence[Mapping[str, Any]], predictions: Sequence[float], bins: int = 10) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("ece_input_length_mismatch")
    if not _is_int(bins) or bins < 2:
        raise ValueError("ece_bins_must_be_integer_at_least_2")
    bucket: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for row, probability in zip(rows, predictions):
        label = row.get("label")
        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("ece_requires_binary_labels")
        p = _clip(probability)
        bucket[min(bins - 1, int(p * bins))].append((p, int(label)))
    reliability, weighted_gap, max_gap = [], 0.0, 0.0
    for index, values in enumerate(bucket):
        if not values:
            reliability.append({"bin": index, "count": 0, "mean_probability": None, "observed_rate": None, "absolute_gap": None})
            continue
        mean_probability = sum(p for p, _ in values) / len(values)
        observed_rate = sum(y for _, y in values) / len(values)
        gap = abs(mean_probability - observed_rate)
        weighted_gap += len(values) / max(1, len(rows)) * gap
        max_gap = max(max_gap, gap)
        reliability.append({"bin": index, "count": len(values), "mean_probability": mean_probability,
                            "observed_rate": observed_rate, "absolute_gap": gap})
    return {"method": "equal_width_bins_descriptive", "bins": bins, "rows": len(rows),
            "ece": weighted_gap if rows else None, "max_bin_gap": max_gap if rows else None,
            "reliability": reliability}


def _valid_records(raw_records: Sequence[Mapping[str, Any]], as_of_index: int) -> list[dict[str, Any]]:
    if not _is_int(as_of_index) or as_of_index < 1:
        raise ValueError("as_of_index_must_be_positive_integer")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("records_must_be_a_sequence")
    rows, seen = [], set()
    for position, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"record_must_be_mapping_at_{position}")
        row = dict(raw)
        decision_index = row.get("decision_index")
        maturity_index = row.get("label_available_index")
        if not _is_int(decision_index) or not 0 <= decision_index < as_of_index:
            raise ValueError(f"decision_index_out_of_asof_range_at_{position}")
        if decision_index in seen:
            raise ValueError(f"duplicate_decision_index_{decision_index}")
        seen.add(decision_index)
        if not _is_int(maturity_index) or maturity_index <= decision_index:
            raise ValueError(f"label_available_index_must_follow_decision_at_{position}")
        if not isinstance(row.get("label_mature"), bool):
            raise ValueError(f"label_mature_must_be_boolean_at_{position}")
        label = row.get("label")
        if row["label_mature"] and (label not in (0, 1) or isinstance(label, bool)):
            raise ValueError(f"mature_label_must_be_binary_at_{position}")
        if not row["label_mature"] and label is not None:
            raise ValueError(f"unmature_label_must_be_null_at_{position}")
        row["baseline_probability"] = _clip(row.get("baseline_probability", 0.5))
        experts = row.get("expert_probabilities", {})
        if not isinstance(experts, Mapping):
            raise ValueError(f"expert_probabilities_must_be_mapping_at_{position}")
        row["expert_probabilities"] = {key: _clip(experts.get(key, row["baseline_probability"])) for key in EXPERT_KEYS}
        for key in SUPPORT_KEYS:
            value = row.get(key, 0)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{key}_must_be_nonnegative_integer_at_{position}")
        rows.append(row)
    return sorted(rows, key=lambda row: (row["decision_index"], str(row.get("decision_id", ""))))


def _version_spec(step: int) -> dict[str, Any]:
    if step == 1:
        return {"version": ANCHOR_VERSION, "family": "legacy_v4.5_recipe_alias",
                "renamed_from": "legacy v4.5", "specialist_mass_before_support": LEGACY_V45_SCALE,
                "support_rule": "legacy min(1,n/40)", "promotion_status": "exploratory_anchor_not_validated",
                "description": "Reproduce legacy v4.5 blend recipe; exploratory anchor only."}
    if 2 <= step <= 10:
        k = 20 * (step - 1)
        return {"version": f"{step / 10:.1f}a", "family": "support_prior_sweep", "support_k": k,
                "specialist_mass_before_support": LEGACY_V45_SCALE,
                "description": "Change support shrinkage only; keep legacy v4.5 specialist shares."}
    if 11 <= step <= 20:
        cap = 0.20 + (step - 11) * 0.05
        return {"version": f"{step / 10:.1f}a", "family": "specialist_mass_cap_sweep", "mass_cap": round(cap, 8),
                "support_k": 40, "description": "Change total specialist mass only; preserve causal-baseline residual mass."}
    if 21 <= step <= 30:
        t = (step - 21) / 9.0
        shares = (0.15 + 0.40 * t, 0.55 - 0.25 * t, 0.30 - 0.15 * t)
        return {"version": f"{step / 10:.1f}a", "family": "expert_contribution_ablation",
                "mass_cap": 0.50, "shares": [round(value, 8) for value in shares], "support_k": 60,
                "description": "Vary rolling/regime/volatility allocation with fixed total cap."}
    if 31 <= step <= 40:
        threshold = 0.04 + (step - 31) * 0.02
        return {"version": f"{step / 10:.1f}a", "family": "expert_disagreement_guard",
                "mass_cap": LEGACY_V45_SCALE, "support_k": 60, "disagreement_threshold": round(threshold, 8),
                "description": "Reduce specialist weight when the experts disagree beyond the preregistered range."}
    if 41 <= step <= 49:
        k = 40 + (step - 41) * 20
        cap = 0.50 - (step - 41) * 0.025
        return {"version": f"{step / 10:.1f}a", "family": "robust_low_mass_sweep",
                "mass_cap": round(cap, 8), "support_k": k, "disagreement_threshold": 0.12,
                "description": "Combine stronger support shrinkage, lower mass caps and a fixed disagreement guard."}
    raise ValueError("unsupported_candidate_step")


def _candidate_probability(row: Mapping[str, Any], step: int) -> float:
    base = _clip(row["baseline_probability"])
    experts_map = row["expert_probabilities"]
    experts = [_clip(experts_map[key]) for key in EXPERT_KEYS]
    supports = [int(row.get(key, 0)) for key in SUPPORT_KEYS]
    spec = _version_spec(step)
    if step == 1:
        confidences = [min(1.0, n / 40.0) for n in supports]
        weights = [LEGACY_V45_SCALE * share * conf for share, conf in zip(SPECIALIST_SHARES, confidences)]
    elif 2 <= step <= 10:
        k = spec["support_k"]
        confidences = [n / (n + k) if n else 0.0 for n in supports]
        weights = [LEGACY_V45_SCALE * share * conf for share, conf in zip(SPECIALIST_SHARES, confidences)]
    elif 11 <= step <= 20:
        k, cap = spec["support_k"], spec["mass_cap"]
        confidences = [n / (n + k) if n else 0.0 for n in supports]
        weights = [cap * share * conf for share, conf in zip(SPECIALIST_SHARES, confidences)]
    elif 21 <= step <= 30:
        k, cap = spec["support_k"], spec["mass_cap"]
        confidences = [n / (n + k) if n else 0.0 for n in supports]
        weights = [cap * share * conf for share, conf in zip(spec["shares"], confidences)]
    else:
        k, cap = spec["support_k"], spec["mass_cap"]
        confidences = [n / (n + k) if n else 0.0 for n in supports]
        disagreement = max(experts) - min(experts)
        agreement_factor = min(1.0, spec["disagreement_threshold"] / max(disagreement, EPS))
        weights = [cap * share * conf * agreement_factor for share, conf in zip(SPECIALIST_SHARES, confidences)]
    total = sum(weights)
    if total < -EPS or total > 1.0 + EPS:
        raise ValueError("candidate_specialist_weights_out_of_bounds")
    return _clip((1.0 - total) * base + sum(weight * value for weight, value in zip(weights, experts)))


def _candidate_row(row: Mapping[str, Any]) -> dict[str, float]:
    return {version: _candidate_probability(row, step) for step, version in enumerate(FIXED_VERSIONS, start=1)}


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
            return {"blocks": self.n, "mean_gain": None, "standard_error": None,
                    "lower_screen": None, "positive_block_share": None}
        mean = self.total / self.n
        variance = max(0.0, (self.squares - self.n * mean * mean) / (self.n - 1))
        standard_error = math.sqrt(variance / self.n)
        return {"blocks": self.n, "mean_gain": mean, "standard_error": standard_error,
                "lower_screen": mean - critical_z * standard_error,
                "positive_block_share": self.positive / self.n}


class _CalibrationAccumulator:
    def __init__(self, bins: int = 10) -> None:
        self.bins = bins
        self.count, self.sum_p, self.sum_y = [0] * bins, [0.0] * bins, [0.0] * bins
        self.rows = 0

    def add(self, probability: float, label: int) -> None:
        p = _clip(probability)
        index = min(self.bins - 1, int(p * self.bins))
        self.count[index] += 1
        self.sum_p[index] += p
        self.sum_y[index] += int(label)
        self.rows += 1

    def diagnostics(self) -> dict[str, Any]:
        reliability, ece, max_gap = [], 0.0, 0.0
        for index, count in enumerate(self.count):
            if not count:
                reliability.append({"bin": index, "count": 0, "mean_probability": None, "observed_rate": None, "absolute_gap": None})
                continue
            mean_p, observed = self.sum_p[index] / count, self.sum_y[index] / count
            gap = abs(mean_p - observed)
            ece += count / max(1, self.rows) * gap
            max_gap = max(max_gap, gap)
            reliability.append({"bin": index, "count": count, "mean_probability": mean_p,
                                "observed_rate": observed, "absolute_gap": gap})
        return {"method": "equal_width_10_bin_descriptive", "rows": self.rows,
                "ece": ece if self.rows else None, "max_bin_gap": max_gap if self.rows else None,
                "reliability": reliability}


class _Governor:
    """Prequential candidate governor with strict maturity and closed-block selection."""
    def __init__(self, expected: Mapping[int, int]) -> None:
        self.expected = dict(expected)
        self.pending: list[tuple[int, int, dict[str, Any]]] = []
        self.open: dict[int, dict[str, dict[str, dict[str, list[float]]]]] = defaultdict(dict)
        self.processed: dict[int, int] = defaultdict(int)
        self.moments = {
            version: {reference: {metric: _Moments() for metric in ("brier", "log_loss")} for reference in REFERENCES}
            for version in CANDIDATE_VERSIONS
        }
        self.cal = {version: _CalibrationAccumulator() for version in FIXED_VERSIONS}
        self.base_cal = _CalibrationAccumulator()
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

    def _observe(self, row: Mapping[str, Any], forecasts: Mapping[str, float]) -> None:
        index, label = int(row["decision_index"]), int(row["label"])
        # Calibration eligibility must match the score-comparison cohort: only
        # matured labels in non-embargo source windows can affect the online gate.
        if index % BLOCK_PERIOD >= BLOCK_SPAN_BARS:
            return
        base_probability = _clip(row["baseline_probability"])
        self.base_cal.add(base_probability, label)
        for version in FIXED_VERSIONS:
            self.cal[version].add(forecasts[version], label)
        self.matured_scored += 1
        block = index // BLOCK_PERIOD
        for version in CANDIDATE_VERSIONS:
            for reference, reference_probability in ((CAUSAL_BASELINE, base_probability),
                                                     (ANCHOR_VERSION, forecasts[ANCHOR_VERSION])):
                target = self.open.setdefault(block, {}).setdefault(version, {}).setdefault(
                    reference, {"brier": [0.0, 0.0], "log_loss": [0.0, 0.0]})
                target["brier"][0] += _brier(reference_probability, label) - _brier(forecasts[version], label)
                target["brier"][1] += 1.0
                target["log_loss"][0] += _log_loss(reference_probability, label) - _log_loss(forecasts[version], label)
                target["log_loss"][1] += 1.0
        self.processed[block] += 1

    def advance(self, current: int, forecasts: Mapping[int, Mapping[str, float]]) -> None:
        while self.pending and self.pending[0][0] < current:
            maturity, decision, row = heapq.heappop(self.pending)
            if decision >= current or maturity >= current:
                raise ValueError("prequential_queue_violated_strict_maturity_cutoff")
            if decision not in forecasts:
                raise ValueError("matured_forecast_missing_from_ledger")
            self._observe(row, forecasts[decision])
        while (self.cursor + 1) * BLOCK_PERIOD <= current:
            block, required = self.cursor, self.expected.get(self.cursor, 0)
            if self.processed.get(block, 0) < required:
                break
            block_data = self.open.get(block, {})
            for version in CANDIDATE_VERSIONS:
                for reference in REFERENCES:
                    for metric in ("brier", "log_loss"):
                        pair = block_data.get(version, {}).get(reference, {}).get(metric)
                        if pair and pair[1]:
                            self.moments[version][reference][metric].add(pair[0] / pair[1])
            self.cursor += 1

    def select(self) -> dict[str, Any]:
        nblocks = min(self.moments[version][CAUSAL_BASELINE]["brier"].n for version in CANDIDATE_VERSIONS)
        if self.cached_gate is not None and nblocks == self.last_evaluated_blocks:
            return self.cached_gate
        if self.matured_scored < MIN_HISTORY_ROWS or nblocks < MIN_BLOCKS:
            self.last_evaluated_blocks = nblocks
            self.cached_gate = {"selected": CAUSAL_BASELINE, "status": "BASELINE_ONLY_INSUFFICIENT_MATURED_BLOCK_SUPPORT",
                                "history_scored_rows": self.matured_scored, "independent_blocks": nblocks, "candidates": {}}
            return self.cached_gate
        # Re-evaluate only every eight closed blocks; intermediate decisions reuse the last governed choice.
        if nblocks % SELECTION_LOOK_BLOCK_INTERVAL != 0:
            return self.cached_gate or {"selected": CAUSAL_BASELINE,
                                        "status": "BASELINE_ONLY_INSUFFICIENT_MATURED_BLOCK_SUPPORT",
                                        "history_scored_rows": self.matured_scored, "independent_blocks": nblocks, "candidates": {}}
        self.last_evaluated_blocks = nblocks
        self.selection_look_count += 1
        critical_z, alpha_look, alpha_per_test = _sequential_critical_z(
            self.selection_look_count, len(CANDIDATE_VERSIONS), endpoint_count=4)
        self.last_critical_z, self.last_alpha_look = critical_z, alpha_look
        base_ece = self.base_cal.diagnostics()["ece"]
        anchor_ece = self.cal[ANCHOR_VERSION].diagnostics()["ece"]
        diagnostics, accepted = {}, []
        for version in CANDIDATE_VERSIONS:
            screens: dict[str, Any] = {}
            lowers, pass_flags = [], []
            for reference in REFERENCES:
                brier = self.moments[version][reference]["brier"].stats(critical_z)
                log_loss = self.moments[version][reference]["log_loss"].stats(critical_z)
                screens[reference] = {"brier": brier, "log_loss": log_loss}
                for item in (brier, log_loss):
                    lowers.append(item["lower_screen"])
                    pass_flags.append(item["lower_screen"] is not None and item["lower_screen"] > 0.0
                                      and item["positive_block_share"] is not None
                                      and item["positive_block_share"] >= MIN_POSITIVE_BLOCK_SHARE)
            ece = self.cal[version].diagnostics()["ece"]
            # ECE versus a global-prevalence forecast is diagnostic, not a hard gate:
            # an informative regime-conditioned model may be less globally flat.
            # Still require calibration not to degrade versus the v0.1a anchor.
            calibration_ok = ece is not None and anchor_ece is not None and ece <= anchor_ece + ECE_TOLERANCE
            eligible = all(pass_flags) and calibration_ok
            diagnostics[version] = {"eligible": eligible, "screens_vs_reference": screens,
                                    "ece": ece, "causal_baseline_ece": base_ece,
                                    "v0_1a_anchor_ece": anchor_ece, "ece_tolerance": ECE_TOLERANCE,
                                    "calibration_gate_passed": calibration_ok,
                                    "look_index": self.selection_look_count, "critical_z": critical_z,
                                    "alpha_look": alpha_look, "alpha_per_one_sided_test": alpha_per_test}
            if eligible:
                accepted.append((min(float(value) for value in lowers if value is not None), version))
        if not accepted:
            self.cached_gate = {"selected": CAUSAL_BASELINE,
                                "status": "BASELINE_ONLY_NO_CONFIRMED_DUAL_REFERENCE_DUAL_SCORE_GAIN",
                                "history_scored_rows": self.matured_scored, "independent_blocks": nblocks,
                                "look_index": self.selection_look_count, "critical_z": critical_z,
                                "alpha_look": alpha_look, "candidates": diagnostics}
            return self.cached_gate
        margin, selected = sorted(accepted, key=lambda item: (-item[0], item[1]))[0]
        self.cached_gate = {"selected": selected, "status": "SPECIALIST_PASSES_DUAL_REFERENCE_DUAL_SCORE_AND_CALIBRATION_GATE",
                            "history_scored_rows": self.matured_scored, "independent_blocks": nblocks,
                            "joint_conservative_margin": margin, "look_index": self.selection_look_count,
                            "critical_z": critical_z, "alpha_look": alpha_look, "candidates": diagnostics}
        return self.cached_gate


def _metric(rows: Sequence[Mapping[str, Any]], predictions: Sequence[float]) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("metric_input_length_mismatch")
    if not rows:
        return {"rows": 0, "positive_rate": None, "brier": None, "log_loss": None, "calibration": _ece([], [])}
    labels = [int(row["label"]) for row in rows]
    return {"rows": len(rows), "positive_rate": sum(labels) / len(labels),
            "brier": sum(_brier(p, y) for p, y in zip(predictions, labels)) / len(labels),
            "log_loss": sum(_log_loss(p, y) for p, y in zip(predictions, labels)) / len(labels),
            "calibration": _ece(rows, predictions)}


def _paired_block_bootstrap(rows: Sequence[Mapping[str, Any]], reference: Sequence[float], candidate: Sequence[float],
                            *, replicates: int, seed: int) -> dict[str, Any]:
    """Bootstrap paired model gains on non-embargo source windows.

    Full-cohort point estimates and source-window bootstrap estimates are different
    estimands by design; both are emitted explicitly so their signs are not conflated.
    Each replicate uses the same sampled block indices for Brier and log-loss.
    """
    if len(rows) != len(reference) or len(rows) != len(candidate):
        raise ValueError("bootstrap_input_lengths_must_match")
    full_brier_diffs: list[float] = []
    full_loss_diffs: list[float] = []
    source_brier_diffs: list[float] = []
    source_loss_diffs: list[float] = []
    brier_by_block: dict[int, list[float]] = defaultdict(list)
    loss_by_block: dict[int, list[float]] = defaultdict(list)
    for row, reference_p, candidate_p in zip(rows, reference, candidate):
        label = int(row["label"])
        index = int(row["decision_index"])
        brier_gain = _brier(reference_p, label) - _brier(candidate_p, label)
        loss_gain = _log_loss(reference_p, label) - _log_loss(candidate_p, label)
        full_brier_diffs.append(brier_gain)
        full_loss_diffs.append(loss_gain)
        if index % BLOCK_PERIOD >= BLOCK_SPAN_BARS:
            continue
        block = index // BLOCK_PERIOD
        source_brier_diffs.append(brier_gain)
        source_loss_diffs.append(loss_gain)
        brier_by_block[block].append(brier_gain)
        loss_by_block[block].append(loss_gain)

    paired_block_means: list[tuple[float, float]] = []
    for block in sorted(set(brier_by_block) & set(loss_by_block)):
        brier_values, loss_values = brier_by_block[block], loss_by_block[block]
        if brier_values and loss_values:
            paired_block_means.append((sum(brier_values) / len(brier_values),
                                       sum(loss_values) / len(loss_values)))
    count = len(paired_block_means)
    source_rows = len(source_brier_diffs)
    report_base = {
        "scored_rows_full_cohort": len(rows),
        "source_window_rows": source_rows,
        "source_window_blocks": count,
        "full_cohort_brier_mean_improvement": sum(full_brier_diffs) / len(full_brier_diffs) if full_brier_diffs else None,
        "full_cohort_log_loss_mean_improvement": sum(full_loss_diffs) / len(full_loss_diffs) if full_loss_diffs else None,
        "source_window_row_weighted_brier_mean_improvement": (
            sum(source_brier_diffs) / source_rows if source_rows else None),
        "source_window_row_weighted_log_loss_mean_improvement": (
            sum(source_loss_diffs) / source_rows if source_rows else None),
        "bootstrap_point_estimand": "equal_weight_mean_of_non_embargo_source_window_block_means",
        "bootstrap_resampling": "shared_paired_block_indices_for_brier_and_log_loss",
    }
    if list(reference) == list(candidate):
        return {**report_base, "status": "IDENTICAL_REFERENCE", "blocks": 0, "replicates": 0,
                "brier_improvement_ci95": [0.0, 0.0], "log_loss_improvement_ci95": [0.0, 0.0],
                "brier_mean_improvement": 0.0, "log_loss_mean_improvement": 0.0}
    if count < MIN_BLOCKS:
        return {**report_base, "status": "INSUFFICIENT_EMBARGOED_BLOCK_SUPPORT", "blocks": count,
                "replicates": 0, "brier_improvement_ci95": None, "log_loss_improvement_ci95": None,
                "brier_mean_improvement": None, "log_loss_mean_improvement": None}

    brier_blocks = [pair[0] for pair in paired_block_means]
    loss_blocks = [pair[1] for pair in paired_block_means]
    block_means_brier = sum(brier_blocks) / count
    block_means_loss = sum(loss_blocks) / count
    rng = random.Random(seed)
    boot_brier, boot_loss = [], []
    for _ in range(replicates):
        sampled = [rng.randrange(count) for _ in range(count)]
        boot_brier.append(sum(brier_blocks[index] for index in sampled) / count)
        boot_loss.append(sum(loss_blocks[index] for index in sampled) / count)
    boot_brier.sort()
    boot_loss.sort()
    lower, upper = int(0.025 * replicates), min(replicates - 1, int(0.975 * replicates))
    return {**report_base, "status": "DIAGNOSTIC_PAIRED_SOURCE_TIME_BLOCK_BOOTSTRAP",
            "blocks": count, "replicates": replicates,
            "brier_mean_improvement": block_means_brier,
            "brier_improvement_ci95": [boot_brier[lower], boot_brier[upper]],
            "log_loss_mean_improvement": block_means_loss,
            "log_loss_improvement_ci95": [boot_loss[lower], boot_loss[upper]],
            "block_span_bars": BLOCK_SPAN_BARS, "embargo_bars": EMBARGO_BARS, "seed": seed,
            "interpretation": "Intervals estimate the equal-weight mean gain on non-embargo source windows. Full-cohort gains are reported separately and may differ, including in sign; these are diagnostic intervals, not universal finite-sample guarantees."}

def evaluate_ledger(ledger: Mapping[str, Any], *, as_of_index: int, score_partition: str = "final_holdout",
                    bootstrap_replicates: int = 500) -> dict[str, Any]:
    """Evaluate 50 fixed hypotheses and a causal prequential governor on one ledger."""
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("records"), Sequence):
        raise ValueError("ledger_must_contain_records_sequence")
    if not _is_int(bootstrap_replicates) or not 100 <= bootstrap_replicates <= 10000:
        raise ValueError("bootstrap_replicates_must_be_100_to_10000")
    if not isinstance(score_partition, str) or not score_partition:
        raise ValueError("score_partition_must_be_nonempty")
    records = _valid_records(ledger["records"], as_of_index)
    if not records:
        raise ValueError("empty_forecast_ledger")

    last_decision_index = max(int(row["decision_index"]) for row in records)
    expected: dict[int, int] = defaultdict(int)
    for row in records:
        index = int(row["decision_index"])
        # A label whose availability lies beyond the final decision cannot ever
        # enter this run's causal queue. Excluding it from expected support prevents
        # it from permanently holding an earlier source block open.
        if (row["label_mature"] and row["label"] in (0, 1)
                and int(row["label_available_index"]) < last_decision_index
                and index % BLOCK_PERIOD < BLOCK_SPAN_BARS):
            expected[index // BLOCK_PERIOD] += 1
    governor, forecasts = _Governor(expected), {}
    gate_counts, selected_counts, trace = defaultdict(int), defaultdict(int), []
    trace_by_block: dict[int, list[Any]] = defaultdict(list)
    for row in records:
        index = int(row["decision_index"])
        governor.advance(index, forecasts)
        predictions = _candidate_row(row)
        gate = governor.select()
        selected = gate["selected"]
        predictions[GOVERNED_VERSION] = predictions[selected] if selected in predictions else _clip(row["baseline_probability"])
        forecasts[index] = predictions
        gate_counts[gate["status"]] += 1
        selected_counts[selected] += 1
        trace.append([row.get("decision_id", index), index, selected, gate["status"], gate.get("independent_blocks", 0)])
        trace_by_block[index // BLOCK_PERIOD].append(trace[-1])
        governor.queue(row)

    expected_source_window_rows = sum(expected.values())
    maturity_window_reconciled = governor.matured_scored == expected_source_window_rows
    selection_by_index = {int(entry[1]): entry[2] for entry in trace}
    baseline_fallback_rows = [
        row for row in records
        if selection_by_index[int(row["decision_index"])] == CAUSAL_BASELINE
    ]
    baseline_fallback_exact = bool(baseline_fallback_rows) and all(
        forecasts[int(row["decision_index"])][GOVERNED_VERSION] == _clip(row["baseline_probability"])
        for row in baseline_fallback_rows
    )
    all_decisions_selected_baseline = len(baseline_fallback_rows) == len(records)
    all_governed_probabilities_match_baseline = all(
        forecasts[int(row["decision_index"])][GOVERNED_VERSION] == _clip(row["baseline_probability"])
        for row in records
    )
    baseline_fallback_manifest = _sha([
        [row.get("decision_id", row["decision_index"]), int(row["decision_index"]),
         forecasts[int(row["decision_index"])][GOVERNED_VERSION], _clip(row["baseline_probability"])]
        for row in baseline_fallback_rows
    ])

    scored = [row for row in records if row.get("partition") == score_partition
              and int(row["decision_index"]) < as_of_index
              and int(row["label_available_index"]) < as_of_index
              and row["label_mature"] and row["label"] in (0, 1)]
    scored.sort(key=lambda row: int(row["decision_index"]))
    if not scored:
        raise ValueError("no_matured_rows_for_requested_score_partition")
    indices = [int(row["decision_index"]) for row in scored]
    base_predictions = [_clip(row["baseline_probability"]) for row in scored]
    scored_forecasts = {version: [forecasts[index][version] for index in indices] for version in VERSION_ORDER}
    baseline_metrics = _metric(scored, base_predictions)
    metrics = {version: _metric(scored, scored_forecasts[version]) for version in VERSION_ORDER}
    anchor_predictions, anchor_metrics = scored_forecasts[ANCHOR_VERSION], metrics[ANCHOR_VERSION]
    prediction_hashes = {}
    for step, version in enumerate(VERSION_ORDER, start=1):
        model_predictions = scored_forecasts[version]
        prediction_hashes[version] = _sha([[row.get("decision_id", row["decision_index"]), prediction]
                                           for row, prediction in zip(scored, model_predictions)])
        item = metrics[version]
        item.update({
            "status": "RESEARCH_DIAGNOSTIC_ONLY",
            "baseline_brier_gain": baseline_metrics["brier"] - item["brier"],
            "baseline_log_loss_improvement": baseline_metrics["log_loss"] - item["log_loss"],
            "baseline_ece_improvement": baseline_metrics["calibration"]["ece"] - item["calibration"]["ece"],
            "v0_1a_anchor_brier_gain": anchor_metrics["brier"] - item["brier"],
            "v0_1a_anchor_log_loss_improvement": anchor_metrics["log_loss"] - item["log_loss"],
            "prediction_manifest_sha256": prediction_hashes[version],
            "paired_block_interval_vs_causal_baseline": _paired_block_bootstrap(
                scored, base_predictions, model_predictions, replicates=bootstrap_replicates, seed=20261020 + step * 13),
            "paired_block_interval_vs_v0_1a_anchor": _paired_block_bootstrap(
                scored, anchor_predictions, model_predictions, replicates=bootstrap_replicates, seed=20261020 + step * 17),
        })

    slices: dict[str, dict[str, Any]] = {"regime": {}, "volatility_bucket": {}}
    audit_versions = (ANCHOR_VERSION, "4.5a", GOVERNED_VERSION)
    for dimension, key_fn in (
        ("regime", lambda row: str(row.get("regime_id", "unknown"))),
        ("volatility_bucket", lambda row: str((row.get("state_tags") or {}).get("volatility_bucket", "unknown"))),
    ):
        groups: dict[str, list[int]] = defaultdict(list)
        for position, row in enumerate(scored):
            groups[key_fn(row)].append(position)
        for group, positions in sorted(groups.items()):
            if len(positions) < 30:
                slices[dimension][group] = {"rows": len(positions), "status": "INSUFFICIENT_SUPPORT"}
            else:
                subset = [scored[position] for position in positions]
                slices[dimension][group] = {version: _metric(subset, [scored_forecasts[version][position] for position in positions])
                                            for version in audit_versions}
                slices[dimension][group][CAUSAL_BASELINE] = _metric(subset, [base_predictions[position] for position in positions])

    baseline_reference_normalization = all(
        base_predictions[pos] == _clip(scored[pos]["baseline_probability"]) for pos in range(len(scored))
    )
    baseline_reproduction = baseline_reference_normalization and baseline_fallback_exact
    same_cohort = all(metrics[version]["rows"] == len(scored) for version in VERSION_ORDER)
    score_cohort_sha = _sha([[row.get("decision_id", row["decision_index"]), int(row["decision_index"]),
                              int(row["label_available_index"]), int(row["label"])] for row in scored])
    trace_block_hashes = {str(block): _sha(entries) for block, entries in sorted(trace_by_block.items())}
    metric_hash = _sha(metrics)
    report_digest = _sha({"model_version": MODEL_VERSION, "score_partition": score_partition,
                          "score_cohort_sha256": score_cohort_sha, "prediction_hashes": prediction_hashes,
                          "metrics_sha256": metric_hash, "selection_trace_block_sha256": trace_block_hashes,
                          "selected_counts": dict(sorted(selected_counts.items())),
                          "expected_source_window_rows": expected_source_window_rows,
                          "observed_source_window_rows": governor.matured_scored,
                          "maturity_window_reconciled": maturity_window_reconciled,
                          "baseline_fallback_manifest_sha256": baseline_fallback_manifest,
                          "baseline_fallback_exact": baseline_fallback_exact,
                          "all_decisions_selected_baseline": all_decisions_selected_baseline,
                          "all_governed_probabilities_match_baseline": all_governed_probabilities_match_baseline})

    calibration_cohort_matches = (
        governor.base_cal.rows == governor.matured_scored
        and all(accumulator.rows == governor.matured_scored for accumulator in governor.cal.values())
    )

    report = {
        "protocol_version": MODEL_VERSION,
        "version_order": list(VERSION_ORDER),
        "version_reset_lineage": {
            "0.1a": {"renamed_from": "legacy v4.5", "meaning": "same deterministic blend recipe, new experimental lineage",
                      "promotion_status": "exploratory_anchor_not_validated"},
            "causal_reference": "original expanding matured-label baseline, retained outside the a-series",
            "anchor_selection_caveat": "The legacy v4.5 recipe was selected after inspecting the prior final-holdout results. Comparisons against v0.1a on that same historical cohort are post-selection and descriptive, not independent confirmatory evidence.",
            "governed_version": "5.0a", "fixed_hypothesis_count": len(FIXED_VERSIONS),
            "prequential_candidate_count": len(CANDIDATE_VERSIONS),
        },
        "status": "BASELINE_GOVERNED_RESEARCH_ONLY",
        "score_partition": score_partition,
        "as_of_index_exclusive": as_of_index,
        "scored_rows": len(scored),
        "anchors": {CAUSAL_BASELINE: {"status": "REFERENCE_ONLY", **baseline_metrics},
                    ANCHOR_VERSION: {"status": "RENAMED_EXPLORATORY_ANCHOR", **anchor_metrics}},
        "versions": metrics,
        "candidate_specs": [_version_spec(step) for step in range(1, 50)],
        "score_cohort_sha256": score_cohort_sha,
        "prediction_hashes": prediction_hashes,
        "metrics_sha256": metric_hash,
        "prequential_governor": {
            "status": "FIXED_PROTOCOL_PREQUENTIAL_SELECTION",
            "selected_version_counts": dict(sorted(selected_counts.items())),
            "gate_status_counts": dict(sorted(gate_counts.items())),
            "registered_candidate_versions": list(CANDIDATE_VERSIONS),
            "baseline_fallback_audit": {
                "baseline_selected_decision_count": len(baseline_fallback_rows),
                "fallback_probabilities_match_source_baseline_exactly": baseline_fallback_exact,
                "all_decisions_selected_causal_baseline": all_decisions_selected_baseline,
                "all_governed_probabilities_match_source_baseline_exactly": all_governed_probabilities_match_baseline,
                "fallback_prediction_manifest_sha256": baseline_fallback_manifest,
            },
            "incremental_audit": {"matured_scored_window_rows": governor.matured_scored,
                                   "expected_matured_source_window_rows": expected_source_window_rows,
                                   "observed_matured_source_window_rows": governor.matured_scored,
                                   "last_decision_index": last_decision_index,
                                   "prequential_maturity_window_reconciled": maturity_window_reconciled,
                                   "closed_independent_blocks": min(governor.moments[version][CAUSAL_BASELINE]["brier"].n
                                                                     for version in CANDIDATE_VERSIONS),
                                   "selection_look_count": governor.selection_look_count,
                                   "last_critical_z": governor.last_critical_z,
                                   "last_alpha_look": governor.last_alpha_look,
                                   "calibration_rows_by_version": {key: value.rows for key, value in governor.cal.items()},
                                   "calibration_cohort_matches_dual_score_gate": calibration_cohort_matches},
            "candidate_gate": {"matured_labels_only": True,
                               "strict_maturity_cutoff": "label_available_index < current_decision_index",
                               "source_scoring_window_bars": BLOCK_SPAN_BARS,
                               "target_horizon_embargo_bars": EMBARGO_BARS,
                               "minimum_matured_scored_rows": MIN_HISTORY_ROWS,
                               "minimum_independent_blocks": MIN_BLOCKS,
                               "familywise_alpha": FAMILYWISE_ALPHA,
                               "candidate_count": len(CANDIDATE_VERSIONS),
                               "one_sided_endpoints_per_candidate_per_look": 4,
                               "selection_look_rule": "alpha_look(k)=0.05/(k*(k+1)); sum_k alpha_look(k)=0.05",
                               "selection_look_block_interval": SELECTION_LOOK_BLOCK_INTERVAL,
                               "minimum_positive_block_share": MIN_POSITIVE_BLOCK_SHARE,
                               "calibration_gate_evidence": "matured non-embargo source-window labels; same cohort as Brier/log-loss screens",
                               "dual_reference_required": [CAUSAL_BASELINE, ANCHOR_VERSION],
                               "dual_score_required": ["brier", "log_loss"],
                               "max_ece_degradation": ECE_TOLERANCE,
                               "calibration_requires_not_degrading_vs_anchor": True,
                               "fallback": "exact_rowwise_causal_baseline; never fallback to renamed v0.1a anchor",
                               "final_holdout_used_for_static_parameter_selection": False,
                               "prior matured holdout labels may inform later prequential decisions": True},
            "selection_trace_sha256": _sha(trace),
            "selection_trace_block_sha256": trace_block_hashes,
            "baseline_reproduction_verified": baseline_reproduction,
            "baseline_reference_normalization_verified": baseline_reference_normalization,
            "baseline_fallback_prediction_reproduction_verified": baseline_fallback_exact,
            "prequential_maturity_window_reconciled": maturity_window_reconciled,
            "same_scored_cohort_verified": same_cohort,
            "future_label_used_for_own_decision": False,
        },
        "multi_calibration_audit": {
            "metrics": ["brier", "log_loss", "ece", "max_bin_gap"],
            "reliability_bins": 10,
            "scores_by_regime_and_volatility": slices,
            "candidate_calibration_rows": {version: governor.cal[version].rows for version in FIXED_VERSIONS},
            "ece_and_subgroup_reliability_are_descriptive": True,
            "interpretation": "ECE, maximum-bin gaps, and subgroup reliability are diagnostics; sparse slices do not prove conditional calibration.",
        },
        "multi_audit": {
            "input_contract": "PASS" if baseline_reproduction else "FAIL",
            "all_versions_same_scored_cohort": same_cohort,
            "prequential_calibration_cohort_matches_dual_score_cohort": calibration_cohort_matches,
            "prequential_maturity_window_reconciled": maturity_window_reconciled,
            "causal_baseline_fallback_exact": baseline_fallback_exact,
            "baseline_fallback_prediction_manifest_sha256": baseline_fallback_manifest,
            "score_cohort_sha256": score_cohort_sha,
            "unique_decision_indices": len({row["decision_index"] for row in records}) == len(records),
            "chronological_order_normalized": True,
            "expected_source_window_maturity_reconciled": maturity_window_reconciled,
            "source_time_block_bootstrap": {"block_span_bars": BLOCK_SPAN_BARS, "embargo_bars": EMBARGO_BARS,
                                            "replicates": bootstrap_replicates,
                                            "interpretation": "paired diagnostic bootstrap on non-embargo source windows; not a universal finite-sample guarantee"},
            "bootstrap_estimand_alignment": {
                "full_cohort_gain_fields": ["baseline_brier_gain", "baseline_log_loss_improvement"],
                "interval_aligned_gain_fields": ["paired_block_interval_vs_causal_baseline.brier_mean_improvement",
                                                  "paired_block_interval_vs_causal_baseline.log_loss_mean_improvement"],
                "bootstrap_cohort": "mature scored rows with decision_index % 24 < 12; embargo-window rows are excluded",
                "weighting": "equal weight per source-window block for interval point estimate",
                "brier_and_log_loss_share_resampled_block_indices": True,
                "full_cohort_and_bootstrap_estimands_can_differ_in_sign": True
            },
            "source_provenance_independently_verified": False,
            "historical_feature_availability_verified": False,
            "broker_quote_equivalence_verified": False,
        },
        "validation": {"baseline_reproduction": "PASS" if baseline_reproduction else "FAIL",
                       "identical_scored_cohort": "PASS" if same_cohort else "FAIL",
                       "strict_maturity_filter_in_gate": True,
                       "prequential_calibration_cohort_matches_dual_score_gate": "PASS" if calibration_cohort_matches else "FAIL",
                       "prequential_maturity_window_reconciled": "PASS" if maturity_window_reconciled else "FAIL",
                       "causal_baseline_fallback_exact": "PASS" if baseline_fallback_exact else "FAIL",
                       "holdout_scores_used_to_choose_static_parameters": False,
                       "live_trade_authority": False, "decision_output": "NO_TRADE",
                       "model_validated": False, "deployment_calibrated": False,
                       "live_signal_enabled": False, "live_order_execution_enabled": False},
        "report_sha256": report_digest,
    }
    if (not baseline_reproduction or not same_cohort or not calibration_cohort_matches
            or not maturity_window_reconciled or not baseline_fallback_exact
            or (all_decisions_selected_baseline and not all_governed_probabilities_match_baseline)):
        raise ValueError("baseline_fallback_or_maturity_or_cohort_contract_failed")
    return report
