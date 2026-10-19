"""Cross-source XAUUSD OHLC comparison without merging unlike feeds.

Naive timestamps are rejected by default. A caller may explicitly allow a UTC assumption
for descriptive diagnostics only; that assumption never verifies source timezone semantics.
No source is promoted for trading by this module.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

PRICE_FIELDS = ("open", "high", "low", "close")
# Screening heuristics, not an executable-price truth standard.
DEFAULT_MAX_MEDIAN_CLOSE_DIFFERENCE_BPS = 5.0
DEFAULT_MAX_P95_CLOSE_DIFFERENCE_BPS = 20.0


def _utc_timestamp(value: Any, *, naive_timestamp_policy: str = "reject") -> datetime:
    if not isinstance(value, datetime):
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        value = datetime.fromisoformat(raw)
    if value.tzinfo is None or value.utcoffset() is None:
        if naive_timestamp_policy != "assume_utc_for_diagnostic_only":
            raise ValueError("naive_timestamp_timezone_unverified")
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 30:
        return None
    mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    sum_left = sum(value * value for value in centered_left)
    sum_right = sum(value * value for value in centered_right)
    if sum_left <= 0 or sum_right <= 0:
        return None
    covariance = sum(x * y for x, y in zip(centered_left, centered_right))
    return covariance / math.sqrt(sum_left * sum_right)



def _numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    """Compact deterministic distribution summary; no source normalization is applied."""
    return {
        "median": statistics.median(values) if values else None,
        "p05": _percentile(values, 0.05),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _price_basis_diagnostics(
    common_times: Sequence[datetime],
    primary_map: Mapping[datetime, Mapping[str, Any]],
    secondary_map: Mapping[datetime, Mapping[str, Any]],
    *,
    expected_interval_seconds: int,
) -> dict[str, Any]:
    """Describe possible level/scale discrepancies without proposing a source correction."""
    primary_closes = [float(primary_map[ts]["close"]) for ts in common_times]
    secondary_closes = [float(secondary_map[ts]["close"]) for ts in common_times]
    signed_differences = [p - s for p, s in zip(primary_closes, secondary_closes)]
    level_correlation = _correlation(primary_closes, secondary_closes)
    fit: dict[str, Any] = {"status": "INSUFFICIENT_OVERLAP_OR_VARIANCE"}
    residual_abs_bps: list[float] = []
    residual_abs_units: list[float] = []

    if len(common_times) >= 30:
        x_mean = statistics.fmean(secondary_closes)
        y_mean = statistics.fmean(primary_closes)
        x_variance = sum((x - x_mean) ** 2 for x in secondary_closes)
        if x_variance > 0:
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(secondary_closes, primary_closes)) / x_variance
            intercept = y_mean - slope * x_mean
            residuals = [y - (intercept + slope * x) for x, y in zip(secondary_closes, primary_closes)]
            residual_abs_units = [abs(value) for value in residuals]
            residual_abs_bps = [abs(value) / abs(y) * 10000.0 for value, y in zip(residuals, primary_closes) if y]
            fit = {
                "status": "DIAGNOSTIC_ONLY",
                "mapping": "primary_close ~= intercept + slope * secondary_close",
                "intercept_price_units": intercept,
                "slope": slope,
                "level_correlation": level_correlation,
                "r_squared_from_level_correlation": level_correlation ** 2 if level_correlation is not None else None,
                "absolute_residual_price_units": _numeric_summary(residual_abs_units),
                "absolute_residual_basis_points_of_primary_close": _numeric_summary(residual_abs_bps),
                "interpretation": "descriptive fit only; not a calibrated price conversion, quote-side adjustment, or source selection",
            }

    quartiles = []
    n = len(common_times)
    for quartile in range(4):
        start = quartile * n // 4
        end = (quartile + 1) * n // 4
        values = signed_differences[start:end]
        quartiles.append({
            "chronological_quartile": quartile + 1,
            "rows": len(values),
            "signed_primary_minus_secondary_close_difference_units": _numeric_summary(values),
        })

    return_difference_bps: list[float] = []
    paired_rows = 0
    paired_primary_returns: list[float] = []
    paired_secondary_returns: list[float] = []
    for previous, current in zip(common_times, common_times[1:]):
        if (current - previous).total_seconds() != expected_interval_seconds:
            continue
        p0, p1 = float(primary_map[previous]["close"]), float(primary_map[current]["close"])
        s0, s1 = float(secondary_map[previous]["close"]), float(secondary_map[current]["close"])
        if p0 <= 0 or s0 <= 0:
            continue
        p_return, s_return = p1 / p0 - 1.0, s1 / s0 - 1.0
        paired_primary_returns.append(p_return)
        paired_secondary_returns.append(s_return)
        return_difference_bps.append(abs(p_return - s_return) * 10000.0)
        paired_rows += 1

    fit_metrics = fit.get("absolute_residual_basis_points_of_primary_close", {})
    residual_p95 = fit_metrics.get("p95")
    fit_r2 = fit.get("r_squared_from_level_correlation")
    abs_level_diffs_bps = [
        abs(value) / abs(close) * 10000.0
        for value, close in zip(signed_differences, primary_closes) if close
    ]
    material_level_difference = bool(
        len(common_times) >= 30
        and fit_r2 is not None and fit_r2 >= 0.995
        and residual_p95 is not None and residual_p95 <= 5.0
        and abs_level_diffs_bps
        and statistics.median(abs_level_diffs_bps) > 5.0
    )
    return {
        "sample_size": len(common_times),
        "signed_primary_minus_secondary_close_difference_units": _numeric_summary(signed_differences),
        "level_correlation": level_correlation,
        "ols_level_relationship": fit,
        "signed_level_difference_by_chronological_quartile": quartiles,
        "paired_return_difference": {
            "consecutive_five_minute_pairs": paired_rows,
            "absolute_return_difference_basis_points": _numeric_summary(return_difference_bps),
            "correlation": _correlation(paired_primary_returns, paired_secondary_returns),
            "interpretation": "return differences are measured directly; price-level correlation is not a substitute for return correlation",
        },
        "near_affine_level_difference_candidate": material_level_difference,
        "candidate_rule": "diagnostic flag only when level-fit R-squared >= 0.995, residual p95 <= 5 bps, and median absolute level difference > 5 bps",
        "source_correction_applied": False,
        "source_promoted": False,
        "limitations": [
            "A linear relationship cannot establish quote-side, provider, timezone, rollover or licensing correctness.",
            "The fitted coefficients must not be used to rescale historical prices or backtest results.",
            "Timestamp alignment and source identity need independent verification before interpreting this diagnostic.",
        ],
    }

def _timestamp_offset_sensitivity(
    primary_map: Mapping[datetime, Mapping[str, Any]],
    secondary_map: Mapping[datetime, Mapping[str, Any]],
    *,
    expected_interval_seconds: int,
    minimum_overlap_rows: int,
    max_abs_lag_bars: int,
) -> dict[str, Any]:
    """Scan a fixed small timestamp-offset window for diagnosis only; never alters source rows."""
    results = []
    for offset_bars in range(-max_abs_lag_bars, max_abs_lag_bars + 1):
        delta = timedelta(seconds=offset_bars * expected_interval_seconds)
        shifted_map = {timestamp + delta: row for timestamp, row in secondary_map.items()}
        overlap = sorted(primary_map.keys() & shifted_map.keys())
        close_bps = []
        exact_ohlc = 0
        primary_returns, secondary_returns = [], []
        for ts in overlap:
            left, right = primary_map[ts], shifted_map[ts]
            left_close, right_close = float(left["close"]), float(right["close"])
            close_bps.append(abs(left_close - right_close) / abs(left_close) * 10000 if left_close else math.inf)
            exact_ohlc += int(all(float(left[field]) == float(right[field]) for field in PRICE_FIELDS))
        for previous, current in zip(overlap, overlap[1:]):
            if (current - previous).total_seconds() != expected_interval_seconds:
                continue
            p0, p1 = float(primary_map[previous]["close"]), float(primary_map[current]["close"])
            s0, s1 = float(shifted_map[previous]["close"]), float(shifted_map[current]["close"])
            if p0 > 0 and s0 > 0:
                primary_returns.append(p1 / p0 - 1.0)
                secondary_returns.append(s1 / s0 - 1.0)
        eligible = len(overlap) >= minimum_overlap_rows
        results.append({
            "secondary_timestamp_offset_bars_for_diagnostic_only": offset_bars,
            "secondary_timestamp_offset_seconds_for_diagnostic_only": offset_bars * expected_interval_seconds,
            "overlap_rows": len(overlap),
            "eligible_for_comparison": eligible,
            "median_absolute_close_difference_bps": statistics.median(close_bps) if close_bps else None,
            "p95_absolute_close_difference_bps": _percentile(close_bps, 0.95),
            "one_bar_return_correlation": _correlation(primary_returns, secondary_returns),
            "exact_ohlc_match_count": exact_ohlc,
        })
    eligible_results = [r for r in results if r["eligible_for_comparison"] and r["median_absolute_close_difference_bps"] is not None]
    best = min(eligible_results, key=lambda r: (
        r["median_absolute_close_difference_bps"],
        r["p95_absolute_close_difference_bps"],
        abs(r["secondary_timestamp_offset_bars_for_diagnostic_only"]),
    )) if eligible_results else None
    zero = next((r for r in eligible_results if r["secondary_timestamp_offset_bars_for_diagnostic_only"] == 0), None)
    zero_median = zero["median_absolute_close_difference_bps"] if zero else None
    best_median = best["median_absolute_close_difference_bps"] if best else None
    reduction = (zero_median - best_median) / zero_median if zero_median and best_median is not None else 0.0
    p95_materially_reduced = bool(zero and best and zero.get("p95_absolute_close_difference_bps") and best.get("p95_absolute_close_difference_bps") <= 0.8 * zero["p95_absolute_close_difference_bps"])
    potential_offset = bool(best and zero and best["secondary_timestamp_offset_bars_for_diagnostic_only"] != 0 and reduction >= 0.5 and p95_materially_reduced)
    best_at_boundary = bool(best and abs(best["secondary_timestamp_offset_bars_for_diagnostic_only"]) == max_abs_lag_bars)
    if best is None or zero is None:
        alignment_assessment = "INSUFFICIENT_OVERLAP_FOR_OFFSET_ASSESSMENT"
    elif best_at_boundary:
        # A boundary optimum means the tested domain did not bracket the minimum.
        # Do not turn a finite-window search into an unqualified negative finding.
        alignment_assessment = "INDETERMINATE_SCAN_BOUNDARY"
    elif potential_offset:
        alignment_assessment = "POTENTIAL_ALIGNMENT_MISMATCH_CANDIDATE"
    else:
        alignment_assessment = "NO_STRONG_OFFSET_EVIDENCE_WITHIN_CONFIGURED_SCAN"
    return {
        "method": "sensitivity scan only; original timestamps and the primary exact-time report are unchanged",
        "alignment_assessment": alignment_assessment,
        "zero_offset_median_absolute_close_difference_bps": zero_median,
        "best_candidate_median_absolute_close_difference_bps": best_median,
        "relative_median_discrepancy_reduction_vs_zero_offset": reduction,
        "potential_timestamp_alignment_mismatch": potential_offset,
        "best_candidate_at_scan_boundary": best_at_boundary,
        "scan_boundary_warning": "best candidate is at the configured scan boundary; extend diagnostics or review timestamp semantics before drawing conclusions" if best_at_boundary else None,
        "potential_timestamp_alignment_mismatch_rule": "non-zero offset reduces median absolute close discrepancy by >=50% and p95 by >=20%; exploratory only",
        "max_abs_offset_bars": max_abs_lag_bars,
        "expected_interval_seconds": expected_interval_seconds,
        "offsets_tested_bars": list(range(-max_abs_lag_bars, max_abs_lag_bars + 1)),
        "results": results,
        "best_candidate_offset": best,
        "automatically_shifted_source_timestamps": False,
        "automatically_promoted_source": False,
        "limitations": [
            "Small offset scans cannot prove intended candle-label semantics.",
            "Selecting the minimum discrepancy across offsets is exploratory multiple testing, not a source-correction rule.",
            "A better offset does not establish which feed is correct or executable.",
        ],
    }


def compare_source_datasets(
    primary_rows: Sequence[Mapping[str, Any]],
    secondary_rows: Sequence[Mapping[str, Any]],
    *,
    expected_interval_seconds: int = 300,
    minimum_overlap_rows: int = 1000,
    sample_limit: int = 250,
    max_median_close_difference_bps: float = DEFAULT_MAX_MEDIAN_CLOSE_DIFFERENCE_BPS,
    max_p95_close_difference_bps: float = DEFAULT_MAX_P95_CLOSE_DIFFERENCE_BPS,
    max_abs_timestamp_lag_bars: int = 12,
    naive_timestamp_policy: str = "reject",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not primary_rows or not secondary_rows:
        raise ValueError("both_source_datasets_must_be_nonempty")
    if expected_interval_seconds < 1 or minimum_overlap_rows < 1 or sample_limit < 1:
        raise ValueError("comparison_limits_must_be_positive")
    for name, value in (("max_median_close_difference_bps", max_median_close_difference_bps), ("max_p95_close_difference_bps", max_p95_close_difference_bps)):
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(name + "_must_be_finite_nonnegative")
    if isinstance(max_abs_timestamp_lag_bars, bool) or not isinstance(max_abs_timestamp_lag_bars, int) or not 0 <= max_abs_timestamp_lag_bars <= 24:
        raise ValueError("max_abs_timestamp_lag_bars_must_be_integer_between_0_and_24")
    if naive_timestamp_policy not in {"reject", "assume_utc_for_diagnostic_only"}:
        raise ValueError("naive_timestamp_policy_invalid")

    secondary_map: dict[datetime, Mapping[str, Any]] = {}
    for row in secondary_rows:
        ts = _utc_timestamp(row["timestamp"], naive_timestamp_policy=naive_timestamp_policy)
        if ts in secondary_map:
            raise ValueError("duplicate_timestamp_in_secondary_source")
        secondary_map[ts] = row
    primary_map: dict[datetime, Mapping[str, Any]] = {}
    for row in primary_rows:
        ts = _utc_timestamp(row["timestamp"], naive_timestamp_policy=naive_timestamp_policy)
        if ts in primary_map:
            raise ValueError("duplicate_timestamp_in_primary_source")
        primary_map[ts] = row

    common_times = sorted(primary_map.keys() & secondary_map.keys())
    lag_sensitivity = _timestamp_offset_sensitivity(
        primary_map, secondary_map, expected_interval_seconds=expected_interval_seconds,
        minimum_overlap_rows=minimum_overlap_rows, max_abs_lag_bars=max_abs_timestamp_lag_bars,
    )
    primary_times, secondary_times = set(primary_map), set(secondary_map)
    close_abs_diffs, close_diff_bps, close_signed_diffs = [], [], []
    field_diffs: dict[str, list[float]] = {field: [] for field in PRICE_FIELDS}
    field_signed_diffs: dict[str, list[float]] = {field: [] for field in PRICE_FIELDS}
    exact_rows = 0
    sample: list[dict[str, Any]] = []
    paired_primary_returns, paired_secondary_returns = [], []

    for ts in common_times:
        left, right = primary_map[ts], secondary_map[ts]
        diffs = {field: abs(float(left[field]) - float(right[field])) for field in PRICE_FIELDS}
        for field, diff in diffs.items():
            field_diffs[field].append(diff)
            field_signed_diffs[field].append(float(left[field]) - float(right[field]))
        left_close = float(left["close"])
        close_diff = diffs["close"]
        close_abs_diffs.append(close_diff)
        signed_diff = left_close - float(right["close"])
        close_signed_diffs.append(signed_diff)
        close_diff_bps.append(close_diff / abs(left_close) * 10000 if left_close else math.inf)
        exact_rows += int(all(float(left[field]) == float(right[field]) for field in PRICE_FIELDS))
        sample.append({
            "timestamp_utc_assumed": ts.isoformat(),
            "primary_close": left_close,
            "secondary_close": float(right["close"]),
            "absolute_close_difference": close_diff,
            "absolute_close_difference_bps": close_diff_bps[-1],
            "exact_ohlc_match": all(float(left[field]) == float(right[field]) for field in PRICE_FIELDS),
        })

    for previous, current in zip(common_times, common_times[1:]):
        interval = (current - previous).total_seconds()
        if interval != expected_interval_seconds:
            continue
        p0, p1 = float(primary_map[previous]["close"]), float(primary_map[current]["close"])
        s0, s1 = float(secondary_map[previous]["close"]), float(secondary_map[current]["close"])
        if p0 <= 0 or s0 <= 0:
            continue
        paired_primary_returns.append(p1 / p0 - 1.0)
        paired_secondary_returns.append(s1 / s0 - 1.0)

    if len(sample) > sample_limit:
        positions = sorted({round(i * (len(sample) - 1) / max(1, sample_limit - 1)) for i in range(sample_limit)})
        sample = [sample[index] for index in positions]

    n_common = len(common_times)
    median_close_bps = statistics.median(close_diff_bps) if close_diff_bps else None
    p95_close_bps = _percentile(close_diff_bps, 0.95)
    if n_common < minimum_overlap_rows:
        status = "INSUFFICIENT_OVERLAP"
    elif (median_close_bps is not None and median_close_bps > float(max_median_close_difference_bps)) or (p95_close_bps is not None and p95_close_bps > float(max_p95_close_difference_bps)):
        status = "SOURCE_DISAGREEMENT_REVIEW_REQUIRED"
    else:
        status = "CROSS_SOURCE_DIAGNOSTIC_ONLY"
    report = {
        "status": status,
        "comparison_mode": "exact_timestamp_overlap_only_no_dataset_concatenation",
        "timestamp_interpretation_policy": naive_timestamp_policy,
        "timestamp_semantics_verified": False,
        "utc_assumption_is_diagnostic_only": naive_timestamp_policy == "assume_utc_for_diagnostic_only",
        "timestamp_offset_sensitivity": lag_sensitivity,
        "primary_source": {
            "rows": len(primary_rows),
            "start_utc_assumed": min(primary_map).isoformat(),
            "end_utc_assumed": max(primary_map).isoformat(),
        },
        "secondary_source": {
            "rows": len(secondary_rows),
            "start_utc_assumed": min(secondary_map).isoformat(),
            "end_utc_assumed": max(secondary_map).isoformat(),
        },
        "common_timestamps": n_common,
        "primary_only_timestamps": len(primary_times - secondary_times),
        "secondary_only_timestamps": len(secondary_times - primary_times),
        "primary_overlap_fraction": n_common / len(primary_rows),
        "secondary_overlap_fraction": n_common / len(secondary_rows),
        "timezone_normalization": "naive timestamps treated as UTC for matching only; source claim is not independently verified",
        "volume_comparison": "omitted because volume semantics differ or are unverified across providers",
        "exact_ohlc_match_count": exact_rows,
        "exact_ohlc_match_fraction": exact_rows / n_common if n_common else None,
        "signed_close_difference_units_primary_minus_secondary": {
            "median": statistics.median(close_signed_diffs) if close_signed_diffs else None,
            "p05": _percentile(close_signed_diffs, 0.05),
            "p95": _percentile(close_signed_diffs, 0.95),
            "min": min(close_signed_diffs) if close_signed_diffs else None,
            "max": max(close_signed_diffs) if close_signed_diffs else None,
        },
        "source_disagreement_screen": {
            "max_median_close_difference_bps": float(max_median_close_difference_bps),
            "max_p95_close_difference_bps": float(max_p95_close_difference_bps),
            "observed_median_close_difference_bps": median_close_bps,
            "observed_p95_close_difference_bps": p95_close_bps,
            "material_disagreement_flag": status == "SOURCE_DISAGREEMENT_REVIEW_REQUIRED",
            "thresholds_are_heuristics_not_execution_price_truth": True,
            "interpretation": "material difference requires source/time convention review; never auto-shift, rescale or merge prices",
        },
        "signed_price_difference_units_primary_minus_secondary": {
            field: _numeric_summary(values) for field, values in field_signed_diffs.items()
        },
        "absolute_price_difference_units": {
            field: {
                "median": statistics.median(values) if values else None,
                "p95": _percentile(values, 0.95),
                "p99": _percentile(values, 0.99),
                "max": max(values) if values else None,
            }
            for field, values in field_diffs.items()
        },
        "absolute_close_difference_basis_points": {
            "median": statistics.median(close_diff_bps) if close_diff_bps else None,
            "p95": _percentile(close_diff_bps, 0.95),
            "p99": _percentile(close_diff_bps, 0.99),
            "max": max(close_diff_bps) if close_diff_bps else None,
        },
        "paired_one_bar_returns": len(paired_primary_returns),
        "one_bar_return_correlation": _correlation(paired_primary_returns, paired_secondary_returns),
        "price_basis_diagnostics": _price_basis_diagnostics(
            common_times, primary_map, secondary_map, expected_interval_seconds=expected_interval_seconds,
        ),
        "provider_provenance_independently_verified": False,
        "upstream_data_rights_independently_verified": False,
        "reference_only": True,
        "decision_output": "NO_TRADE",
        "limitations": [
            "Timestamp alignment is not proof that either provider is an authoritative executable quote.",
            "OHLC differences can arise from feed composition, quote side, aggregation, timestamps or genuine source errors.",
            "Repository MIT licensing does not independently settle rights in upstream market data.",
            "Correlation and price differences are diagnostics, not a source-selection rule or predictive edge.",
            "Sources are not concatenated; source-specific backtests remain separate.",
        ],
        "sample_rows": len(sample),
    }
    return report, sample
