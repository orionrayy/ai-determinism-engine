"""Cross-source XAUUSD OHLC comparison without merging unlike feeds.

Naive timestamps from a source whose public README claims UTC are interpreted as UTC only
for matching. That convention is recorded as unverified; no source is promoted for trading.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

PRICE_FIELDS = ("open", "high", "low", "close")


def _utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        value = datetime.fromisoformat(raw)
    if value.tzinfo is None or value.utcoffset() is None:
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


def compare_source_datasets(
    primary_rows: Sequence[Mapping[str, Any]],
    secondary_rows: Sequence[Mapping[str, Any]],
    *,
    expected_interval_seconds: int = 300,
    minimum_overlap_rows: int = 1000,
    sample_limit: int = 250,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not primary_rows or not secondary_rows:
        raise ValueError("both_source_datasets_must_be_nonempty")
    if expected_interval_seconds < 1 or minimum_overlap_rows < 1 or sample_limit < 1:
        raise ValueError("comparison_limits_must_be_positive")

    secondary_map: dict[datetime, Mapping[str, Any]] = {}
    for row in secondary_rows:
        ts = _utc_timestamp(row["timestamp"])
        if ts in secondary_map:
            raise ValueError("duplicate_timestamp_in_secondary_source")
        secondary_map[ts] = row
    primary_map: dict[datetime, Mapping[str, Any]] = {}
    for row in primary_rows:
        ts = _utc_timestamp(row["timestamp"])
        if ts in primary_map:
            raise ValueError("duplicate_timestamp_in_primary_source")
        primary_map[ts] = row

    common_times = sorted(primary_map.keys() & secondary_map.keys())
    primary_times, secondary_times = set(primary_map), set(secondary_map)
    close_abs_diffs, close_diff_bps = [], []
    field_diffs: dict[str, list[float]] = {field: [] for field in PRICE_FIELDS}
    exact_rows = 0
    sample: list[dict[str, Any]] = []
    paired_primary_returns, paired_secondary_returns = [], []

    for ts in common_times:
        left, right = primary_map[ts], secondary_map[ts]
        diffs = {field: abs(float(left[field]) - float(right[field])) for field in PRICE_FIELDS}
        for field, diff in diffs.items():
            field_diffs[field].append(diff)
        left_close = float(left["close"])
        close_diff = diffs["close"]
        close_abs_diffs.append(close_diff)
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
    status = "CROSS_SOURCE_DIAGNOSTIC_ONLY" if n_common >= minimum_overlap_rows else "INSUFFICIENT_OVERLAP"
    report = {
        "status": status,
        "comparison_mode": "exact_timestamp_overlap_only_no_dataset_concatenation",
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
