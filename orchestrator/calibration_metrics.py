from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def _probability(value: Any) -> float:
    try:
        p = float(value)
    except (TypeError, ValueError):
        p = 0.0
    return max(0.0, min(1.0, p))


def _label(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return 1 if float(value) >= 0.5 else 0
    except (TypeError, ValueError):
        return 0


def _samples(
    samples: Iterable[Mapping[str, Any]],
) -> list[tuple[float, int]]:
    result = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        if "confidence" not in sample:
            continue
        if "correct" not in sample:
            continue
        label_source = str(sample.get("label_source") or "").strip().lower()
        if label_source not in {"ground_truth", "benchmark", "human_eval"}:
            continue
        result.append(
            (
                _probability(sample["confidence"]),
                _label(sample["correct"]),
            )
        )
    return result


def brier_score(samples: Iterable[Mapping[str, Any]]) -> float | None:
    values = _samples(samples)
    if not values:
        return None
    return sum((p - y) ** 2 for p, y in values) / len(values)


def reliability_bins(
    samples: Iterable[Mapping[str, Any]],
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    values = _samples(samples)
    bins = max(1, min(int(bins), 100))
    result = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        bucket = [
            (p, y)
            for p, y in values
            if (p >= low and p < high)
            or (index == bins - 1 and p == high)
        ]
        count = len(bucket)
        mean_confidence = (
            sum(p for p, _ in bucket) / count if count else 0.0
        )
        accuracy = (
            sum(y for _, y in bucket) / count if count else 0.0
        )
        result.append(
            {
                "bin": index,
                "lower": round(low, 6),
                "upper": round(high, 6),
                "count": count,
                "mean_confidence": round(mean_confidence, 6),
                "accuracy": round(accuracy, 6),
                "gap": round(abs(mean_confidence - accuracy), 6),
            }
        )
    return result


def expected_calibration_error(
    samples: Iterable[Mapping[str, Any]],
    *,
    bins: int = 10,
) -> float | None:
    values = _samples(samples)
    if not values:
        return None
    total = len(values)
    gaps = reliability_bins(samples, bins=bins)
    return sum(
        (item["count"] / total) * item["gap"]
        for item in gaps
    )


def maximum_calibration_error(
    samples: Iterable[Mapping[str, Any]],
    *,
    bins: int = 10,
) -> float | None:
    values = _samples(samples)
    if not values:
        return None
    gaps = reliability_bins(samples, bins=bins)
    return max(item["gap"] for item in gaps if item["count"]) if any(
        item["count"] for item in gaps
    ) else 0.0


def values_as_mappings(
    values: Sequence[tuple[float, int]],
) -> list[dict[str, Any]]:
    return [
        {"confidence": confidence, "correct": bool(correct)}
        for confidence, correct in values
    ]


def selective_metrics(
    samples: Iterable[Mapping[str, Any]],
    *,
    thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> list[dict[str, Any]]:
    values = _samples(samples)
    if not values:
        return []

    result = []
    for threshold in thresholds:
        t = _probability(threshold)
        selected = [(p, y) for p, y in values if p >= t]
        count = len(selected)
        correct = sum(y for _, y in selected)
        result.append(
            {
                "threshold": round(t, 6),
                "selected": count,
                "coverage": round(count / len(values), 6),
                "accuracy": round(correct / count, 6) if count else None,
                "risk": round(1.0 - (correct / count), 6)
                if count
                else None,
            }
        )
    return result


def calibration_summary(
    samples: Iterable[Mapping[str, Any]],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    values = _samples(samples)
    if not values:
        return {
            "available": False,
            "sample_count": 0,
            "reason": "no_explicit_confidence_outcome_labels",
        }

    materialized = values_as_mappings(values)
    return {
        "available": True,
        "sample_count": len(values),
        "brier_score": round(brier_score(materialized) or 0.0, 6),
        "ece": round(
            expected_calibration_error(materialized, bins=bins) or 0.0,
            6,
        ),
        "mce": round(
            maximum_calibration_error(materialized, bins=bins) or 0.0,
            6,
        ),
        "reliability_bins": reliability_bins(
            materialized,
            bins=bins,
        ),
        "selective": selective_metrics(materialized),
    }


__all__ = [
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_bins",
    "selective_metrics",
    "calibration_summary",
]
