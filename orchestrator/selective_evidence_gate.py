from __future__ import annotations

from typing import Any, Mapping


DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_MIN_EVIDENCE_COVERAGE = 0.90
DEFAULT_MIN_SUPPORTED_COVERAGE = 0.80
DEFAULT_MAX_UNRESOLVED_RATE = 0.20
DEFAULT_MIN_INDEPENDENCE_PROXY_CONFIDENCE = 0.50


def _bounded_probability(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def selective_evidence_gate(
    verdict: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    confidence_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    min_evidence_coverage: float = DEFAULT_MIN_EVIDENCE_COVERAGE,
    min_supported_coverage: float = DEFAULT_MIN_SUPPORTED_COVERAGE,
    max_unresolved_rate: float = DEFAULT_MAX_UNRESOLVED_RATE,
    min_independence_proxy_confidence: float = (
        DEFAULT_MIN_INDEPENDENCE_PROXY_CONFIDENCE
    ),
) -> dict[str, Any]:
    confidence = _bounded_probability(verdict.get("confidence"))
    if confidence is None:
        return {
            "available": False,
            "abstain": False,
            "reason": "confidence_missing",
        }

    coverage = validation.get("coverage")
    evidence_coverage = _bounded_probability(
        coverage.get("evidence_coverage")
        if isinstance(coverage, Mapping)
        else None
    )
    supported_coverage = _bounded_probability(
        coverage.get("supported_coverage")
        if isinstance(coverage, Mapping)
        else None
    )

    claims = verdict.get("claims")
    material = [
        claim for claim in claims
        if isinstance(claim, Mapping)
        and bool(claim.get("material", True))
    ] if isinstance(claims, list) else []
    unresolved = sum(
        1
        for claim in material
        if str(claim.get("status") or "") in {
            "CONTESTED",
            "UNSUPPORTED",
            "UNKNOWN",
        }
    )
    unresolved_rate = unresolved / len(material) if material else 0.0

    selective = {
        "available": True,
        "confidence": confidence,
        "confidence_threshold": float(confidence_threshold),
        "evidence_coverage": evidence_coverage,
        "min_evidence_coverage": float(min_evidence_coverage),
        "supported_coverage": supported_coverage,
        "min_supported_coverage": float(min_supported_coverage),
        "unresolved_rate": round(unresolved_rate, 6),
        "max_unresolved_rate": float(max_unresolved_rate),
        "independence_proxy_confidence": None,
        "min_independence_proxy_confidence": float(
            min_independence_proxy_confidence
        ),
        "abstain": False,
        "reason": "not_triggered",
    }

    independence = validation.get("independence")
    if isinstance(independence, Mapping):
        selective["independence_proxy_confidence"] = independence.get(
            "independence_proxy_confidence"
        )
    indep_conf = _bounded_probability(
        selective["independence_proxy_confidence"]
    )

    if confidence < float(confidence_threshold):
        selective["reason"] = "confidence_below_selective_threshold"
        return selective

    failures = []
    if (
        evidence_coverage is None
        or evidence_coverage < float(min_evidence_coverage)
    ):
        failures.append("evidence_coverage")
    if (
        supported_coverage is None
        or supported_coverage < float(min_supported_coverage)
    ):
        failures.append("supported_coverage")
    if unresolved_rate > float(max_unresolved_rate):
        failures.append("unresolved_rate")
    if (
        indep_conf is not None
        and indep_conf < float(min_independence_proxy_confidence)
    ):
        failures.append("independence_proxy_confidence")

    if failures:
        selective["abstain"] = True
        selective["reason"] = "high_confidence_evidence_mismatch"
        selective["failed_checks"] = failures
    else:
        selective["reason"] = "high_confidence_evidence_consistent"

    return selective


__all__ = ["selective_evidence_gate"]
