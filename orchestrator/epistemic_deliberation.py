#!/usr/bin/env python3
"""Deterministic, conditional deliberation policy."""
from __future__ import annotations

from typing import Any, Mapping

try:
    from .evidence_independence import independence_summary
except ImportError:
    from evidence_independence import independence_summary


DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_STABLE_CONFIDENCE = 0.80
DEFAULT_MIN_INDEPENDENT_SOURCES = 2


def _computed_independent_source_count(item: Mapping[str, Any]) -> int:
    evidence = item.get("evidence_records")
    if isinstance(evidence, list):
        return int(
            independence_summary(
                [record for record in evidence if isinstance(record, Mapping)]
            ).get("distinct_work_count")
            or 0
        )
    independence = item.get("independence")
    if isinstance(independence, Mapping):
        try:
            return max(0, int(independence.get("distinct_work_count") or 0))
        except (TypeError, ValueError):
            return 0
    return 0


def debate_decision(
    proposals: list[Mapping[str, Any]],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    stable_confidence: float = DEFAULT_STABLE_CONFIDENCE,
    min_independent_sources: int = DEFAULT_MIN_INDEPENDENT_SOURCES,
) -> dict[str, Any]:
    if not proposals:
        return {
            "required": True,
            "reason": "no_proposals",
            "proposal_count": 0,
            "distinct_answers": 0,
            "min_confidence": 0.0,
            "min_independent_sources": max(0, int(min_independent_sources)),
            "challenge_mode": "blind",
            "max_rounds": 2,
        }

    answers = {
        str(item.get("answer") or item.get("result") or "").strip()
        for item in proposals
    }
    confidences = []
    evidence_missing = False
    evidence_weak = False

    for item in proposals:
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.0
        confidences.append(max(0.0, min(1.0, confidence)))

        refs = item.get("evidence_refs", [])
        if not isinstance(refs, list) or not refs:
            evidence_missing = True
        independent_sources = _computed_independent_source_count(item)
        if independent_sources < max(0, int(min_independent_sources)):
            evidence_weak = True

    if len(answers) > 1:
        reason = "material_disagreement"
        required = True
    elif any(value < confidence_threshold for value in confidences):
        reason = "low_confidence"
        required = True
    elif evidence_missing or evidence_weak:
        reason = "insufficient_evidence"
        required = True
    elif min(confidences) >= stable_confidence:
        reason = "stable_consensus"
        required = False
    else:
        reason = "uncertain_consensus"
        required = True

    return {
        "required": required,
        "reason": reason,
        "proposal_count": len(proposals),
        "distinct_answers": len(answers),
        "min_confidence": min(confidences) if confidences else 0.0,
        "min_independent_sources": max(0, int(min_independent_sources)),
        "challenge_mode": "blind" if required else "none",
        "max_rounds": 2 if required else 0,
    }


def blind_challenge_view(
    proposals: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for item in sorted(proposals, key=lambda value: str(value.get("agent_id") or "")):
        view = {
            str(key): value
            for key, value in item.items()
            if key not in {"vote_count", "votes", "majority", "consensus_count"}
        }
        result.append(view)
    return result


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_STABLE_CONFIDENCE",
    "DEFAULT_MIN_INDEPENDENT_SOURCES",
    "debate_decision",
    "blind_challenge_view",
]
