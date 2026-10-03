#!/usr/bin/env python3
"""Deterministic, conditional deliberation policy."""
from __future__ import annotations

from typing import Any, Mapping


DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_STABLE_CONFIDENCE = 0.80


def debate_decision(
    proposals: list[Mapping[str, Any]],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    stable_confidence: float = DEFAULT_STABLE_CONFIDENCE,
) -> dict[str, Any]:
    if not proposals:
        return {
            "required": True,
            "reason": "no_proposals",
            "max_rounds": 2,
        }

    answers = {
        str(item.get("answer") or item.get("result") or "").strip()
        for item in proposals
    }
    confidences = []
    evidence_missing = False
    for item in proposals:
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        confidences.append(confidence)
        refs = item.get("evidence_refs", [])
        if not isinstance(refs, list) or not refs:
            evidence_missing = True

    if len(answers) > 1:
        reason = "material_disagreement"
        required = True
    elif any(value < confidence_threshold for value in confidences):
        reason = "low_confidence"
        required = True
    elif evidence_missing:
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
