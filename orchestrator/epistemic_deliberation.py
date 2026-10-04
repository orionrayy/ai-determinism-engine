#!/usr/bin/env python3
"""Deterministic, conditional deliberation policy."""
from __future__ import annotations

from typing import Any, Mapping
import hashlib
import json

try:
    from .evidence_independence import count_distinct_evidence_works
except ImportError:
    from evidence_independence import count_distinct_evidence_works


DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_STABLE_CONFIDENCE = 0.80
DEFAULT_MIN_INDEPENDENT_SOURCES = 2


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

        records = item.get("evidence_records", [])
        if not isinstance(records, list):
            records = []
        independent_sources = count_distinct_evidence_works(
            [
                record for record in records
                if isinstance(record, Mapping)
            ]
        )
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
    redacted: list[tuple[str, dict[str, Any]]] = []
    identity_fields = {
        "agent_id",
        "agent_role",
        "role",
        "role_instruction",
        "candidate_id",
        "vote_count",
        "votes",
        "majority",
        "consensus_count",
    }
    for item in proposals:
        view = {
            str(key): value
            for key, value in item.items()
            if str(key) not in identity_fields
        }
        raw = json.dumps(
            view,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        redacted.append((hashlib.sha256(raw).hexdigest(), view))

    redacted.sort(key=lambda pair: pair[0])
    result = []
    for index, (_, view) in enumerate(redacted, start=1):
        result.append({
            "candidate_id": f"candidate_{index}",
            **view,
        })
    return result


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_STABLE_CONFIDENCE",
    "DEFAULT_MIN_INDEPENDENT_SOURCES",
    "debate_decision",
    "blind_challenge_view",
]
