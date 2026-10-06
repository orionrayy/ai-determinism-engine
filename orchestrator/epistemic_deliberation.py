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


def _evidence_strength(proposal: Mapping[str, Any]) -> float:
    records = proposal.get("evidence_records")
    if not isinstance(records, list):
        return 0.0
    scores = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        authority = max(0.0, min(1.0, float(record.get("authority_score") or 0.0)))
        independence = max(
            0.0,
            min(1.0, float(record.get("independence_confidence") or 0.0)),
        )
        access = 1.0 if str(record.get("access_verification") or "") == "verified" else 0.0
        integrity = 1.0 if str(record.get("publication_status") or "normal") == "normal" else 0.0
        scores.append(
            0.45 * authority
            + 0.25 * independence
            + 0.15 * access
            + 0.15 * integrity
        )
    return max(scores, default=0.0)


def _answer_key(proposal: Mapping[str, Any]) -> str:
    return str(
        proposal.get("answer")
        or proposal.get("result")
        or ""
    ).strip()


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
        _answer_key(item)
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

    answer_groups: dict[str, list[Mapping[str, Any]]] = {}
    for item in proposals:
        answer_groups.setdefault(_answer_key(item), []).append(item)
    strengths = {
        _answer_key(item): max(_evidence_strength(candidate) for candidate in group)
        for item in proposals
        for group in [answer_groups[_answer_key(item)]]
    }
    minority_escalation = False
    minority_answers: list[str] = []
    if len(answers) > 1:
        counts = {
            answer: len(group)
            for answer, group in answer_groups.items()
        }
        majority_count = max(counts.values())
        majority_answers = {
            answer for answer, count in counts.items() if count == majority_count
        }
        majority_strength = max(
            (strengths.get(answer, 0.0) for answer in majority_answers),
            default=0.0,
        )
        for answer, count in counts.items():
            if count >= majority_count:
                continue
            if strengths.get(answer, 0.0) >= max(0.85, majority_strength + 0.15):
                minority_escalation = True
                minority_answers.append(answer)

    if len(proposals) < 2:
        reason = "insufficient_independent_proposals"
        required = True
    elif len(answers) > 1:
        reason = (
            "minority_evidence_escalation"
            if minority_escalation
            else "material_disagreement"
        )
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
        "minimum_proposals_for_consensus": 2,
        "minority_escalation": minority_escalation,
        "minority_answers": sorted(set(minority_answers)),
        "answer_evidence_strength": {
            answer: round(score, 4)
            for answer, score in sorted(strengths.items())
        },
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
