from __future__ import annotations

from typing import Any, Mapping

try:
    from .epistemic_deliberation import (
        blind_challenge_view,
        debate_decision,
    )
except ImportError:
    from epistemic_deliberation import (
        blind_challenge_view,
        debate_decision,
    )


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def proposal_from_verdict(
    verdict: Mapping[str, Any],
    *,
    agent_id: str,
    evidence_records: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    claims = verdict.get("claims")
    evidence_refs: list[str] = []
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, Mapping):
                continue
            refs = claim.get("evidence_refs")
            if isinstance(refs, list):
                evidence_refs.extend(str(item) for item in refs if str(item).strip())
    direct_refs = verdict.get("evidence_refs")
    if isinstance(direct_refs, list):
        evidence_refs.extend(str(item) for item in direct_refs if str(item).strip())
    evidence_refs = sorted(set(evidence_refs))

    records = evidence_records if isinstance(evidence_records, list) else []
    try:
        independent_count = max(
            0,
            int(
                _first_nonempty(
                    verdict.get("independent_source_count"),
                    len(records),
                )
                or 0
            ),
        )
    except (TypeError, ValueError):
        independent_count = len(records)

    try:
        confidence = float(verdict.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "agent_id": str(agent_id),
        "answer": str(
            _first_nonempty(
                verdict.get("result"),
                verdict.get("answer"),
                verdict.get("summary"),
            )
            or ""
        ),
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence_refs": evidence_refs,
        "independent_source_count": independent_count,
        "claims": claims if isinstance(claims, list) else [],
    }


def deliberation_context(
    proposals: list[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = [
        dict(item)
        for item in proposals
        if isinstance(item, Mapping)
    ]
    decision = debate_decision(normalized)

    # Remove voting/majority signals, then anonymize agent identity for the
    # adjudicator. Candidate ordering is deterministic, but no candidate is
    # privileged by an exposed agent identifier.
    blind = blind_challenge_view(normalized)
    anonymized = []
    for index, item in enumerate(blind, start=1):
        candidate = dict(item)
        candidate.pop("agent_id", None)
        candidate["candidate_id"] = f"candidate_{index}"
        anonymized.append(candidate)

    return {
        "policy_version": 1,
        "decision": decision,
        "candidates": anonymized[:4],
        "candidate_count": len(anonymized),
    }



__all__ = [
    "deliberation_context",
    "proposal_from_verdict",
]
