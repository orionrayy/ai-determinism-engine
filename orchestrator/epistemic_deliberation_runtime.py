from __future__ import annotations

import hashlib
import json
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

    evidence_cards = []
    for record in records[:8]:
        if not isinstance(record, Mapping):
            continue
        card = {
            "canonical_id": str(
                _first_nonempty(
                    record.get("canonical_id"),
                    record.get("id"),
                )
                or ""
            ),
            "title": str(record.get("title") or "")[:500],
            "provider": str(record.get("provider") or "")[:120],
            "venue": str(record.get("venue") or "")[:200],
            "year": record.get("year"),
            "doi": str(record.get("doi") or "")[:200],
            "url": str(
                _first_nonempty(
                    record.get("url"),
                    record.get("landing_url"),
                )
                or ""
            )[:500],
            "authority_signals": record.get("authority_signals", {}),
        }
        evidence_cards.append(card)

    return {
        "agent_id": str(agent_id),
        "answer": str(
            _first_nonempty(
                verdict.get("result"),
                verdict.get("answer"),
                verdict.get("summary"),
            )
            or ""
        )[:6000],
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence_refs": evidence_refs[:32],
        "independent_source_count": independent_count,
        "claims": claims if isinstance(claims, list) else [],
        "evidence_cards": evidence_cards,
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
    blind_inputs = []
    for item in normalized:
        candidate = dict(item)
        real_agent = str(candidate.pop("agent_id", "") or "")
        canonical = json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        candidate["agent_id"] = hashlib.sha256(canonical).hexdigest()
        blind_inputs.append(candidate)

    blind = blind_challenge_view(blind_inputs)
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
