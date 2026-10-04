from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping
try:
    from .evidence_independence import independence_summary
except ImportError:
    from evidence_independence import independence_summary


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


def _normalized_statement(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _proposal_claims(
    proposals: list[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for proposal_index, proposal in enumerate(proposals):
        claims = proposal.get("claims")
        if not isinstance(claims, list):
            continue
        for index, raw in enumerate(claims):
            if not isinstance(raw, Mapping):
                continue
            claim_id = str(
                raw.get("claim_id") or f"claim-{proposal_index + 1}-{index + 1}"
            ).strip()
            if not claim_id:
                continue
            refs = raw.get("evidence_refs")
            refs = (
                sorted({str(ref).strip() for ref in refs if str(ref).strip()})
                if isinstance(refs, list)
                else []
            )
            groups.setdefault(claim_id, []).append({
                "statement": _normalized_statement(raw.get("statement")),
                "status": str(raw.get("status") or "UNKNOWN"),
                "material": bool(raw.get("material", True)),
                "evidence_refs": refs,
            })
    return groups


def build_claim_challenges(
    proposals: list[Mapping[str, Any]],
    *,
    max_challenges: int = 12,
) -> list[dict[str, Any]]:
    """Build deterministic claim-level challenges without exposing candidate identity."""
    groups = _proposal_claims(proposals)
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for proposal in proposals:
        records = proposal.get("evidence_records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            cid = str(record.get("canonical_id") or "").strip()
            if cid:
                current = evidence_by_id.get(cid)
                if current is None or float(record.get("authority_score") or 0.0) > float(
                    current.get("authority_score") or 0.0
                ):
                    evidence_by_id[cid] = record

    challenges: list[dict[str, Any]] = []
    for claim_id in sorted(groups):
        items = groups[claim_id]
        statements = sorted({item["statement"] for item in items if item["statement"]})
        statuses = sorted({item["status"] for item in items})
        refs = sorted({
            ref
            for item in items
            for ref in item["evidence_refs"]
        })
        material = any(item["material"] for item in items)

        challenge_type = None
        if len(statements) > 1:
            challenge_type = "claim_disagreement"
            question = (
                "Compare the competing statements for this claim. Determine which wording, "
                "if any, is supported by the strongest evidence; otherwise preserve the dispute."
            )
        elif len(statuses) > 1:
            challenge_type = "claim_status_conflict"
            question = (
                "Resolve the conflicting claim statuses only from the cited evidence. "
                "Do not use confidence or candidate agreement as proof."
            )
        elif material and not refs:
            challenge_type = "evidence_gap"
            question = (
                "This material claim lacks cited evidence. Find no implicit support: "
                "either attach valid evidence or mark the claim unresolved."
            )
        elif material and any(
            item["status"] in {"SUPPORTED_DIRECT", "SUPPORTED_INDIRECT"}
            for item in items
        ):
            authoritative = [
                ref for ref in refs
                if float(evidence_by_id.get(ref, {}).get("authority_score") or 0.0) >= 0.65
            ]
            if not authoritative:
                challenge_type = "authority_gap"
                question = (
                    "Check whether the claim has authoritative support. Index presence or "
                    "model assertions alone are insufficient; preserve uncertainty when stronger evidence is absent."
                )

        if challenge_type:
            challenges.append({
                "claim_id": claim_id,
                "challenge_type": challenge_type,
                "question": question,
                "statements": statements[:3],
                "statuses": statuses[:5],
                "evidence_refs": refs[:16],
                "requires_rebuttal": True,
            })

    # A claim occurring in only one independent proposal is an additional coverage
    # challenge when there are multiple candidates: absence is not falsity, but it
    # warrants explicit review before adjudication.
    if len(proposals) > 1:
        for claim_id in sorted(groups):
            if len(groups[claim_id]) == 1 and all(
                item.get("material", True) for item in groups[claim_id]
            ):
                if not any(item["claim_id"] == claim_id for item in challenges):
                    challenges.append({
                        "claim_id": claim_id,
                        "challenge_type": "coverage_gap",
                        "question": (
                            "This material claim appears in only one candidate analysis. "
                            "Check whether the other evidence lanes support, weaken, or leave it unresolved."
                        ),
                        "statements": [
                            groups[claim_id][0]["statement"]
                        ] if groups[claim_id][0]["statement"] else [],
                        "statuses": [groups[claim_id][0]["status"]],
                        "evidence_refs": groups[claim_id][0]["evidence_refs"][:16],
                        "requires_rebuttal": True,
                    })

    return sorted(
        challenges,
        key=lambda item: (
            item["claim_id"],
            item["challenge_type"],
            tuple(item.get("evidence_refs") or []),
        ),
    )[: max(0, int(max_challenges))]


def validate_deliberation_responses(
    verdict: Mapping[str, Any],
    deliberation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(deliberation, Mapping):
        return {"passed": True, "required": False, "missing": []}
    challenges = deliberation.get("challenges", [])
    if not isinstance(challenges, list) or not challenges:
        return {"passed": True, "required": False, "missing": []}

    responses = verdict.get("deliberation_responses")
    if not isinstance(responses, list):
        return {
            "passed": False,
            "required": True,
            "missing": [str(item.get("claim_id") or "") for item in challenges if isinstance(item, Mapping)],
            "reason": "deliberation_responses must be an array",
        }

    expected = {
        str(item.get("claim_id") or "").strip()
        for item in challenges
        if isinstance(item, Mapping) and str(item.get("claim_id") or "").strip()
    }
    seen: set[str] = set()
    invalid: list[str] = []
    allowed_refs = {
        str(item.get("canonical_id") or "").strip()
        for item in (verdict.get("evidence_records") or [])
        if isinstance(item, Mapping) and str(item.get("canonical_id") or "").strip()
    }
    for response in responses:
        if not isinstance(response, Mapping):
            invalid.append("<non-object>")
            continue
        claim_id = str(response.get("claim_id") or "").strip()
        status = str(response.get("status") or "").strip().lower()
        refs = response.get("evidence_refs", [])
        refs = (
            [str(ref).strip() for ref in refs if str(ref).strip()]
            if isinstance(refs, list)
            else []
        )
        if claim_id not in expected or claim_id in seen:
            invalid.append(claim_id or "<missing-claim-id>")
            continue
        seen.add(claim_id)
        if status not in {"resolved", "contested", "unknown"}:
            invalid.append(claim_id)
            continue
        if any(ref not in allowed_refs for ref in refs):
            invalid.append(claim_id)
            continue
        if status in {"resolved", "contested"} and not refs:
            invalid.append(claim_id)

    missing = sorted(expected - seen)
    return {
        "passed": not invalid and not missing,
        "required": True,
        "expected_count": len(expected),
        "response_count": len(seen),
        "missing": missing,
        "invalid": sorted(set(invalid)),
    }


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
    independence = independence_summary(
        [item for item in records if isinstance(item, Mapping)]
    )
    independent_count = int(
        independence.get("distinct_work_count") or 0
    )

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
        "independence": independence,
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
    challenges = build_claim_challenges(normalized)

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

    max_rounds = int(decision.get("max_rounds") or 0)
    return {
        "policy_version": 2,
        "decision": decision,
        "candidates": anonymized[:4],
        "candidate_count": len(anonymized),
        "challenges": challenges,
        "adaptive_rounds": [
            {
                "round": 1,
                "mode": "blind_claim_challenge",
                "challenges": challenges,
            }
        ] + (
            [{
                "round": 2,
                "mode": "rebuttal_and_evidence_recheck",
                "challenges": challenges,
            }]
            if max_rounds >= 2 and challenges
            else []
        ),
        "stop_conditions": [
            "all_material_challenges_resolved",
            "no_authoritative_evidence_can_be_established",
            "new_material_conflict_requires_preservation_of_uncertainty",
        ],
    }



__all__ = [
    "build_claim_challenges",
    "deliberation_context",
    "proposal_from_verdict",
    "validate_deliberation_responses",
]
