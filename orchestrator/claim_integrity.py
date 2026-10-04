from __future__ import annotations

from typing import Any, Mapping

STATUS_STRENGTH = {
    "UNKNOWN": 0,
    "UNSUPPORTED": 1,
    "CONTESTED": 1,
    "SUPPORTED_INDIRECT": 2,
    "SUPPORTED_DIRECT": 3,
}


def validate_truth_lock(
    draft: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    draft_claims = draft.get("claims")
    source_claims = source.get("claims")
    if not isinstance(draft_claims, list):
        return {"passed": False, "reason": "draft claims are missing"}
    if not isinstance(source_claims, list):
        return {"passed": False, "reason": "source claims are missing"}

    source_by_id = {
        str(claim.get("claim_id") or "").strip(): claim
        for claim in source_claims
        if isinstance(claim, Mapping)
        and str(claim.get("claim_id") or "").strip()
    }
    source_evidence = {
        str(ref).strip()
        for claim in source_claims
        if isinstance(claim, Mapping)
        for ref in (claim.get("evidence_refs") or [])
        if str(ref).strip()
    }

    violations: list[dict[str, Any]] = []
    checked = 0
    for index, claim in enumerate(draft_claims):
        if not isinstance(claim, Mapping) or not bool(claim.get("material", True)):
            continue
        checked += 1
        claim_id = str(claim.get("claim_id") or f"draft-{index + 1}").strip()
        lineage = claim.get("derives_from_claims")
        lineage_ids = [
            str(item).strip()
            for item in lineage
            if str(item).strip()
        ] if isinstance(lineage, list) else []
        if not lineage_ids:
            violations.append({
                "claim_id": claim_id,
                "reason": "missing_claim_lineage",
            })
            continue
        if any(item not in source_by_id for item in lineage_ids):
            violations.append({
                "claim_id": claim_id,
                "reason": "unknown_source_claim",
                "derives_from_claims": lineage_ids,
            })
            continue

        draft_status = str(claim.get("status") or "UNKNOWN")
        source_statuses = {
            str(source_by_id[item].get("status") or "UNKNOWN")
            for item in lineage_ids
        }

        weak_source_statuses = {
            status
            for status in source_statuses
            if status in {"CONTESTED", "UNSUPPORTED", "UNKNOWN"}
        }
        if weak_source_statuses and any(
            draft_status != source_status for source_status in weak_source_statuses
        ):
            violations.append({
                "claim_id": claim_id,
                "reason": "uncertainty_upgrade",
                "source_statuses": sorted(source_statuses),
                "draft_status": draft_status,
            })
            continue

        if source_statuses and draft_status in STATUS_STRENGTH:
            min_source_strength = min(
                STATUS_STRENGTH.get(status, 0) for status in source_statuses
            )
            if STATUS_STRENGTH[draft_status] > min_source_strength:
                violations.append({
                    "claim_id": claim_id,
                    "reason": "status_strength_upgrade",
                    "source_statuses": sorted(source_statuses),
                    "draft_status": draft_status,
                })
                continue

        refs = claim.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            violations.append({
                "claim_id": claim_id,
                "reason": "missing_evidence_refs",
            })
            continue
        if any(str(ref).strip() not in source_evidence for ref in refs):
            violations.append({
                "claim_id": claim_id,
                "reason": "evidence_not_in_adjudicated_set",
            })

    return {
        "passed": not violations,
        "checked_material_claims": checked,
        "violations": violations,
    }
