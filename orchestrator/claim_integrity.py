from __future__ import annotations

from typing import Any, Mapping
import re

STATUS_STRENGTH = {
    "UNKNOWN": 0,
    "UNSUPPORTED": 1,
    "CONTESTED": 1,
    "SUPPORTED_INDIRECT": 2,
    "SUPPORTED_DIRECT": 3,
}

_NEGATION_TOKENS = {
    "no", "not", "never", "without", "none", "neither", "nor",
    "lack", "lacks", "unlikely", "ineffective",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?", re.IGNORECASE)


def _statement_tokens(value: Any) -> list[str]:
    return _TOKEN_RE.findall(str(value or "").lower())


def _statement_similarity(left: Any, right: Any) -> float:
    a = set(_statement_tokens(left))
    b = set(_statement_tokens(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _numeric_tokens(value: Any) -> set[str]:
    return {
        token for token in _statement_tokens(value)
        if any(char.isdigit() for char in token)
    }


def _negation_profile(value: Any) -> set[str]:
    return {
        token for token in _statement_tokens(value)
        if token in _NEGATION_TOKENS
    }


def _validate_statement_preservation(
    draft_statement: Any,
    source_statements: list[Any],
) -> dict[str, Any]:
    candidates = [
        statement for statement in source_statements
        if str(statement or "").strip()
    ]
    if not candidates:
        return {
            "passed": False,
            "reason": "source_statements_missing",
            "similarity": 0.0,
        }

    similarities = [
        _statement_similarity(draft_statement, source)
        for source in candidates
    ]
    best_index = max(range(len(similarities)), key=similarities.__getitem__)
    best_source = candidates[best_index]
    similarity = similarities[best_index]

    source_numbers = set().union(*(_numeric_tokens(source) for source in candidates))
    draft_numbers = _numeric_tokens(draft_statement)
    if source_numbers - draft_numbers:
        return {
            "passed": False,
            "reason": "material_numeric_detail_dropped",
            "similarity": round(similarity, 4),
            "missing_numeric_tokens": sorted(source_numbers - draft_numbers),
        }

    source_negated = bool(
        _negation_profile(best_source)
    )
    draft_negated = bool(
        _negation_profile(draft_statement)
    )
    if source_negated != draft_negated:
        return {
            "passed": False,
            "reason": "negation_polarity_changed",
            "similarity": round(similarity, 4),
        }

    token_count = len(_statement_tokens(best_source))
    threshold = 0.35 if token_count >= 10 else 0.50
    return {
        "passed": similarity >= threshold,
        "reason": (
            "preserved"
            if similarity >= threshold
            else "semantic_overlap_too_low"
        ),
        "similarity": round(similarity, 4),
        "threshold": threshold,
        "matched_source_index": best_index,
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
        source_statements = [
            source_by_id[item].get("statement")
            for item in lineage_ids
        ]
        statement_check = _validate_statement_preservation(
            claim.get("statement"),
            source_statements,
        )
        if statement_check.get("passed") is not True:
            violations.append({
                "claim_id": claim_id,
                "reason": "statement_preservation_failed",
                "statement_check": statement_check,
            })
            continue

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
