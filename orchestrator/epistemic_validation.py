#!/usr/bin/env python3
"""Deterministic claim/evidence validation primitives."""
# v67+: explicit epistemic boundary for claim-level contracts.
from __future__ import annotations

from typing import Any, Mapping
try:
    from .evidence_independence import independence_summary
    from .selective_evidence_gate import selective_evidence_gate
except ImportError:
    from evidence_independence import independence_summary
    from selective_evidence_gate import selective_evidence_gate


VALID_STATUSES = {
    "SUPPORTED_DIRECT",
    "SUPPORTED_INDIRECT",
    "CONTESTED",
    "UNSUPPORTED",
    "UNKNOWN",
}


def validate_evidence_records(
    evidence_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    seen: set[str] = set()
    invalid_indexes: list[int] = []
    for index, record in enumerate(evidence_records):
        if not isinstance(record, Mapping):
            invalid_indexes.append(index)
            continue
        canonical_id = str(record.get("canonical_id") or "").strip()
        if not canonical_id or canonical_id in seen:
            invalid_indexes.append(index)
            continue
        seen.add(canonical_id)
    return {
        "passed": not invalid_indexes,
        "checked_records": len(evidence_records),
        "invalid_indexes": invalid_indexes,
    }


def validate_claims(
    claims: list[Mapping[str, Any]],
    evidence_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence_ids = {
        str(item.get("canonical_id") or "").strip()
        for item in evidence_records
        if isinstance(item, Mapping) and str(item.get("canonical_id") or "").strip()
    }
    invalid: list[str] = []
    seen_claim_ids: set[str] = set()
    checked = 0
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            invalid.append(f"claim-{index + 1}")
            continue
        claim_id = str(claim.get("claim_id") or f"claim-{index + 1}").strip()
        statement = str(claim.get("statement") or "").strip()
        refs = claim.get("evidence_refs", [])
        status = str(claim.get("status") or "UNKNOWN")
        material = claim.get("material", True)

        if not claim_id or claim_id in seen_claim_ids:
            invalid.append(claim_id or f"claim-{index + 1}")
            continue
        seen_claim_ids.add(claim_id)

        if not statement or not isinstance(material, bool):
            invalid.append(claim_id)
        if not isinstance(refs, list):
            invalid.append(claim_id)
            continue
        if status not in VALID_STATUSES:
            invalid.append(claim_id)
            continue
        if not all(str(ref).strip() in evidence_ids for ref in refs):
            invalid.append(claim_id)
        if status in {"SUPPORTED_DIRECT", "SUPPORTED_INDIRECT", "CONTESTED"} and not refs:
            invalid.append(claim_id)
        checked += 1

    return {
        "passed": not invalid,
        "checked_claims": checked,
        "invalid_claim_ids": sorted(set(invalid)),
    }


def claim_coverage(claims: list[Mapping[str, Any]]) -> dict[str, Any]:
    material = [
        claim for claim in claims
        if isinstance(claim, Mapping) and bool(claim.get("material", True))
    ]
    supported = [
        claim for claim in material
        if str(claim.get("status") or "") in {
            "SUPPORTED_DIRECT",
            "SUPPORTED_INDIRECT",
        }
    ]
    evidence_linked = [
        claim for claim in material
        if isinstance(claim.get("evidence_refs"), list) and bool(claim.get("evidence_refs"))
    ]
    total = len(material)
    return {
        "material_claims": total,
        "supported_material_claims": len(supported),
        "evidence_linked_material_claims": len(evidence_linked),
        # Retained as the semantic "supported claim coverage" metric.
        "coverage": (len(supported) / total) if total else 1.0,
        # Used for minimum coverage because contested/unknown claims can be
        # honest while still being explicitly evidence-linked.
        "evidence_coverage": (len(evidence_linked) / total) if total else 1.0,
        "contested_material_claims": sum(
            1 for claim in material if str(claim.get("status") or "") == "CONTESTED"
        ),
        "unsupported_material_claims": sum(
            1 for claim in material if str(claim.get("status") or "") == "UNSUPPORTED"
        ),
        "unknown_material_claims": sum(
            1 for claim in material if str(claim.get("status") or "") == "UNKNOWN"
        ),
    }


def validate_epistemic_output(
    output: Mapping[str, Any],
    *,
    min_coverage: float | None = None,
) -> dict[str, Any]:
    claims = output.get("claims", [])
    evidence = output.get("evidence_records", [])
    if not isinstance(claims, list):
        return {"passed": False, "reason": "claims must be an array"}
    if not isinstance(evidence, list):
        return {"passed": False, "reason": "evidence_records must be an array"}

    evidence_result = validate_evidence_records(evidence)
    if not evidence_result["passed"]:
        return {
            "passed": False,
            "evidence": evidence_result,
            "reason": "evidence_records are malformed or duplicate",
        }

    claim_result = validate_claims(claims, evidence)
    coverage = claim_coverage(claims)
    threshold_ok = (
        min_coverage is None
        or coverage["evidence_coverage"] >= float(min_coverage)
    )
    independence = independence_summary(
        [
            item for item in evidence
            if isinstance(item, Mapping)
        ]
    )
    base_result = {
        "evidence": evidence_result,
        "claims": claim_result,
        "coverage": coverage,
        "min_coverage": min_coverage,
        "min_coverage_basis": "evidence_coverage",
        "independence": independence,
    }
    selective = selective_evidence_gate(
        output,
        base_result,
    )
    passed = bool(
        evidence_result["passed"]
        and claim_result["passed"]
        and threshold_ok
    )
    return {
        **base_result,
        "passed": passed,
        "selective": selective,
        "selective_abstention": bool(selective.get("abstain")),
    }


__all__ = [
    "VALID_STATUSES",
    "validate_evidence_records",
    "validate_claims",
    "claim_coverage",
    "validate_epistemic_output",
]
