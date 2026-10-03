#!/usr/bin/env python3
"""Deterministic claim/evidence validation primitives."""
# v67: explicit epistemic boundary for claim-level contracts.
from __future__ import annotations

from typing import Any, Mapping

VALID_STATUSES = {
    "SUPPORTED_DIRECT",
    "SUPPORTED_INDIRECT",
    "CONTESTED",
    "UNSUPPORTED",
    "UNKNOWN",
}


def validate_claims(
    claims: list[Mapping[str, Any]],
    evidence_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence_ids = {
        str(item.get("canonical_id") or "").strip()
        for item in evidence_records
        if str(item.get("canonical_id") or "").strip()
    }
    invalid: list[str] = []
    checked = 0
    for index, claim in enumerate(claims):
        claim_id = str(claim.get("claim_id") or f"claim-{index + 1}")
        refs = claim.get("evidence_refs", [])
        status = str(claim.get("status") or "UNKNOWN")
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
        if bool(claim.get("material", True))
    ]
    supported = [
        claim for claim in material
        if str(claim.get("status") or "") in {
            "SUPPORTED_DIRECT",
            "SUPPORTED_INDIRECT",
        }
    ]
    total = len(material)
    return {
        "material_claims": total,
        "supported_material_claims": len(supported),
        "coverage": (len(supported) / total) if total else 1.0,
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
    claim_result = validate_claims(claims, evidence)
    coverage = claim_coverage(claims)
    threshold_ok = (
        min_coverage is None
        or coverage["coverage"] >= float(min_coverage)
    )
    return {
        "passed": bool(claim_result["passed"] and threshold_ok),
        "claims": claim_result,
        "coverage": coverage,
        "min_coverage": min_coverage,
    }