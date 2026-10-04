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

MAX_EVIDENCE_RECORDS = 64
MAX_CLAIMS = 48
MAX_EVIDENCE_REFS_PER_CLAIM = 32


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


def _normalize_passage(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def validate_claim_passages(
    claims: list[Mapping[str, Any]],
    evidence_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence_by_id = {
        str(record.get("canonical_id") or "").strip(): record
        for record in evidence_records
        if isinstance(record, Mapping)
        and str(record.get("canonical_id") or "").strip()
    }
    invalid: list[str] = []
    checked = 0
    missing = 0

    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        passages = claim.get("evidence_passages")
        if passages is None:
            continue
        if not isinstance(passages, list):
            invalid.append(str(claim.get("claim_id") or "unknown"))
            continue
        for passage in passages[:8]:
            if not isinstance(passage, Mapping):
                invalid.append(str(claim.get("claim_id") or "unknown"))
                continue
            ref = str(
                passage.get("evidence_ref")
                or passage.get("ref")
                or ""
            ).strip()
            text = _normalize_passage(passage.get("text"))
            if not ref or not text or ref not in evidence_by_id:
                invalid.append(str(claim.get("claim_id") or "unknown"))
                continue
            record = evidence_by_id[ref]
            corpus_parts = [
                record.get("abstract"),
                record.get("full_text"),
                record.get("text"),
            ]
            corpus = _normalize_passage(" ".join(
                str(item or "") for item in corpus_parts
            ))
            checked += 1
            if not corpus:
                missing += 1
                continue
            if text not in corpus:
                invalid.append(str(claim.get("claim_id") or "unknown"))

    return {
        "passed": not invalid,
        "checked_passages": checked,
        "missing_corpus": missing,
        "invalid_claim_ids": sorted(set(invalid)),
        "mode": "exact_normalized_substring",
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
        "supported_coverage": (len(supported) / total) if total else 1.0,
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


def _bind_evidence_to_trusted_records(
    output: Mapping[str, Any],
    trusted_evidence_records: list[Mapping[str, Any]] | None,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if trusted_evidence_records is None:
        records = output.get("evidence_records", [])
        return (
            [item for item in records if isinstance(item, Mapping)],
            {"trusted": False, "untrusted_ids": []},
        )

    trusted = {
        str(item.get("canonical_id") or "").strip(): item
        for item in trusted_evidence_records
        if isinstance(item, Mapping) and str(item.get("canonical_id") or "").strip()
    }
    claimed_records = output.get("evidence_records", [])
    claimed_ids = {
        str(item.get("canonical_id") or "").strip()
        for item in claimed_records
        if isinstance(item, Mapping) and str(item.get("canonical_id") or "").strip()
    }
    claim_refs = set()
    claims = output.get("claims", [])
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, Mapping):
                continue
            refs = claim.get("evidence_refs", [])
            if isinstance(refs, list):
                claim_refs.update(
                    str(ref).strip() for ref in refs if str(ref).strip()
                )

    unknown_ids = sorted(
        (claimed_ids | claim_refs) - set(trusted)
    )
    if unknown_ids:
        return [], {
            "trusted": True,
            "untrusted_ids": unknown_ids,
            "passed": False,
        }

    bound_ids = sorted(claimed_ids | claim_refs)
    return [trusted[cid] for cid in bound_ids], {
        "trusted": True,
        "untrusted_ids": [],
        "bound_ids": bound_ids,
        "passed": True,
    }


def validate_epistemic_output(
    output: Mapping[str, Any],
    *,
    min_coverage: float | None = None,
    trusted_evidence_records: list[Mapping[str, Any]] | None = None,
    require_passages: bool = False,
) -> dict[str, Any]:
    claims = output.get("claims", [])
    evidence = output.get("evidence_records", [])
    if not isinstance(claims, list):
        return {"passed": False, "reason": "claims must be an array"}
    if not isinstance(evidence, list):
        return {"passed": False, "reason": "evidence_records must be an array"}
    if len(evidence) > MAX_EVIDENCE_RECORDS:
        return {
            "passed": False,
            "reason": f"evidence_records exceeds limit {MAX_EVIDENCE_RECORDS}",
        }
    if len(claims) > MAX_CLAIMS:
        return {
            "passed": False,
            "reason": f"claims exceeds limit {MAX_CLAIMS}",
        }
    for claim in claims:
        if (
            isinstance(claim, Mapping)
            and isinstance(claim.get("evidence_refs"), list)
            and len(claim.get("evidence_refs") or []) > MAX_EVIDENCE_REFS_PER_CLAIM
        ):
            return {
                "passed": False,
                "reason": (
                    "claim evidence_refs exceeds limit "
                    f"{MAX_EVIDENCE_REFS_PER_CLAIM}"
                ),
            }

    bound_evidence, binding = _bind_evidence_to_trusted_records(
        output,
        trusted_evidence_records,
    )
    if binding.get("trusted") and not binding.get("passed", False):
        return {
            "passed": False,
            "evidence": {
                "passed": False,
                "reason": "evidence_refs_crossed_trust_boundary",
                "untrusted_ids": binding.get("untrusted_ids", []),
            },
            "claims": {"passed": False, "invalid_claim_ids": []},
            "evidence_binding": binding,
            "reason": "LLM attempted to cite evidence outside trusted dependency records",
        }

    evidence = bound_evidence
    evidence_result = validate_evidence_records(evidence)
    if not evidence_result["passed"]:
        return {
            "passed": False,
            "evidence": evidence_result,
            "reason": "evidence_records are malformed or duplicate",
        }

    claim_result = validate_claims(claims, evidence)
    passage_result = validate_claim_passages(claims, evidence)
    passage_required_failure = (
        require_passages
        and any(
            isinstance(claim, Mapping)
            and str(claim.get("status") or "") == "SUPPORTED_DIRECT"
            and bool(claim.get("evidence_refs"))
            and not isinstance(claim.get("evidence_passages"), list)
            for claim in claims
        )
    )
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
        "passage_validation": passage_result,
        "evidence_binding": binding,
        "bound_evidence_records": evidence,
    }
    selective_input = dict(output)
    selective_input["evidence_records"] = evidence
    selective = selective_evidence_gate(
        selective_input,
        base_result,
    )
    passed = bool(
        evidence_result["passed"]
        and claim_result["passed"]
        and passage_result["passed"]
        and not passage_required_failure
        and threshold_ok
        and not bool(selective.get("abstain"))
    )
    return {
        **base_result,
        "passed": passed,
        "selective": selective,
        "selective_abstention": bool(selective.get("abstain")),
        "passage_required": bool(require_passages),
    }


__all__ = [
    "VALID_STATUSES",
    "validate_evidence_records",
    "validate_claims",
    "claim_coverage",
    "validate_epistemic_output",
]
