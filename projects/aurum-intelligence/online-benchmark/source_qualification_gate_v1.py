"""Evidence-backed source qualification gates for research and execution use.

A separate URL, a successful download, a permissive repository license label, or
a green software test does not establish independent provenance or lawful rights
in the underlying observations. This module records verified evidence explicitly
and fails closed when a required claim is missing or unreferenced.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

RESEARCH_EVIDENCE_FIELDS = (
    "provenance_verified",
    "intended_use_rights_verified",
    "timestamp_semantics_verified",
    "bar_convention_verified",
    "quote_basis_verified",
)
EXECUTION_EVIDENCE_FIELDS = (
    "broker_authorized",
    "execution_costs_measured",
    "fill_semantics_verified",
)
ALL_EVIDENCE_FIELDS = RESEARCH_EVIDENCE_FIELDS + EXECUTION_EVIDENCE_FIELDS


def _nonempty_reference(value: object) -> bool:
    """Accept stable evidence references, not booleans or implicit source names."""
    if not isinstance(value, str):
        return False
    reference = value.strip()
    return bool(reference) and len(reference) <= 2048 and (
        reference.startswith("https://")
        or reference.startswith("evidence://")
        or reference.startswith("repo://")
    )


def evaluate_source_qualification(
    source_id: str,
    evidence: Mapping[str, object] | None = None,
    evidence_refs: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Evaluate evidence without inferring verification from source claims.

    A field qualifies only when it is literally True and has a stable evidence
    reference. Strings such as "true", truthy numbers and missing values fail
    closed. This evaluates evidence readiness, not model quality or profitability.
    """
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id_required")
    if evidence is not None and not isinstance(evidence, Mapping):
        raise ValueError("evidence_must_be_mapping")
    if evidence_refs is not None and not isinstance(evidence_refs, Mapping):
        raise ValueError("evidence_refs_must_be_mapping")

    evidence = evidence or {}
    evidence_refs = evidence_refs or {}
    unknown_fields = sorted((set(evidence) | set(evidence_refs)) - set(ALL_EVIDENCE_FIELDS))
    if unknown_fields:
        raise ValueError("unknown_evidence_fields:" + ",".join(unknown_fields))

    checks: dict[str, dict[str, object]] = {}
    blockers: list[str] = []
    for field in ALL_EVIDENCE_FIELDS:
        declared = evidence.get(field, False)
        if not isinstance(declared, bool):
            raise ValueError(f"{field}_must_be_boolean")
        reference = evidence_refs.get(field)
        referenced = _nonempty_reference(reference)
        qualified = declared is True and referenced
        checks[field] = {
            "verified": qualified,
            "reference_attached": referenced,
            "evidence_reference": reference.strip() if referenced else None,
        }
        if not declared:
            blockers.append(f"{field}_not_verified")
        elif not referenced:
            blockers.append(f"{field}_evidence_reference_missing")

    research_missing = [
        field for field in RESEARCH_EVIDENCE_FIELDS if not checks[field]["verified"]
    ]
    execution_missing = [
        field for field in EXECUTION_EVIDENCE_FIELDS if not checks[field]["verified"]
    ]
    research_qualified = not research_missing
    execution_qualified = research_qualified and not execution_missing

    return {
        "source_id": source_id.strip(),
        "schema_version": "1.0.0",
        "research_qualification_status": (
            "QUALIFIED_FOR_RESEARCH_REVIEW" if research_qualified else "REFERENCE_ONLY"
        ),
        "execution_qualification_status": (
            "ELIGIBLE_FOR_SEPARATE_EXECUTION_REVIEW" if execution_qualified else "BLOCKED"
        ),
        "research_missing_evidence": research_missing,
        "execution_missing_evidence": execution_missing,
        "checks": checks,
        "source_promotion_allowed": research_qualified,
        "live_execution_authorized": False,
        "decision_output": "NO_TRADE",
        "blockers": blockers,
        "interpretation": (
            "Evidence readiness only; qualification does not establish predictive skill, "
            "calibration, profitability, or production fitness."
        ),
    }
