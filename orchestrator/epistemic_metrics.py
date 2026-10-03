#!/usr/bin/env python3
"""Small deterministic epistemic/research quality metrics."""
from __future__ import annotations

from typing import Any, Mapping


def _empty() -> dict[str, Any]:
    return {
        "epistemic_nodes": 0,
        "research_nodes": 0,
        "material_claims": 0,
        "supported_material_claims": 0,
        "contested_material_claims": 0,
        "unsupported_material_claims": 0,
        "unknown_material_claims": 0,
        "evidence_linked_material_claims": 0,
        "supported_coverage": 1.0,
        "evidence_coverage": 1.0,
        "claim_evidence_ref_count": 0,
        "research_evidence_record_count": 0,
        "independent_source_count_max": 0,
        "by_node": {},
    }


def _summarize_epistemic(verdict: Mapping[str, Any]) -> dict[str, Any]:
    claims = verdict.get("claims") if isinstance(verdict.get("claims"), list) else []
    material = [
        item for item in claims
        if isinstance(item, Mapping) and bool(item.get("material", True))
    ]
    supported = [
        item for item in material
        if str(item.get("status") or "") in {"SUPPORTED_DIRECT", "SUPPORTED_INDIRECT"}
    ]
    contested = [item for item in material if str(item.get("status") or "") == "CONTESTED"]
    unsupported = [item for item in material if str(item.get("status") or "") == "UNSUPPORTED"]
    unknown = [item for item in material if str(item.get("status") or "") == "UNKNOWN"]
    linked = [
        item for item in material
        if isinstance(item.get("evidence_refs"), list) and bool(item.get("evidence_refs"))
    ]
    evidence = verdict.get("evidence_records")
    evidence_count = len(evidence) if isinstance(evidence, list) else 0
    try:
        independent_count = max(0, int(verdict.get("independent_source_count") or 0))
    except (TypeError, ValueError):
        independent_count = 0
    if independent_count == 0:
        independent_count = evidence_count
    total = len(material)
    linked_total = len(linked)
    return {
        "type": "epistemic",
        "material_claims": total,
        "supported_material_claims": len(supported),
        "contested_material_claims": len(contested),
        "unsupported_material_claims": len(unsupported),
        "unknown_material_claims": len(unknown),
        "evidence_linked_material_claims": linked_total,
        "supported_coverage": (len(supported) / total) if total else 1.0,
        "evidence_coverage": (linked_total / total) if total else 1.0,
        "claim_evidence_ref_count": sum(
            len(item.get("evidence_refs") or [])
            for item in claims
            if isinstance(item, Mapping) and isinstance(item.get("evidence_refs"), list)
        ),
        "research_evidence_record_count": evidence_count,
        "independent_source_count": independent_count,
    }


def record_node_metrics(
    workflow: dict[str, Any],
    node_id: str,
    *,
    verdict: Mapping[str, Any] | None = None,
    research_output: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = dict(workflow.get("epistemic_metrics") or {})
    by_node = dict(current.get("by_node") or {})
    if verdict is not None:
        summary = _summarize_epistemic(verdict)
    elif research_output is not None:
        evidence = research_output.get("evidence_records")
        evidence_count = len(evidence) if isinstance(evidence, list) else 0
        try:
            independent_count = max(0, int(research_output.get("independent_source_count") or 0))
        except (TypeError, ValueError):
            independent_count = 0
        summary = {
            "type": "research",
            "research_evidence_record_count": evidence_count,
            "independent_source_count": independent_count,
            "budget": str((research_output.get("research_budget") or {}).get("name") or ""),
            "extended_providers_used": list(research_output.get("extended_providers_used") or [])[:8],
        }
    else:
        return current

    by_node[str(node_id)] = summary
    aggregate = _empty()
    aggregate["by_node"] = by_node
    for item in by_node.values():
        if item.get("type") == "epistemic":
            aggregate["epistemic_nodes"] += 1
            for key in (
                "material_claims", "supported_material_claims",
                "contested_material_claims", "unsupported_material_claims",
                "unknown_material_claims", "evidence_linked_material_claims",
                "claim_evidence_ref_count", "research_evidence_record_count",
            ):
                aggregate[key] += int(item.get(key) or 0)
            aggregate["independent_source_count_max"] = max(
                aggregate["independent_source_count_max"],
                int(item.get("independent_source_count") or 0),
            )
        elif item.get("type") == "research":
            aggregate["research_nodes"] += 1
            aggregate["research_evidence_record_count"] += int(
                item.get("research_evidence_record_count") or 0
            )
            aggregate["independent_source_count_max"] = max(
                aggregate["independent_source_count_max"],
                int(item.get("independent_source_count") or 0),
            )
    material = aggregate["material_claims"]
    linked = aggregate["evidence_linked_material_claims"]
    supported = aggregate["supported_material_claims"]
    aggregate["supported_coverage"] = (supported / material) if material else 1.0
    aggregate["evidence_coverage"] = (linked / material) if material else 1.0
    workflow["epistemic_metrics"] = aggregate
    return aggregate


__all__ = ["record_node_metrics"]
