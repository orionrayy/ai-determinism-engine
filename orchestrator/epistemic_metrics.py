#!/usr/bin/env python3
"""Small deterministic epistemic/research quality metrics."""
from __future__ import annotations

from typing import Any, Mapping
try:
    from .calibration_metrics import calibration_summary
    from .evidence_independence import independence_summary
except ImportError:
    from calibration_metrics import calibration_summary
    from evidence_independence import independence_summary



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
        "distinct_evidence_work_count_max": 0,
        "independence_proxy_confidence_min": 1.0,
        "calibration": {
            "available": False,
            "sample_count": 0,
            "reason": "no_explicit_confidence_outcome_labels",
        },
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
    evidence_records = (
        [item for item in evidence if isinstance(item, Mapping)]
        if isinstance(evidence, list)
        else []
    )
    independence = independence_summary(evidence_records)
    evidence_count = len(evidence_records)
    independent_count = int(independence["distinct_work_count"])

    calibration_inputs = verdict.get("calibration_samples")
    if not isinstance(calibration_inputs, list):
        calibration_outcome = None
        if "ground_truth_correct" in verdict:
            calibration_outcome = verdict.get("ground_truth_correct")
        calibration_inputs = (
            [{
                "confidence": verdict.get("confidence"),
                "correct": calibration_outcome,
                "label_source": "ground_truth",
            }]
            if calibration_outcome is not None and "confidence" in verdict
            else []
        )
    calibration_inputs = [
        item for item in calibration_inputs
        if isinstance(item, Mapping) and "confidence" in item and "correct" in item
    ][:64]
    calibration = calibration_summary(calibration_inputs)
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
        "distinct_evidence_work_count": independent_count,
        "independence_proxy": True,
        "independence_proxy_confidence": float(
            independence["independence_proxy_confidence"]
        ),
        "calibration": calibration,
        "calibration_samples": calibration_inputs,
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
        evidence_records = (
            [item for item in evidence if isinstance(item, Mapping)]
            if isinstance(evidence, list)
            else []
        )
        evidence_count = len(evidence_records)
        independence = independence_summary(evidence_records)
        independent_count = int(independence["distinct_work_count"])
        summary = {
            "type": "research",
            "research_evidence_record_count": evidence_count,
            "independent_source_count": independent_count,
            "distinct_evidence_work_count": independent_count,
            "independence_proxy": True,
            "independence_proxy_confidence": float(
                independence["independence_proxy_confidence"]
            ),
            "budget": str((research_output.get("research_budget") or {}).get("name") or ""),
            "extended_providers_used": list(research_output.get("extended_providers_used") or [])[:8],
        }
    else:
        return current

    by_node[str(node_id)] = summary
    aggregate = _empty()
    aggregate["by_node"] = by_node
    calibration_samples: list[Mapping[str, Any]] = []
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
            aggregate["distinct_evidence_work_count_max"] = max(
                aggregate["distinct_evidence_work_count_max"],
                int(item.get("distinct_evidence_work_count") or 0),
            )
            samples = item.get("calibration_samples")
            if isinstance(samples, list):
                calibration_samples.extend(
                    sample for sample in samples
                    if isinstance(sample, Mapping)
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
    aggregate["calibration"] = calibration_summary(
        calibration_samples,
    )
    material = aggregate["material_claims"]
    linked = aggregate["evidence_linked_material_claims"]
    supported = aggregate["supported_material_claims"]
    aggregate["supported_coverage"] = (supported / material) if material else 1.0
    aggregate["evidence_coverage"] = (linked / material) if material else 1.0
    workflow["epistemic_metrics"] = aggregate
    return aggregate


__all__ = ["record_node_metrics"]
