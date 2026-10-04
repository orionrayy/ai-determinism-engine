#!/usr/bin/env python3
"""Deterministic context packing for bounded agent handoffs."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

MAX_CONTEXT_BYTES = 48 * 1024
DEFAULT_DEPENDENCY_BYTES = 6 * 1024
MAX_DEPENDENCIES = 24
MAX_CONTRACT_BYTES = 8 * 1024
MAX_REPAIR_BYTES = 4 * 1024
FINAL_METADATA_RESERVE_BYTES = 256


class ContextBudgetError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def bounded_json(value: Any, max_bytes: int) -> tuple[str, bool]:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    encoded = raw.encode("utf-8")
    if len(encoded) <= max_bytes:
        return raw, False
    clipped = encoded[:max_bytes].decode("utf-8", "ignore")
    return clipped + "...[truncated]", True


def _compact_evidence_output(
    output: Any,
    max_bytes: int,
) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(output, dict) or not isinstance(output.get("evidence_records"), list):
        return None, False

    compact: dict[str, Any] = {
        "query": str(output.get("query") or ""),
        "independent_source_count": int(output.get("independent_source_count") or 0),
        "evidence_records": [],
    }
    if isinstance(output.get("provider_counts"), dict):
        compact["provider_counts"] = {
            str(k): int(v)
            for k, v in sorted(output["provider_counts"].items())
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    if isinstance(output.get("extended_errors"), dict):
        compact["extended_errors"] = {
            str(k): bounded_json(v, 512)[0]
            for k, v in sorted(output["extended_errors"].items())
        }

    truncated = False
    for raw in sorted(
        (item for item in output["evidence_records"] if isinstance(item, dict)),
        key=lambda item: str(item.get("canonical_id") or item.get("title") or ""),
    ):
        record = {
            "canonical_id": str(raw.get("canonical_id") or ""),
            "title": str(raw.get("title") or ""),
            "providers": sorted(str(v) for v in (raw.get("providers") or []) if str(v)),
            "year": raw.get("year"),
            "doi": str(raw.get("doi") or ""),
            "arxiv_id": str(raw.get("arxiv_id") or ""),
            "pmid": str(raw.get("pmid") or ""),
            "pmcid": str(raw.get("pmcid") or ""),
            "venue": str(raw.get("venue") or ""),
            "citation_count": int(raw.get("citation_count") or 0),
            "open_access": bool(raw.get("open_access")),
            "full_text_url": str(raw.get("full_text_url") or ""),
            "primaryity": str(raw.get("primaryity") or "unknown"),
            "authority_signals": sorted(str(v) for v in (raw.get("authority_signals") or []) if str(v)),
        }
        probe = dict(compact)
        probe["evidence_records"] = compact["evidence_records"] + [record]
        if len(canonical_json(probe)) > max_bytes:
            truncated = True
            break
        compact["evidence_records"].append(record)

    if len(canonical_json(compact)) > max_bytes:
        compact["evidence_records"] = []
        compact["evidence_records_truncated"] = True
        truncated = True
    return compact, truncated


def _dependency_record(
    dependency_id: str,
    dependency: Mapping[str, Any],
    *,
    max_output_bytes: int,
) -> dict[str, Any]:
    output = dependency.get("output")
    structured_output, structured_truncated = _compact_evidence_output(
        output,
        max_output_bytes,
    )
    if structured_output is not None:
        output_json = structured_output
        truncated = structured_truncated
    else:
        output_json, truncated = bounded_json(output, max_output_bytes)
    output_sha256 = digest(output)
    evidence_sha256 = dependency.get("evidence_sha256")
    record = {
        "capability": str(dependency.get("capability") or ""),
        "tool": str(dependency.get("tool") or ""),
        "status": str(dependency.get("status") or ""),
        "output": output_json,
        "output_sha256": output_sha256,
        "evidence_sha256": str(evidence_sha256) if evidence_sha256 else None,
    }
    if truncated:
        record["output_truncated"] = True
        record["output_digest_only"] = True
    error = dependency.get("error")
    if error:
        record["error"], record["error_truncated"] = bounded_json(error, 2 * 1024)
    return record


def pack_node_context(
    *,
    goal: Any,
    dependencies: Mapping[str, Any] | None,
    contract: Mapping[str, Any] | None,
    repair_feedback: Mapping[str, Any] | None,
    max_bytes: int = MAX_CONTEXT_BYTES,
    dependency_bytes: int = DEFAULT_DEPENDENCY_BYTES,
    trusted_evidence_records: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if max_bytes < 8 * 1024:
        raise ContextBudgetError("context budget is too small")
    if dependency_bytes < 512:
        raise ContextBudgetError("dependency budget is too small")

    deps = dependencies or {}
    if not isinstance(deps, Mapping):
        raise ContextBudgetError("dependencies must be an object")
    if len(deps) > MAX_DEPENDENCIES:
        raise ContextBudgetError(
            f"dependency count exceeds {MAX_DEPENDENCIES}"
        )

    contract_json, contract_truncated = bounded_json(
        contract or {},
        min(MAX_CONTRACT_BYTES, max_bytes // 4),
    )
    repair_json, repair_truncated = bounded_json(
        repair_feedback or {},
        min(MAX_REPAIR_BYTES, max_bytes // 8),
    )

    effective_max_bytes = max_bytes - FINAL_METADATA_RESERVE_BYTES
    if effective_max_bytes < 8 * 1024:
        raise ContextBudgetError('context budget leaves insufficient metadata reserve')

    packed: dict[str, Any] = {
        "goal": str(goal or ""),
        "dependencies": {},
        "contract": json.loads(contract_json) if not contract_truncated else {
            "truncated": True,
            "sha256": digest(contract or {}),
            "preview": contract_json,
        },
        "repair_feedback": json.loads(repair_json) if not repair_truncated else {
            "truncated": True,
            "sha256": digest(repair_feedback or {}),
            "preview": repair_json,
        },
    }
    trusted_evidence_truncated = False
    if trusted_evidence_records is not None:
        raw_records = sorted(
            (
                item for item in trusted_evidence_records
                if isinstance(item, Mapping)
            ),
            key=lambda item: str(item.get("canonical_id") or item.get("title") or ""),
        )
        evidence = []
        for index, raw in enumerate(raw_records[:64]):
            record = {
                "canonical_id": str(raw.get("canonical_id") or ""),
                "provider": str(raw.get("provider") or ""),
                "provider_id": str(raw.get("provider_id") or ""),
                "title": str(raw.get("title") or "")[:500],
                "year": raw.get("year"),
                "doi": str(raw.get("doi") or "")[:200],
                "arxiv_id": str(raw.get("arxiv_id") or "")[:200],
                "pmid": str(raw.get("pmid") or "")[:100],
                "pmcid": str(raw.get("pmcid") or "")[:100],
                "venue": str(raw.get("venue") or "")[:300],
                "full_text_url": str(raw.get("full_text_url") or "")[:500],
                "authority_class": str(raw.get("authority_class") or "unknown"),
                "authority_score": float(raw.get("authority_score") or 0.0),
                "authority_tier": str(raw.get("authority_tier") or ""),
                "publication_status": str(raw.get("publication_status") or "normal"),
                "independence_key": str(raw.get("independence_key") or ""),
                "independence_confidence": float(raw.get("independence_confidence") or 0.0),
            }
            # Passage validation gets an abstract window only for the first
            # bounded evidence lane; all canonical IDs remain available.
            if index < 32 and raw.get("abstract"):
                record["abstract"] = str(raw.get("abstract") or "")[:700]
            evidence.append(record)
        if len(raw_records) > 64:
            trusted_evidence_truncated = True
        packed["trusted_evidence"] = evidence

    omitted: list[str] = []
    dep_items = {str(key): value for key, value in deps.items()}
    for dep_id in sorted(dep_items):
        candidate = _dependency_record(dep_id, dep_items[dep_id], max_output_bytes=dependency_bytes)
        packed['dependencies'][dep_id] = candidate
        if len(canonical_json(packed)) <= effective_max_bytes:
            continue
        packed['dependencies'].pop(dep_id, None)
        compact = {
            'capability': candidate['capability'], 'tool': candidate['tool'],
            'status': candidate['status'], 'output_sha256': candidate['output_sha256'],
            'evidence_sha256': candidate['evidence_sha256'],
            'omitted': True, 'reason': 'context_budget',
        }
        packed['dependencies'][dep_id] = compact
        if len(canonical_json(packed)) > effective_max_bytes:
            packed['dependencies'].pop(dep_id, None)
            omitted.append(dep_id)

    while len(canonical_json(packed)) > max_bytes and packed['dependencies']:
        ranked = sorted(packed['dependencies'].items(), key=lambda item: (-len(canonical_json(item[1])), item[0]))
        dep_id, dep_value = ranked[0]
        if isinstance(dep_value, dict) and 'output' in dep_value:
            compact = dict(dep_value)
            compact.pop('output', None)
            compact['omitted'] = True
            compact['reason'] = 'final_context_budget'
            packed['dependencies'][dep_id] = compact
        else:
            packed['dependencies'].pop(dep_id, None)
        if dep_id not in omitted:
            omitted.append(dep_id)

    packed['context_budget'] = {
        'max_bytes': max_bytes,
        'used_bytes': len(canonical_json(packed)),
        'dependency_bytes': dependency_bytes,
        'omitted_dependencies': sorted(omitted),
        'truncated_contract': contract_truncated,
        'truncated_repair_feedback': repair_truncated,
        'truncated_trusted_evidence': trusted_evidence_truncated,
    }

    def projected_final_size(value: Mapping[str, Any]) -> int:
        # context_digest is fixed-width (64 hex characters). Use a conservative
        # five-digit used_bytes placeholder because the complete payload is <50 KiB.
        candidate = dict(value)
        budget = dict(candidate.get('context_budget') or {})
        budget['used_bytes'] = max_bytes
        candidate['context_budget'] = budget
        candidate['context_digest'] = '0' * 64
        return len(canonical_json(candidate))

    if projected_final_size(packed) > max_bytes:
        packed['contract'] = {'sha256': digest(contract or {}), 'omitted': True}
        packed['repair_feedback'] = {'sha256': digest(repair_feedback or {}), 'omitted': True}
        packed['context_budget']['truncated_contract'] = contract_truncated
        packed['context_budget']['truncated_repair_feedback'] = repair_truncated

    while projected_final_size(packed) > max_bytes and packed['dependencies']:
        ranked = sorted(
            packed['dependencies'].items(),
            key=lambda item: (-len(canonical_json(item[1])), item[0]),
        )
        dep_id, dep_value = ranked[0]
        if isinstance(dep_value, dict) and 'output' in dep_value:
            compact = dict(dep_value)
            compact.pop('output', None)
            compact['omitted'] = True
            compact['reason'] = 'final_context_budget'
            packed['dependencies'][dep_id] = compact
        else:
            packed['dependencies'].pop(dep_id, None)
        if dep_id not in omitted:
            omitted.append(dep_id)
        packed['context_budget']['omitted_dependencies'] = sorted(omitted)

    if projected_final_size(packed) > max_bytes:
        raise ContextBudgetError('unable to pack context within budget')

    # The digest intentionally excludes the self-referential digest field and the
    # mutable diagnostic byte counter. This makes the provenance digest stable:
    # a verifier removes both fields and hashes the remaining canonical object.
    digestable = dict(packed)
    digest_budget = dict(digestable.get('context_budget') or {})
    digest_budget.pop('used_bytes', None)
    digestable['context_budget'] = digest_budget
    packed['context_digest'] = digest(digestable)

    packed['context_budget']['used_bytes'] = len(canonical_json(packed))
    if len(canonical_json(packed)) > max_bytes:
        raise ContextBudgetError('context digest metadata exceeds budget')
    return packed