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


def _dependency_record(
    dependency_id: str,
    dependency: Mapping[str, Any],
    *,
    max_output_bytes: int,
) -> dict[str, Any]:
    output = dependency.get("output")
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

    omitted: list[str] = []
    for dep_id in sorted(str(key) for key in deps):
        candidate = _dependency_record(
            dep_id,
            deps[dep_id],
            max_output_bytes=dependency_bytes,
        )
        packed["dependencies"][dep_id] = candidate
        size = len(canonical_json(packed))
        if size <= max_bytes:
            continue

        packed["dependencies"].pop(dep_id, None)
        compact = {
            "capability": candidate["capability"],
            "tool": candidate["tool"],
            "status": candidate["status"],
            "output_sha256": candidate["output_sha256"],
            "evidence_sha256": candidate["evidence_sha256"],
            "omitted": True,
            "reason": "context_budget",
        }
        if len(canonical_json({**packed, "dependencies": {dep_id: compact}})) <= max_bytes:
            packed["dependencies"][dep_id] = compact
        else:
            omitted.append(dep_id)

    packed["context_budget"] = {
        "max_bytes": max_bytes,
        "used_bytes": len(canonical_json(packed)),
        "dependency_bytes": dependency_bytes,
        "omitted_dependencies": omitted,
        "truncated_contract": contract_truncated,
        "truncated_repair_feedback": repair_truncated,
    }
    if len(canonical_json(packed)) > max_bytes:
        # Contract/repair previews can still be too large after metadata.
        packed["contract"] = {"sha256": digest(contract or {}), "omitted": True}
        packed["repair_feedback"] = {
            "sha256": digest(repair_feedback or {}),
            "omitted": True,
        }
        packed["context_budget"]["used_bytes"] = len(canonical_json(packed))
    if len(canonical_json(packed)) > max_bytes:
        raise ContextBudgetError("unable to pack context within budget")
    packed["context_digest"] = digest(packed)
    return packed
