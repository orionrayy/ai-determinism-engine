from __future__ import annotations

import hashlib
import json
from typing import Any

MAX_SUMMARY_BYTES = 16 * 1024


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bounded_summary(value: Any, limit: int = MAX_SUMMARY_BYTES) -> str:
    raw = canonical_json(value)
    if len(raw.encode("utf-8")) <= limit:
        return raw
    clipped = raw.encode("utf-8")[:limit].decode("utf-8", "ignore")
    return clipped + "...[truncated]"


def build_evidence(
    workflow_id: str,
    node_id: str,
    capability: str,
    tool: str,
    output: Any,
    validation: Any,
    artifacts: Any | None = None,
) -> dict[str, Any]:
    payload = {
        "workflow_id": workflow_id,
        "node_id": node_id,
        "capability": capability,
        "tool": tool,
        "output_sha256": sha256_value(output),
        "output_summary": bounded_summary(output),
        "validation": validation,
        "artifacts": artifacts or [],
    }
    payload["evidence_sha256"] = sha256_value(payload)
    return payload
