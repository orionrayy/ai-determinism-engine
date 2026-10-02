from __future__ import annotations

import hashlib
import json
import re
from typing import Any

MAX_SUMMARY_BYTES = 16 * 1024
MAX_DURABLE_DEPTH = 8
MAX_DURABLE_ITEMS = 128
MAX_DURABLE_STRING_BYTES = 4096
DIAGNOSTIC_TRACE_KEY_RE = re.compile(
    r"^(?:trace|traceback|stack|stack_trace)$",
    re.IGNORECASE,
)
SECRET_VALUE_RE = re.compile(
    r"""(?ix)
    (?:
        bearer\s+[^\s,;]+
        |(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|
           password|passwd|secret|client[_-]?secret|credential)\s*[:=]\s*
           ['"]?[^\s,;'""]+
    )
    """
)
SENSITIVE_KEY_RE = re.compile(
    r"(?:authorization|password|passwd|secret|token|api[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|cookie|set-cookie|private[_-]?key|"
    r"credential|client[_-]?secret)",
    re.IGNORECASE,
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sanitize_for_durable(
    value: Any,
    *,
    _depth: int = 0,
) -> Any:
    """Return a bounded, secret-redacted representation for durable storage."""
    if _depth > MAX_DURABLE_DEPTH:
        return {"redacted": True, "reason": "max_depth"}

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        items = list(value.items())
        for index, (raw_key, raw_value) in enumerate(items):
            if index >= MAX_DURABLE_ITEMS:
                result["_truncated_items"] = len(items) - MAX_DURABLE_ITEMS
                break
            key = str(raw_key)
            if DIAGNOSTIC_TRACE_KEY_RE.fullmatch(key):
                result[key] = {
                    "redacted": True,
                    "reason": "diagnostic_trace",
                    "sha256": sha256_value(raw_value),
                }
                continue
            if SENSITIVE_KEY_RE.search(key):
                result[key] = {
                    "redacted": True,
                    "sha256": sha256_value(raw_value),
                }
                continue
            result[key] = sanitize_for_durable(
                raw_value,
                _depth=_depth + 1,
            )
        return result

    if isinstance(value, list):
        result = [
            sanitize_for_durable(item, _depth=_depth + 1)
            for item in value[:MAX_DURABLE_ITEMS]
        ]
        if len(value) > MAX_DURABLE_ITEMS:
            result.append({
                "_truncated_items": len(value) - MAX_DURABLE_ITEMS,
            })
        return result

    if isinstance(value, str):
        value = SECRET_VALUE_RE.sub(
            lambda match: "[REDACTED]",
            value,
        )
        encoded = value.encode("utf-8")
        if len(encoded) <= MAX_DURABLE_STRING_BYTES:
            return value
        clipped = encoded[:MAX_DURABLE_STRING_BYTES].decode(
            "utf-8",
            "ignore",
        )
        return clipped + "...[truncated]"

    if isinstance(value, (bytes, bytearray)):
        return {
            "redacted": True,
            "reason": "binary",
            "sha256": sha256_value(value.hex()),
        }

    if value is None or isinstance(value, (bool, int, float)):
        return value

    return str(value)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bounded_summary(value: Any, limit: int = MAX_SUMMARY_BYTES) -> str:
    raw = canonical_json(sanitize_for_durable(value))
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
