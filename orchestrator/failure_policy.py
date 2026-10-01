#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import urllib.error
from typing import Any

FAILURE_CLASSES = {
    "transient",
    "intermittent",
    "dependency",
    "contract",
    "semantic",
    "policy",
    "uncertain",
    "permanent",
}

_NON_RETRYABLE_CLASSES = {"contract", "semantic", "policy", "permanent", "uncertain"}
_TRANSIENT_TEXT = (
    "transient",
    "timeout",
    "timed out",
    "temporarily",
    "temporary",
    "connection reset",
    "connection refused",
    "unreachable",
    "429",
    "502",
    "503",
    "504",
    "rate limit",
)
_DEPENDENCY_TEXT = ("discovery", "upstream", "provider", "service unavailable")
_POLICY_TEXT = ("disabled by", "approval", "forbidden", "not configured", "not allowlisted")
_CONTRACT_TEXT = ("invalid", "missing required", "must be", "expected", "unsupported", "schema")
_SEMANTIC_TEXT = ("semantic validation", "validator rejected", "semantic validator")


def classify_failure(exc: Exception) -> str:
    if getattr(exc, "uncertain", False):
        return "uncertain"

    if isinstance(exc, urllib.error.HTTPError):
        code = int(exc.code)
        if code == 429:
            return "intermittent"
        if code in {408, 425, 500, 502, 503, 504}:
            return "transient"
        if code in {401, 403}:
            return "policy"
        if 400 <= code < 500:
            return "contract"

    if isinstance(exc, (TimeoutError, urllib.error.URLError, ConnectionError)):
        return "transient"

    text = str(exc).strip().lower()
    if any(token in text for token in _SEMANTIC_TEXT):
        return "semantic"
    if any(token in text for token in _POLICY_TEXT):
        return "policy"
    if any(token in text for token in _TRANSIENT_TEXT):
        return "intermittent" if "429" in text or "rate limit" in text else "transient"
    if any(token in text for token in _DEPENDENCY_TEXT):
        return "dependency"
    if any(token in text for token in _CONTRACT_TEXT):
        return "contract"
    return "permanent"


def retry_allowed(
    failure_class: str,
    *,
    explicitly_retryable: bool | None = None,
) -> bool:
    if failure_class in _NON_RETRYABLE_CLASSES:
        return False
    if explicitly_retryable is not None:
        return bool(explicitly_retryable)
    return failure_class in {"transient", "intermittent", "dependency"}


def deterministic_retry_delay(
    workflow_id: str,
    node_id: str,
    attempt: int,
    *,
    base_max: float = 8.0,
) -> float:
    attempt = max(1, int(attempt))
    base = min(float(2 ** attempt), float(base_max))
    digest = hashlib.sha256(
        f"{workflow_id}:{node_id}:{attempt}".encode("utf-8")
    ).digest()
    # 0.000-0.250 seconds: deterministic jitter, not wall-clock randomness.
    jitter = int.from_bytes(digest[:2], "big") % 251
    return base + (jitter / 1000.0)


def describe_failure(exc: Exception, *, uncertain: bool = False) -> dict[str, Any]:
    failure_class = "uncertain" if uncertain else classify_failure(exc)
    return {
        "class": failure_class,
        "retry_allowed": retry_allowed(
            failure_class,
            explicitly_retryable=getattr(exc, "retry_allowed", None),
        ),
        "uncertain": failure_class == "uncertain",
        "message": str(exc),
    }
