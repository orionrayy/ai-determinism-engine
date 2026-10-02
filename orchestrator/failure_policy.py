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


def decide_retry(
    failure_class: str,
    *,
    explicitly_retryable: bool | None = None,
    uncertain: bool = False,
    side_effect_started: bool = False,
    idempotent: bool = False,
) -> dict[str, Any]:
    """Return the single authoritative retry/recovery decision."""
    normalized = failure_class if failure_class in FAILURE_CLASSES else "permanent"
    is_uncertain = bool(uncertain or normalized == "uncertain")

    if is_uncertain:
        normalized = "uncertain"
        if side_effect_started and idempotent:
            return {
                "failure_class": normalized,
                "retry_allowed": True,
                "uncertain": True,
                "reason": "uncertain_idempotent_side_effect",
            }
        return {
            "failure_class": normalized,
            "retry_allowed": False,
            "uncertain": True,
            "reason": "uncertain_requires_reconciliation",
        }

    allowed = retry_allowed(
        normalized,
        explicitly_retryable=explicitly_retryable,
    )
    if side_effect_started:
        return {
            "failure_class": "uncertain",
            "retry_allowed": bool(idempotent and allowed),
            "uncertain": True,
            "reason": (
                "side_effect_started_idempotent"
                if idempotent and allowed
                else "side_effect_started_requires_reconciliation"
            ),
        }

    return {
        "failure_class": normalized,
        "retry_allowed": allowed,
        "uncertain": False,
        "reason": "failure_class_policy",
    }


def deterministic_retry_delay(
    workflow_id: str,
    node_id: str,
    attempt: int,
    *,
    jitter_seed: str | None = None,
    base_max: float = 8.0,
    jitter_ratio: float = 0.5,
) -> float:
    """Compute reproducible backoff with a wider workflow-scoped jitter window."""
    attempt = max(1, int(attempt))
    base = min(float(2 ** attempt), float(base_max))
    ratio = max(0.0, min(float(jitter_ratio), 1.0))
    seed = str(jitter_seed or workflow_id or "orchestrator")
    digest = hashlib.sha256(
        f"{seed}:{workflow_id}:{node_id}:{attempt}".encode("utf-8")
    ).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    jitter = base * ratio * fraction
    return round(base + jitter, 3)


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
