#!/usr/bin/env python3
"""Free-first durable execution lease client.

Uses the same authenticated private-input Worker so a workflow can coordinate
workers and persist its global attempt budget without a paid state service.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:
    from .private_input import (
        PrivateInputError,
        _signed_headers,
        private_input_config,
    )
except ImportError:
    from private_input import (
        PrivateInputError,
        _signed_headers,
        private_input_config,
    )


class ExecutionLeaseError(RuntimeError):
    pass


def lease_enabled() -> bool:
    value = os.environ.get("ORCHESTRATOR_EXECUTION_LEASE_ENABLED", "false").strip().lower()
    return value in {"1", "true", "yes", "on", "enabled"}


def lease_subject(workflow: dict[str, Any]) -> str:
    raw = str(workflow.get("execution_id") or workflow.get("id") or "").strip()
    if not raw:
        raise ExecutionLeaseError("lease subject is missing")
    return raw


def lease_owner() -> str:
    owner = os.environ.get("ORCHESTRATOR_LEASE_OWNER", "").strip()
    if not owner:
        run_id = os.environ.get("ORCHESTRATOR_GITHUB_RUN_ID", "").strip()
        attempt = os.environ.get("ORCHESTRATOR_GITHUB_RUN_ATTEMPT", "").strip() or "1"
        owner = f"github:{run_id}:{attempt}" if run_id else f"local:{os.getpid()}"
    if not 1 <= len(owner) <= 128:
        raise ExecutionLeaseError("lease owner is invalid")
    return owner


def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    url, secret = private_input_config()
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    request_url = url + path
    request_path = urllib.parse.urlparse(request_url).path or "/"
    timestamp = int(__import__("time").time())
    request = urllib.request.Request(
        request_url,
        data=encoded,
        headers=_signed_headers(
            method="POST",
            path=request_path,
            timestamp=timestamp,
            body=encoded,
            secret=secret,
        ),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read(32 * 1024)
            status = response.status
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read(32 * 1024).decode("utf-8", "replace"))
        except Exception:
            payload = {}
        error = str(payload.get("error") or f"http_{exc.code}") if isinstance(payload, dict) else f"http_{exc.code}"
        raise ExecutionLeaseError(error) from exc
    except Exception as exc:
        raise ExecutionLeaseError("execution lease transport unavailable") from exc

    if not 200 <= status < 300:
        raise ExecutionLeaseError(f"execution lease returned HTTP {status}")
    try:
        result = json.loads(raw.decode("utf-8", "replace")) if raw else {}
    except json.JSONDecodeError as exc:
        raise ExecutionLeaseError("execution lease returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise ExecutionLeaseError("execution lease response must be an object")
    return result


def acquire_execution_lease(
    workflow: dict[str, Any],
    *,
    ttl_seconds: int = 300,
) -> dict[str, Any]:
    if ttl_seconds < 60 or ttl_seconds > 900:
        raise ExecutionLeaseError("execution lease TTL must be 60..900 seconds")
    result = _post(
        "/v1/leases/acquire",
        {
            "subject": lease_subject(workflow),
            "owner_id": lease_owner(),
            "ttl_seconds": int(ttl_seconds),
        },
    )
    if result.get("ok") is not True:
        if result.get("conflict"):
            raise ExecutionLeaseError("execution lease is held by another worker")
        raise ExecutionLeaseError("execution lease acquisition failed")
    return result


def reserve_remote_execution_attempt(
    workflow: dict[str, Any],
    *,
    count: int,
    max_attempts: int,
) -> dict[str, Any]:
    if count < 1 or count > 16:
        raise ExecutionLeaseError("attempt reservation count must be 1..16")
    if max_attempts < 1 or max_attempts > 256:
        raise ExecutionLeaseError("max attempts must be 1..256")
    result = _post(
        "/v1/leases/reserve-attempt",
        {
            "subject": lease_subject(workflow),
            "owner_id": lease_owner(),
            "count": int(count),
            "max_attempts": int(max_attempts),
        },
    )
    if result.get("ok") is not True:
        if result.get("budget_exhausted"):
            raise ExecutionLeaseError("execution attempt budget exhausted")
        if result.get("lease_lost"):
            raise ExecutionLeaseError("execution lease lost")
        raise ExecutionLeaseError("execution attempt reservation failed")
    return result


def release_execution_lease(workflow: dict[str, Any]) -> bool:
    try:
        result = _post(
            "/v1/leases/release",
            {
                "subject": lease_subject(workflow),
                "owner_id": lease_owner(),
            },
        )
        return result.get("ok") is True
    except (ExecutionLeaseError, PrivateInputError):
        return False
