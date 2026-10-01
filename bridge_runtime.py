#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from typing import Any

PROTOCOL = "ai-orchestrator.connector/v1"
MAX_SKEW_SECONDS = 300
MAX_BODY_BYTES = 64 * 1024
_LOCK = threading.Lock()
_COMPLETED: dict[str, tuple[float, dict[str, Any]]] = {}


class BridgeRuntimeError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign(timestamp: int, body: bytes, secret: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        str(timestamp).encode("utf-8") + b"\n" + body,
        hashlib.sha256,
    ).hexdigest()
    return "sha256=" + digest


def verify_signature(headers: dict[str, str], body: bytes, secret: str, now: int | None = None) -> bool:
    timestamp = headers.get("x-orchestrator-timestamp", "")
    signature = headers.get("x-orchestrator-signature", "")
    try:
        sent_at = int(timestamp)
    except ValueError:
        return False
    current = int(time.time()) if now is None else now
    if abs(current - sent_at) > MAX_SKEW_SECONDS:
        return False
    expected = sign(sent_at, body, secret)
    return hmac.compare_digest(signature, expected)


def load_routes() -> dict[str, dict[str, Any]]:
    raw = os.environ.get("ORCHESTRATOR_CONNECTOR_ROUTES", "{}")
    try:
        routes = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BridgeRuntimeError("ORCHESTRATOR_CONNECTOR_ROUTES is invalid JSON") from exc
    if not isinstance(routes, dict):
        raise BridgeRuntimeError("connector routes must be an object")
    return routes


def validate_envelope(payload: dict[str, Any], routes: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    if payload.get("protocol") != PROTOCOL:
        raise BridgeRuntimeError("unsupported connector protocol")
    request_id = str(payload.get("request_id") or "").strip()
    connector = str(payload.get("connector") or "").strip().lower()
    action = str(payload.get("action") or "").strip().lower()
    if len(request_id) != 64 or any(ch not in "0123456789abcdef" for ch in request_id):
        raise BridgeRuntimeError("invalid request_id")
    route = routes.get(connector)
    if not isinstance(route, dict):
        raise BridgeRuntimeError("connector is not allowlisted")
    allowed = route.get("actions", [])
    if action not in allowed:
        raise BridgeRuntimeError("connector action is not allowlisted")
    return request_id, connector, action


def cleanup_idempotency(now: float) -> None:
    expired = [key for key, (expires, _) in _COMPLETED.items() if expires <= now]
    for key in expired:
        _COMPLETED.pop(key, None)


def cached_result(request_id: str) -> dict[str, Any] | None:
    now = time.time()
    with _LOCK:
        cleanup_idempotency(now)
        item = _COMPLETED.get(request_id)
        return None if item is None else item[1]


def cache_result(request_id: str, result: dict[str, Any], ttl: int = 900) -> None:
    with _LOCK:
        cleanup_idempotency(time.time())
        _COMPLETED[request_id] = (time.time() + ttl, result)


def upstream_secret(route: dict[str, Any]) -> str:
    env_name = str(route.get("secret_env") or "").strip()
    if env_name:
        return os.environ.get(env_name, "")
    return str(route.get("secret") or "")


def dispatch_upstream(route: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    url = str(route.get("url") or "").strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise BridgeRuntimeError("upstream connector URL must use HTTPS")
    secret = upstream_secret(route)
    body = canonical_json(payload)
    timestamp = int(time.time())
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "ai-orchestrator-bridge-runtime/1.0",
        "X-Orchestrator-Protocol": PROTOCOL,
        "X-Orchestrator-Timestamp": str(timestamp),
        "X-Orchestrator-Signature": sign(timestamp, body, secret) if secret else "",
        "Idempotency-Key": str(payload["request_id"]),
    }
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=55) as response:
            raw = response.read().decode("utf-8", "replace")
            status = response.status
    except Exception as exc:
        raise BridgeRuntimeError(f"upstream connector call failed: {exc}") from exc
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        data = {"raw": raw}
    return {"status_code": status, "data": data}


def handle_request(payload: dict[str, Any], shared_secret: str) -> dict[str, Any]:
    body = canonical_json(payload)
    if len(body) > MAX_BODY_BYTES:
        raise BridgeRuntimeError("request exceeds 64 KiB")
    # Signature verification is performed by the HTTP handler using the exact raw body.
    routes = load_routes()
    request_id, connector, action = validate_envelope(payload, routes)

    cached = cached_result(request_id)
    if cached is not None:
        return {**cached, "idempotent_replay": True}

    route = routes[connector]
    result = dispatch_upstream(route, payload)
    response = {
        "ok": True,
        "protocol": PROTOCOL,
        "request_id": request_id,
        "connector": connector,
        "action": action,
        "upstream": result,
    }
    cache_result(request_id, response)
    return response
