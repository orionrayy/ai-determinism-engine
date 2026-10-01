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


ACTION_TYPE_NAMES = {"string", "number", "integer", "boolean", "object", "array"}


def load_routes() -> dict[str, dict[str, Any]]:
    raw = os.environ.get("ORCHESTRATOR_CONNECTOR_ROUTES", "{}")
    try:
        routes = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BridgeRuntimeError("ORCHESTRATOR_CONNECTOR_ROUTES is invalid JSON") from exc
    if not isinstance(routes, dict):
        raise BridgeRuntimeError("connector routes must be an object")
    return routes


def _normalize_action_spec(raw: Any) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    required = raw.get("required", [])
    if not isinstance(required, list):
        required = []
    normalized_required = sorted({str(item).strip() for item in required if str(item).strip()})
    types = raw.get("types", {})
    if not isinstance(types, dict):
        types = {}
    normalized_types = {}
    for key in sorted(types):
        value = str(types[key] or "").strip().lower()
        if str(key).strip() and value in ACTION_TYPE_NAMES:
            normalized_types[str(key).strip()] = value
    return {
        "required": normalized_required,
        "types": normalized_types,
        "idempotent": bool(raw.get("idempotent", False)),
    }


def _resolve_payload_path(payload: Any, path: str) -> tuple[bool, Any]:
    current = payload
    for part in str(path).split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _matches_payload_type(value: Any, type_name: str) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    return False


def validate_action_input(route: dict[str, Any], action: str, value: Any) -> None:
    action_specs = route.get("action_specs", {})
    raw_spec = action_specs.get(action, {}) if isinstance(action_specs, dict) else {}
    spec = _normalize_action_spec(raw_spec)
    if not isinstance(value, dict):
        raise BridgeRuntimeError("connector input must be an object")
    for field_name in spec["required"]:
        found, _ = _resolve_payload_path(value, field_name)
        if not found:
            raise BridgeRuntimeError(f"connector input missing required field: {field_name}")
    for field_name, type_name in spec["types"].items():
        found, field_value = _resolve_payload_path(value, field_name)
        if found and not _matches_payload_type(field_value, type_name):
            raise BridgeRuntimeError(f"connector input field {field_name} must be {type_name}")


def describe_routes(routes: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    routes = load_routes() if routes is None else routes
    described: dict[str, dict[str, Any]] = {}
    for connector in sorted(routes):
        route = routes[connector]
        if not isinstance(route, dict):
            continue
        actions = route.get("actions", [])
        capabilities = route.get("capabilities", [])
        if not isinstance(actions, list):
            actions = []
        if not isinstance(capabilities, list):
            capabilities = []
        secret_env = str(route.get("secret_env") or "").strip()
        configured = bool(str(route.get("url") or "").strip())
        if secret_env:
            configured = configured and bool(os.environ.get(secret_env))
        elif route.get("secret"):
            configured = configured and True
        normalized_actions = sorted(str(item) for item in actions)
        raw_specs = route.get("action_specs", {})
        described[connector] = {
            "actions": normalized_actions,
            "capabilities": sorted(str(item) for item in capabilities),
            "action_specs": {
                action: _normalize_action_spec(
                    raw_specs.get(action, {}) if isinstance(raw_specs, dict) else {}
                )
                for action in normalized_actions
            },
            "risk": str(route.get("risk") or "high"),
            "free_tier": bool(route.get("free_tier", False)),
            "configured": configured,
        }
    return described


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
    validate_action_input(route, action, payload.get("input"))
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
