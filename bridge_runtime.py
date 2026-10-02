#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

PROTOCOL = "ai-orchestrator.connector/v1"
MAX_SKEW_SECONDS = 300
MAX_BODY_BYTES = 64 * 1024
MAX_UPSTREAM_RESPONSE_BYTES = 128 * 1024
IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60
MAX_COMPLETED_ENTRIES = 128
_LOCK = threading.Lock()
# request_id -> (expires_at, semantic_request_digest, cached_response)
_COMPLETED: dict[str, tuple[float, str, dict[str, Any]]] = {}


class BridgeRuntimeError(RuntimeError):
    pass


class BridgeUpstreamError(BridgeRuntimeError):
    def __init__(self, message: str, *, status_code: int = 503, uncertain: bool = True) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.uncertain = bool(uncertain)


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


def _normalize_action_spec(
    raw: Any,
    *,
    default_free_tier: bool = False,
) -> dict[str, Any]:
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
        "free_tier": bool(raw.get("free_tier", default_free_tier)),
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
    spec = _normalize_action_spec(
        raw_spec,
        default_free_tier=bool(route.get("free_tier", False)),
    )
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
                    raw_specs.get(action, {}) if isinstance(raw_specs, dict) else {},
                    default_free_tier=bool(route.get("free_tier", False)),
                )
                for action in normalized_actions
            },
            "risk": str(route.get("risk") or "high"),
            "free_tier": bool(route.get("free_tier", False)),
            "configured": configured,
            "reconciliation": bool(str(route.get("reconciliation_url") or "").strip()),
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


def semantic_request_digest(payload: dict[str, Any]) -> str:
    semantic = {
        "protocol": payload.get("protocol"),
        "workflow_id": payload.get("workflow_id"),
        "node_id": payload.get("node_id"),
        "connector": str(payload.get("connector") or "").strip().lower(),
        "action": str(payload.get("action") or "").strip().lower(),
        "input": payload.get("input", {}),
    }
    return hashlib.sha256(canonical_json(semantic)).hexdigest()


def cached_result(request_id: str, semantic_digest: str) -> dict[str, Any] | None:
    now = time.time()
    with _LOCK:
        cleanup_idempotency(now)
        item = _COMPLETED.get(request_id)
        if item is None:
            return None
        if item[1] != semantic_digest:
            raise BridgeRuntimeError(
                "idempotency key conflicts with an existing request payload"
            )
        return item[2]


def cache_result(
    request_id: str,
    semantic_digest: str,
    result: dict[str, Any],
    ttl: int = IDEMPOTENCY_TTL_SECONDS,
) -> None:
    with _LOCK:
        cleanup_idempotency(time.time())
        while len(_COMPLETED) >= MAX_COMPLETED_ENTRIES:
            _COMPLETED.pop(next(iter(_COMPLETED)))
        _COMPLETED[request_id] = (time.time() + ttl, semantic_digest, result)


def reconciliation_url(route: dict[str, Any]) -> str:
    value = str(route.get("reconciliation_url") or "").strip()
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "https":
        raise BridgeRuntimeError("reconciliation upstream URL must use HTTPS")
    return value


def reconciliation_secret(route: dict[str, Any]) -> str:
    env_name = str(
        route.get("reconciliation_secret_env")
        or route.get("secret_env")
        or ""
    ).strip()
    if env_name:
        return os.environ.get(env_name, "")
    return str(route.get("secret") or "")


def dispatch_reconciliation(
    route: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    url = reconciliation_url(route)
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query.update({
        "request_id": str(payload["request_id"]),
        "connector": str(payload["connector"]),
        "action": str(payload["action"]),
    })
    url = urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, "", urllib.parse.urlencode(query), "")
    )
    secret = reconciliation_secret(route)
    timestamp = int(time.time())
    headers = {
        "Accept": "application/json",
        "User-Agent": "ai-orchestrator-bridge-reconciliation/1.0",
        "X-Orchestrator-Protocol": PROTOCOL,
        "X-Orchestrator-Timestamp": str(timestamp),
        "X-Orchestrator-Signature": sign(timestamp, b"", secret) if secret else "",
        "Idempotency-Key": str(payload["request_id"]),
    }
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read().decode("utf-8", "replace")
            status = response.status
    except Exception as exc:
        raise BridgeRuntimeError(f"reconciliation upstream call failed: {exc}") from exc
    if not (200 <= status < 300):
        raise BridgeRuntimeError(f"reconciliation upstream returned HTTP {status}")
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise BridgeRuntimeError("reconciliation upstream returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise BridgeRuntimeError("reconciliation upstream response must be an object")
    state = str(data.get("state") or "").strip().lower()
    if state not in {"applied", "not_applied", "unknown"}:
        raise BridgeRuntimeError("reconciliation upstream returned invalid state")
    return {"status_code": status, "state": state, "data": data}


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
            raw_bytes = response.read(MAX_UPSTREAM_RESPONSE_BYTES + 1)
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raise BridgeUpstreamError(
            f"upstream connector returned HTTP {exc.code}",
            status_code=502 if exc.code >= 500 else 424,
            uncertain=exc.code >= 500,
        ) from exc
    except Exception as exc:
        raise BridgeUpstreamError(
            f"upstream connector call failed: {exc}",
            status_code=503,
            uncertain=True,
        ) from exc
    if len(raw_bytes) > MAX_UPSTREAM_RESPONSE_BYTES:
        raise BridgeUpstreamError(
            "upstream connector response exceeds 128 KiB",
            status_code=502,
            uncertain=True,
        )
    if not (200 <= status < 300):
        raise BridgeUpstreamError(
            f"upstream connector returned HTTP {status}",
            status_code=502 if status >= 500 else 424,
            uncertain=status >= 500,
        )
    raw = raw_bytes.decode("utf-8", "replace")
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        data = {"raw": raw}
    return {"status_code": status, "data": data}


def handle_reconciliation(payload: dict[str, Any]) -> dict[str, Any]:
    routes = load_routes()
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
    if not str(route.get("reconciliation_url") or "").strip():
        raise BridgeRuntimeError("connector reconciliation is not configured")
    if action not in route.get("actions", []):
        raise BridgeRuntimeError("connector action is not allowlisted")
    raw_specs = route.get("action_specs", {})
    action_spec = _normalize_action_spec(
        raw_specs.get(action, {}) if isinstance(raw_specs, dict) else {},
        default_free_tier=bool(route.get("free_tier", False)),
    )
    if (
        os.environ.get("ORCHESTRATOR_FREE_ONLY", "true").lower() == "true"
        and (
            route.get("free_tier") is not True
            or action_spec.get("free_tier") is not True
        )
    ):
        raise BridgeRuntimeError(
            "connector reconciliation is not certified for free-only execution"
        )
    result = dispatch_reconciliation(
        route,
        {
            "request_id": request_id,
            "connector": connector,
            "action": action,
        },
    )
    return {
        "ok": True,
        "protocol": PROTOCOL,
        "request_id": request_id,
        "connector": connector,
        "action": action,
        "state": result["state"],
        "upstream": result,
    }


def handle_request(payload: dict[str, Any], shared_secret: str) -> dict[str, Any]:
    body = canonical_json(payload)
    if len(body) > MAX_BODY_BYTES:
        raise BridgeRuntimeError("request exceeds 64 KiB")
    # Signature verification is performed by the HTTP handler using the exact raw body.
    routes = load_routes()
    request_id, connector, action = validate_envelope(payload, routes)

    route = routes[connector]
    validate_action_input(route, action, payload.get("input"))
    raw_specs = route.get("action_specs", {})
    action_spec = _normalize_action_spec(
        raw_specs.get(action, {}) if isinstance(raw_specs, dict) else {},
        default_free_tier=bool(route.get("free_tier", False)),
    )
    if (
        os.environ.get("ORCHESTRATOR_FREE_ONLY", "true").lower() == "true"
        and (
            route.get("free_tier") is not True
            or action_spec.get("free_tier") is not True
        )
    ):
        raise BridgeRuntimeError(
            "connector action is not certified for free-only execution"
        )
    semantic_digest = semantic_request_digest(payload)
    cached = cached_result(request_id, semantic_digest)
    if cached is not None:
        return {**cached, "idempotent_replay": True}
    result = dispatch_upstream(route, payload)
    response = {
        "ok": True,
        "protocol": PROTOCOL,
        "request_id": request_id,
        "connector": connector,
        "action": action,
        "upstream": result,
    }
    cache_result(request_id, semantic_digest, response)
    return response
