#!/usr/bin/env python3
from __future__ import annotations

import copy
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
_LOCK = threading.Lock()
_IDEMPOTENCY_CONDITION = threading.Condition(_LOCK)
_COMPLETED: dict[str, tuple[float, str, dict[str, Any]]] = {}
_INFLIGHT: dict[str, str] = {}


class BridgeRuntimeError(RuntimeError):
    pass


class UpstreamConnectorError(BridgeRuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        uncertain: bool = True,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.uncertain = uncertain


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


def route_target_fingerprint(url: str) -> str | None:
    value = str(url or "").strip()
    if not value:
        return None
    parsed = urllib.parse.urlsplit(value)
    normalized = urllib.parse.urlunsplit((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path or "/",
        parsed.query,
        "",
    ))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
            "target_fingerprint": route_target_fingerprint(str(route.get("url") or "")),
            "action_specs": {
                action: _normalize_action_spec(
                    raw_specs.get(action, {}) if isinstance(raw_specs, dict) else {}
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


def request_intent_fingerprint(
    payload: dict[str, Any],
    target_fingerprint: str | None = None,
) -> str:
    intent = {
        "protocol": payload.get("protocol"),
        "workflow_id": payload.get("workflow_id"),
        "node_id": payload.get("node_id"),
        "connector": payload.get("connector"),
        "action": payload.get("action"),
        "input": payload.get("input"),
        "target_fingerprint": target_fingerprint,
    }
    return hashlib.sha256(canonical_json(intent)).hexdigest()


def cleanup_idempotency(now: float) -> None:
    expired = [key for key, (expires, _, _) in _COMPLETED.items() if expires <= now]
    for key in expired:
        _COMPLETED.pop(key, None)


def acquire_idempotency_slot(
    request_id: str,
    intent_fingerprint: str,
) -> dict[str, Any] | None:
    """Return a cached response or claim exclusive execution for this request intent."""
    with _IDEMPOTENCY_CONDITION:
        while True:
            cleanup_idempotency(time.time())
            item = _COMPLETED.get(request_id)
            if item is not None:
                _, stored_fingerprint, result = item
                if stored_fingerprint != intent_fingerprint:
                    raise BridgeRuntimeError(
                        "idempotency key conflicts with request intent"
                    )
                return copy.deepcopy(result)

            inflight = _INFLIGHT.get(request_id)
            if inflight is None:
                _INFLIGHT[request_id] = intent_fingerprint
                return None
            if inflight != intent_fingerprint:
                raise BridgeRuntimeError(
                    "idempotency key conflicts with in-flight request intent"
                )
            _IDEMPOTENCY_CONDITION.wait()


def cache_result(
    request_id: str,
    intent_fingerprint: str,
    result: dict[str, Any],
    ttl: int = 900,
) -> None:
    with _IDEMPOTENCY_CONDITION:
        cleanup_idempotency(time.time())
        _COMPLETED[request_id] = (
            time.time() + ttl,
            intent_fingerprint,
            copy.deepcopy(result),
        )
        _INFLIGHT.pop(request_id, None)
        _IDEMPOTENCY_CONDITION.notify_all()


def release_idempotency_slot(
    request_id: str,
    intent_fingerprint: str,
) -> None:
    with _IDEMPOTENCY_CONDITION:
        if _INFLIGHT.get(request_id) == intent_fingerprint:
            _INFLIGHT.pop(request_id, None)
            _IDEMPOTENCY_CONDITION.notify_all()


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
            raw = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        raise UpstreamConnectorError(
            f"upstream connector call failed: HTTP {exc.code}",
            status_code=exc.code,
            uncertain=exc.code >= 500,
        ) from exc
    except Exception as exc:
        raise UpstreamConnectorError(
            f"upstream connector call failed: {exc}",
            uncertain=True,
        ) from exc
    if not (200 <= status < 300):
        raise UpstreamConnectorError(
            f"upstream connector returned HTTP {status}",
            status_code=status,
            uncertain=status >= 500,
        )
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
    intent_fingerprint = request_intent_fingerprint(
        payload,
        route_target_fingerprint(str(route.get("url") or "")),
    )

    cached = acquire_idempotency_slot(request_id, intent_fingerprint)
    if cached is not None:
        return {**cached, "idempotent_replay": True}

    try:
        result = dispatch_upstream(route, payload)
        response = {
            "ok": True,
            "protocol": PROTOCOL,
            "request_id": request_id,
            "connector": connector,
            "action": action,
            "upstream": result,
        }
        cache_result(request_id, intent_fingerprint, response)
        return response
    except Exception:
        release_idempotency_slot(request_id, intent_fingerprint)
        raise
