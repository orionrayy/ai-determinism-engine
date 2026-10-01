#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

PROTOCOL = "ai-orchestrator.connector/v1"
MAX_PAYLOAD_BYTES = 64 * 1024
CONNECTOR_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
ACTION_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,127}$")
_ACTION_TYPE_NAMES = {"string", "number", "integer", "boolean", "object", "array"}
_DISCOVERY_CACHE_TTL = 60
_DISCOVERY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


class ConnectorBridgeError(RuntimeError):
    pass


class ConnectorRequestError(ConnectorBridgeError):
    def __init__(self, message: str, *, uncertain: bool = False, retry_allowed: bool = False) -> None:
        super().__init__(message)
        self.uncertain = uncertain
        self.retry_allowed = retry_allowed


@dataclass(frozen=True)
class ConnectorRequest:
    protocol: str
    request_id: str
    workflow_id: str
    node_id: str
    connector: str
    action: str
    goal: str
    input: dict[str, Any]
    sent_at: int


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def execution_id(workflow_id: str, node_id: str) -> str:
    return hashlib.sha256(
        f"{workflow_id}:{node_id}".encode("utf-8")
    ).hexdigest()


def build_request(node: Any, goal: str) -> ConnectorRequest:
    workflow_id = str(node.input.get("workflow_id") or "").strip()
    node_id = str(node.id or "").strip()
    connector = str(node.input.get("connector") or "").strip().lower()
    action = str(node.input.get("action") or "").strip().lower()
    payload = node.input.get("payload", {})

    if not workflow_id:
        raise ConnectorBridgeError("workflow_id is required")
    if not node_id:
        raise ConnectorBridgeError("node_id is required")
    if not CONNECTOR_RE.fullmatch(connector):
        raise ConnectorBridgeError("invalid connector name")
    if not ACTION_RE.fullmatch(action):
        raise ConnectorBridgeError("invalid connector action")
    if not isinstance(payload, dict):
        raise ConnectorBridgeError("connector payload must be an object")

    request = ConnectorRequest(
        protocol=PROTOCOL,
        request_id=execution_id(workflow_id, node_id),
        workflow_id=workflow_id,
        node_id=node_id,
        connector=connector,
        action=action,
        goal=str(goal)[:4000],
        input=payload,
        sent_at=int(time.time()),
    )
    encoded = canonical_json(asdict(request))
    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise ConnectorBridgeError("connector payload exceeds 64 KiB safety limit")
    return request


def sign(timestamp: int, body: bytes, secret: str) -> str:
    message = str(timestamp).encode("utf-8") + b"\n" + body
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return "sha256=" + digest


def bridge_config() -> tuple[str, str]:
    url = os.environ.get("ORCHESTRATOR_CONNECTOR_BRIDGE_URL", "").strip()
    secret = os.environ.get("ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET", "")
    if not url:
        raise ConnectorBridgeError(
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL is not configured"
        )
    if not secret:
        raise ConnectorBridgeError(
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET is not configured"
        )
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ConnectorBridgeError("connector bridge HTTPS is required")
    return url, secret


def discovery_url(bridge_url: str) -> str:
    parsed = urllib.parse.urlparse(bridge_url)
    if parsed.scheme != "https":
        raise ConnectorBridgeError("connector discovery HTTPS is required")
    path = parsed.path.rstrip("/")
    if path.endswith("/api/bridge") or path.endswith("/bridge"):
        path = path + "/capabilities"
    else:
        path = path + "/capabilities"
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, path, "", "", "")
    )


def discover_capabilities(
    bridge_url: str,
    *,
    force_refresh: bool = False,
    timeout: int = 20,
) -> dict[str, dict[str, Any]]:
    now = time.time()
    cached = _DISCOVERY_CACHE.get(bridge_url)
    if cached and not force_refresh and cached[0] > now:
        return cached[1]

    url = discovery_url(bridge_url)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ai-orchestrator-connector-discovery/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            status = response.status
    except Exception as exc:
        raise ConnectorBridgeError(f"connector discovery request failed: {exc}") from exc
    if not (200 <= status < 300):
        raise ConnectorBridgeError(f"connector discovery returned HTTP {status}")
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ConnectorBridgeError("connector discovery returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise ConnectorBridgeError("connector discovery response is invalid")
    connectors = payload.get("connectors")
    if not isinstance(connectors, dict):
        raise ConnectorBridgeError("connector discovery did not return connectors")
    normalized: dict[str, dict[str, Any]] = {}
    for name in sorted(connectors):
        spec = connectors[name]
        if not isinstance(spec, dict):
            continue
        actions = spec.get("actions", [])
        capabilities = spec.get("capabilities", [])
        normalized[str(name)] = {
            "actions": sorted(str(item) for item in actions) if isinstance(actions, list) else [],
            "capabilities": sorted(str(item) for item in capabilities) if isinstance(capabilities, list) else [],
            "action_specs": {
                action: _normalize_action_spec(
                    spec.get("action_specs", {}).get(action, {})
                    if isinstance(spec.get("action_specs", {}), dict)
                    else {}
                )
                for action in sorted(str(item) for item in actions)
            },
            "risk": str(spec.get("risk") or "high"),
            "free_tier": bool(spec.get("free_tier", False)),
            "configured": bool(spec.get("configured", False)),
        }
    _DISCOVERY_CACHE[bridge_url] = (now + _DISCOVERY_CACHE_TTL, normalized)
    return normalized


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
        if str(key).strip() and value in _ACTION_TYPE_NAMES:
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


def validate_discovered_action(
    connector: str,
    action: str,
    inventory: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    connector = str(connector or "").strip().lower()
    action = str(action or "").strip().lower()
    spec = inventory.get(connector)
    if not isinstance(spec, dict):
        raise ConnectorBridgeError(f"connector {connector!r} is not advertised by the bridge")
    if spec.get("configured") is False:
        raise ConnectorBridgeError(f"connector {connector!r} is not configured on the bridge")
    actions = spec.get("actions", [])
    if action not in actions:
        raise ConnectorBridgeError(
            f"connector action {connector}:{action} is not advertised by the bridge"
        )
    action_specs = spec.get("action_specs", {})
    raw_action_spec = action_specs.get(action, {}) if isinstance(action_specs, dict) else {}
    return _normalize_action_spec(raw_action_spec)


def validate_discovered_payload(
    connector: str,
    action: str,
    payload: Any,
    inventory: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    action_spec = validate_discovered_action(connector, action, inventory)
    if not isinstance(payload, dict):
        raise ConnectorBridgeError("connector payload must be an object")
    for field_name in action_spec["required"]:
        found, _ = _resolve_payload_path(payload, field_name)
        if not found:
            raise ConnectorBridgeError(f"connector payload missing required field: {field_name}")
    for field_name, type_name in action_spec["types"].items():
        found, value = _resolve_payload_path(payload, field_name)
        if found and not _matches_payload_type(value, type_name):
            raise ConnectorBridgeError(f"connector payload field {field_name} must be {type_name}")
    return action_spec


def build_discovery_snapshot(inventory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "connectors": inventory,
        "count": len(inventory),
    }


def post_request(url: str, secret: str, request: ConnectorRequest) -> dict[str, Any]:
    body = canonical_json(asdict(request))
    timestamp = request.sent_at
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "ai-orchestrator-connector-bridge/1.0",
        "X-Orchestrator-Protocol": PROTOCOL,
        "X-Orchestrator-Timestamp": str(timestamp),
        "X-Orchestrator-Signature": sign(timestamp, body, secret),
        "Idempotency-Key": request.request_id,
    }
    http = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(http, timeout=60) as response:
            raw = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        raise ConnectorRequestError(
            f"connector bridge request failed: HTTP {exc.code}",
            uncertain=exc.code >= 500,
        ) from exc
    except Exception as exc:
        raise ConnectorRequestError(
            f"connector bridge request failed: {exc}",
            uncertain=True,
        ) from exc
    if not (200 <= status < 300):
        raise ConnectorRequestError(
            f"connector bridge request returned HTTP {status}",
            uncertain=status >= 500,
        )

    try:
        result = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ConnectorBridgeError("connector bridge returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise ConnectorBridgeError("connector bridge response must be an object")
    if result.get("ok") is False:
        raise ConnectorBridgeError(
            str(result.get("error") or "connector bridge rejected request")
        )
    return result


def execute_connector_bridge(node: Any, goal: str, dry_run: bool) -> dict[str, Any]:
    request = build_request(node, goal)
    envelope = asdict(request)
    if dry_run:
        return {
            "simulated": True,
            "protocol": PROTOCOL,
            "request_id": request.request_id,
            "connector": request.connector,
            "action": request.action,
            "envelope": envelope,
        }

    url, secret = bridge_config()
    inventory = discover_capabilities(url, force_refresh=True)
    action_spec = validate_discovered_payload(
        request.connector,
        request.action,
        request.input,
        inventory,
    )
    try:
        response = post_request(url, secret, request)
    except ConnectorRequestError as exc:
        exc.retry_allowed = bool(action_spec.get("idempotent"))
        raise
    return {
        "simulated": False,
        "protocol": PROTOCOL,
        "request_id": request.request_id,
        "bridge_url": url,
        "discovery": build_discovery_snapshot(inventory),
        "action_spec": action_spec,
        "response": response,
    }
