#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    from orchestrator.http_safety import read_response_limited
except ImportError:
    from http_safety import read_response_limited

PROTOCOL = "ai-orchestrator.connector/v1"
MAX_SKEW_SECONDS = 300
MAX_BODY_BYTES = 64 * 1024
MAX_UPSTREAM_RESPONSE_BYTES = 128 * 1024
IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60
IDEMPOTENCY_WAIT_TIMEOUT_SECONDS = 65
MAX_COMPLETED_ENTRIES = 128
_LOCK = threading.Lock()
# request_id -> (expires_at, semantic_request_digest, cached_response)
_COMPLETED: dict[str, tuple[float, str, dict[str, Any]]] = {}
# request_id -> (semantic_request_digest, completion_event)
_INFLIGHT: dict[str, tuple[str, threading.Event]] = {}


class BridgeRuntimeError(RuntimeError):
    pass


class BridgeUpstreamError(BridgeRuntimeError):
    def __init__(self, message: str, *, status_code: int = 503, uncertain: bool = True) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.uncertain = bool(uncertain)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign(
    timestamp: int,
    body: bytes,
    secret: str,
    *,
    method: str = "POST",
    path: str = "/bridge",
) -> str:
    message = b"\n".join([
        str(timestamp).encode("utf-8"),
        str(method).upper().encode("utf-8"),
        str(path).encode("utf-8"),
        body,
    ])
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return "sha256=" + digest


def verify_signature(
    headers: dict[str, str],
    body: bytes,
    secret: str,
    now: int | None = None,
    *,
    method: str = "POST",
    path: str = "/bridge",
) -> bool:
    timestamp = headers.get("x-orchestrator-timestamp", "")
    signature = headers.get("x-orchestrator-signature", "")
    try:
        sent_at = int(timestamp)
    except ValueError:
        return False
    current = int(time.time()) if now is None else now
    if abs(current - sent_at) > MAX_SKEW_SECONDS:
        return False
    expected = sign(sent_at, body, secret, method=method, path=path)
    return hmac.compare_digest(signature, expected)



IDEMPOTENCY_DB_ENV = "ORCHESTRATOR_BRIDGE_IDEMPOTENCY_DB"


def idempotency_db_path() -> str | None:
    value = os.environ.get(IDEMPOTENCY_DB_ENV, "").strip()
    return value or None


def _sqlite_connection() -> sqlite3.Connection | None:
    path = idempotency_db_path()
    if not path:
        return None
    db_path = Path(path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=5.0)
    connection.execute("PRAGMA busy_timeout=5000")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS idempotency (
            request_id TEXT PRIMARY KEY,
            semantic_digest TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('inflight', 'completed')),
            expires_at REAL,
            response_json TEXT
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idempotency_expiry_idx ON idempotency(status, expires_at)"
    )
    connection.commit()
    return connection


def _sqlite_cleanup(connection: sqlite3.Connection, now: float) -> None:
    connection.execute(
        "DELETE FROM idempotency WHERE status = 'completed' AND expires_at IS NOT NULL AND expires_at <= ?",
        (now,),
    )


def _sqlite_response(row: tuple[Any, ...], semantic_digest: str) -> dict[str, Any] | None:
    _, stored_digest, status, expires_at, response_json = row
    if str(stored_digest) != semantic_digest:
        raise BridgeRuntimeError(
            "idempotency key conflicts with an existing request payload"
        )
    if status != "completed":
        return None
    try:
        value = json.loads(str(response_json or "{}"))
    except json.JSONDecodeError as exc:
        raise BridgeRuntimeError("durable idempotency response is corrupt") from exc
    if not isinstance(value, dict):
        raise BridgeRuntimeError("durable idempotency response must be an object")
    return value


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
    result_required = raw.get("result_required", [])
    if not isinstance(result_required, list):
        result_required = []
    normalized_result_required = sorted(
        {str(item).strip() for item in result_required if str(item).strip()}
    )
    result_types = raw.get("result_types", {})
    if not isinstance(result_types, dict):
        result_types = {}
    normalized_result_types = {}
    for key in sorted(result_types):
        value = str(result_types[key] or "").strip().lower()
        if str(key).strip() and value in ACTION_TYPE_NAMES:
            normalized_result_types[str(key).strip()] = value
    return {
        "required": normalized_required,
        "types": normalized_types,
        "result_required": normalized_result_required,
        "result_types": normalized_result_types,
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


def validate_action_output(route: dict[str, Any], action: str, value: Any) -> None:
    action_specs = route.get("action_specs", {})
    raw_spec = action_specs.get(action, {}) if isinstance(action_specs, dict) else {}
    spec = _normalize_action_spec(
        raw_spec,
        default_free_tier=bool(route.get("free_tier", False)),
    )
    if not isinstance(value, dict):
        raise BridgeRuntimeError("connector output must be an object")
    for field_name in spec["result_required"]:
        found, _ = _resolve_payload_path(value, field_name)
        if not found:
            raise BridgeRuntimeError(f"connector output missing required field: {field_name}")
    for field_name, type_name in spec["result_types"].items():
        found, field_value = _resolve_payload_path(value, field_name)
        if found and not _matches_payload_type(field_value, type_name):
            raise BridgeRuntimeError(f"connector output field {field_name} must be {type_name}")


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
    expired = [key for key, (expires, _, _) in _COMPLETED.items() if expires <= now]
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
        if item is not None:
            if item[1] != semantic_digest:
                raise BridgeRuntimeError(
                    "idempotency key conflicts with an existing request payload"
                )
            return item[2]
    connection = _sqlite_connection()
    if connection is None:
        return None
    try:
        _sqlite_cleanup(connection, now)
        row = connection.execute(
            "SELECT request_id, semantic_digest, status, expires_at, response_json "
            "FROM idempotency WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        return _sqlite_response(row, semantic_digest)
    finally:
        connection.close()


def cache_result(
    request_id: str,
    semantic_digest: str,
    result: dict[str, Any],
    ttl: int = IDEMPOTENCY_TTL_SECONDS,
) -> None:
    expires_at = time.time() + ttl
    with _LOCK:
        cleanup_idempotency(time.time())
        while len(_COMPLETED) >= MAX_COMPLETED_ENTRIES:
            _COMPLETED.pop(next(iter(_COMPLETED)))
        _COMPLETED[request_id] = (expires_at, semantic_digest, result)
    connection = _sqlite_connection()
    if connection is None:
        return
    try:
        payload = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        connection.execute(
            """
            INSERT INTO idempotency (
                request_id, semantic_digest, status, expires_at, response_json
            ) VALUES (?, ?, 'completed', ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                semantic_digest=excluded.semantic_digest,
                status='completed',
                expires_at=excluded.expires_at,
                response_json=excluded.response_json
            """,
            (request_id, semantic_digest, expires_at, payload),
        )
        _sqlite_cleanup(connection, time.time())
        connection.commit()
    finally:
        connection.close()


def acquire_idempotency_slot(
    request_id: str,
    semantic_digest: str,
) -> tuple[dict[str, Any] | None, threading.Event | None, bool]:
    """Return cached result, waiter event, and owner flag for one request identity.

    When ORCHESTRATOR_BRIDGE_IDEMPOTENCY_DB is configured, SQLite becomes the
    durable single-flight/idempotency ledger. In-flight rows are intentionally
    not expired automatically: after an owner disappears mid-side-effect, the
    next caller must remain fail-closed rather than guessing that replay is safe.
    """
    now = time.time()
    connection = _sqlite_connection()
    if connection is not None:
        try:
            connection.execute("BEGIN IMMEDIATE")
            _sqlite_cleanup(connection, now)
            row = connection.execute(
                "SELECT request_id, semantic_digest, status, expires_at, response_json "
                "FROM idempotency WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is not None:
                cached = _sqlite_response(row, semantic_digest)
                if cached is not None:
                    connection.commit()
                    with _LOCK:
                        _COMPLETED[request_id] = (
                            float(row[3] or now),
                            semantic_digest,
                            cached,
                        )
                    return cached, None, False
                with _LOCK:
                    local = _INFLIGHT.get(request_id)
                connection.commit()
                return None, local[1] if local and local[0] == semantic_digest else None, False

            active_count = connection.execute(
                "SELECT COUNT(*) FROM idempotency"
            ).fetchone()[0]
            if int(active_count) >= MAX_COMPLETED_ENTRIES:
                connection.commit()
                raise BridgeRuntimeError(
                    "durable idempotency store is full; refusing a new side effect"
                )
            connection.execute(
                """
                INSERT INTO idempotency (
                    request_id, semantic_digest, status, expires_at, response_json
                ) VALUES (?, ?, 'inflight', NULL, NULL)
                """,
                (request_id, semantic_digest),
            )
            connection.commit()
            event = threading.Event()
            with _LOCK:
                _INFLIGHT[request_id] = (semantic_digest, event)
            return None, event, True
        except Exception:
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    with _LOCK:
        cleanup_idempotency(now)
        item = _COMPLETED.get(request_id)
        if item is not None:
            if item[1] != semantic_digest:
                raise BridgeRuntimeError(
                    "idempotency key conflicts with an existing request payload"
                )
            return item[2], None, False

        inflight = _INFLIGHT.get(request_id)
        if inflight is not None:
            existing_digest, event = inflight
            if existing_digest != semantic_digest:
                raise BridgeRuntimeError(
                    "idempotency key conflicts with an in-flight request payload"
                )
            return None, event, False

        event = threading.Event()
        _INFLIGHT[request_id] = (semantic_digest, event)
        return None, event, True


def release_idempotency_slot(
    request_id: str,
    event: threading.Event,
) -> None:
    # A durable in-flight row is deliberately retained on owner failure. Removing
    # it here would turn an unknown external outcome into an unsafe replay.
    with _LOCK:
        current = _INFLIGHT.get(request_id)
        if current is not None and current[1] is event:
            _INFLIGHT.pop(request_id, None)
        event.set()


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
            raw = read_response_limited(
                response,
                MAX_UPSTREAM_RESPONSE_BYTES,
                error_message="reconciliation upstream response exceeds 128 KiB",
            ).decode("utf-8", "replace")
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
    cached, event, owner = acquire_idempotency_slot(request_id, semantic_digest)
    if cached is not None:
        return {**cached, "idempotent_replay": True}
    if not owner:
        deadline = time.time() + IDEMPOTENCY_WAIT_TIMEOUT_SECONDS
        if event is not None:
            event.wait(max(0.0, IDEMPOTENCY_WAIT_TIMEOUT_SECONDS))
        while time.time() < deadline:
            cached = cached_result(request_id, semantic_digest)
            if cached is not None:
                return {**cached, "idempotent_replay": True}
            time.sleep(0.10)
        raise BridgeUpstreamError(
            "original idempotent request did not complete successfully; durable "
            "idempotency state remains fail-closed",
            status_code=503,
            uncertain=True,
        )

    assert event is not None
    try:
        result = dispatch_upstream(route, payload)
        validate_action_output(route, action, result.get("data") if isinstance(result, dict) and "data" in result else result)
        response = {
            "ok": True,
            "protocol": PROTOCOL,
            "request_id": request_id,
            "connector": connector,
            "action": action,
            "result": result.get("data", result),
            "upstream": result,
        }
        cache_result(request_id, semantic_digest, response)
        return response
    finally:
        release_idempotency_slot(request_id, event)