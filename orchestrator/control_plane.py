from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_LEASE_TTL_SECONDS = 1200
CONTROL_PLANE_TIMEOUT_SECONDS = 20
CLOCK_SKEW_SECONDS = 300

class ControlPlaneError(RuntimeError):
    pass

class ControlPlaneConfigurationError(ControlPlaneError):
    pass

class ControlPlaneUnavailable(ControlPlaneError):
    pass

class ControlPlaneRejected(ControlPlaneError):
    pass

@dataclass(frozen=True)
class Lease:
    workflow_id: str
    owner: str
    fence_epoch: int
    expires_at: int

@dataclass(frozen=True)
class ResourceLease:
    resource_key: str
    owner: str
    fence_epoch: int
    expires_at: int

@dataclass(frozen=True)
class EffectClaim:
    status: str
    effect_id: str
    semantic_digest: str

@dataclass(frozen=True)
class WorkflowState:
    state: dict[str, Any]
    state_version: int

def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")

def worker_id() -> str:
    explicit = os.environ.get("ORCHESTRATOR_WORKER_ID", "").strip()
    if explicit:
        return explicit[:128]
    run_id = os.environ.get("GITHUB_RUN_ID", "").strip()
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1").strip() or "1"
    if run_id:
        return f"github-actions:{run_id}:{attempt}"[:128]
    host = socket.gethostname().strip()[:64] or "unknown"
    return f"local:{host}:{secrets.token_hex(8)}"[:128]

class ControlPlaneClient:
    """Stdlib-only signed client for leases, fenced effects, state CAS, and outbox."""

    def __init__(
        self,
        base_url: str,
        secret: str,
        *,
        owner: str | None = None,
        timeout: int = CONTROL_PLANE_TIMEOUT_SECONDS,
        lease_ttl_seconds: int | None = None,
    ) -> None:
        parsed = urllib.parse.urlparse(str(base_url).strip())
        if parsed.scheme != "https" or not parsed.netloc:
            raise ControlPlaneConfigurationError("control-plane URL must use HTTPS")
        if not secret:
            raise ControlPlaneConfigurationError("control-plane secret must not be empty")
        self.base_url = str(base_url).rstrip("/")
        self.secret = secret
        self.owner = owner or worker_id()
        self.timeout = max(1, int(timeout))
        raw_ttl = lease_ttl_seconds or int(
            os.environ.get(
                "ORCHESTRATOR_CONTROL_PLANE_LEASE_TTL",
                str(DEFAULT_LEASE_TTL_SECONDS),
            )
        )
        self.lease_ttl_seconds = max(60, min(int(raw_ttl), 3600))

    @classmethod
    def from_env(cls) -> "ControlPlaneClient":
        url = os.environ.get("ORCHESTRATOR_CONTROL_PLANE_URL", "").strip()
        secret = os.environ.get("ORCHESTRATOR_CONTROL_PLANE_SECRET", "").strip()
        if bool(url) != bool(secret):
            raise ControlPlaneConfigurationError(
                "ORCHESTRATOR_CONTROL_PLANE_URL and "
                "ORCHESTRATOR_CONTROL_PLANE_SECRET must be configured together"
            )
        if not url:
            raise ControlPlaneConfigurationError(
                "distributed control plane is not configured"
            )
        return cls(url, secret)

    @staticmethod
    def _workflow_path(workflow_id: str, suffix: str) -> str:
        return (
            "/v1/workflows/"
            + urllib.parse.quote(str(workflow_id), safe="")
            + suffix
        )

    @staticmethod
    def _resource_path(resource_key: str, suffix: str) -> str:
        return (
            "/v1/resources/"
            + urllib.parse.quote(str(resource_key), safe="")
            + suffix
        )

    def _signature(
        self,
        timestamp: str,
        method: str,
        path: str,
        body: bytes,
    ) -> str:
        data = b"\n".join([
            timestamp.encode("utf-8"),
            method.upper().encode("utf-8"),
            path.encode("utf-8"),
            body,
        ])
        return hmac.new(
            self.secret.encode("utf-8"),
            data,
            hashlib.sha256,
        ).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = canonical_json(payload) if payload is not None else b""
        timestamp = str(int(time.time()))
        req = urllib.request.Request(
            self.base_url + path,
            data=body if method.upper() != "GET" else None,
            method=method.upper(),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "ai-determinism-engine-control-plane/2.0",
                "X-Control-Plane-Timestamp": timestamp,
                "X-Control-Plane-Signature": self._signature(
                    timestamp,
                    method,
                    path,
                    body,
                ),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read(768 * 1024)
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read(128 * 1024)
                detail = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                detail = {}
            msg = (
                str(detail.get("error") or detail.get("message") or "")
                if isinstance(detail, dict)
                else ""
            )
            if 400 <= exc.code < 500:
                raise ControlPlaneRejected(
                    msg or f"control-plane rejected: HTTP {exc.code}"
                ) from exc
            raise ControlPlaneUnavailable(
                f"control-plane HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ControlPlaneUnavailable(
                f"control-plane unavailable: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControlPlaneUnavailable(
                "control-plane returned invalid JSON"
            ) from exc
        if not isinstance(result, dict):
            raise ControlPlaneUnavailable(
                "control-plane response must be an object"
            )
        return result

    def acquire_lease(self, workflow_id: str) -> Lease:
        result = self._request(
            "POST",
            self._workflow_path(workflow_id, "/lease/acquire"),
            {
                "owner": self.owner,
                "workflow_id": str(workflow_id),
                "ttl_seconds": self.lease_ttl_seconds,
            },
        )
        if result.get("status") not in {"acquired", "renewed"}:
            raise ControlPlaneRejected(
                f"unexpected lease status: {result.get('status')}"
            )
        return Lease(
            str(workflow_id),
            self.owner,
            int(result["fence_epoch"]),
            int(result["expires_at"]),
        )

    def renew_lease(self, workflow_id: str, fence_epoch: int) -> Lease:
        result = self._request(
            "POST",
            self._workflow_path(workflow_id, "/lease/renew"),
            {
                "owner": self.owner,
                "fence_epoch": int(fence_epoch),
                "ttl_seconds": self.lease_ttl_seconds,
            },
        )
        if result.get("status") != "renewed":
            raise ControlPlaneRejected(
                f"unexpected lease status: {result.get('status')}"
            )
        return Lease(
            str(workflow_id),
            self.owner,
            int(result["fence_epoch"]),
            int(result["expires_at"]),
        )

    def release_lease(self, workflow_id: str, fence_epoch: int) -> None:
        result = self._request(
            "POST",
            self._workflow_path(workflow_id, "/lease/release"),
            {
                "owner": self.owner,
                "workflow_id": str(workflow_id),
                "fence_epoch": int(fence_epoch),
            },
        )
        if result.get("status") not in {"released", "already_released"}:
            raise ControlPlaneRejected(
                f"unexpected release status: {result.get('status')}"
            )

    def acquire_resource(
        self,
        resource_key: str,
        *,
        workflow_id: str,
    ) -> ResourceLease:
        result = self._request(
            "POST",
            self._resource_path(resource_key, "/lease/acquire"),
            {
                "owner": self.owner,
                "workflow_id": str(workflow_id),
                "ttl_seconds": self.lease_ttl_seconds,
            },
        )
        if result.get("status") != "acquired":
            raise ControlPlaneRejected(
                f"resource lock unavailable: {resource_key}"
            )
        return ResourceLease(
            str(resource_key),
            self.owner,
            int(result["fence_epoch"]),
            int(result["expires_at"]),
        )

    def renew_resource(
        self,
        resource_key: str,
        *,
        workflow_id: str,
        fence_epoch: int,
    ) -> ResourceLease:
        result = self._request(
            "POST",
            self._resource_path(resource_key, "/lease/renew"),
            {
                "owner": self.owner,
                "workflow_id": str(workflow_id),
                "fence_epoch": int(fence_epoch),
                "ttl_seconds": self.lease_ttl_seconds,
            },
        )
        if result.get("status") != "renewed":
            raise ControlPlaneRejected(
                f"resource lock renewal rejected: {resource_key}"
            )
        return ResourceLease(
            str(resource_key),
            self.owner,
            int(result["fence_epoch"]),
            int(result["expires_at"]),
        )

    def release_resource(
        self,
        resource_key: str,
        *,
        workflow_id: str,
        fence_epoch: int,
    ) -> None:
        result = self._request(
            "POST",
            self._resource_path(resource_key, "/lease/release"),
            {
                "owner": self.owner,
                "workflow_id": str(workflow_id),
                "fence_epoch": int(fence_epoch),
            },
        )
        if result.get("status") not in {"released", "already_released"}:
            raise ControlPlaneRejected(
                f"resource release rejected: {resource_key}"
            )

    def arm_recovery(
        self,
        workflow_id: str,
        *,
        owner: str,
        fence_epoch: int,
        due_at: int,
        event_id: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            self._workflow_path(workflow_id, "/recovery/arm"),
            {
                "owner": str(owner),
                "workflow_id": str(workflow_id),
                "fence_epoch": int(fence_epoch),
                "due_at": int(due_at),
                "event_id": str(event_id),
            },
        )

    def ack_recovery(
        self,
        workflow_id: str,
        event_id: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            self._workflow_path(workflow_id, "/recovery/ack"),
            {
                "workflow_id": str(workflow_id),
                "event_id": str(event_id),
            },
        )

    def get_recovery(
        self,
        workflow_id: str,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            self._workflow_path(workflow_id, "/recovery"),
        )

    def get_workflow_state(self, workflow_id: str) -> WorkflowState | None:
        result = self._request(
            "GET",
            self._workflow_path(workflow_id, "/state"),
        )
        if result.get("status") == "absent":
            return None
        state = result.get("state")
        if not isinstance(state, dict):
            raise ControlPlaneUnavailable(
                "control-plane workflow state is not an object"
            )
        return WorkflowState(
            state=state,
            state_version=int(result.get("state_version", 0)),
        )

    def put_workflow_state(
        self,
        workflow_id: str,
        *,
        owner: str,
        fence_epoch: int,
        expected_state_version: int,
        state: dict[str, Any],
    ) -> int:
        result = self._request(
            "PUT",
            self._workflow_path(workflow_id, "/state"),
            {
                "owner": str(owner),
                "fence_epoch": int(fence_epoch),
                "expected_state_version": int(expected_state_version),
                "state": state,
            },
        )
        if result.get("status") != "stored":
            raise ControlPlaneRejected(
                f"workflow state CAS rejected: {result.get('status')}"
            )
        return int(result["state_version"])

    def append_outbox_event(
        self,
        workflow_id: str,
        *,
        owner: str,
        fence_epoch: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> int:
        result = self._request(
            "POST",
            self._workflow_path(workflow_id, "/outbox"),
            {
                "owner": str(owner),
                "fence_epoch": int(fence_epoch),
                "event_type": str(event_type),
                "payload": payload,
            },
        )
        if result.get("status") != "appended":
            raise ControlPlaneRejected(
                f"outbox append rejected: {result.get('status')}"
            )
        return int(result["sequence"])

    def claim_effect(
        self,
        workflow_id: str,
        effect_id: str,
        semantic_digest: str,
        fence_epoch: int,
    ) -> EffectClaim:
        result = self._request(
            "POST",
            self._workflow_path(workflow_id, "/effects/claim"),
            {
                "owner": self.owner,
                "fence_epoch": int(fence_epoch),
                "effect_id": str(effect_id),
                "semantic_digest": str(semantic_digest),
            },
        )
        status = str(result.get("status") or "")
        if status not in {"claimed", "inflight", "completed"}:
            raise ControlPlaneRejected(f"unexpected claim status: {status}")
        return EffectClaim(
            status,
            str(result.get("effect_id") or effect_id),
            str(result.get("semantic_digest") or semantic_digest),
        )

    def complete_effect(
        self,
        workflow_id: str,
        effect_id: str,
        semantic_digest: str,
        fence_epoch: int,
        output_sha256: str = "",
    ) -> None:
        result = self._request(
            "POST",
            self._workflow_path(workflow_id, "/effects/complete"),
            {
                "owner": self.owner,
                "fence_epoch": int(fence_epoch),
                "effect_id": str(effect_id),
                "semantic_digest": str(semantic_digest),
                "output_sha256": str(output_sha256),
            },
        )
        if result.get("status") not in {"completed", "already_completed"}:
            raise ControlPlaneRejected(
                f"unexpected completion status: {result.get('status')}"
            )

    def resolve_effect(
        self,
        workflow_id: str,
        effect_id: str,
        semantic_digest: str,
        fence_epoch: int,
        *,
        outcome: str,
        output_sha256: str = "",
    ) -> None:
        outcome = str(outcome).lower().strip()
        if outcome not in {"completed", "not_applied", "unknown"}:
            raise ValueError("invalid effect outcome")
        result = self._request(
            "POST",
            self._workflow_path(workflow_id, "/effects/resolve"),
            {
                "owner": self.owner,
                "fence_epoch": int(fence_epoch),
                "effect_id": str(effect_id),
                "semantic_digest": str(semantic_digest),
                "outcome": outcome,
                "output_sha256": str(output_sha256),
            },
        )
        if result.get("status") not in {
            "completed",
            "not_applied",
            "unknown",
        }:
            raise ControlPlaneRejected(
                f"unexpected resolution status: {result.get('status')}"
            )

    def inspect_effect(self, workflow_id: str, effect_id: str) -> dict[str, Any]:
        q = urllib.parse.urlencode({"effect_id": str(effect_id)})
        return self._request(
            "GET",
            self._workflow_path(workflow_id, "/effects/inspect") + "?" + q,
        )
