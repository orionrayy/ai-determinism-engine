from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping


DEFAULT_STALE_SECONDS = 1500
ACTIVE_RUN_STATUSES = frozenset({
    "queued",
    "in_progress",
    "waiting",
    "requested",
    "pending",
})


def _parse_timestamp(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_stale_running(
    workflow: Mapping[str, Any],
    now: datetime,
    *,
    stale_seconds: int = DEFAULT_STALE_SECONDS,
) -> bool:
    if str(workflow.get("status") or "") != "running":
        return False
    updated = _parse_timestamp(workflow.get("updated_at"))
    if updated is None:
        return True
    current = now
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    return (current - updated).total_seconds() >= max(1, int(stale_seconds))


def _barrier_failed(workflow: Mapping[str, Any]) -> bool:
    executions = workflow.get("executions")
    if not isinstance(executions, Mapping):
        return False
    for node in workflow.get("nodes", []):
        if not isinstance(node, Mapping) or node.get("status") != "failed":
            continue
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        key = hashlib.sha256(
            f"{workflow.get('id')}:{node_id}".encode("utf-8")
        ).hexdigest()
        record = executions.get(key)
        if isinstance(record, Mapping) and record.get("status") == "barrier_failed":
            return True
    return False


def _execution_uncertain(workflow: Mapping[str, Any]) -> bool:
    for node in workflow.get("nodes", []):
        if not isinstance(node, Mapping):
            continue
        error = node.get("error")
        if (
            node.get("status") == "failed"
            and isinstance(error, Mapping)
            and bool(error.get("execution_uncertain"))
            and node.get("tool") in {"connector_bridge", "github"}
        ):
            return True
    return False


def is_recovery_candidate(
    workflow: Mapping[str, Any],
    now: datetime,
    *,
    active_run_status: str | None = None,
    stale_seconds: int = DEFAULT_STALE_SECONDS,
    recovery_due: bool | None = None,
) -> bool:
    status = str(workflow.get("status") or "")
    if not str(workflow.get("id") or "").strip():
        return False
    if active_run_status == "unknown":
        return False
    if active_run_status in ACTIVE_RUN_STATUSES and recovery_due is not True:
        return False
    if status == "waiting_approval":
        return False
    if status == "running":
        if recovery_due is not True and not is_stale_running(
            workflow,
            now,
            stale_seconds=stale_seconds,
        ):
            return False
        if active_run_status in ACTIVE_RUN_STATUSES and recovery_due is not True:
            return False
        return True
    if status == "waiting_agents":
        if recovery_due is False:
            return False
        federation = workflow.get("federation")
        federation_status = federation.get("status") if isinstance(federation, Mapping) else None
        return federation_status in {"prepared", "dispatched"}
    if status == "failed":
        return _execution_uncertain(workflow) or _barrier_failed(workflow)
    return False


def recovery_generation(workflow: Mapping[str, Any]) -> str:
    nodes = []
    for node in workflow.get("nodes", []):
        if not isinstance(node, Mapping):
            continue
        error = node.get("error")
        node_error = {}
        if isinstance(error, Mapping):
            node_error = {
                "execution_uncertain": bool(error.get("execution_uncertain")),
                "durability_barrier_failed": bool(error.get("durability_barrier_failed")),
            }
        nodes.append({
            "id": str(node.get("id") or ""),
            "status": str(node.get("status") or ""),
            "tool": str(node.get("tool") or ""),
            "error": node_error,
        })
    nodes.sort(key=lambda item: item["id"])
    executions = []
    raw_executions = workflow.get("executions")
    if isinstance(raw_executions, Mapping):
        for key, record in raw_executions.items():
            if not isinstance(record, Mapping):
                continue
            status = str(record.get("status") or "")
            if status in {"barrier_failed", "inflight", "completed", "not_applied"}:
                executions.append({
                    "id": str(key),
                    "status": status,
                    "effect_id": str(record.get("effect_id") or ""),
                    "effect_semantic_digest": str(
                        record.get("effect_semantic_digest") or ""
                    ),
                    "fence_epoch": int(record.get("fence_epoch") or 0),
                })
    executions.sort(key=lambda item: (item["id"], item["status"]))
    federation = workflow.get("federation")
    federation_state = {}
    if isinstance(federation, Mapping):
        federation_state = {
            "id": str(federation.get("id") or ""),
            "status": str(federation.get("status") or ""),
        }
    payload = {
        "workflow_id": str(workflow.get("id") or ""),
        "status": str(workflow.get("status") or ""),
        "failed_node": str(workflow.get("failed_node") or ""),
        "plan_fingerprint": str(workflow.get("plan_fingerprint") or ""),
        "execution_id": str(workflow.get("execution_id") or ""),
        "nodes": nodes,
        "executions": executions,
        "federation": federation_state,
        "control_plane_state_version": int(workflow.get("control_plane_state_version") or 0),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def recovery_event_id(workflow: Mapping[str, Any]) -> str:
    return "scheduled-recovery:" + str(workflow.get("id") or "") + ":" + recovery_generation(workflow)[:32]


__all__ = [
    "ACTIVE_RUN_STATUSES",
    "DEFAULT_STALE_SECONDS",
    "is_recovery_candidate",
    "is_stale_running",
    "recovery_event_id",
    "recovery_generation",
]
