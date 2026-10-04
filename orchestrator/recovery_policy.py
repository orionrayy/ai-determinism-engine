from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any, Mapping

ACTIVE_RUN_STATUSES = frozenset({"queued", "in_progress"})
RECOVERABLE_FAILED_TOOLS = frozenset({"connector_bridge", "github"})


def run_is_active(status: str | None) -> bool:
    return str(status or "").strip().lower() in ACTIVE_RUN_STATUSES


def _parse_updated_at(workflow: Mapping[str, Any]) -> datetime | None:
    raw = str(workflow.get("updated_at") or "").strip()
    if not raw:
        return None
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _has_uncertain_effect(workflow: Mapping[str, Any]) -> bool:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return False
    return any(
        isinstance(node, Mapping)
        and node.get("status") == "failed"
        and node.get("tool") in RECOVERABLE_FAILED_TOOLS
        and isinstance(node.get("error"), Mapping)
        and node["error"].get("execution_uncertain") is True
        for node in nodes
    )


def _has_barrier_failure(workflow: Mapping[str, Any]) -> bool:
    if workflow.get("barrier_failed") is True:
        return True
    nodes = workflow.get("nodes")
    executions = workflow.get("executions")
    if not isinstance(nodes, list) or not isinstance(executions, Mapping):
        return False
    for node in nodes:
        if not isinstance(node, Mapping) or node.get("status") != "failed":
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue
        execution_id = hashlib.sha256(
            f"{workflow.get('id')}:{node_id}".encode("utf-8")
        ).hexdigest()
        record = executions.get(execution_id)
        if isinstance(record, Mapping) and record.get("status") == "barrier_failed":
            return True
    return False


def recovery_reasons(
    workflow: Mapping[str, Any],
    *,
    now: datetime,
    active_run_status: str | None = None,
    stale_after_seconds: int = 300,
) -> list[str]:
    status = str(workflow.get("status") or "").strip().lower()
    if not str(workflow.get("id") or "").strip():
        return []
    if status in {"waiting_approval", "completed", "cancelled"}:
        return []

    if status == "failed":
        reasons = []
        if _has_uncertain_effect(workflow):
            reasons.append("uncertain_effect")
        if _has_barrier_failure(workflow):
            reasons.append("durability_barrier")
        return reasons

    if status == "waiting_agents":
        federation = workflow.get("federation")
        if isinstance(federation, Mapping) and federation.get("status") in {
            "prepared",
            "dispatched",
        }:
            return ["waiting_agents"]
        return []

    if status != "running":
        return []

    if run_is_active(active_run_status):
        return []

    run_id = str(workflow.get("github_run_id") or "").strip()
    if run_id and active_run_status is None:
        # If a tracked Actions run cannot be inspected, fail closed rather than
        # launching a duplicate continuation.
        return []

    updated = _parse_updated_at(workflow)
    if updated is None:
        return ["stale_running"] if not run_id else []

    age = (now - updated).total_seconds()
    return ["stale_running"] if age >= max(1, int(stale_after_seconds)) else []


def recovery_generation(
    workflow: Mapping[str, Any],
    generation: str | None = None,
) -> str:
    if generation:
        return str(generation)
    for field in (
        "control_plane_state_version",
        "state_generation",
        "github_run_attempt",
        "updated_at",
    ):
        value = workflow.get(field)
        if value not in (None, ""):
            return str(value)
    return "0"


def recovery_event_id(
    workflow: Mapping[str, Any],
    reason: str,
    generation: str | None = None,
) -> str:
    workflow_id = str(workflow.get("id") or "").strip()
    reason_value = str(reason or "unknown").strip().lower()
    generation_value = recovery_generation(workflow, generation)
    raw = "\0".join((workflow_id, reason_value, generation_value))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"scheduled-recovery:{workflow_id}:{reason_value}:{digest}"


__all__ = [
    "ACTIVE_RUN_STATUSES",
    "RECOVERABLE_FAILED_TOOLS",
    "recovery_event_id",
    "recovery_generation",
    "recovery_reasons",
    "run_is_active",
]
