from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
from typing import Any, Mapping

try:
    from .deterministic_codec import digest
except ImportError:
    from deterministic_codec import digest



DEFAULT_STALE_SECONDS = 1500
FEDERATION_STALE_SECONDS = 10 * 60
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


def is_stale_federation(
    workflow: Mapping[str, Any],
    now: datetime,
    *,
    stale_seconds: int = FEDERATION_STALE_SECONDS,
) -> bool:
    federation = workflow.get("federation")
    if not isinstance(federation, Mapping):
        return False
    if federation.get("status") not in {"prepared", "dispatched"}:
        return False
    created = _parse_timestamp(federation.get("created_at"))
    if created is None:
        return False
    current = now
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    return (current - created).total_seconds() >= max(1, int(stale_seconds))


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
        federation_status = (
            federation.get("status")
            if isinstance(federation, Mapping)
            else None
        )
        if federation_status not in {"prepared", "dispatched"}:
            return False
        if recovery_due is not True and not is_stale_federation(
            workflow,
            now,
            stale_seconds=FEDERATION_STALE_SECONDS,
        ):
            return False
        return True
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
    return digest(payload)


def recovery_event_id(workflow: Mapping[str, Any]) -> str:
    return "scheduled-recovery:" + str(workflow.get("id") or "") + ":" + recovery_generation(workflow)[:32]


def _github_run_status(repository: str, run_id: str) -> str:
    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "GET",
                f"repos/{repository}/actions/runs/{run_id}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if "HTTP 404" in stderr or '"status":404' in stderr:
            return "missing"
        return "unknown"
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return "unknown"
    return str(payload.get("status") or "unknown").strip().lower()


def run() -> tuple[int, int, list[str]]:
    repository = os.environ.get("REPOSITORY", "").strip()
    schedule_run_id = os.environ.get("SCHEDULE_RUN_ID", "").strip()
    if not repository or not schedule_run_id:
        raise RuntimeError("REPOSITORY and SCHEDULE_RUN_ID are required")

    try:
        from .orchestrator import (
            AUTHORITY_DISTRIBUTED_CONTROL_PLANE,
            DEFAULT_TERMINAL_COMPACTION_DAYS,
            compact_terminal_workflows,
            load_state,
            workflow_authority_mode,
        )
        from .control_plane import ControlPlaneClient, ControlPlaneError
    except ImportError:
        from orchestrator import (
            AUTHORITY_DISTRIBUTED_CONTROL_PLANE,
            DEFAULT_TERMINAL_COMPACTION_DAYS,
            compact_terminal_workflows,
            load_state,
            workflow_authority_mode,
        )
        from control_plane import ControlPlaneClient, ControlPlaneError

    state = load_state()
    now = datetime.now(timezone.utc)
    try:
        stale_seconds = max(
            1,
            int(
                os.environ.get(
                    "ORCHESTRATOR_RECOVERY_STALE_SECONDS",
                    str(DEFAULT_STALE_SECONDS),
                )
            ),
        )
    except ValueError:
        stale_seconds = DEFAULT_STALE_SECONDS

    control_plane = None
    if (
        os.environ.get("ORCHESTRATOR_CONTROL_PLANE_URL", "").strip()
        and os.environ.get("ORCHESTRATOR_CONTROL_PLANE_SECRET", "").strip()
    ):
        try:
            control_plane = ControlPlaneClient.from_env()
        except ControlPlaneError:
            control_plane = None

    dispatched = 0
    failures: list[str] = []
    for workflow in state.get("workflows", {}).values():
        if not isinstance(workflow, dict):
            continue
        workflow_id = str(workflow.get("id") or "").strip()
        if not workflow_id:
            continue

        candidate_workflow: dict[str, Any] = workflow
        recovery_due: bool | None = None
        distributed = workflow_authority_mode(workflow) == AUTHORITY_DISTRIBUTED_CONTROL_PLANE
        if distributed:
            if control_plane is None:
                failures.append(f"{workflow_id}: distributed state authority unavailable")
                continue
            try:
                remote = control_plane.get_workflow_state(workflow_id)
            except ControlPlaneError as exc:
                failures.append(f"{workflow_id}: control-plane read failed: {type(exc).__name__}: {exc}")
                continue
            if remote is None:
                failures.append(f"{workflow_id}: control-plane state absent")
                continue
            if str(remote.state.get("id") or "").strip() != workflow_id:
                failures.append(f"{workflow_id}: control-plane workflow identity mismatch")
                continue
            candidate_workflow = dict(remote.state)
            candidate_workflow["control_plane_state_version"] = remote.state_version
            recovery_due = bool(remote.recovery_due)

        active_run_status = None
        if candidate_workflow.get("status") == "running" and candidate_workflow.get("github_run_id"):
            active_run_status = _github_run_status(
                repository,
                str(candidate_workflow.get("github_run_id")),
            )

        if not is_recovery_candidate(
            candidate_workflow,
            now,
            active_run_status=active_run_status,
            stale_seconds=stale_seconds,
            recovery_due=recovery_due,
        ):
            continue

        event_id = recovery_event_id(candidate_workflow)
        claim_owner = "scheduled-recovery:" + schedule_run_id
        claimed = False
        if distributed:
            try:
                claim = control_plane.claim_recovery(
                    workflow_id,
                    event_id,
                    owner=claim_owner,
                    ttl_seconds=300,
                )
            except ControlPlaneError as exc:
                failures.append(f"{workflow_id}: recovery claim failed: {type(exc).__name__}: {exc}")
                continue
            if str(claim.get("status") or "") != "claimed":
                continue
            claimed = True

        payload = {
            "event_type": "orchestrator.continue",
            "client_payload": {
                "workflow_id": workflow_id,
                "event_id": event_id,
                "scheduled_recovery": True,
                "schedule_run_id": schedule_run_id,
            },
        }
        try:
            result = subprocess.run(
                [
                    "gh",
                    "api",
                    "--method",
                    "POST",
                    f"repos/{repository}/dispatches",
                    "--input",
                    "-",
                ],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=False,
            )
        except Exception as exc:
            failures.append(f"{workflow_id}: dispatch exception: {type(exc).__name__}: {exc}")
            continue
        if result.returncode != 0:
            stderr = (result.stderr or "").strip().replace("\\n", " ")
            failures.append(f"{workflow_id}: dispatch exit={result.returncode} {stderr[:300]}")
            continue

        dispatched += 1
        if distributed and claimed:
            try:
                control_plane.ack_recovery(
                    workflow_id,
                    event_id,
                    owner=claim_owner,
                )
            except ControlPlaneError as exc:
                failures.append(
                    f"{workflow_id}: recovery ack failed: {type(exc).__name__}: {exc}"
                )

    retention_raw = os.environ.get(
        "ORCHESTRATOR_TERMINAL_COMPACTION_DAYS",
        str(DEFAULT_TERMINAL_COMPACTION_DAYS),
    )
    try:
        retention_days = int(retention_raw)
    except ValueError:
        retention_days = DEFAULT_TERMINAL_COMPACTION_DAYS
    compacted = compact_terminal_workflows(
        state,
        now=now,
        retention_days=retention_days,
    )
    return dispatched, len(compacted), failures


def main() -> int:
    dispatched, compacted, failures = run()
    marker = "/tmp/orchestrator_recovery_failures.json"
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump(failures[:16], handle, ensure_ascii=False, sort_keys=True)
    print(f"dispatched={dispatched}")
    print(f"compacted={compacted}")
    print(f"recovery_failures={len(failures)}")
    for failure in failures[:16]:
        print(f"recovery_failure={failure}")
    # The workflow shell owns the final failure decision after compaction.
    return 0


__all__ = [
    "ACTIVE_RUN_STATUSES",
    "DEFAULT_STALE_SECONDS",
    "FEDERATION_STALE_SECONDS",
    "is_stale_federation",
    "is_recovery_candidate",
    "is_stale_running",
    "recovery_event_id",
    "recovery_generation",
    "run",
    "main",
]
