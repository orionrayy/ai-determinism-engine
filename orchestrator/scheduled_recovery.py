#!/usr/bin/env python3
"""Scheduled recovery adapter for GitHub Actions.

The scheduler only decides which workflows need another continuation dispatch.
Execution authority remains in the orchestrator/control plane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

from .orchestrator import (
    DEFAULT_TERMINAL_COMPACTION_DAYS,
    compact_terminal_workflows,
    load_state,
)


def parse_timestamp(value: Any) -> datetime | None:
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


def stale_running(workflow: dict[str, Any], *, now: datetime, stale_seconds: int = 300) -> bool:
    updated = parse_timestamp(workflow.get("updated_at"))
    if updated is None:
        return True
    return (now - updated).total_seconds() >= stale_seconds


def barrier_failed(workflow: dict[str, Any]) -> bool:
    executions = workflow.get("executions")
    if not isinstance(executions, dict):
        return False
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict) or node.get("status") != "failed":
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        key = hashlib.sha256(
            f"{workflow.get('id')}:{node_id}".encode("utf-8")
        ).hexdigest()
        record = executions.get(key)
        if isinstance(record, dict) and record.get("status") == "barrier_failed":
            return True
    return False


def recovery_candidate(workflow: dict[str, Any], *, now: datetime) -> bool:
    if not isinstance(workflow, dict):
        return False
    status = workflow.get("status")
    federation = workflow.get("federation") or {}
    uncertain = any(
        isinstance(node, dict)
        and node.get("status") == "failed"
        and isinstance(node.get("error"), dict)
        and bool(node["error"].get("execution_uncertain"))
        and node.get("tool") == "connector_bridge"
        for node in workflow.get("nodes", [])
    )
    return bool(
        status == "waiting_approval"
        or (status == "running" and stale_running(workflow, now=now))
        or (
            status == "waiting_agents"
            and federation.get("status") in {"prepared", "dispatched"}
        )
        or (status == "failed" and (uncertain or barrier_failed(workflow)))
    )


def recovery_event_id(workflow: dict[str, Any]) -> str:
    return (
        "scheduled-recovery:"
        + str(workflow.get("id") or "").strip()
        + ":"
        + str(workflow.get("updated_at") or "")
    )


def dispatch_continuation(
    *,
    repository: str,
    workflow_id: str,
    event_id: str,
    schedule_run_id: str,
) -> None:
    payload = {
        "event_type": "orchestrator.continue",
        "client_payload": {
            "workflow_id": workflow_id,
            "event_id": event_id,
            "scheduled_recovery": True,
            "schedule_run_id": schedule_run_id,
        },
    }
    subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repository}/dispatches",
            "--input",
            "-",
        ],
        input=json.dumps(payload),
        text=True,
        check=True,
    )


def run() -> tuple[int, int]:
    repository = os.environ.get("REPOSITORY", "").strip()
    schedule_run_id = os.environ.get("SCHEDULE_RUN_ID", "").strip()
    if not repository or not schedule_run_id:
        raise RuntimeError("REPOSITORY and SCHEDULE_RUN_ID are required")

    state = load_state()
    now = datetime.now(timezone.utc)
    dispatched = 0

    for workflow in state.get("workflows", {}).values():
        if not isinstance(workflow, dict):
            continue
        workflow_id = str(workflow.get("id") or "").strip()
        if not workflow_id or not recovery_candidate(workflow, now=now):
            continue
        dispatch_continuation(
            repository=repository,
            workflow_id=workflow_id,
            event_id=recovery_event_id(workflow),
            schedule_run_id=schedule_run_id,
        )
        dispatched += 1

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
    return dispatched, len(compacted)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    dispatched, compacted = run()
    print(f"dispatched={dispatched}")
    print(f"compacted={compacted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
