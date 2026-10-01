#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

CURRENT_STATE_VERSION = 3
CURRENT_WORKFLOW_SCHEMA_VERSION = 2
MAX_PARALLEL = 8


class StateSchemaError(RuntimeError):
    pass


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def migrate_state(state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise StateSchemaError("state must be an object")

    version = _as_int(state.get("version"), 1)
    if version > CURRENT_STATE_VERSION:
        raise StateSchemaError(
            f"state version {version} is newer than supported version {CURRENT_STATE_VERSION}"
        )

    workflows = state.get("workflows")
    if workflows is None:
        workflows = {}
    if not isinstance(workflows, dict):
        raise StateSchemaError("state.workflows must be an object")

    for workflow_id, workflow in workflows.items():
        if not isinstance(workflow, dict):
            raise StateSchemaError(f"workflow {workflow_id!r} must be an object")

        workflow.setdefault("schema_version", CURRENT_WORKFLOW_SCHEMA_VERSION)
        workflow.setdefault("repair_feedback", {})
        workflow.setdefault("evidence", {})
        workflow.setdefault("reconciliations", {})
        workflow.setdefault("replan_count", 0)
        workflow.setdefault("execution_mode", "live" if workflow.get("live") else "dry-run")
        workflow.setdefault("plan_fingerprint", None)
        if "plan_integrity" not in workflow:
            workflow["plan_integrity"] = (
                "legacy_unverified" if workflow.get("plan_fingerprint") is None else "pending"
            )

        raw_parallel = _as_int(workflow.get("max_parallel"), 4)
        workflow["max_parallel"] = max(1, min(raw_parallel, MAX_PARALLEL))

        nodes = workflow.get("nodes", [])
        if not isinstance(nodes, list):
            raise StateSchemaError(f"workflow {workflow_id!r}.nodes must be an array")
        for node in nodes:
            if not isinstance(node, dict):
                raise StateSchemaError(f"workflow {workflow_id!r} contains a non-object node")
            node.setdefault("depends_on", [])
            node.setdefault("risk", "low")
            node.setdefault("status", "pending")
            node.setdefault("retry_count", 0)
            node.setdefault("max_retries", 2)
            node.setdefault("input", {})
            node.setdefault("output", {})
            node.setdefault("error", {})
            node.setdefault("contract", {})

    state["workflows"] = workflows
    state.setdefault("last_workflow_id", None)
    state["version"] = CURRENT_STATE_VERSION
    return state
