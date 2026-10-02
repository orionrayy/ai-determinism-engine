#!/usr/bin/env python3
from __future__ import annotations

from typing import Any
import hashlib

CURRENT_STATE_VERSION = 4
CURRENT_WORKFLOW_SCHEMA_VERSION = 4
MAX_PARALLEL = 8
DEFAULT_MAX_ATTEMPTS_PER_WORKFLOW = 64
MAX_ATTEMPTS_PER_WORKFLOW = 128


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

        workflow_version = _as_int(
            workflow.get("schema_version"),
            1,
        )
        if workflow_version > CURRENT_WORKFLOW_SCHEMA_VERSION:
            raise StateSchemaError(
                f"workflow {workflow_id!r} schema version {workflow_version} "
                f"is newer than supported version {CURRENT_WORKFLOW_SCHEMA_VERSION}"
            )
        workflow["schema_version"] = CURRENT_WORKFLOW_SCHEMA_VERSION
        workflow.setdefault("repair_feedback", {})
        workflow.setdefault("evidence", {})
        workflow.setdefault("reconciliations", {})
        workflow.setdefault("replan_count", 0)
        workflow.setdefault("attempts_used", 0)
        workflow["attempts_used"] = max(0, _as_int(workflow.get("attempts_used"), 0))
        workflow.setdefault("max_attempts", DEFAULT_MAX_ATTEMPTS_PER_WORKFLOW)
        workflow["max_attempts"] = max(
            1,
            min(
                _as_int(workflow.get("max_attempts"), DEFAULT_MAX_ATTEMPTS_PER_WORKFLOW),
                MAX_ATTEMPTS_PER_WORKFLOW,
            ),
        )
        workflow.setdefault(
            "retry_jitter_seed",
            hashlib.sha256(
                str(workflow.get("id") or workflow_id).encode("utf-8")
            ).hexdigest()[:32],
        )
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
            node.setdefault("agent_role", "")

    state["workflows"] = workflows
    state.setdefault("last_workflow_id", None)
    state["version"] = CURRENT_STATE_VERSION
    return state
