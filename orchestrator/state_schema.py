#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Any

CURRENT_STATE_VERSION = 4
CURRENT_WORKFLOW_SCHEMA_VERSION = 5
MAX_PARALLEL = 8
MAX_REPLANS = 2
MAX_NODE_RETRIES = 8
MAX_GOAL_CHARS = 4000
MAX_NODE_INTENT_BYTES = 32 * 1024
MAX_NODE_STATE_BYTES = 128 * 1024
MAX_ARTIFACTS_PER_NODE = 32
RISK_LEVELS = {"low", "medium", "high", "critical"}
NODE_STATUSES = {
    "pending", "ready", "running", "validating", "waiting_approval",
    "retrying", "replanning", "reconciling", "completed", "failed", "cancelled",
}
WORKFLOW_STATUSES = {"planning", "ready", "running", "waiting_approval", "failed", "completed", "cancelled"}
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class StateSchemaError(RuntimeError):
    pass


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _strict_schema_version(value: Any, *, default: int, field_name: str) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise StateSchemaError(f"{field_name} must be an integer") from exc


def _strict_optional_positive_int(
    value: Any,
    *,
    field_name: str,
) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StateSchemaError(f"{field_name} must be an integer or null") from exc
    if parsed < 1:
        raise StateSchemaError(f"{field_name} must be >= 1")
    return parsed


def migrate_state(state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise StateSchemaError("state must be an object")

    version = _strict_schema_version(
        state.get("version"),
        default=1,
        field_name="state.version",
    )
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

        if workflow.get("id") is None:
            workflow["id"] = str(workflow_id)
        if str(workflow.get("id")) != str(workflow_id):
            raise StateSchemaError(
                f"workflow key {workflow_id!r} does not match workflow.id {workflow.get('id')!r}"
            )
        if not SAFE_ID_RE.fullmatch(str(workflow.get("id") or "")):
            raise StateSchemaError(
                f"workflow {workflow_id!r}.id contains unsafe characters"
            )
        workflow_version = _strict_schema_version(
            workflow.get("schema_version"),
            default=1,
            field_name=f"workflow {workflow_id!r}.schema_version",
        )
        if workflow_version > CURRENT_WORKFLOW_SCHEMA_VERSION:
            raise StateSchemaError(
                f"workflow {workflow_id!r} schema version {workflow_version} "
                f"is newer than supported version {CURRENT_WORKFLOW_SCHEMA_VERSION}"
            )
        workflow["schema_version"] = CURRENT_WORKFLOW_SCHEMA_VERSION
        goal = workflow.get("goal")
        if goal is not None:
            if not isinstance(goal, str):
                raise StateSchemaError(f"workflow {workflow_id!r}.goal must be a string")
            if not goal.strip():
                raise StateSchemaError(f"workflow {workflow_id!r}.goal must not be empty")
            if len(goal) > MAX_GOAL_CHARS:
                raise StateSchemaError(
                    f"workflow {workflow_id!r}.goal exceeds {MAX_GOAL_CHARS} characters"
                )
        workflow.setdefault("repair_feedback", {})
        workflow.setdefault("evidence", {})
        workflow.setdefault("reconciliations", {})
        workflow.setdefault("policy_fingerprint", None)
        workflow.setdefault("route_snapshot", {})
        workflow.setdefault("policy_integrity", "legacy_unverified")
        workflow.setdefault(
            "origin_github_run_id",
            workflow.get("github_run_id"),
        )
        workflow.setdefault(
            "origin_github_run_attempt",
            workflow.get("github_run_attempt"),
        )
        workflow.setdefault("github_run_attempt", None)
        workflow["github_run_attempt"] = _strict_optional_positive_int(
            workflow.get("github_run_attempt"),
            field_name=f"workflow {workflow_id!r}.github_run_attempt",
        )
        workflow["origin_github_run_attempt"] = _strict_optional_positive_int(
            workflow.get("origin_github_run_attempt"),
            field_name=f"workflow {workflow_id!r}.origin_github_run_attempt",
        )
        workflow.setdefault("execution_budget", {
            "max_steps": 96,
            "used_steps": 0,
        })
        if not isinstance(workflow.get("execution_budget"), dict):
            raise StateSchemaError(f"workflow {workflow_id!r}.execution_budget must be an object")
        budget = workflow["execution_budget"]
        try:
            max_steps = int(budget.get("max_steps", 96))
            used_steps = int(budget.get("used_steps", 0))
        except (TypeError, ValueError) as exc:
            raise StateSchemaError(f"workflow {workflow_id!r}.execution_budget contains non-integer values") from exc
        if max_steps < 1 or max_steps > 256:
            raise StateSchemaError(f"workflow {workflow_id!r}.execution_budget.max_steps out of bounds")
        if used_steps < 0 or used_steps > 256:
            raise StateSchemaError(f"workflow {workflow_id!r}.execution_budget.used_steps out of bounds")
        if used_steps > max_steps:
            raise StateSchemaError(f"workflow {workflow_id!r}.execution_budget.used_steps exceeds max_steps")
        budget["max_steps"] = max_steps
        budget["used_steps"] = used_steps
        try:
            replan_count = int(workflow.get("replan_count", 0))
        except (TypeError, ValueError) as exc:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.replan_count must be an integer"
            ) from exc
        if replan_count < 0 or replan_count > MAX_REPLANS:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.replan_count out of bounds"
            )
        workflow["replan_count"] = replan_count
        status = workflow.get("status")
        if status is not None and status not in WORKFLOW_STATUSES:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.status is invalid"
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
            if not isinstance(node.get("id"), str) or not node["id"].strip():
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node id must be a non-empty string"
                )
            if not isinstance(node.get("capability"), str) or not node["capability"].strip():
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node capability must be a non-empty string"
                )
            if not isinstance(node.get("tool"), str) or not node["tool"].strip():
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node tool must be a non-empty string"
                )
            if not isinstance(node.get("depends_on"), list):
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node {node['id']!r}.depends_on must be an array"
                )
            if any(not isinstance(dep, str) or not dep.strip() for dep in node["depends_on"]):
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node {node['id']!r}.depends_on contains an invalid id"
                )
            if node.get("risk") not in RISK_LEVELS:
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node {node['id']!r}.risk is invalid"
                )
            if node.get("status") not in NODE_STATUSES:
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node {node['id']!r}.status is invalid"
                )
            try:
                retry_count = int(node.get("retry_count", 0))
                max_retries = int(node.get("max_retries", 2))
            except (TypeError, ValueError) as exc:
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node {node['id']!r} retry fields must be integers"
                ) from exc
            if (
                retry_count < 0
                or max_retries < 0
                or max_retries > MAX_NODE_RETRIES
                or retry_count > max_retries
            ):
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node {node['id']!r} retry fields are out of bounds"
                )
            node["retry_count"] = retry_count
            node["max_retries"] = max_retries
            for field_name in ("input", "output", "error", "contract"):
                if not isinstance(node.get(field_name), dict):
                    raise StateSchemaError(
                        f"workflow {workflow_id!r} node {node['id']!r}.{field_name} must be an object"
                    )
            artifacts = node["input"].get("artifacts")
            if artifacts is not None:
                if not isinstance(artifacts, list):
                    raise StateSchemaError(
                        f"workflow {workflow_id!r} node {node['id']!r}.input.artifacts must be an array"
                    )
                if len(artifacts) > MAX_ARTIFACTS_PER_NODE:
                    raise StateSchemaError(
                        f"workflow {workflow_id!r} node {node['id']!r} exceeds artifact limit"
                    )
            intent_serialized = json.dumps(
                {
                    "id": node["id"],
                    "capability": node["capability"],
                    "tool": node["tool"],
                    "depends_on": node["depends_on"],
                    "risk": node["risk"],
                    "input": node["input"],
                    "contract": node["contract"],
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            )
            if len(intent_serialized.encode("utf-8")) > MAX_NODE_INTENT_BYTES:
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node {node['id']!r} intent exceeds {MAX_NODE_INTENT_BYTES} bytes"
                )
            state_serialized = json.dumps(
                node,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            )
            if len(state_serialized.encode("utf-8")) > MAX_NODE_STATE_BYTES:
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node {node['id']!r} state exceeds {MAX_NODE_STATE_BYTES} bytes"
                )

    state["workflows"] = workflows
    state.setdefault("last_workflow_id", None)
    state["version"] = CURRENT_STATE_VERSION
    return state
