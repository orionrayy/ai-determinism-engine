#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Any

CURRENT_STATE_VERSION = 4
CURRENT_WORKFLOW_SCHEMA_VERSION = 6
MAX_PARALLEL = 8
MAX_REPLANS = 2
MAX_NODE_RETRIES = 8
MAX_CALLBACK_ATTEMPTS = 12
MAX_GOAL_CHARS = 4000
MAX_NODE_INTENT_BYTES = 32 * 1024
MAX_NODE_STATE_BYTES = 128 * 1024
MAX_ARTIFACTS_PER_NODE = 32
MAX_NODES = 24
RISK_LEVELS = {"low", "medium", "high", "critical"}
NODE_STATUSES = {
    "pending", "ready", "running", "validating", "waiting_approval",
    "retrying", "replanning", "reconciling", "completed", "failed", "cancelled",
}
WORKFLOW_STATUSES = {"planning", "ready", "running", "waiting_approval", "failed", "completed", "cancelled"}
EXECUTION_MODES = {"live", "dry-run"}
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
DOMAIN_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
OPERATION_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
VOLATILE_INPUT_KEYS = {
    "workflow_id",
    "context",
    "repair_feedback",
    "approval_issue",
    "approval_granted",
    "approval_fingerprint",
    "approval_actor",
    "approval_approved_at",
}


class StateSchemaError(RuntimeError):
    pass


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _strict_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise StateSchemaError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[+-]?\\d+", stripped):
            return int(stripped)
    raise StateSchemaError(f"{field_name} must be an integer")


def _strict_schema_version(value: Any, *, default: int, field_name: str) -> int:
    if value is None:
        return default
    return _strict_int(value, field_name=field_name)


def _strict_bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
    field_name: str,
) -> int:
    parsed = default if value is None else _strict_int(value, field_name=field_name)
    if parsed < minimum or parsed > maximum:
        raise StateSchemaError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return parsed


def _strict_optional_positive_int(
    value: Any,
    *,
    field_name: str,
) -> int | None:
    if value is None:
        return None
    parsed = _strict_int(value, field_name=field_name)
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
    if version < 1:
        raise StateSchemaError("state.version must be >= 1")
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
        if workflow_version < 1:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.schema_version must be >= 1"
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
        workflow.setdefault("executions", {})
        workflow.setdefault(
            "callback",
            {"status": "pending" if workflow.get("execution_id") else "disabled"},
        )
        workflow.setdefault("route_snapshot", {})
        for field_name in (
            "repair_feedback",
            "evidence",
            "reconciliations",
            "executions",
            "route_snapshot",
            "callback",
        ):
            if not isinstance(workflow.get(field_name), dict):
                raise StateSchemaError(
                    f"workflow {workflow_id!r}.{field_name} must be an object"
                )
        workflow.setdefault("policy_fingerprint", None)
        workflow.setdefault("route_snapshot", {})
        for field_name in ("plan_fingerprint", "policy_fingerprint"):
            value = workflow.get(field_name)
            if value not in (None, ""):
                if not isinstance(value, str) or not HEX64_RE.fullmatch(value):
                    raise StateSchemaError(
                        f"workflow {workflow_id!r}.{field_name} must be a 64-character lowercase hex digest"
                    )
        for field_name in ("event_id", "idempotency_key"):
            value = workflow.get(field_name)
            if value not in (None, ""):
                if not isinstance(value, str) or len(value) > 128:
                    raise StateSchemaError(
                        f"workflow {workflow_id!r}.{field_name} must be a string of <= 128 characters"
                    )
        for field_name in ("execution_id", "intent_fingerprint", "input_digest"):
            value = workflow.get(field_name)
            if value is not None and value != "":
                if not isinstance(value, str) or not HEX64_RE.fullmatch(value):
                    raise StateSchemaError(
                        f"workflow {workflow_id!r}.{field_name} must be a 64-character lowercase hex digest"
                    )
        for field_name in ("external_workflow_id", "parent_execution_id"):
            value = workflow.get(field_name)
            if value is not None and value != "" and not SAFE_ID_RE.fullmatch(str(value)):
                raise StateSchemaError(
                    f"workflow {workflow_id!r}.{field_name} contains unsafe characters"
                )
        external_domain = workflow.get("external_domain")
        if external_domain:
            if not isinstance(external_domain, str) or not DOMAIN_RE.fullmatch(external_domain):
                raise StateSchemaError(
                    f"workflow {workflow_id!r}.external_domain is invalid"
                )
        external_operation = workflow.get("external_operation")
        if external_operation:
            if not isinstance(external_operation, str) or not OPERATION_RE.fullmatch(external_operation):
                raise StateSchemaError(
                    f"workflow {workflow_id!r}.external_operation is invalid"
                )
        idempotency_key = workflow.get("idempotency_key")
        if idempotency_key is not None and len(str(idempotency_key)) > 128:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.idempotency_key is too long"
            )
        try:
            external_attempt = int(workflow.get("external_attempt", 1))
        except (TypeError, ValueError) as exc:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.external_attempt must be an integer"
            ) from exc
        if external_attempt < 1 or external_attempt > 1000:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.external_attempt out of bounds"
            )
        workflow["external_attempt"] = external_attempt
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
        callback = workflow["callback"]
        callback_statuses = {"disabled", "pending", "sent", "dead_letter"}
        callback_status = str(callback.get("status") or "disabled")
        if callback_status not in callback_statuses:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.callback.status is invalid"
            )
        try:
            callback_attempts = int(callback.get("attempts", 0))
        except (TypeError, ValueError) as exc:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.callback.attempts must be an integer"
            ) from exc
        if callback_attempts < 0 or callback_attempts > MAX_CALLBACK_ATTEMPTS:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.callback.attempts out of bounds"
            )
        target_fingerprint = callback.get("target_fingerprint")
        if target_fingerprint not in (None, ""):
            if not isinstance(target_fingerprint, str) or not HEX64_RE.fullmatch(target_fingerprint):
                raise StateSchemaError(
                    f"workflow {workflow_id!r}.callback.target_fingerprint is invalid"
                )
        callback["status"] = callback_status
        callback["attempts"] = callback_attempts
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
        if workflow.get("live") is not None and not isinstance(workflow.get("live"), bool):
            raise StateSchemaError(f"workflow {workflow_id!r}.live must be boolean")
        if workflow.get("execution_mode") not in EXECUTION_MODES:
            raise StateSchemaError(f"workflow {workflow_id!r}.execution_mode is invalid")
        workflow.setdefault("plan_fingerprint", None)
        if "plan_integrity" not in workflow:
            workflow["plan_integrity"] = (
                "legacy_unverified" if workflow.get("plan_fingerprint") is None else "pending"
            )

        workflow["max_parallel"] = _strict_bounded_int(
            workflow.get("max_parallel"),
            default=4,
            minimum=1,
            maximum=MAX_PARALLEL,
            field_name=f"workflow {workflow_id!r}.max_parallel",
        )

        nodes = workflow.get("nodes", [])
        if not isinstance(nodes, list):
            raise StateSchemaError(f"workflow {workflow_id!r}.nodes must be an array")
        if len(nodes) > MAX_NODES:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.nodes count exceeds {MAX_NODES}"
            )
        node_ids: set[str] = set()
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
            if not SAFE_ID_RE.fullmatch(node["id"]):
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node {node['id']!r} contains unsafe characters"
                )
            if node["id"] in node_ids:
                raise StateSchemaError(
                    f"workflow {workflow_id!r} contains duplicate node id {node['id']!r}"
                )
            node_ids.add(node["id"])
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
            if any(dep == node["id"] for dep in node["depends_on"]):
                raise StateSchemaError(
                    f"workflow {workflow_id!r} node {node['id']!r} depends on itself"
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
            intent_input = {
                key: value
                for key, value in node["input"].items()
                if key not in VOLATILE_INPUT_KEYS
            }
            intent_serialized = json.dumps(
                {
                    "id": node["id"],
                    "capability": node["capability"],
                    "tool": node["tool"],
                    "depends_on": node["depends_on"],
                    "risk": node["risk"],
                    "input": intent_input,
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
        for node in nodes:
            for dependency in node["depends_on"]:
                if dependency not in node_ids:
                    raise StateSchemaError(
                        f"workflow {workflow_id!r} node {node['id']!r} references unknown dependency {dependency!r}"
                    )

    state["workflows"] = workflows
    state.setdefault("last_workflow_id", None)
    state["version"] = CURRENT_STATE_VERSION
    return state
