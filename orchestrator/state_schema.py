#!/usr/bin/env python3
from __future__ import annotations

from typing import Any
import hashlib
import re

try:
    from .federation_scheduler import (
        DEFAULT_MAX_BATCHES_PER_WORKFLOW,
        DEFAULT_MAX_TASKS_PER_WORKFLOW,
        MAX_BATCHES_PER_WORKFLOW,
        MAX_TASKS_PER_WORKFLOW,
    )
except ImportError:
    from federation_scheduler import (
        DEFAULT_MAX_BATCHES_PER_WORKFLOW,
        DEFAULT_MAX_TASKS_PER_WORKFLOW,
        MAX_BATCHES_PER_WORKFLOW,
        MAX_TASKS_PER_WORKFLOW,
    )

CURRENT_STATE_VERSION = 4
CURRENT_WORKFLOW_SCHEMA_VERSION = 8
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
        workflow.setdefault("agent_team", {})
        workflow.setdefault("workload", {})
        workflow.setdefault("github_run_attempt", None)
        workflow.setdefault("origin_github_run_attempt", workflow.get("github_run_attempt"))
        workflow.setdefault("private_input_ref", None)
        workflow.setdefault("idempotency_key", None)
        for field_name in (
            "private_input_ref",
            "input_digest",
            "intent_fingerprint",
            "ingress_intent_digest",
            "plan_intent_fingerprint",
            "provider_resolution_fingerprint",
        ):
            value = workflow.get(field_name)
            if value in (None, ""):
                continue
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise StateSchemaError(
                    f"workflow {workflow_id!r}.{field_name} must be a 64-character lowercase hexadecimal string"
                )
        private_ref = workflow.get("private_input_ref")
        if private_ref not in (None, ""):
            if workflow.get("execution_id") in (None, "") or workflow.get("input_digest") in (None, ""):
                raise StateSchemaError(
                    f"workflow {workflow_id!r} private_input_ref requires execution_id and input_digest"
                )
        workload = workflow.get("workload")
        if not isinstance(workload, dict):
            raise StateSchemaError(
                f"workflow {workflow_id!r}.workload must be an object"
            )
        for field_name in ("blueprint_digest", "manifest_digest"):
            value = workload.get(field_name)
            if value in (None, ""):
                continue
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise StateSchemaError(
                    f"workflow {workflow_id!r}.workload.{field_name} must be a 64-character lowercase hexadecimal string"
                )
        completed_units = workload.get("completed_unit_ids", [])
        if not isinstance(completed_units, list) or len(completed_units) > 256:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.workload.completed_unit_ids must be a list of <=256 items"
            )
        if workflow.get("idempotency_key") not in (None, ""):
            if not isinstance(workflow["idempotency_key"], str) or len(workflow["idempotency_key"]) > 128:
                raise StateSchemaError(
                    f"workflow {workflow_id!r}.idempotency_key must be at most 128 characters"
                )
        for field_name in ("github_run_attempt", "origin_github_run_attempt"):
            value = workflow.get(field_name)
            if value is not None:
                if isinstance(value, bool):
                    raise StateSchemaError(
                        f"workflow {workflow_id!r}.{field_name} must be a positive integer or null"
                    )
                try:
                    parsed = int(value)
                except (TypeError, ValueError) as exc:
                    raise StateSchemaError(
                        f"workflow {workflow_id!r}.{field_name} must be a positive integer or null"
                    ) from exc
                if parsed < 1:
                    raise StateSchemaError(
                        f"workflow {workflow_id!r}.{field_name} must be a positive integer or null"
                    )
                workflow[field_name] = parsed
        workflow.setdefault("federation", {})
        workflow.setdefault(
            "max_federation_batches",
            DEFAULT_MAX_BATCHES_PER_WORKFLOW,
        )
        workflow["max_federation_batches"] = max(
            0,
            min(
                _as_int(
                    workflow.get("max_federation_batches"),
                    DEFAULT_MAX_BATCHES_PER_WORKFLOW,
                ),
                MAX_BATCHES_PER_WORKFLOW,
            ),
        )
        workflow.setdefault(
            "max_federation_tasks",
            DEFAULT_MAX_TASKS_PER_WORKFLOW,
        )
        workflow["max_federation_tasks"] = max(
            0,
            min(
                _as_int(
                    workflow.get("max_federation_tasks"),
                    DEFAULT_MAX_TASKS_PER_WORKFLOW,
                ),
                MAX_TASKS_PER_WORKFLOW,
            ),
        )
        workflow.setdefault("federation_batches_used", 0)
        workflow["federation_batches_used"] = max(
            0, _as_int(workflow.get("federation_batches_used"), 0)
        )
        workflow.setdefault("federation_tasks_used", 0)
        workflow["federation_tasks_used"] = max(
            0, _as_int(workflow.get("federation_tasks_used"), 0)
        )
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
        try:
            from .authority import infer_legacy_authority, VALID_AUTHORITY_MODES
        except ImportError:
            from authority import infer_legacy_authority, VALID_AUTHORITY_MODES
        workflow.setdefault("authority_mode", infer_legacy_authority(workflow))
        authority_mode = str(workflow.get("authority_mode") or "").strip()
        if authority_mode not in VALID_AUTHORITY_MODES:
            raise StateSchemaError(
                f"workflow {workflow_id!r}.authority_mode is invalid"
            )
        workflow.setdefault("plan_fingerprint", None)
        workflow.setdefault("plan_intent_fingerprint", None)
        workflow.setdefault("provider_resolution_fingerprint", None)
        provider_change = workflow.get("provider_resolution_change")
        if provider_change is not None and not isinstance(provider_change, dict):
            raise StateSchemaError(
                f"workflow {workflow_id!r}.provider_resolution_change must be an object or null"
            )
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
            node.setdefault("resources", [])

    state["workflows"] = workflows
    state.setdefault("last_workflow_id", None)
    state["version"] = CURRENT_STATE_VERSION
    return state
