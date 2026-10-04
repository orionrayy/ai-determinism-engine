#!/usr/bin/env python3
"""Typed, bounded protocol for federated agent workers.

Artifacts are transport only. The supervisor/state ledger remains authoritative.
Only low/medium-risk, parallel-safe tasks may cross this boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

try:
    from .agent_fabric import AGENT_PROTOCOL_VERSION, AGENTS, RISK_RANK, agent_id, validate_role
except ImportError:
    from agent_fabric import AGENT_PROTOCOL_VERSION, AGENTS, RISK_RANK, agent_id, validate_role


FEDERATION_PROTOCOL_VERSION = 2
MAX_FEDERATED_TASKS = 8
MAX_TASK_JSON_BYTES = 8 * 1024
MAX_FEDERATION_MANIFEST_BYTES = 48 * 1024
MAX_RESULT_JSON_BYTES = 16 * 1024
MAX_RESULT_OUTPUT_BYTES = 12 * 1024
MAX_RESULT_ERROR_BYTES = 4 * 1024
MAX_TASK_CONTEXT_BYTES = 8 * 1024
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
ALLOWED_TASK_RISK = {"low", "medium"}


class FederationProtocolError(ValueError):
    pass


try:
    from .deterministic_codec import canonical_json, digest
except ImportError:
    from deterministic_codec import canonical_json, digest


def _bounded_dict(value: Any, max_bytes: int, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FederationProtocolError(f"{label} must be an object")
    if len(canonical_json(value)) > max_bytes:
        raise FederationProtocolError(f"{label} exceeds {max_bytes} bytes")
    return value


def _validate_safe_id(label: str, value: str) -> None:
    if not _SAFE_ID.fullmatch(str(value).strip()):
        raise FederationProtocolError(
            f"{label} contains unsupported characters or length"
        )


@dataclass(frozen=True)
class AgentTask:
    federation_id: str
    workflow_id: str
    task_id: str
    agent_id: str
    role: str
    capability: str
    tool: str
    risk: str
    instruction: str
    context: dict[str, Any]
    contract: dict[str, Any]
    attempt: int
    input_digest: str
    protocol_version: int = FEDERATION_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "federation_id": self.federation_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "role": self.role,
            "capability": self.capability,
            "tool": self.tool,
            "risk": self.risk,
            "instruction": self.instruction,
            "context": self.context,
            "contract": self.contract,
            "attempt": self.attempt,
            "input_digest": self.input_digest,
        }


@dataclass(frozen=True)
class AgentResult:
    federation_id: str
    workflow_id: str
    task_id: str
    agent_id: str
    role: str
    capability: str
    tool: str
    attempt: int
    input_digest: str
    status: str
    output: dict[str, Any]
    output_sha256: str
    error: dict[str, Any] | None = None
    worker_run_id: str = ""
    protocol_version: int = FEDERATION_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "federation_id": self.federation_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "role": self.role,
            "capability": self.capability,
            "tool": self.tool,
            "attempt": self.attempt,
            "input_digest": self.input_digest,
            "status": self.status,
            "output": self.output,
            "output_sha256": self.output_sha256,
            "error": self.error,
            "worker_run_id": self.worker_run_id,
        }


def build_task(
    *,
    federation_id: str,
    workflow_id: str,
    task_id: str,
    role: str,
    capability: str,
    tool: str,
    risk: str,
    instruction: str,
    context: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
    attempt: int = 1,
) -> AgentTask:
    federation_id = str(federation_id).strip()
    workflow_id = str(workflow_id).strip()
    task_id = str(task_id).strip()
    role = str(role).strip()
    capability = str(capability).strip()
    tool = str(tool).strip()
    risk = str(risk).strip()
    instruction = str(instruction).strip()

    for label, value in (
        ("federation_id", federation_id),
        ("workflow_id", workflow_id),
        ("task_id", task_id),
    ):
        _validate_safe_id(label, value)

    if risk not in ALLOWED_TASK_RISK:
        raise FederationProtocolError(
            f"federated task risk must be low/medium, got {risk}"
        )

    profile = validate_role(role, capability, risk)
    if not profile.parallel_safe:
        raise FederationProtocolError(
            f"agent role {role} is not parallel-safe"
        )
    if RISK_RANK[risk] > RISK_RANK[profile.risk_ceiling]:
        raise FederationProtocolError(
            f"risk {risk} exceeds agent role ceiling"
        )
    if not instruction:
        raise FederationProtocolError("instruction is required")

    context = _bounded_dict(
        context or {},
        MAX_TASK_CONTEXT_BYTES,
        "context",
    )
    contract = _bounded_dict(
        contract or {},
        4 * 1024,
        "contract",
    )

    try:
        attempt = int(attempt)
    except (TypeError, ValueError) as exc:
        raise FederationProtocolError("attempt must be an integer") from exc
    if attempt < 1:
        raise FederationProtocolError("attempt must be >= 1")

    identity_payload = {
        "workflow_id": workflow_id,
        "task_id": task_id,
        "role": role,
        "capability": capability,
        "tool": tool,
        "risk": risk,
        "instruction": instruction,
        "context": context,
        "contract": contract,
        "attempt": attempt,
    }
    task = AgentTask(
        federation_id=federation_id,
        workflow_id=workflow_id,
        task_id=task_id,
        agent_id=agent_id(workflow_id, task_id, role),
        role=role,
        capability=capability,
        tool=tool,
        risk=risk,
        instruction=instruction,
        context=context,
        contract=contract,
        attempt=attempt,
        input_digest=digest(identity_payload),
    )
    validate_task(task)
    if len(canonical_json(task.to_dict())) > MAX_TASK_JSON_BYTES:
        raise FederationProtocolError("serialized task exceeds task size limit")
    return task


def validate_task(task: AgentTask) -> None:
    if task.protocol_version != FEDERATION_PROTOCOL_VERSION:
        raise FederationProtocolError("unsupported federation task protocol version")
    if task.risk not in ALLOWED_TASK_RISK:
        raise FederationProtocolError("federated task contains disallowed risk")
    validate_role(task.role, task.capability, task.risk)
    if not AGENTS[task.role].parallel_safe:
        raise FederationProtocolError("federated task role is not parallel-safe")
    if task.agent_id != agent_id(task.workflow_id, task.task_id, task.role):
        raise FederationProtocolError("agent identity mismatch")
    for label, value in (
        ("federation_id", task.federation_id),
        ("workflow_id", task.workflow_id),
        ("task_id", task.task_id),
    ):
        _validate_safe_id(label, value)
    if task.attempt < 1:
        raise FederationProtocolError("task attempt must be >= 1")

    identity_payload = {
        "workflow_id": task.workflow_id,
        "task_id": task.task_id,
        "role": task.role,
        "capability": task.capability,
        "tool": task.tool,
        "risk": task.risk,
        "instruction": task.instruction,
        "context": task.context,
        "contract": task.contract,
        "attempt": task.attempt,
    }
    if task.input_digest != digest(identity_payload):
        raise FederationProtocolError("task input digest mismatch")
    _bounded_dict(task.context, MAX_TASK_CONTEXT_BYTES, "context")
    _bounded_dict(task.contract, 4 * 1024, "contract")


def task_from_dict(value: dict[str, Any]) -> AgentTask:
    value = _bounded_dict(value, MAX_TASK_JSON_BYTES, "task")
    try:
        task = AgentTask(
            federation_id=str(value.get("federation_id") or ""),
            workflow_id=str(value.get("workflow_id") or ""),
            task_id=str(value.get("task_id") or ""),
            agent_id=str(value.get("agent_id") or ""),
            role=str(value.get("role") or ""),
            capability=str(value.get("capability") or ""),
            tool=str(value.get("tool") or ""),
            risk=str(value.get("risk") or ""),
            instruction=str(value.get("instruction") or ""),
            context=dict(value.get("context") or {}),
            contract=dict(value.get("contract") or {}),
            attempt=int(value.get("attempt") or 0),
            input_digest=str(value.get("input_digest") or ""),
            protocol_version=int(value.get("protocol_version") or 0),
        )
    except (TypeError, ValueError) as exc:
        raise FederationProtocolError("invalid task field types") from exc
    validate_task(task)
    return task


def build_result(
    task: AgentTask,
    *,
    status: str,
    output: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    worker_run_id: str = "",
) -> AgentResult:
    validate_task(task)
    status = str(status)
    if status not in {"completed", "failed"}:
        raise FederationProtocolError(
            "result status must be completed or failed"
        )
    output = _bounded_dict(output or {}, MAX_RESULT_OUTPUT_BYTES, "output")
    if error is not None:
        _bounded_dict(error, MAX_RESULT_ERROR_BYTES, "error")

    result = AgentResult(
        federation_id=task.federation_id,
        workflow_id=task.workflow_id,
        task_id=task.task_id,
        agent_id=task.agent_id,
        role=task.role,
        capability=task.capability,
        tool=task.tool,
        attempt=task.attempt,
        input_digest=task.input_digest,
        status=status,
        output=output,
        output_sha256=digest(output),
        error=error,
        worker_run_id=str(worker_run_id),
    )
    validate_result(result, expected_task=task)
    if len(canonical_json(result.to_dict())) > MAX_RESULT_JSON_BYTES:
        raise FederationProtocolError(
            "serialized result exceeds result size limit"
        )
    return result


def validate_result(
    result: AgentResult,
    *,
    expected_task: AgentTask | None = None,
) -> None:
    if result.protocol_version != FEDERATION_PROTOCOL_VERSION:
        raise FederationProtocolError("unsupported federation result protocol version")
    if result.status not in {"completed", "failed"}:
        raise FederationProtocolError("invalid federation result status")
    _validate_safe_id("federation_id", result.federation_id)
    _validate_safe_id("workflow_id", result.workflow_id)
    _validate_safe_id("task_id", result.task_id)
    validate_role(result.role, result.capability, "low")
    if not AGENTS[result.role].parallel_safe:
        raise FederationProtocolError("result role is not parallel-safe")
    if result.agent_id != agent_id(result.workflow_id, result.task_id, result.role):
        raise FederationProtocolError("result agent identity mismatch")
    if result.attempt < 1:
        raise FederationProtocolError("result attempt must be >= 1")
    _bounded_dict(result.output, MAX_RESULT_OUTPUT_BYTES, "output")
    if result.output_sha256 != digest(result.output):
        raise FederationProtocolError("result output digest mismatch")
    if result.error is not None:
        _bounded_dict(result.error, MAX_RESULT_ERROR_BYTES, "error")
    if expected_task is not None:
        validate_task(expected_task)
        fields = (
            "federation_id",
            "workflow_id",
            "task_id",
            "agent_id",
            "role",
            "capability",
            "tool",
            "attempt",
            "input_digest",
        )
        if any(
            getattr(result, field) != getattr(expected_task, field)
            for field in fields
        ):
            raise FederationProtocolError("result does not match expected task")


def build_manifest(
    federation_id: str,
    tasks: list[AgentTask],
) -> dict[str, Any]:
    federation_id = str(federation_id).strip()
    _validate_safe_id("federation_id", federation_id)
    if not tasks or len(tasks) > MAX_FEDERATED_TASKS:
        raise FederationProtocolError("federation must contain 1..8 tasks")
    task_ids = {task.task_id for task in tasks}
    if len(task_ids) != len(tasks):
        raise FederationProtocolError("federation contains duplicate task ids")
    for task in tasks:
        validate_task(task)
        if task.federation_id != federation_id:
            raise FederationProtocolError("task federation identity mismatch")
    workflow_ids = {task.workflow_id for task in tasks}
    if len(workflow_ids) != 1:
        raise FederationProtocolError("tasks must belong to one workflow")

    manifest = {
        "protocol_version": FEDERATION_PROTOCOL_VERSION,
        "federation_id": federation_id,
        "workflow_id": tasks[0].workflow_id,
        "tasks": [task.to_dict() for task in tasks],
    }
    if len(canonical_json(manifest)) > MAX_FEDERATION_MANIFEST_BYTES:
        raise FederationProtocolError("federation manifest exceeds size limit")
    return manifest
