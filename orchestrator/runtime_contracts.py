#!/usr/bin/env python3
"""Portable execution contracts for the enterprise runtime boundary.

These contracts intentionally depend only on the Python standard library.
They are transport-neutral: GitHub Actions, a queue, an HTTP worker, or a
future durable-execution backend can carry the same envelopes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping


CONTRACT_SCHEMA_VERSION = 1


class ContractError(ValueError):
    """Raised when a portable runtime contract is malformed."""


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonically serializable: {exc}") from exc


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _text(value: Any, field_name: str, *, max_length: int = 512) -> str:
    text = str(value or "").strip()
    if not text:
        raise ContractError(f"{field_name} is required")
    if len(text) > max_length:
        raise ContractError(f"{field_name} exceeds {max_length} characters")
    return text


def _digest(value: Any, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ContractError(f"{field_name} must be a 64-character lowercase SHA-256")
    return text


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{field_name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field_name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ContractError(f"{field_name} must be a non-negative integer")
    return parsed


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContractError(f"{field_name} must be an object")
    return dict(value)


def _string_list(value: Any, field_name: str, *, max_items: int = 128) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, list | tuple):
        raise ContractError(f"{field_name} must be an array of strings")
    if len(value) > max_items:
        raise ContractError(f"{field_name} exceeds {max_items} items")
    result = tuple(_text(item, f"{field_name}[]", max_length=256) for item in value)
    if len(set(result)) != len(result):
        raise ContractError(f"{field_name} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    sha256: str
    schema_version: str
    workflow_id: str
    producer_task_id: str
    location: str
    media_type: str = "application/octet-stream"
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id", max_length=256))
        object.__setattr__(self, "sha256", _digest(self.sha256, "sha256"))
        object.__setattr__(self, "schema_version", _text(self.schema_version, "schema_version", max_length=64))
        object.__setattr__(self, "workflow_id", _text(self.workflow_id, "workflow_id", max_length=256))
        object.__setattr__(self, "producer_task_id", _text(self.producer_task_id, "producer_task_id", max_length=256))
        object.__setattr__(self, "location", _text(self.location, "location", max_length=2048))
        object.__setattr__(self, "media_type", _text(self.media_type, "media_type", max_length=256))
        if self.size_bytes is not None:
            object.__setattr__(self, "size_bytes", _nonnegative_int(self.size_bytes, "size_bytes"))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "producer_task_id": self.producer_task_id,
            "location": self.location,
            "media_type": self.media_type,
        }
        if self.size_bytes is not None:
            result["size_bytes"] = self.size_bytes
        return result

    @property
    def digest(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    evidence_id: str
    work_id: str
    source_uri: str
    content_sha256: str
    retrieved_at: str
    access_verification: str = "unknown"
    passage_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id", max_length=256))
        object.__setattr__(self, "work_id", _text(self.work_id, "work_id", max_length=512))
        object.__setattr__(self, "source_uri", _text(self.source_uri, "source_uri", max_length=4096))
        object.__setattr__(self, "content_sha256", _digest(self.content_sha256, "content_sha256"))
        object.__setattr__(self, "retrieved_at", _text(self.retrieved_at, "retrieved_at", max_length=64))
        access = _text(self.access_verification, "access_verification", max_length=64).lower()
        if access not in {"verified", "declared_only", "metadata_only", "unknown"}:
            raise ContractError("access_verification is unsupported")
        object.__setattr__(self, "access_verification", access)
        object.__setattr__(
            self,
            "passage_ids",
            _string_list(list(self.passage_ids), "passage_ids", max_items=32),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "work_id": self.work_id,
            "source_uri": self.source_uri,
            "content_sha256": self.content_sha256,
            "retrieved_at": self.retrieved_at,
            "access_verification": self.access_verification,
            "passage_ids": list(self.passage_ids),
        }

    @property
    def digest(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True, slots=True)
class SkillManifest:
    skill_id: str
    version: str
    digest: str
    capabilities: tuple[str, ...]
    input_schema: str
    output_schema: str
    runtime: str
    entrypoint: str
    trust_class: str = "unreviewed"
    cost_class: str = "unknown"
    risk: str = "medium"
    required_resources: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", _text(self.skill_id, "skill_id", max_length=256))
        object.__setattr__(self, "version", _text(self.version, "version", max_length=64))
        object.__setattr__(self, "digest", _digest(self.digest, "digest"))
        object.__setattr__(self, "capabilities", _string_list(list(self.capabilities), "capabilities"))
        object.__setattr__(self, "input_schema", _text(self.input_schema, "input_schema", max_length=512))
        object.__setattr__(self, "output_schema", _text(self.output_schema, "output_schema", max_length=512))
        object.__setattr__(self, "runtime", _text(self.runtime, "runtime", max_length=128))
        object.__setattr__(self, "entrypoint", _text(self.entrypoint, "entrypoint", max_length=1024))
        trust = _text(self.trust_class, "trust_class", max_length=64).lower()
        risk = _text(self.risk, "risk", max_length=32).lower()
        if risk not in {"low", "medium", "high", "critical"}:
            raise ContractError("risk is unsupported")
        object.__setattr__(self, "trust_class", trust)
        object.__setattr__(self, "cost_class", _text(self.cost_class, "cost_class", max_length=64).lower())
        object.__setattr__(self, "risk", risk)
        object.__setattr__(
            self,
            "required_resources",
            _string_list(list(self.required_resources), "required_resources"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "digest": self.digest,
            "capabilities": list(self.capabilities),
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "runtime": self.runtime,
            "entrypoint": self.entrypoint,
            "trust_class": self.trust_class,
            "cost_class": self.cost_class,
            "risk": self.risk,
            "required_resources": list(self.required_resources),
            "metadata": dict(self.metadata),
        }

    @property
    def manifest_digest(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    version: str
    digest: str
    free_only: bool
    tenant_id: str
    limits: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _text(self.version, "version", max_length=64))
        object.__setattr__(self, "digest", _digest(self.digest, "digest"))
        object.__setattr__(self, "free_only", bool(self.free_only))
        object.__setattr__(self, "tenant_id", _text(self.tenant_id, "tenant_id", max_length=256))
        object.__setattr__(self, "limits", _mapping(self.limits, "limits"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "digest": self.digest,
            "free_only": self.free_only,
            "tenant_id": self.tenant_id,
            "limits": dict(self.limits),
        }

    @property
    def snapshot_digest(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True, slots=True)
class WorkflowCommand:
    workflow_id: str
    tenant_id: str
    goal: str
    input_digest: str
    intent_digest: str
    policy: PolicySnapshot
    command_type: str = "start"
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _text(self.workflow_id, "workflow_id", max_length=256))
        object.__setattr__(self, "tenant_id", _text(self.tenant_id, "tenant_id", max_length=256))
        object.__setattr__(self, "goal", _text(self.goal, "goal", max_length=8192))
        object.__setattr__(self, "input_digest", _digest(self.input_digest, "input_digest"))
        object.__setattr__(self, "intent_digest", _digest(self.intent_digest, "intent_digest"))
        object.__setattr__(self, "command_type", _text(self.command_type, "command_type", max_length=64))
        object.__setattr__(self, "schema_version", _nonnegative_int(self.schema_version, "schema_version"))
        if not isinstance(self.policy, PolicySnapshot):
            raise ContractError("policy must be a PolicySnapshot")
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractError(f"unsupported contract schema version: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "tenant_id": self.tenant_id,
            "goal": self.goal,
            "input_digest": self.input_digest,
            "intent_digest": self.intent_digest,
            "command_type": self.command_type,
            "policy": self.policy.to_dict(),
        }

    @property
    def command_digest(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True, slots=True)
class TaskEnvelope:
    workflow_id: str
    task_id: str
    tenant_id: str
    attempt: int
    capability: str
    skill: SkillManifest
    input_ref: ArtifactRef | None
    policy: PolicySnapshot
    idempotency_key: str
    deadline_epoch: int | None = None
    fence_epoch: int = 0
    trace_id: str = ""
    partition: str = ""
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _text(self.workflow_id, "workflow_id", max_length=256))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id", max_length=256))
        object.__setattr__(self, "tenant_id", _text(self.tenant_id, "tenant_id", max_length=256))
        object.__setattr__(self, "attempt", max(1, _nonnegative_int(self.attempt, "attempt")))
        object.__setattr__(self, "capability", _text(self.capability, "capability", max_length=256))
        if not isinstance(self.skill, SkillManifest):
            raise ContractError("skill must be a SkillManifest")
        if self.input_ref is not None and not isinstance(self.input_ref, ArtifactRef):
            raise ContractError("input_ref must be an ArtifactRef or null")
        if not isinstance(self.policy, PolicySnapshot):
            raise ContractError("policy must be a PolicySnapshot")
        object.__setattr__(self, "idempotency_key", _text(self.idempotency_key, "idempotency_key", max_length=256))
        object.__setattr__(self, "fence_epoch", _nonnegative_int(self.fence_epoch, "fence_epoch"))
        if self.deadline_epoch is not None:
            object.__setattr__(self, "deadline_epoch", _nonnegative_int(self.deadline_epoch, "deadline_epoch"))
        object.__setattr__(self, "trace_id", str(self.trace_id or "").strip()[:256])
        object.__setattr__(self, "partition", str(self.partition or "").strip()[:256])
        object.__setattr__(self, "schema_version", _nonnegative_int(self.schema_version, "schema_version"))
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractError(f"unsupported contract schema version: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "tenant_id": self.tenant_id,
            "attempt": self.attempt,
            "capability": self.capability,
            "skill": self.skill.to_dict(),
            "policy": self.policy.to_dict(),
            "idempotency_key": self.idempotency_key,
            "fence_epoch": self.fence_epoch,
            "trace_id": self.trace_id,
            "partition": self.partition,
        }
        if self.input_ref is not None:
            result["input_ref"] = self.input_ref.to_dict()
        if self.deadline_epoch is not None:
            result["deadline_epoch"] = self.deadline_epoch
        return result

    @property
    def envelope_digest(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True, slots=True)
class TaskResultEnvelope:
    workflow_id: str
    task_id: str
    tenant_id: str
    attempt: int
    outcome: str
    retry_class: str
    output_ref: ArtifactRef | None = None
    evidence_refs: tuple[EvidenceRef, ...] = ()
    error_code: str = ""
    worker_id: str = ""
    trace_id: str = ""
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _text(self.workflow_id, "workflow_id", max_length=256))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id", max_length=256))
        object.__setattr__(self, "tenant_id", _text(self.tenant_id, "tenant_id", max_length=256))
        object.__setattr__(self, "attempt", max(1, _nonnegative_int(self.attempt, "attempt")))
        outcome = _text(self.outcome, "outcome", max_length=64).lower()
        retry_class = _text(self.retry_class, "retry_class", max_length=64).lower()
        if outcome not in {"completed", "failed", "blocked", "cancelled", "unknown"}:
            raise ContractError("outcome is unsupported")
        if retry_class not in {"none", "safe", "reconcile", "blocked"}:
            raise ContractError("retry_class is unsupported")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "retry_class", retry_class)
        if self.output_ref is not None and not isinstance(self.output_ref, ArtifactRef):
            raise ContractError("output_ref must be an ArtifactRef or null")
        refs = tuple(self.evidence_refs or ())
        if len(refs) > 128:
            raise ContractError("evidence_refs exceeds 128 items")
        if any(not isinstance(item, EvidenceRef) for item in refs):
            raise ContractError("evidence_refs must contain EvidenceRef values")
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "error_code", str(self.error_code or "").strip()[:256])
        object.__setattr__(self, "worker_id", str(self.worker_id or "").strip()[:256])
        object.__setattr__(self, "trace_id", str(self.trace_id or "").strip()[:256])
        object.__setattr__(self, "schema_version", _nonnegative_int(self.schema_version, "schema_version"))
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractError(f"unsupported contract schema version: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "tenant_id": self.tenant_id,
            "attempt": self.attempt,
            "outcome": self.outcome,
            "retry_class": self.retry_class,
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "error_code": self.error_code,
            "worker_id": self.worker_id,
            "trace_id": self.trace_id,
        }
        if self.output_ref is not None:
            result["output_ref"] = self.output_ref.to_dict()
        return result

    @property
    def result_digest(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True, slots=True)
class WorkerHeartbeat:
    worker_id: str
    pool: str
    capabilities: tuple[str, ...]
    active_tasks: int
    capacity: int
    emitted_at_epoch: int
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_id", _text(self.worker_id, "worker_id", max_length=256))
        object.__setattr__(self, "pool", _text(self.pool, "pool", max_length=256))
        object.__setattr__(self, "capabilities", _string_list(list(self.capabilities), "capabilities"))
        object.__setattr__(self, "active_tasks", _nonnegative_int(self.active_tasks, "active_tasks"))
        object.__setattr__(self, "capacity", max(1, _nonnegative_int(self.capacity, "capacity")))
        object.__setattr__(self, "emitted_at_epoch", _nonnegative_int(self.emitted_at_epoch, "emitted_at_epoch"))
        if self.active_tasks > self.capacity:
            raise ContractError("active_tasks cannot exceed capacity")
        object.__setattr__(self, "schema_version", _nonnegative_int(self.schema_version, "schema_version"))
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractError(f"unsupported contract schema version: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "worker_id": self.worker_id,
            "pool": self.pool,
            "capabilities": list(self.capabilities),
            "active_tasks": self.active_tasks,
            "capacity": self.capacity,
            "emitted_at_epoch": self.emitted_at_epoch,
        }

    @property
    def heartbeat_digest(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True, slots=True)
class QueueClaim:
    queue_id: str
    task_id: str
    worker_id: str
    claim_id: str
    attempt: int
    lease_expires_at: int
    fence_epoch: int = 0
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "queue_id", _text(self.queue_id, "queue_id", max_length=256))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id", max_length=256))
        object.__setattr__(self, "worker_id", _text(self.worker_id, "worker_id", max_length=256))
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id", max_length=256))
        object.__setattr__(self, "attempt", max(1, _nonnegative_int(self.attempt, "attempt")))
        object.__setattr__(self, "lease_expires_at", _nonnegative_int(self.lease_expires_at, "lease_expires_at"))
        object.__setattr__(self, "fence_epoch", _nonnegative_int(self.fence_epoch, "fence_epoch"))
        object.__setattr__(self, "schema_version", _nonnegative_int(self.schema_version, "schema_version"))
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractError(f"unsupported contract schema version: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "queue_id": self.queue_id,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "claim_id": self.claim_id,
            "attempt": self.attempt,
            "lease_expires_at": self.lease_expires_at,
            "fence_epoch": self.fence_epoch,
        }

    @property
    def claim_digest(self) -> str:
        return sha256_hex(self.to_dict())


def contract_digest(value: Any) -> str:
    """Digest a contract or plain JSON-compatible value deterministically."""
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    return sha256_hex(payload)
