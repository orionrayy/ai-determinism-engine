from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

PROTOCOL_VERSION = "1.0"


class ContractError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"not canonically serializable: {exc}") from exc


def canonical_digest(value: Any) -> str:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    return "sha256:" + hashlib.sha256(canonical_json(payload)).hexdigest()


def _text(value: Any, name: str, limit: int = 1024) -> str:
    value = str(value or "").strip()
    if not value:
        raise ContractError(f"{name} is required")
    if len(value) > limit:
        raise ContractError(f"{name} exceeds {limit} characters")
    return value


def _sha(value: Any, name: str) -> str:
    value = str(value or "").strip().lower()
    if not value.startswith("sha256:") or len(value) != 71:
        raise ContractError(f"{name} must be sha256:<64 hex>")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ContractError(f"{name} must be sha256:<64 hex>") from exc
    return value


def _unique_strings(value: Any, name: str, limit: int = 128) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ContractError(f"{name} must be an array")
    if len(value) > limit:
        raise ContractError(f"{name} exceeds {limit} items")
    result = tuple(_text(v, f"{name}[]", 256) for v in value)
    if len(set(result)) != len(result):
        raise ContractError(f"{name} must be unique")
    return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    sha256: str
    schema_version: str
    workflow_id: str
    producer_task_id: str
    tenant_id: str
    location: str
    media_type: str = "application/octet-stream"
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        for n in ("artifact_id","schema_version","workflow_id","producer_task_id","tenant_id","location"):
            object.__setattr__(self, n, _text(getattr(self, n), n))
        object.__setattr__(self, "sha256", _sha(self.sha256, "sha256"))
        object.__setattr__(self, "media_type", _text(self.media_type, "media_type", 256))
        if self.size_bytes is not None:
            if isinstance(self.size_bytes, bool) or int(self.size_bytes) < 0:
                raise ContractError("size_bytes must be non-negative")
            object.__setattr__(self, "size_bytes", int(self.size_bytes))

    def to_dict(self) -> dict[str, Any]:
        out = {
            "artifact_id": self.artifact_id, "sha256": self.sha256,
            "schema_version": self.schema_version, "workflow_id": self.workflow_id,
            "producer_task_id": self.producer_task_id, "tenant_id": self.tenant_id,
            "location": self.location, "media_type": self.media_type,
        }
        if self.size_bytes is not None:
            out["size_bytes"] = self.size_bytes
        return out


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
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id", 256))
        object.__setattr__(self, "work_id", _text(self.work_id, "work_id", 512))
        object.__setattr__(self, "source_uri", _text(self.source_uri, "source_uri", 4096))
        object.__setattr__(self, "content_sha256", _sha(self.content_sha256, "content_sha256"))
        object.__setattr__(self, "retrieved_at", _text(self.retrieved_at, "retrieved_at", 64))
        access = _text(self.access_verification, "access_verification", 64).lower()
        if access not in {"verified","declared_only","metadata_only","unknown"}:
            raise ContractError("unsupported access_verification")
        object.__setattr__(self, "access_verification", access)
        object.__setattr__(self, "passage_ids", _unique_strings(self.passage_ids, "passage_ids", 32))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id, "work_id": self.work_id,
            "source_uri": self.source_uri, "content_sha256": self.content_sha256,
            "retrieved_at": self.retrieved_at, "access_verification": self.access_verification,
            "passage_ids": list(self.passage_ids),
        }


@dataclass(frozen=True, slots=True)
class SkillManifest:
    skill_id: str
    version: str
    digest: str
    description: str
    capabilities: tuple[str, ...]
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    trust_level: str
    billing_class: str
    side_effecting: bool
    required_approval: bool
    permissions: Mapping[str, Any] = field(default_factory=dict)
    resource_profile: Mapping[str, Any] = field(default_factory=dict)
    adapters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", _text(self.skill_id, "skill_id", 160).lower())
        object.__setattr__(self, "version", _text(self.version, "version", 64))
        object.__setattr__(self, "digest", _sha(self.digest, "digest"))
        object.__setattr__(self, "description", _text(self.description, "description", 4096))
        object.__setattr__(self, "capabilities", _unique_strings(self.capabilities, "capabilities", 64))
        object.__setattr__(self, "input_schema", _mapping(self.input_schema, "input_schema"))
        object.__setattr__(self, "output_schema", _mapping(self.output_schema, "output_schema"))
        trust = _text(self.trust_level, "trust_level", 32).lower()
        billing = _text(self.billing_class, "billing_class", 64).lower()
        if trust not in {"untrusted","standard","trusted","isolated"}:
            raise ContractError("unsupported trust_level")
        if billing not in {"free_public","free_allowance","credentialed_optional","paid","unknown"}:
            raise ContractError("unsupported billing_class")
        if not isinstance(self.side_effecting, bool) or not isinstance(self.required_approval, bool):
            raise ContractError("side_effecting/required_approval must be boolean")
        object.__setattr__(self, "trust_level", trust)
        object.__setattr__(self, "billing_class", billing)
        object.__setattr__(self, "permissions", _mapping(self.permissions, "permissions"))
        object.__setattr__(self, "resource_profile", _mapping(self.resource_profile, "resource_profile"))
        object.__setattr__(self, "adapters", _mapping(self.adapters, "adapters"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id, "version": self.version, "digest": self.digest,
            "description": self.description, "capabilities": list(self.capabilities),
            "input_schema": dict(self.input_schema), "output_schema": dict(self.output_schema),
            "permissions": dict(self.permissions), "resource_profile": dict(self.resource_profile),
            "trust": {"level": self.trust_level},
            "policy": {"billing_class": self.billing_class, "side_effecting": self.side_effecting,
                       "required_approval": self.required_approval},
            "adapters": dict(self.adapters),
        }


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    version: str
    digest: str
    tenant_id: str
    free_only: bool
    limits: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _text(self.version, "version", 64))
        object.__setattr__(self, "digest", _sha(self.digest, "digest"))
        object.__setattr__(self, "tenant_id", _text(self.tenant_id, "tenant_id", 128))
        if not isinstance(self.free_only, bool):
            raise ContractError("free_only must be boolean")
        limits = _mapping(self.limits, "limits")
        clean: dict[str, int] = {}
        for k, v in limits.items():
            if isinstance(v, bool) or int(v) < 0:
                raise ContractError(f"limit {k} must be non-negative")
            clean[str(k)] = int(v)
        object.__setattr__(self, "limits", clean)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "digest": self.digest, "tenant_id": self.tenant_id,
                "free_only": self.free_only, "limits": dict(self.limits)}


@dataclass(frozen=True, slots=True)
class TaskEnvelope:
    protocol_version: str
    task_id: str
    workflow_id: str
    tenant_id: str
    capability: str
    task_kind: str
    attempt: int
    input: Mapping[str, Any]
    contract: Mapping[str, Any]
    routing: Mapping[str, Any]
    deadline: Mapping[str, Any]
    trace: Mapping[str, str]
    budget: Mapping[str, Any] = field(default_factory=dict)
    parent: Mapping[str, Any] = field(default_factory=dict)
    skill: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_version", _text(self.protocol_version, "protocol_version", 32))
        for n, lim in (("task_id",128),("workflow_id",128),("tenant_id",128),("capability",128),("task_kind",64)):
            object.__setattr__(self, n, _text(getattr(self,n), n, lim))
        if self.task_kind not in {"workflow","activity","research","agent","validation","recovery"}:
            raise ContractError("unsupported task_kind")
        if isinstance(self.attempt, bool) or not (1 <= int(self.attempt) <= 128):
            raise ContractError("attempt must be 1..128")
        object.__setattr__(self, "attempt", int(self.attempt))
        for n in ("input","contract","routing","deadline","trace","budget","parent","skill","metadata"):
            object.__setattr__(self, n, _mapping(getattr(self,n), n))
        if set(("mode",)).difference(self.input):
            raise ContractError("input.mode is required")
        if self.input["mode"] not in {"inline","reference"}:
            raise ContractError("input.mode unsupported")
        if self.input["mode"] == "reference" and not self.input.get("sha256"):
            raise ContractError("reference input requires sha256")
        if set(("trace_id","span_id")).difference(self.trace):
            raise ContractError("trace requires trace_id/span_id")

    def to_dict(self) -> dict[str, Any]:
        out = {
            "protocol_version": self.protocol_version, "task_id": self.task_id,
            "workflow_id": self.workflow_id, "tenant_id": self.tenant_id,
            "capability": self.capability, "task_kind": self.task_kind,
            "attempt": self.attempt, "input": dict(self.input),
            "contract": dict(self.contract), "routing": dict(self.routing),
            "deadline": dict(self.deadline), "trace": dict(self.trace),
        }
        for key in ("skill","budget","parent","metadata"):
            value = getattr(self, key)
            if value: out[key] = dict(value)
        return out


@dataclass(frozen=True, slots=True)
class TaskResultEnvelope:
    protocol_version: str
    task_id: str
    workflow_id: str
    tenant_id: str
    attempt: int
    outcome: str
    retry_class: str
    output: Mapping[str, Any]
    trace: Mapping[str, str]
    error_code: str = ""
    worker_id: str = ""
    evidence_refs: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_version", _text(self.protocol_version, "protocol_version", 32))
        for n, lim in (("task_id",128),("workflow_id",128),("tenant_id",128)):
            object.__setattr__(self, n, _text(getattr(self,n), n, lim))
        if isinstance(self.attempt, bool) or int(self.attempt) < 1:
            raise ContractError("attempt must be >= 1")
        object.__setattr__(self, "attempt", int(self.attempt))
        outcome = _text(self.outcome, "outcome", 32).lower()
        retry_class = _text(self.retry_class, "retry_class", 32).lower()
        if outcome not in {"completed","failed","blocked","cancelled","unknown"}:
            raise ContractError("unsupported outcome")
        if retry_class not in {"none","safe","reconcile","blocked"}:
            raise ContractError("unsupported retry_class")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "retry_class", retry_class)
        object.__setattr__(self, "output", _mapping(self.output, "output"))
        object.__setattr__(self, "trace", _mapping(self.trace, "trace"))
        if "trace_id" not in self.trace or "span_id" not in self.trace:
            raise ContractError("trace requires trace_id/span_id")
        refs = tuple(self.evidence_refs or ())
        if len(refs) > 128 or any(not isinstance(r, EvidenceRef) for r in refs):
            raise ContractError("invalid evidence_refs")
        object.__setattr__(self, "evidence_refs", refs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version, "task_id": self.task_id,
            "workflow_id": self.workflow_id, "tenant_id": self.tenant_id,
            "attempt": self.attempt, "outcome": self.outcome, "retry_class": self.retry_class,
            "output": dict(self.output), "trace": dict(self.trace),
            "error_code": self.error_code, "worker_id": self.worker_id,
            "evidence_refs": [r.to_dict() for r in self.evidence_refs],
        }


@dataclass(frozen=True, slots=True)
class WorkerHeartbeat:
    worker_id: str
    pool: str
    capabilities: tuple[str, ...]
    active_tasks: int
    capacity: int
    heartbeat_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_id", _text(self.worker_id, "worker_id", 128))
        object.__setattr__(self, "pool", _text(self.pool, "pool", 128))
        object.__setattr__(self, "capabilities", _unique_strings(self.capabilities, "capabilities"))
        if int(self.active_tasks) < 0 or int(self.capacity) < 1 or int(self.active_tasks) > int(self.capacity):
            raise ContractError("invalid worker capacity")
        object.__setattr__(self, "active_tasks", int(self.active_tasks))
        object.__setattr__(self, "capacity", int(self.capacity))
        object.__setattr__(self, "heartbeat_at", float(self.heartbeat_at))

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_version":PROTOCOL_VERSION,"worker_id":self.worker_id,"pool":self.pool,
                "capabilities":list(self.capabilities),"active_tasks":self.active_tasks,
                "capacity":self.capacity,"heartbeat_at":self.heartbeat_at}


@dataclass(frozen=True, slots=True)
class WorkerRegistration:
    protocol_version: str
    worker_id: str
    build_id: str
    capabilities: tuple[str, ...]
    skill_versions: Mapping[str, str]
    trust_level: str
    network_class: str
    resources: Mapping[str, Any]
    health: Mapping[str, Any]
    drain_state: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_version", _text(self.protocol_version, "protocol_version", 32))
        object.__setattr__(self, "worker_id", _text(self.worker_id, "worker_id", 128))
        object.__setattr__(self, "build_id", _text(self.build_id, "build_id", 128))
        object.__setattr__(self, "capabilities", _unique_strings(self.capabilities, "capabilities"))
        object.__setattr__(self, "skill_versions", _mapping(self.skill_versions, "skill_versions"))
        trust = _text(self.trust_level, "trust_level", 32).lower()
        if trust not in {"untrusted","standard","trusted","isolated"}:
            raise ContractError("unsupported trust_level")
        object.__setattr__(self, "trust_level", trust)
        object.__setattr__(self, "network_class", _text(self.network_class, "network_class", 64))
        object.__setattr__(self, "resources", _mapping(self.resources, "resources"))
        object.__setattr__(self, "health", _mapping(self.health, "health"))
        object.__setattr__(self, "drain_state", _mapping(self.drain_state, "drain_state"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version, "worker_id": self.worker_id,
            "build_id": self.build_id, "capabilities": list(self.capabilities),
            "skill_versions": dict(self.skill_versions), "trust_level": self.trust_level,
            "network_class": self.network_class, "resources": dict(self.resources),
            "health": dict(self.health), "drain_state": dict(self.drain_state),
        }
