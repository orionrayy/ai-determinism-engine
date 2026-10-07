"""Portable enterprise-runtime reference seams.

These primitives are deliberately isolated from the production GitHub/Cloudflare
bootstrap until their contracts and acceptance tests are independently green.
"""

from .contracts import (
    ArtifactRef, EvidenceRef, PolicySnapshot, SkillManifest,
    TaskEnvelope, TaskResultEnvelope, WorkerHeartbeat, WorkerRegistration,
    canonical_digest,
)
from .queue import QueueClaimRecord, TaskQueue, SQLiteTaskQueue, PostgresTaskQueue, NATSJetStreamAdapter
from .state import EventRecord, StateIntegrityError, StateManifest, FileObjectStore, SegmentedWorkflowStore
from .workers import WorkerRecord, WorkerRegistry, LongLivedWorker
from .admission import AdmissionDecision, FairAdmissionController
from .skills import SkillAuthorizationError, SkillRecord, SkillRegistry
from .telemetry import TraceContext, JsonlTelemetrySink, MemoryTelemetrySink, TelemetryRecorder
from .scale import CampaignResult, run_w1, run_w2, run_w3

__all__ = [
    "ArtifactRef", "EvidenceRef", "PolicySnapshot", "SkillManifest",
    "TaskEnvelope", "TaskResultEnvelope", "WorkerHeartbeat", "WorkerRegistration",
    "canonical_digest", "QueueClaimRecord", "TaskQueue", "SQLiteTaskQueue",
    "PostgresTaskQueue", "NATSJetStreamAdapter", "EventRecord", "StateIntegrityError",
    "StateManifest", "FileObjectStore", "SegmentedWorkflowStore", "WorkerRecord",
    "WorkerRegistry", "LongLivedWorker", "AdmissionDecision", "FairAdmissionController",
    "SkillAuthorizationError", "SkillRecord", "SkillRegistry", "TraceContext",
    "JsonlTelemetrySink", "MemoryTelemetrySink", "TelemetryRecorder", "CampaignResult",
    "run_w1", "run_w2", "run_w3",
]
