# v75 Recovery, Authority, and Epistemic Boundary Hardening

**Status:** Approved for implementation from the previously agreed recovery/control-plane/epistemic audit scope.

## Goal

Make the v74 orchestration line operationally safe for unattended $0 use by removing the known scheduled-recovery failure modes, making distributed-control-plane authority explicit and non-downgradable, and making epistemic abstention/debate decisions depend on validated evidence rather than self-reported metadata.

## Constraints

- Production orchestration remains GitHub Actions + stdlib Python.
- ORCHESTRATOR_FREE_ONLY=true remains authoritative for live production workflow execution.
- No paid database, broker, queue, workflow engine, or runtime dependency is added.
- Git remains an audit/eventual-snapshot layer; an active distributed control plane is the live authority for workflows assigned to it.
- Existing workflow state must remain readable; migration must be backward-compatible.
- Recovery must be bounded, deterministic, idempotent, and fail closed on ambiguous side effects.

## Design

### 1. Scheduled recovery

Introduce a small pure-Python recovery policy module.

The module exposes deterministic helpers for:
- identifying stale running workflows;
- recognizing recoverable failed/barrier states;
- excluding waiting_approval from polling;
- generating a stable recovery event identity from recovery-relevant workflow state rather than mutable updated_at;
- deciding whether an active GitHub run status permits rearming.

The scheduled workflow remains the coarse watchdog. It queries the stored GitHub run ID before rearming. Queued/in-progress/waiting/requested/pending runs are never rearmed. Missing or unreadable historical runs remain eligible for bounded recovery.

Repository dispatch must explicitly use POST. A failed dispatch for one candidate must not prevent later candidates from being processed; the scheduler reports aggregate failure after attempting all candidates.

Terminal-state compaction must tolerate an empty/nonexistent shard directory.

Default stale threshold is 1500 seconds (25 minutes), derived from the current 1200-second default control-plane lease plus safety margin. It remains configurable.

### 2. Immutable workflow authority

Persist authority_mode at workflow creation:
- git_durable for workflows that are not live under distributed control-plane configuration;
- distributed_control_plane for live workflows created while the control plane is configured.

Existing workflows are migrated deterministically:
- explicit distributed control-plane state => distributed_control_plane;
- otherwise => git_durable.

A workflow assigned to distributed_control_plane may not silently execute as Git-only when control-plane configuration is absent. The runtime blocks and records the control-plane dependency instead.

A workflow assigned to git_durable continues to use Git even if control-plane configuration is later added; later configuration cannot silently change its authority.

load_workflow selects remote state only for workflows whose persisted authority is distributed. For distributed workflows, a missing control plane is a dependency failure rather than permission to read/execute stale Git state.

The existing run_workflow() control-plane context remains the single authority-session path. run_one_step() is refactored to use the same context manager so lease acquisition/release and authority gating cannot drift between execution paths.

### 3. Ingress idempotency

Persist idempotency_key on every newly created workflow when supplied. This restores deterministic request identity matching for idempotency-only repository-dispatch/private-input ingress.

### 4. Epistemic validation and selective abstention

Expose supported_coverage explicitly as the supported-material-claim coverage metric.

Selective abstention is an admission outcome: when its high-confidence evidence checks fail, epistemic validation does not pass.

Low-confidence outputs remain outside the selective high-confidence gate; they may be routed by the existing conditional deliberation/research policy.

### 5. Deliberation

Evidence weakness is computed from validated evidence_records using canonical evidence-work identity, never from an LLM-reported independent_source_count.

Blind challenge views remove identity/role/voting metadata and replace it with opaque deterministic candidate indices. The candidate ordering is derived only from the redacted content digest, not agent identity.

### 6. Scope boundary

This release does not introduce a Durable Object Alarm wakeup subsystem, a new external evidence graph service, or the full ExecutionRuntime consolidation. Those remain next-stage improvements after these invariants are green.

## Acceptance criteria

1. Scheduled recovery sends POST repository_dispatch requests.
2. One failed recovery dispatch does not suppress later candidates.
3. Waiting-approval workflows are not repeatedly resumed by the scheduler.
4. Recovery event IDs are stable while recovery-relevant state is unchanged.
5. Active GitHub runs are not rearmed.
6. Empty workflow-shard state can complete recovery/compaction without pathspec failure.
7. Distributed-control-plane workflows fail closed when the control plane is unavailable.
8. Git-authority workflows are not silently upgraded by later control-plane configuration.
9. idempotency_key survives workflow creation and supports ingress deduplication.
10. High-confidence epistemic abstention blocks validation when evidence checks fail.
11. Sufficient supported coverage no longer fails because of a missing alias field.
12. Self-reported source counts cannot satisfy the deliberation evidence threshold.
13. Blind deliberation exposes no agent identity or role.
14. The branch remains stdlib-only for the orchestrator and free-first in production.
15. Unit, compile, actionlint, and offline evaluation checks pass on the feature branch.
