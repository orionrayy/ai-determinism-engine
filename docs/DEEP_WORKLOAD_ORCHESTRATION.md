# Deep General Workload Orchestration

This layer extends the control plane for large, multi-stage workload specifications without creating a second execution runtime.

## Bounded context handoff
orchestrator/context_budget.py packs dependency evidence into a deterministic global budget. Dependency outputs retain digest and evidence metadata; lower-value bodies are omitted when the aggregate context would exceed the hard envelope.

## Hierarchical blueprint waves
orchestrator/blueprint_compiler.py emits execution units and dependency-safe waves. A wave never exceeds 24 units, matching the current orchestration node cap. Large blueprints can be sliced into resumable boundaries.

## Durable workload lineage
Workflow state contains a generic workload envelope. Blueprint compilation records blueprint identity, version, digests, unit count, and wave count. Workload nodes can identify workload_unit_id and workload_wave_id so progress can be persisted.

## Safety boundary
These layers do not execute external side effects. Existing risk, human approval, connector reconciliation, idempotency, checkpoints, federation, and free-only routing remain authoritative. Blueprint file ingestion is limited to ORCHESTRATOR_WORKLOAD_ROOT.

## Long-horizon lifecycle
ingest -> provenance -> semantic research -> reconcile -> compile units -> compile waves -> bounded agent execution -> evidence/checkpoint -> validation -> retry/replan -> continue -> final verification.