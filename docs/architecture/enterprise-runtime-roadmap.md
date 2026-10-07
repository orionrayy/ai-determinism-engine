# Enterprise Runtime Roadmap and Smartphone-Only Build Plan

## Principle

A smartphone-only development environment is compatible with the target architecture.

The phone should be the operator/developer console. It should not become the production scheduler, database, queue, or worker fleet.

Cloud-hosted CI/build infrastructure provides compute; the phone provides control.

## Phase 0 — architecture freeze

Establish contracts before adding more runtime mechanisms:

- portable workflow command;
- task envelope;
- task result envelope;
- artifact reference;
- evidence reference;
- skill manifest;
- worker heartbeat;
- queue claim;
- policy snapshot;
- deterministic digests;
- tenant identity.

Deliverable: architecture documents plus contract tests.

## Phase 1 — portable contracts

Add dependency-free Python dataclasses/validators for:

- `WorkflowCommand`;
- `TaskEnvelope`;
- `TaskResultEnvelope`;
- `ArtifactRef`;
- `EvidenceRef`;
- `SkillManifest`;
- `WorkerHeartbeat`;
- `QueueClaim`;
- `PolicySnapshot`.

Every contract has:

- schema version;
- canonical JSON form;
- deterministic digest;
- backwards-compatibility rule.

Do not connect another queue yet.

## Phase 2 — substrate adapters

Introduce interfaces:

- `ControlPlaneAdapter`;
- `DispatchAdapter`;
- `ObjectStoreAdapter`;
- `VisibilityAdapter`;
- `SecretProvider`;
- `ToolAdapter`;
- `AgentAdapter`.

Keep current implementations:

- GitHub Actions;
- Cloudflare Durable Objects;
- local filesystem/Git-backed artifacts.

Future implementations can replace the substrate without rewriting planner/evidence logic.

## Phase 3 — history-backed state segmentation

Introduce immutable event records and a compact state manifest.

Required invariant:

`reconstruct(state_manifest, event_history, artifact_refs) == current_state`

Do not remove the existing full-state path until reconstruction, replay, and migration tests pass.

The first version can still store everything on the bootstrap substrate while implementing the contract as if object storage were external.

## Phase 4 — dispatch independence

Implement `DispatchAdapter` while keeping GitHub dispatch as one backend.

Run the same workflow through:

- GitHub repository dispatch;
- an external durable queue.

Compare:

- throughput;
- queue latency;
- duplicate delivery;
- retry behavior;
- fairness;
- recovery;
- control-plane write amplification.

The orchestrator should not know which backend is selected.

## Phase 5 — skill and research platform

Turn the current capability graph into a registry-backed planner:

`task -> capabilities -> eligible skills -> policy -> provider/tool -> worker pool`

Research becomes:

`query planner -> source fanout -> cache -> normalize -> dedupe -> evidence scorer -> verifier -> adjudicator`

Free/public research sources stay first.

MCP is an adapter for tool/resource/skill interoperability.

A2A is an adapter for delegation to external agent systems.

The internal policy registry remains authoritative for trust, risk, cost, compatibility, and version pinning.

## Phase 6 — observability and quality ledger

Add append-only operational events and derive:

- queue pressure;
- latency;
- reliability;
- provider health;
- skill success;
- rework;
- contradiction rate;
- evidence verification;
- human intervention;
- resource usage;
- failure class.

Skill quality must be measured from outcomes, not from LLM confidence alone.

## Phase 7 — production substrate

Only after contract and replay tests pass, introduce:

- scalable dispatch;
- partitioned state/history;
- object storage;
- horizontally scalable workers;
- centralized visibility;
- autoscaling;
- failure-domain controls.

Keep the workflow core unchanged.

## Smartphone-only operating procedure

The complete development loop can be done from Android:

1. Inspect and edit repository code through GitHub web or the connected GitHub tooling.
2. Use GitHub Actions for compile, unit tests, lint, integration checks, and conformance runs.
3. Use Cloudflare dashboard/browser tools for the free-first control plane.
4. Use Termux for local deterministic Python checks, fixtures, protocol simulations, and small load generators.
5. Use cloud-hosted development shells when a temporary Linux environment is required.
6. Keep the phone outside the production critical path.

The phone can safely own:

- source edits;
- branch/PR management;
- CI observation;
- architecture review;
- small synthetic benchmark generation.

The phone should not own:

- the only copy of production state;
- a long-running worker;
- the primary queue;
- the enterprise scheduler;
- the only recovery mechanism.

## Immediate engineering order

1. Contract schemas and deterministic digests.
2. Dispatch adapter.
3. Artifact/object reference adapter.
4. Event/history manifest.
5. Tenant/quota policy.
6. Capacity evaluator.
7. Contract/replay tests.
8. Hot-state segmentation.
9. External queue integration.
10. Skill registry and multi-agent protocol adapters.
11. Observability/quality ledger.
12. Production deployment profile.

This order deliberately delays infrastructure selection until the application contracts are portable.

## Exit criteria for each boundary

### GitHub boundary

The system can complete a workflow when GitHub is unavailable after ingress because the runtime has already accepted and durably persisted the command.

### State boundary

The system can reconstruct state without storing all dependency outputs in one authoritative row.

### Queue boundary

A task can be delivered more than once without producing a duplicate external effect.

### Worker boundary

A crashed worker can be replaced without corrupting workflow ownership or losing task identity.

### Evidence boundary

A claim cannot become authoritative solely because an LLM supplied a citation; trusted evidence and exact passage references are required where the contract demands them.

### Skill boundary

A skill implementation can be upgraded, quarantined, or replaced without rewriting workflows.

### Cost boundary

A workload can be rejected, downgraded, deferred, or rerouted before crossing a free-tier or tenant budget.

### Production boundary

GitHub can be unavailable for the runtime's critical path while release/audit workflows remain operational.
