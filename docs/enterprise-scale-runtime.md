# Enterprise-Scale Runtime Architecture

Status: design baseline only; no changes to main.
Date: 2026-10-05

## Executive decision

The current engine is a strong correctness-oriented durable executor, but it is not yet an enterprise-scale workload platform. The next architecture must treat GitHub Actions as an adapter rather than as the scheduler, durable queue, or primary production state system.

Current strengths include deterministic DAG planning, bounded local execution, fenced distributed authority, effect claims and reconciliation, recovery, evidence trust boundaries, provider budgets, and free-first policy.

The main architectural ceiling is that scheduling, execution and persistence remain coupled to a Python process, GitHub federation is workflow/matrix based, and live workflow state remains a monolithic bounded JSON document. Those constraints become structural at high volume.

Free-first is a deployment principle, not a promise that physical enterprise capacity costs zero. Open-source/self-hosted software can remain $0-licensed, but compute, storage, network, redundancy and operations are real resources.

## Research basis

Temporal's architecture separates workflow logic from activity execution, uses durable event history, shards workflow executions, and dispatches work through task queues to worker processes. PostgreSQL explicitly documents SKIP LOCKED as useful for avoiding row-lock contention when multiple consumers read queue-like tables. NATS JetStream provides durable consumers, acknowledgements/redelivery and work-queue semantics. OpenTelemetry defines common conventions across traces, metrics, logs and resources.

GitHub Actions has hard scaling boundaries: a matrix is limited to 256 jobs per workflow run, a concurrency group can queue at most 100 pending jobs or runs, and workflow runtime is capped at 35 days. GitHub Free includes 2,000 hosted-runner minutes per month for private repositories; self-hosted runners are free as an Actions service but the underlying machines remain the operator's responsibility.

Cloudflare Durable Objects are useful as a lightweight authority adapter, but the Workers Free plan is bounded at 100,000 requests/day, 13,000 GB-s/day, 5 million SQLite rows read/day, 100,000 rows written/day and 5 GB of SQLite data. Exceeding a free-plan dimension causes further operations of that type to fail.

## Target architecture

```text
INGRESS
  -> ADMISSION / AUTH / IDEMPOTENCY / QUOTAS
  -> PARTITIONED CONTROL PLANE
       -> durable workflow state transitions
       -> shard ownership and fencing
       -> deterministic readiness/scheduling
       -> timers and recovery
  -> DURABLE TASK BROKER
       -> priority lanes
       -> partitioning
       -> ack/redelivery
       -> retry and DLQ
  -> STATE / EVENT PLANE
       -> immutable event tail
       -> compact snapshots
       -> dependency/result manifests
  -> STATELESS WORKER FLEETS
       -> LLM
       -> research
       -> tools/connectors
       -> browser/code/sandbox
       -> artifact verification
  -> EFFECT / ARTIFACT PLANE
  -> TELEMETRY

Side planes:
  skill/capability registry
  policy/budget registry
  identity/tenant isolation
  Git audit/export adapter
```

The key rule is that the control plane decides truth and scheduling but never performs long-running provider or tool execution. Workers execute tasks and return correlated results. The broker transports work; it does not become the workflow authority.

## Plane boundaries

| Plane | Owns | Must not own |
|---|---|---|
| Ingress | identity, authentication, input admission | execution |
| Control plane | state transitions, fencing, scheduling decisions, recovery | model/tool execution |
| State/event | snapshots, event tail, projections | worker lifecycle |
| Task broker | durable delivery and retries | workflow truth |
| Workers | computation and effects | authoritative state mutation |
| Skills | capability metadata and compatibility | ungoverned execution |
| Research | source acquisition and provenance | workflow truth |
| Effects/artifacts | output/effect identity and verification | scheduling |
| Telemetry | correlated operational signals | business truth |
| Git adapter | audit, review, CI and export | high-volume queue/state authority |

## Task envelope

Every task should become transport-neutral data. Recommended immutable fields are:

```text
protocol_version
tenant_id
workflow_id
execution_id
task_id
attempt
partition_key
capability
skill_id
skill_version
priority
deadline_at
visibility_timeout
idempotency_key
fence_epoch
input_digest
contract_digest
policy_digest
resource_budget
trace_context
context_refs
artifact_refs
```

The worker must not receive executable authority inside task payloads. It receives a capability identity and resolves implementation from a trusted registry.

## Delivery semantics

Do not promise global exactly-once execution. The target semantics are at-least-once delivery plus deterministic idempotency, fencing, effect claims, visibility timeouts, reconciliation and dead-letter handling.

A side effect must be classified as completed, not_applied or unknown before a retry can safely repeat it. Unknown effects stay in reconciliation rather than being guessed away.

## Partitioning

Partitioning is needed at several levels:

1. Workflow partition: hash of tenant_id plus workflow_id.
2. Task partition: capability, tenant fairness class, runtime/region and optional affinity key.
3. Provider partition: provider plus pool plus credential/profile plus region.
4. Resource partition: independent lock ownership by resource key.

A workflow has one home shard at a time. A shard owns its workflows' state transitions and recovery scheduling. There must be no single global scheduler mutex.

## State model

The current 600 KiB distributed workflow-state ceiling is correctly fail-closed, but the enterprise design must eliminate dependency on one large JSON row.

Target hot state:

```text
Workflow Header
  status / state_version / shard / owner / policy_digest
Snapshot
  compact deterministic materialized state
Event Tail
  append-only state-transition records
Dependency Manifest
  task_id -> immutable result reference + digest
Artifact Manifest
  artifact_id -> content address + verification state
Recovery Manifest
  timers / retries / wake-up records
```

Large outputs, evidence corpora, logs and artifacts must live behind immutable references. Resume becomes snapshot validation, event-tail replay, dependency-digest validation and deterministic reconstruction of the next ready work.

## Durable queue abstraction

Expose a transport-neutral queue interface:

```text
enqueue(task)
claim(batch, worker)
heartbeat(task)
ack(task, result)
nack(task, reason, retry_at)
move_to_dlq(task, reason)
inspect(task)
queue_metrics()
```

Recommended free-first adapters:

| Backend | Position |
|---|---|
| SQLite/in-process | local development and deterministic tests |
| PostgreSQL | first horizontally scaled deployment |
| NATS JetStream | higher-throughput queue/event fabric |
| Cloudflare Durable Objects | bounded distributed authority adapter |
| GitHub Actions | execution/CI adapter and low-volume burst worker |

PostgreSQL is a practical first distributed queue because row locking plus SKIP LOCKED supports multiple consumers without making a global queue lock the bottleneck. NATS is the stronger candidate once queue delivery itself needs high-throughput durable streaming and consumer scaling.

## Scheduler, admission and backpressure

The scheduler becomes a deterministic policy engine. Admission is allowed only when tenant quota, workflow quota, capability quota, provider quota, resource quota, worker capacity and deadline feasibility all pass.

Track queue depth and queue age. Backlog approximately follows arrivals minus completions; increasing concurrency does not help when the provider, state store or queue is the actual bottleneck.

Use independent lanes for interactive, normal, bulk research and maintenance work. Bulk workloads must never consume the entire scheduler budget.

## Multi-tenant fairness

Even a single-user repository should design the contract as multi-tenant. Minimum controls are concurrent workflow limits, concurrent task limits, per-capability and per-provider budgets, storage/byte budgets, LLM-call budgets, retry budgets and priority ceilings.

A high-volume tenant must not starve another tenant by producing a larger DAG. Weighted fair scheduling or an equivalent deterministic policy is required.

## Worker architecture

Workers should become stateless, long-lived processes with capability-specific pools:

```text
research-workers
llm-workers
connector-workers
browser-workers
code-workers
sandbox-workers
artifact-verifiers
maintenance-workers
```

A worker validates the task envelope, obtains an attempt lease, executes, heartbeats when necessary, writes large results to the artifact plane and returns a bounded result envelope. It never becomes the owner of workflow truth.

GitHub Actions remains useful for ephemeral workers, CI and bounded bursts. Enterprise execution should also support long-lived workers outside GitHub.

## Skill and capability registry

Future multi-skill operation must be discover -> validate -> authorize -> budget -> schedule -> execute -> verify, never discover -> execute.

A skill manifest should contain skill identity, semantic version, manifest digest, capabilities, input/output schemas, runtime entrypoint, protocol version, network/file/secret permissions, risk class, free-tier classification, request/runtime limits, quality level and implementation provenance.

Registry trust should be layered: built-in trusted entries, repository-local entries, organization-signed entries, then optional external discovery. A lower-trust source cannot silently override a higher-trust entry.

Suggested lifecycle:

```text
DISCOVERED -> VALIDATED -> ENABLED -> HEALTHY
                         |
                         v
                 DEGRADED -> QUARANTINED -> DISABLED
```

Health can demote a skill but never broaden its permissions.

## Research/exploration fabric

Research should become a reusable workload subsystem:

```text
Goal
 -> query decomposition
 -> source strategy
 -> parallel retrieval
 -> deduplication
 -> source integrity
 -> access verification
 -> passage extraction
 -> claim/evidence graph
 -> contradiction detection
 -> coverage analysis
 -> synthesis
 -> durable research memory
```

Each source should retain canonical identity, provider, publication status, authority signals, independence key, access route, access verification, retrieval timestamp, source digest, corpus digest and claim references.

Exploration must always have explicit bounds: maximum depth, queries, sources, providers, LLM calls, runtime, bytes, retries, allowed skills/domains and stop conditions. The LLM cannot grant itself extra budget.

The current v79 evidence boundary is therefore worth preserving and generalizing: trusted records, retraction awareness, challenge-scoped evidence, passage-level validation and conservative claim preservation should become reusable research primitives.

## Effect and artifact plane

Workflow state should reference immutable artifacts instead of embedding large outputs. Artifact identity can be content-addressed with a canonical metadata digest plus content digest.

Effects should retain effect identity, semantic digest, status, attempts, worker ownership/fence, provider, external reference, output digest and reconciliation state.

## Observability

The internal event vocabulary should map cleanly to OpenTelemetry concepts rather than inventing an incompatible proprietary telemetry model.

At minimum correlate tenant_id, workflow_id, execution_id, task_id, attempt, skill_id, worker_id, partition_id, trace_id and span_id.

Core signals include workflow/task lifecycle, queue depth and age, worker utilization, provider throttle/error rate, effect-unknown count, research integrity failures and evidence validation failures.

Telemetry is non-authoritative. It must never be inserted into the same critical transaction path merely for analytical completeness.

## Deployment ladder

| Profile | Core | Purpose |
|---|---|---|
| A: $0 local | Python + SQLite + local registry | development/tests |
| B: $0 community | public GitHub Actions + bounded Cloudflare + free public providers | demos/low volume |
| C: self-hosted scale | PostgreSQL + NATS + stateless workers + optional object store | real multi-worker load |
| D: enterprise | partitioned control plane + clustered storage/broker + worker pools + HA/observability | large production |

Profiles A and B can be zero-dollar. Profiles C and D can remain free/open-source in software licensing, but their compute and operational resources are not inherently free.

## Migration sequence

Phase E0: introduce transport-neutral interfaces for WorkflowStore, EventStore, TaskQueue, LeaseStore, EffectStore, ArtifactStore, CapabilityRegistry and TelemetrySink while retaining current adapters.

Phase E1: separate scheduler/state-transition code from task execution. Replace cross-worker ThreadPoolExecutor assumptions with task envelopes.

Phase E2: add PostgreSQL queue adapter; keep SQLite for local tests. Add NATS after the queue contract is stable.

Phase E3: replace monolithic hot state with compact header/snapshot/event-tail/manifests and prove replay from clean snapshots.

Phase E4: introduce long-lived stateless worker protocol and keep GitHub as one adapter.

Phase E5: add tenant fairness, capability/provider quotas, queue-age admission and capacity-aware backpressure.

Phase E6: extract the research and evidence fabric; introduce skill manifests, verification and quarantine.

Phase E7: add OTel-compatible traces/metrics/logs and offline aggregation.

Phase E8: execute load and failure-injection campaigns before making scale claims.

## Enterprise readiness gates

| Gate | Evidence required |
|---|---|
| Determinism | replay produces identical next-state/commands |
| Durability | crash/restart recovers without manual repair |
| Delivery | duplicates and late completions are safe |
| Effects | uncertain effects reconcile before retry |
| Scale | declared throughput/concurrency passes load test |
| Fairness | one tenant cannot starve others |
| Backpressure | overload remains bounded and recoverable |
| State | no single monolithic row limits workflow size |
| Security | skills/tools are isolated and policy scoped |
| Evidence | claims remain bound to validated sources |
| Observability | incidents are diagnosable from correlated telemetry |
| Portability | GitHub/local/PostgreSQL/NATS share semantics |

## Explicit unresolved problems

Shard ownership and rebalancing; deterministic snapshot compaction; task leasing across worker fleets; priority/fairness algorithms; hot-partition mitigation; provider quotas without centralized write hotspots; scalable evidence indexing; skill sandboxing and package signing; cross-tenant secret isolation; multi-region failover; telemetry cardinality control; and capacity-based autoscaling.

These are implementation epics, not assumptions.

## Final architectural position

The target is not a bigger GitHub Actions workflow. It is a portable durable-execution platform in which LLMs supply intelligence, Python policies define admissibility and deterministic transitions, a partitioned control plane owns workflow truth, a durable broker owns delivery, workers perform execution, skills define governed capabilities, the research fabric provides provenance-bound knowledge, effects are idempotent and reconcilable, Git remains an audit/export surface, and GitHub Actions is one worker/CI adapter.