# Enterprise-Scale Runtime Foundation

Status: architecture baseline, not production implementation.

## Executive decision

The current GitHub Actions execution fabric is a useful zero-dollar laboratory and bounded pilot substrate, but it must not remain the primary execution fabric for high-volume enterprise workloads.

The target is a portable durable-execution platform with:

1. GitHub as source control, review, CI, release metadata, and selected audit publication.
2. A dedicated control plane as the authority for workflow ownership, fencing, state/version checks, quotas, and admission.
3. A durable dispatch plane separated from workflow state.
4. Stateless or ephemeral worker pools specialized by capability, resource class, trust zone, and tenant.
5. Object storage for large payloads, evidence corpora, artifacts, traces, and checkpoints.
6. A relational/event visibility plane for operational search and analytics once the control-plane store is no longer the correct query surface.
7. A research/evidence plane with provenance, caching, deduplication, freshness, contradiction handling, and passage-level validation.
8. An interoperability plane for tools and agents, with MCP/A2A treated as adapters rather than internal authority.
9. An observability/evaluation plane for traces, SLOs, quality, retries, rework, and resource consumption.

Core separation:

> LLMs propose and reason. Deterministic policy decides admissibility. Durable infrastructure commits state. Workers execute. External data is addressed by immutable identity. Git records definitions and selected audit artifacts, not the live event stream.

## Why GitHub moves to the edge

GitHub Actions is valuable because the public-repository standard hosted runner path is zero-dollar and the repository already has mature CI, workflow triggers, and auditability. It is not a good general-purpose high-volume dispatch database or queue.

The current execution path still has several GitHub-coupled operations:

- workflow dispatch and repository dispatch;
- issue-driven ingress;
- GitHub concurrency groups;
- GitHub API interactions in recovery;
- repository checkout;
- Git commits for live state persistence;
- GitHub artifact/issue surfaces for coordination.

At enterprise volume, the scheduler should not depend on Git commit latency, repository write serialization, issue events, or API polling for each task.

GitHub should instead provide:

- workflow definitions and policies;
- review and change control;
- CI and conformance testing;
- release promotion;
- operator-facing audit snapshots;
- incident evidence and reproducible fixtures.

The runtime should be able to operate without a GitHub repository being reachable on every task transition.

## Target planes

### 1. Ingress plane

Responsibilities:

- authenticate the caller;
- assign tenant/workspace;
- canonicalize goal and input references;
- calculate intent and input digests;
- enforce ingress quotas;
- write a durable command/event;
- return a stable workflow/run ID and asynchronous status handle.

An HTTP request must not be required to remain open for the lifetime of a workflow.

### 2. Control plane

One logical workflow ID maps to one partition/authority owner at a time.

Required primitives:

- workflow lease;
- monotonically increasing fence epoch;
- workflow state version/CAS;
- durable history reference;
- task admission;
- task lease/heartbeat;
- effect identity and reconciliation;
- resource quota/lock;
- cancellation;
- timers;
- human approval signals;
- recovery scheduling;
- tenant/account isolation.

The primary partitioning key should normally be a stable hash such as `hash(tenant_id, workflow_id)`, not a repository-global singleton.

A partition has one logical writer for its owned state, while unrelated partitions proceed independently.

### 3. Dispatch plane

Workflow state must not double as the task queue.

The dispatch abstraction needs:

- enqueue;
- claim;
- visibility timeout or lease;
- heartbeat for long tasks;
- acknowledgement;
- negative acknowledgement;
- retry-at timestamp;
- priority;
- tenant/workload fairness;
- capacity/backpressure;
- dead-letter or quarantine;
- idempotency key;
- partition affinity;
- optional batch operations.

The application must depend on a `DispatchAdapter`, not a vendor-specific queue API.

Bootstrap implementation:

`GitHubRepositoryDispatchAdapter`

Scale implementation:

`QueueAdapter`

The same workflow contract must run over both.

### 4. Worker plane

Workers are disposable execution hosts.

A `TaskEnvelope` should contain at minimum:

- schema version;
- workflow/run/task IDs;
- tenant ID;
- partition;
- attempt;
- fence epoch;
- capability;
- selected skill/implementation;
- resource class;
- deadline;
- input reference;
- output contract;
- effect contract;
- trace/correlation IDs;
- idempotency key;
- policy snapshot digest.

A `TaskResultEnvelope` should contain:

- outcome;
- retry classification;
- output reference;
- output digest;
- evidence references;
- bounded metrics;
- worker/runtime metadata;
- next-action hint.

Large output must never require the control plane to carry the complete JSON payload.

### 5. Knowledge and evidence plane

Research becomes a first-class subsystem:

`discover -> retrieve -> normalize -> deduplicate -> score -> verify -> extract -> challenge -> synthesize`

Canonical evidence records should support:

- immutable evidence ID;
- provider/source;
- work identity such as DOI, PMID, URL, or document ID;
- source version/retrieval timestamp;
- authority profile;
- access route;
- access verification;
- integrity/retraction signal;
- corpus reference;
- exact passage references;
- extraction method;
- query/provenance lineage;
- freshness/expiry;
- deduplication identity.

Free/public providers remain first in the route. Credentialed or metered providers require explicit policy admission.

### 6. Artifact/object plane

Git is source control, not object storage.

Large or high-churn data should live outside Git:

- full research responses;
- PDFs/documents;
- generated artifacts;
- verbose traces;
- model transcripts where policy permits;
- evidence corpora;
- large intermediate results;
- replay fixtures;
- cold workflow snapshots.

The zero-dollar bootstrap can use a free object-storage tier for small pilots. The abstraction must remain portable.

### 7. Visibility and observability plane

Operational queries should be separated from authoritative workflow state.

Useful dimensions:

- tenant;
- workflow;
- workload;
- task;
- capability;
- worker pool;
- provider;
- model;
- status;
- latency;
- attempts;
- retry class;
- queue wait;
- execution time;
- evidence quality;
- adjudication outcome;
- human intervention;
- cost class.

Minimum SLO/error-budget signals:

- ingress acceptance latency;
- queue wait p50/p95/p99;
- task execution p50/p95/p99;
- workflow completion latency;
- recovery detection/convergence;
- duplicate-effect rate;
- ambiguous-effect rate;
- evidence verification failure;
- unsupported-claim rate;
- rework rate;
- provider degradation.

## Hot state versus cold state

The current v79 design is correct to reject lossy projection as a shortcut around the 600 KiB control-plane workflow-state limit.

The enterprise design should split:

### Hot authoritative state

Only:

- lifecycle status;
- scheduling frontier;
- compact task manifests/references;
- leases;
- counters;
- policy snapshot digests;
- recovery metadata;
- small deterministic indexes.

### Immutable history

An ordered sequence of state-transition/event records sufficient to reconstruct the logical workflow state.

### Cold state

Large dependency outputs and intermediate results.

### Object state

Large blobs, evidence passages, documents, generated artifacts, and transcripts.

An authoritative artifact reference must bind:

`artifact_id + content_sha256 + schema_version + workflow_id + producer_task_id`

Resume becomes:

`reconstruct(state_manifest, event_history, artifact_refs) == current_state`

The control plane must reject a missing or mismatched digest instead of silently substituting another payload.

## Multi-tenant isolation

Every ingress, workflow, task, queue claim, resource lock, artifact reference, and trace should carry `tenant_id`.

Quota dimensions:

- tenant;
- workflow class;
- capability;
- provider;
- worker pool;
- global system.

Fair scheduling must prevent one tenant or capability from consuming all shared capacity.

A useful first policy is weighted fair sharing with:

- per-tenant concurrency ceilings;
- per-capability ceilings;
- provider token buckets;
- global circuit breakers.

## Resource control

Use separate budgets for:

1. compute;
2. LLM calls/tokens;
3. external provider requests/concurrency;
4. durable control-plane writes;
5. object-store operations.

Attach the budget snapshot to each admitted task. Reject, defer, or downgrade a task before execution when its estimated cost cannot fit the remaining budget.

This is more scalable than learning about exhaustion after side effects have already started.

## Interoperable skills and multi-agent expansion

The repository already has typed roles and a capability graph. The scalable abstraction is a versioned skill registry.

Example manifest:

```json
{
  "skill_id": "research.claims",
  "version": "1.0.0",
  "digest": "sha256:...",
  "capabilities": ["research", "claim_verification"],
  "input_schema": "schema://...",
  "output_schema": "schema://...",
  "required_resources": ["internet"],
  "risk": "low",
  "trust_class": "reviewed",
  "cost_class": "free_public",
  "runtime": "python",
  "entrypoint": "..."
}
```

The registry should provide:

- discovery;
- compatibility checks;
- policy filtering;
- health;
- version pinning;
- digest verification;
- capability scoring;
- fallback implementation;
- deprecation;
- quarantine;
- shadow/canary execution.

MCP can be an interoperability adapter for tools/resources/prompts and future Skills/Tasks semantics. A2A can be an adapter for delegation to independent agent systems.

Neither protocol is the internal authorization authority.

## Enterprise failure model

Assume all of the following can happen:

- worker crash after an external commit;
- network timeout after provider commit;
- duplicate ingress;
- duplicate task delivery;
- stale completion after lease expiry;
- control-plane outage or partition;
- object-store success followed by worker crash;
- skill version drift;
- source retraction;
- provider rate-limit storm;
- malicious or poisoned tool instruction;
- tenant quota exhaustion;
- schema mismatch during rolling deployment;
- partial deployment;
- duplicate human approval;
- delayed webhook;
- out-of-order external event.

For each failure:

`detect -> classify -> persist -> reconcile -> resume/block`

Never encode recovery as "retry everything".

## Deployment evolution

### Stage A — $0 development

Use:

- public GitHub repository;
- GitHub Actions;
- Cloudflare Workers/Durable Objects for control-plane experiments;
- optional free object storage for small payload tests;
- public research providers;
- Termux on Android for deterministic local tests.

### Stage B — $0 bounded pilot

Move long-lived state and large payloads away from Git.

Use:

- HTTP ingress;
- control-plane authority;
- portable dispatch adapter;
- object storage;
- GitHub Actions only for low-volume/bursty execution and release automation;
- strict quotas and backpressure.

This stage proves architecture, not enterprise SLOs.

### Stage C — scalable production

Replace the execution substrate while retaining the same workflow contracts:

- durable queue/log;
- horizontally scalable worker fleet;
- relational/event history store;
- object storage;
- centralized observability;
- tenant-aware quotas;
- autoscaling;
- failure-domain strategy.

GitHub is now release/audit infrastructure, not runtime state.

### Stage D — enterprise

Add:

- SSO/RBAC;
- secret management;
- encryption/data residency controls as required;
- backup/restore drills;
- multi-region disaster recovery;
- incident runbooks;
- change management;
- schema compatibility policy;
- formal threat model;
- supply-chain security;
- platform SLOs/error budgets;
- tenant-level cost attribution.

## Non-goals

Do not:

- turn a single Cloudflare Durable Object into a giant event bus;
- keep full dependency outputs permanently in hot state;
- make Git commits the per-event source of truth;
- make GitHub Issues the queue;
- add another queue solely for architectural theater;
- treat MCP/A2A interoperability as authorization;
- treat majority voting as truth;
- treat free-tier quotas as enterprise capacity guarantees.

## Decision

The repository should evolve around a portable execution core plus substrate adapters.

The next implementation work should establish contracts and migration seams before selecting a high-volume queue/database/runtime.

