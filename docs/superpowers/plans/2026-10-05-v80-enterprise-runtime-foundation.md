# v80 Enterprise Runtime Foundation — Implementation Plan

## Objective

Convert the existing durable multi-agent orchestrator into a portable execution platform that can scale beyond GitHub Actions while keeping deterministic policy, evidence, effect and recovery semantics intact.

Constraint: development and the reference path remain free-first and usable from a smartphone. Enterprise SLA, high availability and sustained high-volume capacity are a later infrastructure tier, not a false $0 promise.

## Verified baseline

Current branch: feat/v79-boundary-evidence-efficiency

Latest observed head:
5dfb5a1d90286c072d9aa64b770195e25779603e

Latest observed CI:
- Orchestrator Tests run 2034: success
- Private Input Worker Tests run 209: success

The current system already has distributed workflow authority, fencing, state CAS, effect reconciliation, evidence binding, research provider routing and GitHub federation.

## Problem

The current system still couples progress to GitHub workflow runs, repository events, Actions artifacts and GitHub-owned runner lifecycle.

A second hard boundary is the monolithic workflow-state JSON value with an intentionally conservative 600 KiB ceiling.

The target architecture separates:
- correctness/control;
- transport/queue;
- worker compute;
- immutable payload storage;
- skills/tools/agents;
- evidence/research;
- observability.

## Phase A — Portable contracts

Create and version:
1. TaskEnvelope
2. SkillManifest
3. CheckpointManifest
4. WorkerRegistration
5. QueueAdapter
6. TaskResult
7. EffectResolution
8. TraceContext

Acceptance:
- schema files parse with Python standard library only;
- identity fields are bounded;
- large payloads can be referenced by digest/URI;
- the task contract contains no GitHub-specific requirement.

## Phase B — Queue abstraction

Introduce QueueAdapter:

    enqueue(task)
    receive(max_items, visibility_timeout)
    ack(task_id, lease_token)
    nack(task_id, retry_at or dead_letter)
    extend(task_id, lease_token)

Implement in stages:
- GitHubDispatchAdapter;
- MemoryQueueAdapter;
- CloudflareQueueAdapter;
- NatsJetStreamAdapter.

Queue state is transport state, not workflow authority.

Acceptance:
- deterministic tests can enqueue and execute without GitHub;
- duplicate delivery is safe;
- visibility expiry produces redelivery;
- late acknowledgements fail after lease change.

## Phase C — Worker registry

WorkerRegistration fields:

    worker_id
    build_id
    protocol_version
    capabilities
    skill_versions
    trust_level
    network_class
    cpu_class
    memory_mb
    max_slots
    available_slots
    last_heartbeat
    drain_state

Implement:
    register()
    heartbeat()
    acquire_task()
    complete_task()
    begin_drain()
    retire()

Acceptance:
- incompatible work cannot be acquired;
- drained workers receive no new work;
- expired heartbeats remove capacity from routing;
- worker build/version reaches task results.

## Phase D — State segmentation

Define CheckpointManifest.

Hot state becomes a compact manifest. Large node outputs, evidence corpora, research payloads and tool responses become immutable shards.

Resume:
1. read hot manifest with CAS;
2. validate checkpoint generation;
3. load required shards;
4. verify SHA-256;
5. reconstruct deterministic state;
6. resume the frontier;
7. commit a new manifest generation.

Acceptance:
- a workflow over 600 KiB can resume;
- missing or corrupted required shards fail closed;
- stale workers cannot roll back a newer generation.

Do not implement lossy state projection.

## Phase E — Hierarchical quotas

Quota levels:
- tenant;
- workflow;
- capability;
- provider;
- worker pool.

Metrics/states:
- queued;
- running;
- delayed;
- quota_blocked;
- provider_limited;
- capacity_blocked;
- retry_budget_exhausted.

Acceptance:
- one noisy workflow cannot exhaust a tenant pool;
- one tenant cannot exhaust global provider permits;
- retry storms are bounded;
- quota decisions are deterministic and auditable.

## Phase F — Provider permit leases

Use short-lived batch permits for high-volume providers.

Example:
    acquire_permit(provider, count=20, ttl=5s)

The permit is spent locally by a worker. The control plane grants the budget; the worker controls local consumption.

Acceptance:
- aggregate provider request rate stays within configured policy;
- expired permits cannot be over-consumed;
- hard-free mode never consumes paid credentials;
- metered-free providers such as OpenAlex remain budgeted.

## Phase G — Research execution plane

Implement:
    discover
    collect
    normalize
    dedupe
    rank
    fetch
    extract
    verify
    adjudicate
    package

A verified corpus reference is required before a claim is considered passage-verifiable.

Fetcher requirements:
- HTTPS-only;
- DNS/IP validation;
- redirect controls;
- response limits;
- content-type validation;
- SHA-256 digest;
- cache;
- timeout;
- provenance;
- quarantine on malformed content.

## Phase H — Skill registry and interoperability

Internal SkillManifest is authoritative.

Adapters:
- MCP Skills/resources;
- A2A agent/task exchange;
- OpenAPI HTTP services.

Acceptance:
- capability discovery returns an explicitly versioned skill;
- selected skill version is pinned into TaskEnvelope;
- permissions are enforced before execution;
- adapter failure cannot mutate workflow truth;
- provenance survives worker movement.

## Phase I — Observability

Use OpenTelemetry-compatible trace context:
    trace_id
    span_id
    parent_span_id

Attach:
    tenant_id
    workflow_id
    task_id
    attempt
    worker_id
    capability

Measure:
- queue wait;
- execution time;
- retries;
- effect uncertainty/reconciliation;
- CAS conflicts;
- provider throttling;
- worker slot saturation;
- evidence verification failures.

Correctness-critical events are retained without sampling.

## Phase J — GitHub becomes optional

Actions remains useful for:
- manual dispatch;
- event ingress;
- approval;
- CI;
- audit commits;
- fallback execution.

A queue-backed runtime becomes the normal path.

Acceptance:
- queue-backed workflows progress without an Actions run;
- GitHub outage does not destroy live workflow state;
- recovery does not require repository polling.

## Phase K — Production hardening

Only after the portable runtime is working:
- multiple worker pools;
- queue replication;
- autoscaling;
- secret isolation;
- network egress policy;
- SLOs;
- disaster recovery;
- backup/restore;
- progressive rollout/drain;
- security review;
- chaos/failure tests.

## Smartphone-only delivery

The phone is the operator console, not the HA data plane.

Use:
phone → GitHub/Cloudflare/API → CI/build → runtime workers

Termux is appropriate for SSH, smoke tests, manifest generation, emergency workers, log inspection and API calls.

The phone must never be a required always-on component.

## Exit criterion

A small queue-backed workflow executes end-to-end:

phone/API → control plane → queue → non-GitHub worker → state CAS → immutable result → recovery

GitHub can remain a passive audit/CI surface.

## Priority

P0:
- state segmentation contract;
- queue abstraction;
- worker registration/lease protocol;
- hierarchical admission/quota model.

P1:
- non-GitHub worker;
- OTel-compatible lifecycle envelope;
- research fetch/verification;
- SkillManifest registry.

P2:
- MCP/A2A interoperability;
- advanced fairness;
- multi-region;
- enterprise compliance.