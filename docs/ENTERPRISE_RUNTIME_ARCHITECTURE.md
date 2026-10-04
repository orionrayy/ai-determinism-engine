# Enterprise Runtime Architecture — v80

Status: architecture baseline / implementation-ready specification
Parent baseline: v79 distributed control plane + epistemic boundary hardening
Primary constraint: free-first, with a hard $0 operating mode that never silently spills into paid services

## 1. Executive decision

The engine should stop treating GitHub Actions as the long-term execution fabric.

GitHub remains the source-control, CI, audit, release, approval, and operator surface. High-volume execution moves behind a queue-driven runtime with independent workers. The workflow/control model already present in v79 remains the semantic core.

Target separation:

| Plane | Responsibility | $0 path | Enterprise path |
|---|---|---|---|
| Source/audit | Code, contracts, manifests, audit trail | GitHub | GitHub Enterprise / Git |
| API/ingress | Accept workflow requests | Cloudflare Worker | Managed API gateway / ingress |
| Workflow authority | Lease, CAS, effect fencing, recovery | Durable Objects | Sharded durable control service |
| Queue | Durable work dispatch | Cloudflare Queues or NATS lab | NATS/Kafka/SQS/PubSub-class system |
| Blob/data | Large payloads, checkpoints, evidence | R2 | R2/S3/GCS/Azure Blob |
| Workers | Execute tasks and skills | Termux/manual + one free VM/lab worker | Ephemeral containers/VMs/Kubernetes |
| Skill/tool plane | Discover and invoke capabilities | Git registry + HTTP/MCP adapters | Versioned registry + policy gateway |
| Agent plane | Multi-agent delegation | Internal envelope | A2A-compatible federation adapter |
| Observability | Trace/metrics/log correlation | JSONL + OTLP-compatible envelope | OpenTelemetry collector + backend |

The key invariant is that changing the queue, worker platform, or cloud provider must not change workflow semantics.

## 2. What v79 already gives us

The existing code has strong primitives:

- deterministic DAG validation and ready-frontier scheduling;
- bounded parallelism and repair/replanning;
- capability routing and health state;
- evidence binding and claim-level truth locks;
- effect identity, retry classification, reconciliation, and fencing;
- distributed workflow leases with fence epochs;
- workflow state CAS;
- recovery alarms/claims;
- an outbox;
- Git durable/audit state.

The latest observed v79 branch CI is green: Orchestrator Tests run 2034 and Private Input Worker Tests run 209 both completed successfully.

These primitives should be preserved and extracted behind stable contracts rather than replaced.

## 3. The GitHub scaling boundary

The current GitHub implementation still maps orchestration progress to GitHub workflow runs. The supervisor starts jobs, federation uses Actions matrices/artifacts, continuation is represented by repository events, and recovery scans are scheduled by Actions.

That is a good bootstrap architecture but a poor high-volume execution fabric.

GitHub documents a 256-job maximum for a single matrix and a 35-day workflow-run ceiling. Hosted-runner consumption is also metered; the Free plan includes 2,000 standard-runner minutes per month for private repositories. These constraints are orthogonal to the actual capacity of an AI execution engine. Actions should therefore be treated as a control surface and fallback runner, not the primary queue/scheduler for sustained high-volume workloads.

The existing federation workflow deliberately uses max-parallel: 1 to remain friendly to shared free LLM quotas. That is a signal that execution throughput is coupled to platform policy rather than controlled by the engine itself.

Rule:

GitHub events may start or observe a workflow, but a workflow task must not require a GitHub workflow run to exist in order for the scheduler to make progress.

## 4. Target control loop

Replace the large supervisor-job mental model with continuously reconciling controllers:

1. Ingress controller
   - validates requests;
   - assigns workflow and tenant identity;
   - creates immutable workflow manifests;
   - enforces hard $0 or explicit budget policy.

2. Workflow controller
   - owns DAG state and desired state;
   - computes runnable nodes;
   - advances the workflow after durable task results;
   - never executes arbitrary work itself.

3. Admission/quota controller
   - enforces tenant, workflow, capability, provider, concurrency and retry budgets;
   - rejects or delays work before queue admission when limits would be exceeded.

4. Queue dispatcher
   - converts runnable nodes to TaskEnvelope messages;
   - uses queue semantics, not Git commits, for dispatch;
   - supports priorities and fairness.

5. Worker registry
   - tracks worker capabilities, health, build/version, capacity and leases;
   - drains workers safely during deployment;
   - routes only compatible tasks.

6. Recovery controller
   - reconciles leases, inflight effects, timeouts, expired visibility, and stuck workflows;
   - resumes from immutable checkpoint manifests.

7. Evidence/research controller
   - treats research as a pipeline:
     discover → collect → normalize → dedupe → rank → fetch → extract → verify → adjudicate → package.

8. Observability controller
   - converts lifecycle events into trace/span relationships and aggregate metrics;
   - maintains a compact operational ledger separate from payload blobs.

This is the architecture shape used by mature distributed systems: control state is small and authoritative; work is asynchronous; workers are replaceable.

## 5. State architecture

### 5.1 Do not keep the entire workflow in one hot JSON value

The current control plane stores monolithic workflow state and the runtime deliberately protects it with a 600 KiB ceiling.

The safe replacement is a state manifest plus immutable shards.

Hot manifest:

- workflow_id
- tenant_id
- workflow_schema_version
- plan_digest
- state_version
- authority_mode
- workflow_status
- runnable node ids
- node status/attempt cursors
- dependency/result references
- active effect references
- queue/task references
- checkpoint generation
- shard manifest digest
- recovery generation/event id
- policy/budget digest
- trace root id

Cold/immutable shards:

- node outputs;
- large evidence corpora;
- research payloads;
- tool responses;
- artifacts;
- event/history segments;
- worker diagnostics;
- large agent proposals.

Every shard is addressed by content digest plus logical type.

### 5.2 Resume contract

A resume must reconstruct exactly the state needed to decide the next transition.

Checkpoint manifest shape:

    {
      "workflow_id": "…",
      "generation": 42,
      "schema_version": 8,
      "state_digest": "sha256:…",
      "shards": [
        {
          "name": "nodes/0007/output",
          "uri": "r2://…",
          "sha256": "…",
          "bytes": 12345,
          "required_for_resume": true
        }
      ],
      "watermark": "event:000123",
      "parent_generation": 41
    }

The control plane stores the compact manifest, not the large bodies. A shard is accepted only after digest verification. Resume fails closed when a required shard is missing or mismatched.

Git can mirror checkpoint manifests and selected audit events. Git should not be the hot-state database.

### 5.3 State transition protocol

Every state transition is:

read(version) → validate/compute → CAS(version → version+1) → emit event

A task result references:

- task id;
- workflow id;
- attempt;
- worker build/version;
- input digest;
- output digest or immutable result URI;
- trace context;
- effect resolution if applicable.

Late workers are rejected by fence epoch and state version.

## 6. Queue architecture

The TaskEnvelope is the cross-runtime unit.

Queue semantics should be at-least-once delivery. Exactly-once delivery is not assumed. Exactly-once effects are achieved through effect identity + semantic digest + fencing + reconciliation.

Required queue capabilities:

- durable enqueue;
- visibility timeout / lease;
- acknowledgement;
- bounded retries;
- dead-letter/quarantine;
- priority or weighted fairness;
- batching;
- consumer scaling;
- per-queue throughput isolation;
- optional delayed delivery;
- message size suitable for references, not large payload bodies.

Cloudflare Queues currently supports up to 5,000 messages/s per queue, 100-message consumer batches, 250 concurrent push-consumer invocations and a 128 KB message size. Its Workers Free pricing includes only 10,000 operations/day. Therefore Queues fit the $0 sandbox/reference path, but the engine must expose a generic QueueAdapter so an enterprise queue can replace it without changing task semantics.

NATS JetStream is a suitable portable reference because it provides durable consumers, acknowledgements and redelivery, work-queue retention, consumer scaling and replication options.

## 7. Worker fabric

A worker is not a GitHub job. It is a replaceable execution process.

Worker registration:

- worker_id;
- build_id;
- protocol_version;
- supported capabilities;
- skill versions;
- OS/runtime;
- CPU/memory class;
- max concurrent slots;
- current slot availability;
- network policy class;
- trust level;
- heartbeat timestamp;
- drain status.

A task is admitted to a worker only when:

capability_match AND policy_match AND trust_match AND capacity_match AND budget_match AND version_match

Worker lifecycle:

register → ready → leased → execute → result_commit → ack → idle

Deployment lifecycle:

ready → drain_requested → draining → stopped

No new task may be assigned after drain begins. Inflight tasks either complete or expire/reconcile according to their effect contract.

Production scaling should use independent worker pools per workload class. Mature workflow systems separate task queues and worker deployments and use queue-wait latency, task-slot saturation, and CPU/memory as scaling signals.

## 8. Scheduling and fairness

The current per-workflow limits are safety rails, not a production multi-tenant scheduler.

Introduce hierarchical quotas:

tenant → workflow → capability → provider → worker

Each level can constrain:

- max concurrent tasks;
- queue depth;
- CPU seconds;
- memory budget;
- provider requests;
- LLM token budget;
- research source budget;
- retry budget;
- wall-clock deadline;
- artifact bytes.

Admission happens before enqueue.

Fairness should use deterministic weighted queues or a deficit counter. A single tenant must not be able to saturate every worker by flooding one capability.

Backpressure states:

- accepted;
- delayed;
- quota_blocked;
- dependency_blocked;
- provider_limited;
- capacity_blocked;
- cancelled;
- dead_lettered.

These must be explicit state, not inferred from missing jobs.

## 9. Rate limiting

There are two different problems:

1. upstream provider limits;
2. internal worker/capacity limits.

They should not share one mechanism.

For providers, use short permit leases rather than one control-plane write per request at high volume.

Example:

acquire_permit(provider, count=20, ttl=5s)

The worker spends the local permit budget. This reduces control-plane chatter while retaining a global bound.

Provider policy differences remain authoritative. OpenAlex currently provides a $1/day free API allowance per account, so free is a budget class, not an infinite concurrency target.

Crossref list/filter access is also rate-limited, with different public and polite pools. The implementation must keep its configured rate at or below the authoritative provider rate.

## 10. Multi-tenant isolation

Every persistent identifier should be namespaced:

tenant_id / workflow_id / task_id / effect_id

Isolation requirements:

- tenant-aware authorization;
- tenant-scoped secrets;
- capability allowlists;
- network egress classes;
- storage prefixes;
- queue subjects/streams;
- rate-limit buckets;
- audit records;
- retention policies.

A skill must never inherit broader permissions merely because its worker has them. The worker should receive a task-scoped capability or policy projection.

## 11. Skill and tool architecture

The current capability graph is a strong deterministic router but is not yet a full versioned skill registry.

Define a SkillManifest with:

- stable skill_id;
- semantic version;
- capabilities;
- input/output schemas;
- runtime;
- dependencies;
- permissions;
- network egress;
- side-effect class;
- effect contract;
- evidence requirements;
- resource requirements;
- provider affinity;
- max concurrency;
- free-tier eligibility;
- provenance;
- source/digest;
- supported protocol adapters;
- health/version metadata.

The local registry can be files in Git. Production registry storage can be object storage plus a small metadata DB.

External interoperability should be adapters, not the source of truth:

- MCP adapter for tools/resources/prompts and Skills;
- A2A adapter for agent-to-agent task exchange;
- OpenAPI adapter for ordinary HTTP capabilities.

The internal SkillManifest remains authoritative.

The current MCP Skills extension exposes skills through resources and provides list/get operations. A2A 1.0 provides Agent Cards plus task/artifact semantics. The engine should map these into its own deterministic SkillManifest and AgentEnvelope so external protocol changes do not destabilize core policy.

## 12. Research plane

Research is a specialized workload class and should not run as an opaque web-search helper.

Pipeline:

discover → collect → normalize → dedupe → classify → rank → fetch → extract → verify → adjudicate → package

Each evidence record carries:

- canonical identity;
- provider;
- source URL/identifier;
- authority;
- independence group;
- access level;
- access verification;
- retraction/integrity signal;
- timestamps;
- corpus reference;
- corpus digest;
- passages;
- claims supported;
- retrieval trace.

The fetch stage must be separate from discovery. Metadata saying that a PDF exists is not equivalent to fetching and verifying the PDF.

The production research plane should add an actual fetcher with:

- HTTPS-only policy;
- DNS/IP validation;
- redirect controls;
- response size cap;
- content-type policy;
- digesting;
- cache;
- timeout;
- provenance;
- quarantine on malformed content.

Large corpus content goes to blob storage; LLM context gets bounded excerpts.

## 13. Observability

OpenTelemetry-style propagation should be built into the contract even if the first implementation emits plain JSON.

Required identifiers:

- trace_id;
- span_id;
- parent_span_id;
- workflow_id;
- task_id;
- attempt;
- worker_id;
- tenant_id.

Important metrics:

- queue_wait_seconds;
- execution_seconds;
- end_to_end_seconds;
- task_retries_total;
- effect_uncertain_total;
- reconciliation_total;
- state_cas_conflicts_total;
- provider_429_total;
- provider_permit_wait_seconds;
- worker_slots_available;
- worker_utilization;
- workflow_stuck_seconds;
- evidence_records_per_claim;
- evidence_verification_failure_total;
- LLM token counts when available;
- free budget remaining.

Correctness-critical events should never be sampled away. State transitions, effect decisions, security denials and evidence-binding failures are audit events. High-volume informational telemetry can be sampled.

## 14. Reliability tiers

A literal enterprise SLA at $0 is not a credible promise. The architecture must be zero-dollar-compatible, not zero-dollar-magical.

Tier Z0 — development / exploration:
- GitHub;
- Cloudflare free Worker/DO/Queues/R2;
- local/Termux worker;
- optional Oracle Always Free worker VM;
- no HA/SLA guarantee.

Tier Z1 — small production / personal service:
- external worker process;
- durable queue;
- R2/blob payloads;
- DO workflow authority;
- scheduled recovery;
- backups/checkpoint manifests;
- basic OTel-compatible logs.

Tier P1 — production:
- multiple worker pools;
- queue replication;
- durable metadata database;
- secret manager;
- automated rollout/drain;
- alerting;
- explicit SLOs.

Tier E1 — enterprise:
- managed or replicated control plane;
- Kubernetes/VM autoscaling;
- multi-AZ;
- multi-region replication/failover;
- WAF/API gateway;
- KMS/HSM-class secret handling;
- disaster recovery drills;
- tenant isolation;
- compliance/audit controls.

The APIs and contracts should remain identical across tiers.

## 15. Free-first policy

Free-first is a policy mode, not merely provider selection.

economy_mode = hard_free | free_prefer | budgeted

hard_free:
- no paid credential may be used;
- no prepaid balance may be consumed;
- daily/monthly free budgets are hard ceilings;
- when exhausted, work becomes quota_blocked;
- never silently downgrade evidence quality or substitute an untrusted provider.

free_prefer:
- use free capacity first;
- paid fallback requires explicit workflow policy.

budgeted:
- normal quotas and explicit cost ceilings apply.

This policy must be recorded in the workflow manifest and cannot be changed mid-run without a new policy revision.

## 16. Smartphone-only operating model

A smartphone is acceptable as the operator/developer console. It is not a reliable high-availability control plane.

Intended flow:

phone → GitHub/Cloudflare/API → CI/build → runtime workers

Use Termux for:

- SSH administration;
- local smoke tests;
- manifest generation;
- emergency worker execution;
- log inspection;
- API calls;
- small data transformations.

Do not require the phone to remain connected for scheduled orchestration.

A free VM, when used, should be replaceable. Its local disk must not be authoritative. Reproducible bootstrap scripts must recreate the worker from Git and environment configuration.

## 17. Migration strategy

Do not rewrite v79.

Phase 0 — contracts:
- TaskEnvelope;
- SkillManifest;
- CheckpointManifest;
- WorkerRegistration;
- QueueAdapter;
- result/effect envelope;
- TraceContext.

Phase 1 — queue abstraction:
- keep GitHub execution adapter;
- add a queue-backed executor interface;
- route one safe workload through the abstraction.

Phase 2 — worker registry:
- implement registration/heartbeat/drain;
- retain Actions as a worker implementation;
- add one non-GitHub worker.

Phase 3 — state segmentation:
- replace monolithic hot-state payloads with manifest + immutable shards;
- prove deterministic resume and digest verification;
- preserve Git audit replica.

Phase 4 — scheduler:
- move ready-frontier dispatch out of Actions;
- introduce hierarchical quotas and fair scheduling;
- Actions becomes fallback/bootstrap.

Phase 5 — research/skills:
- SkillManifest registry;
- real fetch/verify research stage;
- MCP/A2A/OpenAPI adapters;
- evidence provenance.

Phase 6 — production:
- horizontal worker pools;
- queue replication;
- observability;
- rollout/drain;
- DR;
- security hardening;
- workload SLOs.

## 18. Non-goals

- pretending Cloudflare Free is an enterprise SLA;
- replacing current epistemic validators;
- adding a heavyweight workflow framework dependency;
- forcing Kubernetes into the $0 path;
- making Git a distributed lock manager;
- storing large task payloads in queue messages or DO state;
- claiming exactly-once delivery;
- claiming semantic entailment from lexical passage matching.

## 19. Acceptance criteria

1. A workflow can be dispatched without creating a GitHub Actions run.
2. GitHub can remain an optional worker adapter.
3. A task can move between worker implementations without changing its contract.
4. Large results never need to fit inside hot control-plane state.
5. Resume succeeds using the authoritative manifest plus verified shards.
6. Queue delivery can be at-least-once without duplicate side effects.
7. Tenant/workflow/provider quotas are enforced before work floods the queue.
8. Skills are versioned and permissioned independently of workers.
9. Research provenance survives provider changes.
10. Trace context survives ingress → queue → worker → result → recovery.
11. hard_free mode is provably non-spending.
12. A phone-only operator can deploy, inspect and recover the system without a desktop.

## 20. Priority order

P0:
- checkpoint/state segmentation contract;
- queue abstraction;
- worker registration/lease protocol;
- hierarchical admission/quota model.

P1:
- non-GitHub worker;
- OTel-compatible lifecycle envelope;
- production research fetch/verification;
- SkillManifest registry.

P2:
- MCP/A2A interoperability;
- advanced fairness;
- multi-region;
- enterprise compliance controls.

Central principle:

The workflow engine owns correctness; the infrastructure owns transport and compute.

That separation is the main requirement for making the current foundation portable from GitHub-scale experimentation to high-volume execution.