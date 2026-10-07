# v80 Enterprise-Scale Execution Fabric — Architecture & Capacity Design

Date: 2026-10-05
Status: Draft architecture RFC
Implementation status: Design only; no runtime migration is performed by this document.

## 1. Executive decision

The current engine is a strong durable workflow-orchestration core, but GitHub Actions + Git-backed state must not remain the primary execution, scheduling, or high-volume persistence plane for an enterprise deployment.

The target architecture is a portable execution fabric with five explicit planes:

1. Control plane — authoritative workflow coordination, fencing, quotas, leases, scheduling decisions.
2. Queue plane — durable work distribution and backpressure.
3. Payload plane — large immutable evidence, outputs, checkpoints, and artifacts.
4. Worker plane — pluggable executors; GitHub Actions is only one backend.
5. Evidence/skill plane — research, discovery, validation, skill capability and quality metadata.

The invariant is:

LLM = intelligence
Python/policy = deterministic policy
Control plane = authority
Queue = work transport
Object store = payload/history
Git = source/audit/reproducibility surface
Workers = replaceable execution capacity

The design must preserve this invariant across zero-dollar development, low-volume deployment, and future paid high-volume expansion.

A true enterprise-high-volume workload cannot be guaranteed indefinitely at $0. Free tiers are bounded by hard quotas and concurrency ceilings. The correct objective is a $0-compatible architecture whose interfaces, data contracts, and operational semantics do not need redesign when compute/storage/queue capacity is later purchased.

## 2. Current baseline assessment

The v79 branch has materially improved correctness at the epistemic and distributed-control boundaries. It now has:

- durable DAG execution and bounded parallelism;
- deterministic fallback planning;
- workflow fencing and effect claims;
- state CAS and recovery alarms;
- evidence trust binding;
- claim-level deliberation and truth-locking;
- bounded evidence passage validation;
- provider throttling;
- Git sharding and terminal compaction;
- federation of safe workers;
- offline control-plane evaluation.

That is appropriate as a kernel.

It is not yet an enterprise execution fabric.

### Major blockers

### A. GitHub Actions remains in the hot path

The current workflow still commits .orchestrator and pushes HEAD:main after execution. Multiple concurrent workflows therefore converge on one Git branch write surface and can require fetch/rebase/push retries.

This creates avoidable write contention, Git history amplification, and coupling between durable workflow progress and source-control mutation.

Git should not behave like a queue or database.

### B. GitHub Actions is not the durable high-volume scheduler

The current implementation uses Actions workflow triggers, concurrency groups, repository dispatch, artifacts, and scheduled recovery as major coordination primitives.

That is useful for a zero-service prototype and early worker pool, but it is not the correct long-term abstraction for large task volumes.

GitHub Actions standard GitHub-hosted concurrency is bounded by plan; current documentation lists 20 concurrent standard jobs for Free and 500 for Enterprise. Actions therefore remains a worker backend with a finite scheduler envelope, not an unbounded execution fabric.

### C. Control-plane hot state is still oversized

The Cloudflare Durable Object has an explicit 600 KiB workflow-state ceiling imposed by the current implementation.

The correct solution is not lossy trimming.

The correct solution is a compact authoritative manifest plus durable payload references, with deterministic reconstruction of the worker-facing state.

### D. Queue semantics are not yet first-class

The existing federation path can dispatch independent tasks, but there is no general queue protocol for every ready node.

An enterprise fabric needs:

ready node -> durable task message -> worker claim -> execution -> result reference -> supervisor commit

with duplicate delivery assumed.

### E. Skill discovery is under-modeled

The current registry maps capabilities to tools, but future multi-skill operation requires a richer skill contract.

A skill is not merely a tool name.

It is a versioned executable capability with schemas, resource requirements, cost class, evidence contract, effect contract, worker compatibility, health, and quality history.

### F. Research is still a provider bundle rather than a research fabric

The current provider layer is useful and free-first, but high-scale research needs explicit stages for:

query planning,
provider fan-out,
deduplication,
independence analysis,
access verification,
full-text materialization,
passage extraction,
retraction/correction checks,
evidence utility ranking,
cache reuse,
and bounded synthesis.

The research graph must be able to add providers without rewriting the orchestrator.

## 3. Target topology

    Client / event / API
             |
             v
    +-----------------------+
    | Ingress Gateway       |
    | auth + idempotency    |
    +-----------+-----------+
                |
                v
    +-----------------------+
    | Control Plane         |
    | Worker/API + DO shards|
    | workflow authority    |
    | fencing + quotas      |
    +-----------+-----------+
                |
        +-------+-------+
        |               |
        v               v
   Work Queues       Schedulers
        |
        v
   +----+-------------------------------+
   | Worker execution fabric            |
   |                                    |
   | GitHub Actions | CF Worker | Pull |
   | executor       | executor  | worker|
   +----+-------------+---------+-------+
        |
        +-------------------------------+
        |
        v
   +-------------------------+
   | Payload / Evidence Store|
   | R2 objects by digest    |
   +-----------+-------------+
               |
               v
   +-------------------------+
   | Audit / Replica         |
   | Git refs + manifests    |
   +-------------------------+

Cross-cutting:
- skill registry
- provider registry
- quotas
- rate limits
- resource locks
- trace/events
- evaluation
- quality ledger

## 4. Control plane design

A Durable Object remains an appropriate authority for a single workflow or other logical coordination atom because it gives serialized per-object coordination and can be horizontally sharded by deterministic object ID.

The current routing already has the right basic direction:

workflow:{workflow_id}
resource:{resource_key}
provider:{provider_id}

This must evolve toward explicit logical partitions.

### Required object classes

workflow shard:
- lifecycle/status;
- state version;
- lease/fence epoch;
- ready cursor;
- active task IDs;
- effect summary;
- checkpoint manifest ID;
- payload manifest ID;
- recovery schedule;
- tenant/workflow quota counters;
- deterministic plan/blueprint digests.

resource shard:
- lock owner;
- fence epoch;
- expiration;
- wait metadata;
- optional admission policy.

provider shard:
- rate budget;
- provider health;
- circuit-breaker state;
- quota window;
- retry-after state.

tenant shard:
- aggregate admission control;
- concurrent workflow limit;
- task rate limit;
- token/LLM budget;
- storage budget;
- provider budget.

A tenant shard is optional for the first migration but should exist in the protocol.

## 5. Queue plane

Cloudflare Queues is the preferred next transport abstraction because it is available on Workers Free, supports large numbers of queues, batch delivery, push consumers and pull consumers, and scales consumer concurrency with backlog.

The free tier is not the enterprise envelope: it currently provides 10,000 operations/day and 24-hour retention. A queue can support much higher functional throughput, including a documented 5,000 messages/second per queue limit, but the zero-dollar quota remains the real constraint.

Therefore the implementation must treat queue capacity as configuration, never as a correctness assumption.

### Task message contract

Every task message should contain only small routing data:

{
  "schema": 1,
  "task_id": "...",
  "workflow_id": "...",
  "attempt": 3,
  "claim_token": "...",
  "skill_id": "research.search",
  "skill_version": "1.x",
  "input_digest": "...",
  "context_manifest": "...",
  "output_manifest_target": "...",
  "deadline_at": "...",
  "tenant_id": "...",
  "trace_id": "..."
}

Do not put large evidence, prompts, documents, or node outputs into queue messages.

Queue messages are transport envelopes, not databases.

### Duplicate delivery

The queue protocol must assume at-least-once delivery.

A worker must be safe when receiving the same task multiple times.

Idempotency identity:

H(
  workflow_id,
  task_id,
  attempt,
  input_digest
)

The worker must first claim that identity against the control plane or a durable execution ledger, then execute.

An already-completed claim returns the stored result manifest rather than executing the side effect again.

## 6. Payload plane

R2 should become the default store for bulky immutable data:

- full evidence records;
- downloaded source material;
- evidence passages;
- research bundles;
- node outputs exceeding hot-state limits;
- checkpoint payloads;
- worker logs;
- large federation aggregates;
- evaluation datasets;
- large trace segments.

Store objects by content digest:

sha256/<digest>

Each object should have a deterministic manifest:

{
  "object_sha256": "...",
  "size": 12345,
  "media_type": "application/json",
  "schema": "evidence-records.v2",
  "created_at": "...",
  "producer": "research.search",
  "workflow_id": "...",
  "trace_id": "..."
}

The control plane stores the manifest reference and digest, not the full payload.

R2 currently has a 10 GB/month free storage tier, 1M Class A requests/month, 10M Class B requests/month, and free egress. That is enough to build and exercise the object-store architecture, but it is not an assertion that enterprise-scale retention is permanently $0.

## 7. Hot-state segmentation

The current 600 KiB ceiling is a design signal, not merely a limit to raise.

The new model should separate:

### Hot state
Must be small and authoritative:
- status;
- current node frontier;
- task claims;
- leases/fences;
- counters;
- digests;
- manifests;
- approval status;
- recovery state;
- minimal deterministic resume metadata.

### Warm state
Durable references:
- node input manifests;
- node output manifests;
- checkpoint manifests;
- evidence-corpus manifests;
- event segment references.

### Cold/audit state
Immutable history:
- full event payloads;
- evidence documents;
- large traces;
- historical outputs;
- evaluation data.

A workflow must be resumable from these references alone.

### Resume invariant

For any non-terminal workflow:

resume(workflow_manifest, referenced_payloads) == deterministic continuation

The test must verify both:
1. the same next-node frontier;
2. the same identity and side-effect fencing decisions.

The engine must never silently reconstruct missing payloads from a mutable source.

## 8. Git's new role

Git remains valuable for:

- source code;
- deterministic policy;
- versioned skill definitions;
- architecture;
- test fixtures;
- audit manifests;
- release tags;
- human review;
- reproducibility.

Git stops being:

- the live workflow database;
- the high-frequency queue;
- the primary recovery timer;
- the coordination lock;
- the mandatory per-task write path.

The live worker path should no longer require:

git commit -> git push -> rebase -> retry

for every state transition.

### Migration rule

Phase E is complete only when a live workflow can execute, recover, and finish while Git is temporarily unavailable, provided the configured control/payload planes are healthy.

Git synchronization can be an asynchronous audit replica.

## 9. Worker fabric

The worker protocol must become backend-neutral.

### Worker interface

A worker implementation must expose:

- identity;
- skill capabilities;
- supported schema versions;
- execution mode;
- resource limits;
- trust level;
- network policy;
- free/paid classification;
- health;
- maximum task duration;
- result protocol version.

### Initial backends

1. GitHub Actions worker
   - current implementation;
   - best for repository-native operations and smoke tests;
   - not the long-term universal worker.

2. Cloudflare Worker/Queue consumer
   - ideal for short, stateless, low-CPU tasks;
   - good for routing, normalization, validation, small research transforms.

3. HTTP pull worker
   - smartphone/Termux compatible;
   - ideal for experiments, special tools, provider connectors, and tasks that do not belong on GitHub runners.

4. External compute worker
   - future optional paid backend;
   - same task/result contract.

The control plane should not know which backend executed a task except through worker identity and capability metadata.

## 10. First-class skill model

Replace capability -> tool as the only routing abstraction with:

skill_id
version
capabilities[]
input_schema
output_schema
risk
cost_class
latency_class
network_class
resource_keys[]
effect_contract
evidence_contract
worker_pool
free_tier
max_runtime
max_payload
dependencies[]
health
quality_profile

Example:

skill:
  id: research.openalex.search
  version: 1.2
  capabilities:
    - research.search
  input_schema:
    query: string
  output_schema:
    evidence_records: array
  risk: low
  cost_class: free_allowance
  latency_class: network
  worker_pool: research
  evidence_contract:
    normalized_records: true
    passage_capable: false
  health:
    success_rate: 0.98
    p95_ms: 1200

### Version rule

A task is bound to a concrete skill contract version at admission.

Routing upgrades must not silently mutate the meaning of an in-flight task.

A newer skill version can be selected only for a new attempt/replan under explicit policy.

## 11. Skill routing policy

Routing should score candidates using deterministic policy, for example:

score =
  capability_match
+ schema_compatibility
+ evidence_capability
+ health
+ locality
- cost_penalty
- latency_penalty
- saturation_penalty
- risk_penalty

LLM may propose candidate skills.

Only deterministic policy may authorize final selection.

This preserves the existing design rule that LLMs provide intelligence while policy controls execution.

## 12. Research fabric

Research becomes a graph of independent skills rather than one monolithic provider bundle.

Suggested stages:

1. query decomposition
2. provider fan-out
3. bibliographic normalization
4. DOI/arXiv/PMID identity crosswalk
5. duplicate collapse
6. provider independence graph
7. access-route discovery
8. full-text materialization where permitted
9. retraction/correction screening
10. passage extraction
11. authority/integrity scoring
12. evidence utility ranking
13. evidence bundle materialization
14. claim-level synthesis
15. adjudication

Each stage has a bounded output and deterministic contract.

### Important correction to current behavior

The current research implementation should not claim "verified full text" merely because a provider exposes a full-text URL.

A future full-text skill should explicitly distinguish:

declared_access
metadata_access
retrieved_access
verified_content

The last state requires an actual fetch and validation event.

## 13. Backpressure and quotas

High-volume systems fail without admission control.

Introduce quota dimensions at tenant/workflow/provider/worker-pool levels:

- max active workflows;
- max queued tasks;
- max tasks/sec;
- max concurrent tasks;
- max LLM calls;
- max evidence requests;
- max bytes stored;
- max bytes read per workflow;
- max retries;
- max replan count;
- max side effects;
- max per-resource concurrency.

A scheduler must stop admitting work before downstream systems become saturated.

### Queue depth policy

For each worker pool:

healthy:
  queue_depth < soft limit

throttled:
  queue_depth >= soft limit

blocked:
  queue_depth >= hard limit

When throttled:
- reduce admission;
- prefer cached evidence;
- select cheaper/free skills;
- defer non-critical research;
- avoid speculative parallelism.

When blocked:
- do not create new work;
- allow in-flight work to drain;
- recover or reroute workers.

## 14. Resource-lock evolution

The existing resource objects are already sharded by resource key, which is directionally correct.

The next improvement is to avoid convoy behavior.

Instead of repeatedly polling a lock:

1. request lock;
2. receive either grant or deterministic wait token;
3. enqueue the waiting task;
4. wake/re-admit on release.

This turns lock contention into scheduler work rather than hot polling.

A resource lock should never block unrelated resources.

## 15. Effect protocol

The existing effect contract is one of the strongest foundations and should be generalized.

For every side effect:

effect_id
semantic_digest
provider_identity
attempt
fence_epoch
status
result_manifest
resolution_state

Allowed result states:

claimed
inflight
completed
not_applied
unknown

Unknown must remain fail-closed.

A worker cannot mark an effect completed merely because it received an HTTP 200. Provider-specific contracts must define what constitutes authoritative success.

## 16. Observability

The existing deterministic trace envelope is useful.

At scale, move detailed event payloads out of workflow state.

Use:

trace_id
workflow_id
task_id
worker_id
skill_id
attempt
queue_received_at
started_at
completed_at
result_sha256
failure_class

Generate structured event records and store them in the payload plane.

Required derived metrics:

- admission rate;
- queue wait p50/p95/p99;
- execution p50/p95/p99;
- workflow completion latency;
- task retry rate;
- replan rate;
- duplicate-delivery rate;
- CP CAS conflict rate;
- lease conflict rate;
- resource contention;
- provider 429 rate;
- provider failure rate;
- cache hit rate;
- evidence coverage;
- evidence rejection rate;
- claim adjudication escalation rate;
- cost-class utilization;
- free-budget exhaustion.

The engine should support an offline evaluator and a lightweight live exporter independently.

## 17. Quality ledger for skills and workers

Future routing requires feedback.

Maintain a compact quality record:

skill_id
version
worker_pool
sample_count
success_rate
contract_failure_rate
rework_rate
contradiction_rate
p95_latency
resource_cost_estimate
evidence_quality_score
last_updated

Do not let raw LLM self-ratings define quality.

Quality should be derived from deterministic validation, rework, contradiction, outcome, and latency signals.

## 18. Enterprise failure model

The system must be explicitly tested against these faults:

worker process disappears;
GitHub runner disappears;
queue message is duplicated;
queue consumer crashes after side effect;
control plane becomes temporarily unavailable;
control-plane CAS loses a race;
lease expires while worker is still running;
provider returns 429;
provider returns 5xx repeatedly;
provider changes response schema;
evidence URL disappears;
source becomes retracted;
R2 object is missing;
R2 object digest mismatches;
Git is unavailable;
Git branch has unrelated commits;
worker claims an old skill version;
planner generates an invalid graph;
planner output exceeds bounds;
malicious task attempts SSRF;
untrusted evidence attempts authority escalation.

Each fault must map to one deterministic state transition.

## 19. Capacity model

The design should stop using a single "max parallel" integer as the main capacity model.

For workload W:

admitted_tasks <= min(
  tenant_concurrency,
  queue_capacity,
  provider_budget,
  worker_capacity,
  resource_capacity,
  attempt_budget,
  evidence_budget,
  LLM_budget
)

The effective workflow concurrency is therefore dynamic.

A worker pool can expose:

capacity_now
capacity_soft
capacity_hard
current_load
queue_depth
estimated_drain_time

The scheduler should prefer the least-saturated compatible pool.

## 20. Free-first operating profiles

### Profile F0 — smartphone / development

Use:
- GitHub repository;
- GitHub Actions public standard workers;
- Cloudflare Workers Free;
- SQLite-backed Durable Objects;
- Cloudflare Queues Free where useful;
- R2 Free tier;
- free/public research providers;
- Termux only as an optional external worker.

No desktop is required.

### Profile F1 — small real deployment

Use the same interfaces, but:
- move high-frequency tasks away from Git commits;
- use queue-backed execution;
- store large payloads in R2;
- keep DO state compact;
- use one or more worker pools.

Still target $0 only while quotas are respected.

### Profile P1 — paid expansion without redesign

Increase:
- queue throughput;
- queue retention;
- DO read/write budget;
- R2 storage;
- worker concurrency;
- external worker capacity.

The API contracts must remain unchanged.

This is the most important economic property of the architecture.

## 21. Current free-tier reality

The following current limits must be treated as hard capacity boundaries, not assumptions:

GitHub Actions standard concurrent jobs are bounded by plan; current documentation lists 20 for Free and 500 for Enterprise.

Cloudflare Durable Objects Free uses SQLite and currently includes 100,000 row writes/day, 5 million row reads/day, and 5 GB total stored data. Exceeding a free Durable Objects limit causes further operations of that type to fail.

Cloudflare Queues Free currently includes 10,000 operations/day, with 24-hour retention. Queue functional limits include up to 5,000 messages/sec per queue, but the free operations quota remains the practical F0 bottleneck.

R2 Free currently includes 10 GB-month standard storage, 1 million Class A operations/month, 10 million Class B operations/month, and free egress.

These facts make $0 excellent for architecture development and bounded workloads, not a credible promise of unlimited enterprise throughput.

## 22. Smartphone-only operating model

The architecture must assume the operator may only have an Android phone.

Operator responsibilities should be possible through:

- GitHub mobile/browser for source, PR, issues, Actions status and logs;
- Cloudflare dashboard/browser for Worker/Queue/DO/R2 configuration and status;
- Termux for local smoke tests, pull-worker development and emergency diagnostics;
- browser-based API invocation for controlled administrative operations.

No workflow may require a local desktop checkout to remain healthy.

The phone is the control console.

The cloud is the execution environment.

## 23. Migration plan

### Phase A — v80 architecture contracts

Create:
- execution task protocol;
- result protocol;
- skill manifest schema;
- worker capability handshake;
- payload manifest schema;
- checkpoint manifest schema;
- queue envelope schema;
- SLO definitions;
- capacity test fixtures.

No behavior change.

### Phase B — payload abstraction

Create a storage interface:

put_object
get_object
head_object
delete_object
bind_manifest

Backends:
- local filesystem;
- R2.

Migrate only large payloads first.

### Phase C — queue execution adapter

Create:

enqueue_task
claim_task
complete_task
fail_task
defer_task

Implement:
- local/in-memory test backend;
- GitHub Actions adapter;
- Cloudflare Queue adapter behind a feature flag.

### Phase D — hot-state segmentation

Change workflow state from:

full node outputs/evidence
to:

compact execution manifest + payload references.

Add deterministic resume reconstruction tests.

### Phase E — remove Git from the live hot path

Git becomes:
- asynchronous audit replica;
- source/release control;
- human inspection surface.

A temporary feature flag may retain Git persistence for rollback.

### Phase F — skill/research fabric

Create:
- skill registry v2;
- worker capability discovery;
- quality ledger;
- research stage DAG;
- full-text materialization skill;
- evidence object store.

### Phase G — scale/chaos gate

Before production-scale claims, run synthetic load:

1, 10, 50, 100, 500, 1,000+ logical tasks

and varying:
- 1, 4, 16, 32, 64 workers;
- duplicate delivery;
- 429 storms;
- CP contention;
- worker loss;
- payload corruption.

Record:
- throughput;
- queue wait;
- state conflicts;
- recovery latency;
- duplicate side effects;
- storage growth;
- free-quota consumption.

Do not call the fabric enterprise-ready until the measured envelope is known.

## 24. Architecture decision rules

Rule 1:
Never use Git as a mutex.

Rule 2:
Never put unbounded payloads in the control plane.

Rule 3:
Never assume at-most-once queue delivery.

Rule 4:
Never allow an LLM to bypass deterministic admission or security policy.

Rule 5:
Never treat provider-declared access as fetched/verified access.

Rule 6:
Never make a zero-dollar free tier a correctness dependency.

Rule 7:
Never let a single worker backend define the task protocol.

Rule 8:
Never change an in-flight task's semantic contract implicitly.

Rule 9:
Never make recovery depend on one mutable source when an authoritative durable state exists elsewhere.

Rule 10:
Every scalable subsystem must have an explicit backpressure mode.

## 25. References

GitHub Actions limits:
https://docs.github.com/en/enterprise-cloud@latest/actions/reference/limits

Cloudflare Durable Objects limits:
https://developers.cloudflare.com/durable-objects/platform/limits/

Cloudflare Durable Objects pricing:
https://developers.cloudflare.com/durable-objects/platform/pricing/

Cloudflare Queues limits:
https://developers.cloudflare.com/queues/platform/limits/

Cloudflare Queues pricing:
https://developers.cloudflare.com/queues/platform/pricing/

Cloudflare Queues free-plan announcement:
https://developers.cloudflare.com/changelog/post/2026-02-04-queues-free-plan/

Cloudflare R2 pricing:
https://developers.cloudflare.com/r2/pricing/

## 26. Exit criteria for v80

v80 architecture is considered complete only when:

- Git is no longer required for live high-frequency state transitions;
- a workflow can resume from compact state plus immutable payload manifests;
- task delivery is at-least-once safe;
- worker backends are interchangeable;
- skills are versioned and routable by deterministic policy;
- large evidence/output payloads are outside hot state;
- research can fan out and scale independently;
- quotas/backpressure are first-class;
- chaos tests cover worker, queue, control-plane and payload failures;
- measured throughput/SLOs are recorded;
- the F0 smartphone-operated configuration and the P1 expansion configuration use the same protocols.

No production merge, migration, or replacement of the current v79 runtime is implied by this RFC.
