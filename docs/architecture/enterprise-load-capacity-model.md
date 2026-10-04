# Enterprise Load and Capacity Model

Status: design model; no production-capacity claim.

## Purpose

"Parallelizable" does not mean "high-volume ready".

Capacity is limited by every shared substrate on the critical path:

`ingress -> authority -> dispatch -> worker -> provider/object store -> completion -> visibility`

Every shared quota, lock, database write, provider limit, and queue can become the actual bottleneck.

## Current free-first envelope

The current Cloudflare Durable Objects free tier is quota-bounded. Current documentation lists 100,000 DO requests/day, 13,000 GB-s/day, 5 million SQLite row reads/day, 100,000 row writes/day, and 5 GB stored data.

Cloudflare Queues currently has a 10,000-operations/day free allowance, while R2 currently includes a free monthly storage/operation allowance suitable for small pilots.

GitHub's public standard hosted runners are free, but concurrency and API limits still constrain GitHub-centered orchestration. GitHub documents a 20-job standard concurrency tier for the Free plan and secondary REST API protections such as a 100-concurrent-request ceiling.

These numbers are planning constraints, not throughput guarantees.

## Control-plane arithmetic

A live task can consume some combination of:

- workflow lease acquire/renew;
- state read;
- state CAS;
- effect claim;
- effect completion/resolution;
- resource lock acquire/release;
- recovery;
- outbox;
- provider-rate admission.

Use three planning envelopes until telemetry replaces estimates.

### Light task

Assume 3 authoritative requests/task.

`100,000 / 3 ~= 33,333` task transitions/day.

### Normal task

Assume 5 authoritative requests/task.

`100,000 / 5 = 20,000` task transitions/day.

### Research/effect-heavy task

Assume 8 authoritative requests/task.

`100,000 / 8 = 12,500` task transitions/day.

These are arithmetic ceilings only. Lease renewals, retries, failed requests, reads, recovery, alarms, rate admissions, and operator operations reduce the actual safe envelope substantially.

Until measured, keep a large reserve rather than operating near the free limit.

## The provider-rate trap

A global rate gate is correct when multiple worker pools share one upstream provider, but a gate request for every upstream call becomes a new bottleneck.

The current Crossref path can spend a control-plane request before making the actual Crossref request.

Future hierarchy:

1. local token bucket;
2. worker-pool budget;
3. shared provider permit service only where global coordination is necessary;
4. provider response-header feedback;
5. cache reuse;
6. batched permits/leases at higher volume.

A batch permit must expire and be bounded. Worker loss cannot permanently strand capacity.

## Queue stability

For each queue/partition, observe:

`arrival_rate`, `service_rate`, `queue_depth`, `age_of_oldest`, `retry_rate`, `dead_letter_rate`

A stable system requires:

`arrival_rate < service_rate`

with meaningful safety margin.

When arrival approaches service, queue latency becomes highly sensitive to variance. Admission control should slow producers before workers are completely saturated.

Per-tenant or per-capability partitions become useful when one workload class creates convoying for another.

## Worker capacity

For a worker class:

`throughput ~= concurrency / mean_service_time`

For admission planning, use a conservative service-time percentile:

`required_concurrency ~= target_arrival_rate * p95_service_time`

Example:

- 20 tasks/s;
- p95 execution time 4 s.

Required concurrency is approximately 80 workers before retries, queueing overhead, and failure headroom.

Effective concurrency is:

`min(worker_capacity, provider_limit, tenant_limit, budget_limit)`

Therefore capability routing and provider quotas must be scheduler inputs.

## Hot-state sizing

Do not size the system only by workflow count.

Use:

`hot_state_bytes = active_workflows * average_hot_state_bytes`

and:

`cold_bytes = retained_runs * average_cold_payload_bytes`

The target architecture should keep hot state compact and move large payloads behind verified references.

Reference shape:

`{artifact_id, sha256, schema, location, created_by_task, workflow_id}`

## Enterprise benchmark profiles

The term "large scale" must be replaced by explicit benchmark profiles.

### E1 — sustained SaaS automation

- 10 workflow starts/s;
- 5 tasks/workflow;
- 50 task starts/s;
- mixed 0.5–30 s task duration;
- 10% retries;
- 1–5% human waits.

### E2 — research fan-out

- 5 workflow starts/s;
- 50–500 retrieval/evidence operations/workflow;
- high cache reuse;
- provider-specific rate limits;
- bounded context;
- passage-level extraction and validation;
- asynchronous fan-in.

### E3 — burst

- 0 -> 1,000 workflow starts within 60 s;
- durable buffering;
- bounded worker admission;
- no identity loss;
- no duplicate side effect.

### E4 — sustained high volume

- 100+ workflow starts/s;
- 500+ task starts/s;
- multiple tenants;
- independent worker pools;
- horizontally partitioned state/history;
- centralized observability.

The current GitHub + Cloudflare free-first stack is an engineering/pilot substrate for modest E1/E2 work. It is not an honest E4 execution substrate.

## Acceptance suite before a scale claim

1. Kill a worker after an external commit and before completion.
2. Interrupt state CAS and restart.
3. Deliver one task to two workers.
4. Deliver completion after lease expiry.
5. Saturate tenant A and verify tenant B progresses.
6. Exhaust provider X and verify unrelated capabilities continue.
7. Fill partition P and verify partition Q continues.
8. Grow history until hot-state limits are reached.
9. Reconstruct current state from history plus artifact references.
10. Replay a completed workflow without repeating external side effects.
11. Roll schema versions while workflows remain in flight.
12. Remove an entire worker pool and verify recovery.

Correctness failures and throughput failures must be reported separately.

## Free-first rule

The zero-dollar environment is for proving:

- contracts;
- invariants;
- recovery;
- replay;
- routing;
- portability;
- observability;
- quota behavior;
- research/evidence correctness.

It is not a promise of enterprise-scale throughput once a platform or provider free allowance is exceeded.

The intended progression is:

`$0 bootstrap -> bounded pilot -> substrate swap -> scalable production`

The workflow/evidence contracts remain stable across those stages.
