# Enterprise Capacity and Load Model

Status: design/test specification only; no scale claim.
Date: 2026-10-05

## 1. Purpose

This document defines how the runtime must be capacity-tested before enterprise claims are made. The numbers below are target workloads, not current benchmark results.

## 2. Capacity equation

```text
sustainable_throughput = min(
  worker_capacity,
  broker_capacity,
  state_store_capacity,
  ingress_capacity,
  provider_capacity
)
```

Increasing worker concurrency cannot create throughput when another term is the bottleneck.

## 3. Latency decomposition

```text
task_latency = queue_wait + claim + execution + validation + persistence + effect
```

Every component needs independent p50/p95/p99 measurements. A high-throughput system can still have unacceptable latency when queue wait dominates.

## 4. Little's Law

For a stable workload:

```text
WIP = throughput × average_latency
```

For example, 100 completed tasks/s at 10 s average end-to-end latency implies about 1,000 active/in-flight tasks. This is a planning identity, not a repository benchmark.

## 5. Capacity dimensions

| Dimension | Measure |
|---|---|
| Ingress | requests/s |
| Workflow creation | workflows/s |
| Task production | tasks/s |
| Task completion | tasks/s |
| Queue | depth and oldest age |
| State | writes/s, bytes/write, latency |
| Events | appends/s |
| Artifacts | bytes/s and object count |
| Research | provider calls/s |
| LLM | calls/s and tokens/s |
| Workers | CPU, memory, active slots |
| Effects | external effects/s |
| Reliability | retry/s, DLQ/s, unknown-effect count |

## 6. Workload tiers

| Tier | Proposed validation target |
|---|---|
| W0 | 1 worker, <10 tasks/s, <100 workflows |
| W1 | 2-10 workers, 10-100 tasks/s, 100-1,000 workflows |
| W2 | 10-100 workers, 100-1,000 tasks/s, 1,000-100,000 workflows |
| W3 | 100+ workers, 1,000-10,000+ tasks/s, 100,000-1,000,000+ workflows |

W3 is a future target. It must not be advertised until a reproducible load test demonstrates the actual supported envelope.

## 7. State amplification

If a task result is 50 KiB and a workflow retains 20 dependency results, the embedded state already exceeds 1 MiB before metadata and history overhead. A monolithic state document therefore creates a hard scalability ceiling.

Target relationship:

```text
hot_state = metadata + immutable references + digests
large_payload = external immutable object/result
```

## 8. Write amplification

A single task may create claim, completion, workflow-event, snapshot and telemetry writes. At 1,000 task completions/s this can become thousands of storage operations per second before retries or recovery are counted.

Separate authoritative state writes from analytics. Telemetry must be asynchronously exportable and batchable. It must not make every task completion more expensive merely to preserve observability.

## 9. Queue stability

Approximate backlog dynamics:

```text
dB/dt = arrivals_per_second - completions_per_second
```

If the long-run arrival rate exceeds sustainable completion capacity, backlog grows without bound. Backpressure therefore belongs in admission policy rather than only in worker configuration.

Backpressure order:

1. Reject or delay new bulk work.
2. Preserve interactive capacity.
3. Reduce low-priority dispatch.
4. Throttle provider-heavy work.
5. Reduce speculative research breadth.
6. Pause maintenance.
7. Return explicit retry/deadline information.

## 10. Queue implementation validation

SQLite validates single-process semantics. PostgreSQL should validate the first multi-worker queue. PostgreSQL's SKIP LOCKED is specifically documented as useful for reducing lock contention among consumers of queue-like tables.

NATS JetStream should be validated for the high-throughput adapter where durable acknowledgements, redelivery and consumer scaling become important.

Workflow truth remains outside the broker in both cases.

## 11. GitHub capacity boundary

GitHub Actions is valuable as an ingress/event surface, CI system and worker adapter, but the controller must not interpret workflow count as global queue depth.

Current documented limits include a 256-job matrix ceiling, a 100-pending-run concurrency-group ceiling and a 35-day workflow-run ceiling. GitHub Free includes 2,000 hosted-runner minutes/month for private repositories. Self-hosted runners are free as an Actions service, but the operator supplies the machines.

These limits make GitHub dispatch an unsuitable abstraction for arbitrarily high-volume durable task scheduling even when higher plan/support limits are available.

## 12. Durable Objects capacity boundary

Workers Free Durable Objects currently include 100,000 requests/day, 13,000 GB-s/day, 5 million SQLite rows read/day, 100,000 rows written/day and 5 GB SQLite storage. Free-plan overages fail for the affected resource dimension.

Therefore the current Durable Object control-plane adapter is appropriate for bounded control workloads. It is not an unlimited enterprise scheduler.

## 13. Research workload amplification

Research has multiplicative fan-out:

```text
research_load ≈ queries × providers × pages × retries × claims
```

Every exploration workflow therefore needs explicit limits for sources, provider calls, LLM calls, wall time, retries, bytes and provider selection.

Recommended starting test policy:

```text
max_sources = 32
max_provider_calls = 96
max_llm_calls = 8
max_runtime = 15 minutes
```

These are example limits and should be tuned per workload class.

## 14. Failure-injection suite

Each scale campaign must inject:

- worker death immediately after claim;
- duplicate completion;
- late completion after lease expiry;
- scheduler crash around event/task publication;
- broker disconnect/restart;
- state-store failure;
- provider 429 and 500 bursts;
- malformed skill result;
- stale capability manifest;
- oversized state shard;
- control-plane restart.

Expected behavior is deterministic recovery, no duplicate side effect beyond the declared delivery semantics, bounded queue growth and auditable state transitions.

## 15. Benchmark report

Every run must publish:

```text
offered_load
accepted_load
completed_load
rejected_load
queue_p95_age
task_p95_latency
state_p95_write_latency
worker_utilization
retry_rate
dlq_rate
provider_throttle_rate
memory_peak
storage_growth
recovery_time
```

Run each scenario both in warm steady state and with failure injection.

## 16. Required enterprise gate

Do not promote a workload class until:

```text
throughput target passes
AND p95 latency target passes
AND fairness passes
AND duplicate/late completion passes
AND recovery passes
AND storage growth is bounded
AND provider budget remains bounded
AND operator telemetry is sufficient
```

## 17. Cost model

```text
total_cost ≈ compute + storage + network + provider_usage + telemetry + egress
```

Free-first should therefore expose billing classes and hard budgets for every provider/runtime. The scheduler should reject or downgrade work before an explicit free allowance is crossed.

Zero-dollar software licensing and free-tier operation are achievable for development and small workloads. Zero-dollar high-volume enterprise infrastructure is not a technically honest assumption.

## 18. Recommended proving order

First prove W1 with failure injection. Then W2 with partition skew and broker/state-store stress. Only after W2 is stable should W3 be attempted.

The purpose is to discover architectural limits empirically rather than derive enterprise readiness from code size or theoretical concurrency.