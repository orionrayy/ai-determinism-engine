# Enterprise Runtime Research Baseline — 2026-10-05

This memo records the external evidence used for the v80 architecture decision. Current limits are treated as infrastructure facts, not as design assumptions to hide behind abstractions.

## GitHub Actions

GitHub Actions is appropriate for CI, approvals, ingress events and fallback execution. It should not be the engine's high-volume task scheduler.

Relevant current platform constraints include a 256-job matrix limit and a 35-day workflow-run limit. Hosted-runner usage is metered; the GitHub Free plan includes 2,000 standard-runner minutes per month for private repositories.

Source:
- https://docs.github.com/en/enterprise-cloud@latest/actions/reference/limits

Design consequence:
- use Actions as an adapter, not as the queue;
- never model durable task existence as equivalent to a workflow run;
- do not put high-volume fairness or admission control in Actions concurrency groups.

## Cloudflare Workers / Durable Objects

Durable Objects are inherently single-threaded per object but scale horizontally across many objects. Current documented limits include up to 100 Durable Object classes on Free, 5 GB total SQLite storage on Free, and a soft limit around 1,000 requests/s for an individual object.

Workers Free currently allows 100,000 requests/day with 10 ms CPU per request.

Sources:
- https://developers.cloudflare.com/durable-objects/platform/limits/
- https://developers.cloudflare.com/workers/platform/limits/

Design consequence:
- keep workflow authority sharded by workflow identity;
- keep hot state compact;
- never treat one object as a global scheduler;
- treat Free limits as a development/small-load ceiling.

## Cloudflare Queues / R2

Queues currently document 5,000 messages/s per queue, 100-message consumer batches and 128 KB maximum message size. Free retention is 24 hours and the Free plan includes 10,000 queue operations/day.

R2 currently includes 10 GB-month storage, 1 million Class A requests, 10 million Class B requests and free egress in its standard free tier.

Sources:
- https://developers.cloudflare.com/queues/platform/limits/
- https://developers.cloudflare.com/queues/platform/pricing/
- https://developers.cloudflare.com/r2/pricing/

Design consequence:
- queue messages contain compact references, not large payloads;
- R2 is appropriate for immutable result/evidence/checkpoint shards;
- the queue abstraction must support a later enterprise queue without changing TaskEnvelope semantics.

## Temporal pattern

Temporal's architecture separates the service/control side from application Workers, persists ordered workflow Event History, and dispatches work through Task Queues. Its worker guidance recommends separate queues for distinct workloads and uses queue wait latency, task-slot saturation and worker resource usage as scaling signals.

Sources:
- https://docs.temporal.io/encyclopedia/architecture/temporal-architecture
- https://docs.temporal.io/best-practices/worker

Design consequence:
- separate workflow authority from worker compute;
- model queue wait as a first-class metric;
- use workload-specific worker pools.

## NATS JetStream pattern

NATS JetStream supports durable consumers, acknowledgements/redelivery, work-queue retention, consumer scaling and replication/node-loss recovery.

Source:
- https://docs.nats.io/learn/jetstream/

Design consequence:
- NATS is a good portable self-hosted reference for the QueueAdapter;
- the engine should assume at-least-once delivery;
- duplicate delivery must be absorbed by task/effect idempotency.

## Kubernetes production pattern

Kubernetes documents a production cluster as a control plane plus worker nodes, with multiple computers/nodes normally used for production fault tolerance and high availability.

Sources:
- https://kubernetes.io/docs/concepts/architecture/
- https://kubernetes.io/docs/setup/production-environment/

Design consequence:
- production worker scaling belongs outside the workflow controller;
- autoscaling should respond to queue backlog and resource saturation;
- high availability is an infrastructure tier, not a property of the LLM planner.

## OpenTelemetry

OpenTelemetry context propagation correlates traces, metrics and logs across process and network boundaries using trace/span context.

Source:
- https://opentelemetry.io/docs/concepts/context-propagation/

Design consequence:
- TaskEnvelope must carry trace context;
- queue wait, execution, retry and recovery should share one trace lineage.

## MCP Skills and A2A

The MCP Skills extension defines skills exposed via the Resources primitive and provides list/get operations, while the A2A protocol defines Agent Cards and task/artifact interoperability.

Sources:
- https://skills.extensions.modelcontextprotocol.io/specification/stable/skills
- https://a2a-protocol.org/latest/definitions/
- https://a2a-protocol.org/latest/whats-new-v1/

Design consequence:
- MCP and A2A are interoperability adapters;
- the internal SkillManifest remains authoritative for policy, trust, permissions, version and evidence requirements.

## Free-first research providers

OpenAlex currently describes a free daily API allowance of $1 per account, with usage-based charging beyond the free allowance. The API is therefore a metered-free provider, not an unlimited public service.

Source:
- https://help.openalex.org/access/pricing/

Design consequence:
- the engine's hard_free mode must track provider budgets;
- free provider identity/rate/policy belongs in deterministic admission control;
- “free” must never mean “ignore quota”.

## LLM policy

Gemini 3.8 Flash currently has a Free Tier and Google documents a transition to paid standard pricing from January 1, 2027. The engine therefore needs time-aware model eligibility and must not hard-code “free forever”.

Source:
- https://ai.google.dev/gemini-api/docs/pricing

Design consequence:
- preserve free_until policy in the model registry;
- refuse expired free-only models rather than spending silently.

## Overall conclusion

The external reference architectures converge on the same separation:

control plane → durable queue → replaceable worker pools → immutable payload storage → observability

The v79 code already contains many correctness primitives for this architecture. v80 therefore focuses on portable contracts and migration seams rather than replacing those primitives with a heavyweight framework.
