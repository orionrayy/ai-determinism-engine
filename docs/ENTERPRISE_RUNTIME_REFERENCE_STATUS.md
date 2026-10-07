# Enterprise runtime reference hardening

This PR completes and hardens the isolated enterprise-runtime reference layer for issue #149 without changing the production GitHub/Cloudflare execution path.

Implemented:
- E0 portable contracts and package-level contract boundary
- E2 SQLite at-least-once queue semantics with idempotent enqueue, lease reclaim, ack/nack, PostgreSQL SKIP LOCKED adapter, and NATS JetStream seam
- E3 snapshot + event-tail state segmentation with digest verification, sequence validation, replay, and path confinement
- E4 stateless worker registry/heartbeat/drain protocol and result fencing
- E5 deterministic hierarchical admission/quota checks and weighted fairness reference policy
- E6 versioned skill registry, digest verification, trust/permission checks, hard-free gating, and quarantine
- E7 trace/task identity propagation and JSONL/OTel-compatible reference telemetry
- E8 deterministic local W1/W2/W3 acceptance campaigns; these are reference/failure-injection tests, not enterprise capacity claims

Additional correctness fixes:
- expired tasks at max attempts now become dead rather than remaining processing
- stale successful worker acknowledgements are not blindly requeued
- PostgreSQL nack updates are claim-fenced
- worker heartbeats reject timestamps outside the tolerated window
- draining workers stop receiving new tasks
- state replay verify is no longer a tautology
- object-store keys cannot escape the configured root

Explicit non-goals:
- no Render/payment changes
- no sudden transition / frontend behavior changes
- no claim that the repository now has enterprise-scale physical capacity
- no production integration into the existing scheduler/control plane in this PR
