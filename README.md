# AI Orchestrator Core

Canonical technical memory and operating protocol: `ORCHESTRATION_MEMORY.md`.

Free execution path: GitHub Actions + stdlib Python control plane.

## Connector idempotency-aware execution fabric v8

Connector action contracts now affect runtime recovery: uncertain transport failures and HTTP 5xx responses are marked as potentially side-effecting. An action may be retried automatically only when its discovered contract declares `idempotent: true`. Uncertain non-idempotent failures fail closed and are not replanned to a different provider.

## Connector contract-aware execution fabric v7

Connector discovery now carries a sanitized contract per advertised action: required fields, primitive input types, and an idempotency declaration. In live mode, both the planner and executor validate connector payloads against the discovered contract before the upstream request; the bridge runtime repeats the check server-side. Schemas are optional to preserve compatibility with existing routes, and secret/free-form route data is not exposed.

## Connector-aware execution fabric v6

The live planner can consume the bridge's sanitized capability inventory and emit connector-bridge nodes with an explicit connector, action, and payload. Live execution performs a second preflight against the current bridge inventory before sending the request, so stale or unconfigured connector actions fail before upstream side effects. Discovery results are bounded, normalized, cached briefly, and included in successful connector evidence.

## Evidence-driven v4 layer

Nodes can declare output contracts and expected artifacts. Successful nodes emit SHA-256 evidence records. The free `artifact_verifier` can verify HTTPS URLs, repository files, or files inside the checked-out repository. Validation failures retain bounded repair feedback for fallback tools.

## Orchestration v3

The control plane now supports bounded parallel execution of independent low-risk DAG nodes, deterministic dependency-context propagation, semantic output contracts, free local validation, and ingress idempotency keys.

Safety semantics remain conservative: high-risk and side-effecting nodes are serialized and approval-gated in live mode; uncertain side effects fail closed rather than being replayed automatically. Replanning can return a failed node to the runnable queue after a bounded fallback switch.

Workflow dispatch accepts a `max_parallel` input (1–8). The default is 4. Dry-run remains credential-free and can exercise the complete state machine without executing external side effects.

## Components

- `orchestrator/orchestrator.py`: DAG/state/retry/replan engine plus controlled GitHub file/workflow operations.
- `orchestrator/tools.json`: capability and tool registry.
- `orchestrator/research_bundle.py`: credential-free Wikipedia/arXiv/Crossref research bundle.
- `orchestrator/issue_notify.py`: fail-safe GitHub Issue status notifications.
- `gateway.py`: authenticated event ingress that emits a GitHub repository dispatch.
- `.github/workflows/orchestrator.yml`: execution worker and scheduled resume.
- `.github/workflows/orchestrator-tests.yml`: compile + unit-test gate.

## LLM routing

Gemini is the primary LLM adapter and OpenAI is an optional fallback. Google currently lists Gemini 3.8 Flash as free at the standard API tier. API-key authentication is still required.

## Controlled GitHub operations

The GitHub adapter allowlists metadata/read operations plus `create_issue`, `create_or_update_file`, `delete_file`, and `dispatch_workflow`. Write/dispatch actions are automatically escalated to high risk and require approval in live mode.

## Event / approval flow

- Source issues use the `[ORCHESTRATOR]` prefix.
- High-risk actions create `[ORCHESTRATOR APPROVAL]` issues.
- Adding `orchestrator-approved` or `orchestrator-rejected` resumes the exact stored workflow.
- Each continuation is bound to the originating GitHub Actions run ID, not merely the latest workflow in state.

State is committed to the repository. Do not place secrets or private payloads in workflow goals when using a public repository.

## v66 Trace Contract and Evaluation Harness
The durable event stream now carries a deterministic trace envelope without adding a telemetry backend: workflow events map to workflow spans, node events to node spans, and agent events to agent child spans. The trace IDs are derived from workflow/node/agent identity, so replay and inspection can correlate lifecycle events without depending on wall-clock identity.

A dependency-free control-plane evaluation harness runs six cross-module invariants offline: blueprint repeatability, context-budget enforcement, semantic ingress binding, free-only policy, workload-wave ordering/parallelism, and trace hierarchy. It is executed in CI with no model/API call and fails the build when a control-plane invariant regresses.

## v64–v65 Deep Workload Orchestration and Integrity Hardening
The orchestrator now supports bounded long-horizon workloads through aggregate context budgeting, deterministic execution waves, and durable workload lineage. v65 additionally stabilizes context provenance digests, blocks federation ZIP decompression beyond the aggregate bound before expansion, binds ingress identities to semantic intent, and preserves parallelism metadata for downstream waves.
Regression coverage exercises the context, federation archive, ingress identity, and blueprint-wave integrity boundaries. No database, queue, broker, paid runtime service, or new dependency was introduced.

## v62 Planner Input Bound

The Gemini planner now fails closed before any model call when the goal exceeds 48 KiB. This aligns the planner input with the orchestration context budget and prevents oversized ingress from amplifying prompt size, token use, latency, and free-tier quota consumption.

No dependency or service was added; free-first execution remains unchanged.

## v61 Durable Event Redaction

Event payloads now pass through the same stdlib durable sanitizer used for workflow shards before they are appended to the event log. This closes the remaining path where diagnostic traces or credential-bearing exception text could bypass shard-level redaction and persist in the append-only audit stream.

Regression coverage verifies that bearer/API-key material is absent from durable event output while the diagnostic redaction marker remains auditable.
## v60 Attempt-Budget Accounting Hardening

Retry scheduling no longer consumes the workflow attempt budget ahead of the actual retry execution. The budget is charged exactly once at the execution boundary, preventing double charging and preserving retry/replan capacity under small workflow budgets.

Regression coverage now verifies both exact two-attempt accounting and fail-closed behavior when no attempt slot remains. No new dependency or service is introduced; GitHub Actions + stdlib Python and free-first execution remain unchanged.

## v59 Resource-Safety and Durable-Diagnostics Hardening

All outbound HTTP response bodies now pass through one stdlib bounded-reader helper. The LLM planner and credential-free research providers are capped at 512 KiB, while the connector bridge reconciliation path keeps its tighter 128 KiB limit. This prevents an unexpectedly large remote body from bypassing the existing resource-safety controls.

Durable workflow serialization now removes raw diagnostic traces from persistent state and redacts common credential-bearing patterns in exception text. The runtime keeps the original exception in-process for immediate handling, while durable state retains only bounded, non-secret diagnostics.

Execution callbacks now send a stable Idempotency-Key derived from the workflow execution identity, so bounded callback retries expose the receiver-side deduplication contract explicitly.

No database, broker, proxy, paid service, or runtime dependency is introduced; the execution path remains GitHub Actions + stdlib Python and free-first.

## v57 Connector Concurrent Idempotency

The connector bridge uses a per-request single-flight guard. Concurrent duplicates with the same semantic request identity wait for the original upstream execution instead of issuing a second call. A conflicting payload for the same identity fails closed. The wait is bounded; failure does not trigger a second upstream execution.

This is intentionally scoped per request identity rather than a global bridge lock, preserving concurrency for unrelated requests. Provider-side idempotency and reconciliation remain required across bridge restarts or multiple bridge replicas.

## v56 Connector Bridge Correctness Hardening

The connector bridge now binds its in-memory idempotency cache to a semantic request digest. Reusing the same request ID with a different connector/action/input fails closed instead of returning an unrelated cached response. Validation and the current free-only policy are re-run before replay.

Upstream non-2xx responses, transport failures, and oversized responses are surfaced as explicit upstream failures; 5xx/transport/oversized cases remain marked uncertain so the orchestrator can enter reconciliation rather than mistaking an upstream failure for success. The bridge caps upstream response bodies at 128 KiB and bounds its local replay cache.

The Render and Vercel bridge handlers preserve the distinction between malformed client requests and uncertain upstream failures. No external database, queue, paid service, or runtime dependency is introduced.

## v48 Sharded Workflow State

Workflow snapshots are stored under `.orchestrator/workflows/` using SHA-256(workflow_id) filenames. `.orchestrator/state.json` remains a compact compatibility/index envelope, while `load_state()` hydrates the canonical workflow shards and validates their identities and size bounds.

Normal execution writes only the active workflow shard. Continuation, scheduled recovery, and job-summary paths use the same canonical loader, so sharding does not leave recovery dependent on the empty index file.

## v45 Reliability Hardening

The gateway HMAC covers timestamp, HTTP method, request path, optional `Idempotency-Key`, and the raw body; header names are normalized case-insensitively. Unkeyed unstructured events receive a deterministic event identity that is propagated as the workflow identity for replay serialization.

The orchestrator persists a newly created workflow before its first execution step and no longer writes a stale caller snapshot after execution. Continuations carry the exact GitHub Actions run ID and run attempt into the dispatch step, and source Issue triggers explicitly bind the Issue action event.

## v46 Reliability Hardening

The gateway HMAC covers timestamp, HTTP method, request path, optional `Idempotency-Key`, and the raw body, with case-insensitive header lookup. Unkeyed unstructured events receive a deterministic event identity that is propagated as `workflow_id` for replay serialization.

New workflows are durably persisted before their first execution step. Execution paths no longer perform stale final `save_state()` writes after `persist_workflow()`, preventing one workflow worker from clobbering unrelated state loaded earlier in the process. Source Issue triggers explicitly bind the GitHub Issue action, and continuation dispatch exports the exact `workflow_run.id` and `run_attempt` used by its event identity.

## v49 Targeted Workflow Hydration

When a workflow ID is already known, the orchestrator loads only that workflow's canonical shard instead of hydrating every workflow snapshot. Global resume/list and event-id deduplication continue to use the full loader because those operations require repository-wide visibility.

## v52 Identity-First Ingress

Ingress events with an `Idempotency-Key` or `event_id` receive a deterministic internal workflow identity. New duplicate checks read that single canonical shard first; a full-state scan is retained only as a legacy compatibility fallback. Reused identities with conflicting intent/input digests fail closed. Repository-dispatch concurrency also falls back from workflow ID to event ID and idempotency key, reducing duplicate execution races before state inspection.

## v53 Terminal State Lifecycle

Terminal `completed` and `cancelled` workflow shards older than the configured retention window are compacted instead of deleted. Identity, status, fingerprints, input/idempotency metadata, provenance, node outcome summaries, and pre-compaction hashes remain available for targeted lookup and audit; `failed` workflows are not compacted because they can still be replanned or reconciled.

The default compaction age is 30 days (bounded to 1–365 days). Scheduled recovery performs the maintenance with the same bounded Git rebase/push guard used by the state writer. This reduces current checkout/state size without rewriting Git history or introducing another service.

## v54 Connector Output Verification

Connector actions may optionally advertise `result_required` and `result_types` alongside their input contract. Live responses are validated against that contract before acceptance by the orchestrator. Connector invoke and reconciliation response bodies are capped at 128 KiB. If a response is oversized or violates the declared output contract after a successful upstream call, the result is treated as uncertain so the existing idempotency and reconciliation policy remains in control.

These checks are optional for existing connectors, so the protocol remains backward-compatible. No external schema engine or paid service is required.

## v55 Durable Connector Output Redaction

Connector responses remain available to the current worker, but durable workflow shards and checkpoints now redact sensitive-looking keys such as tokens, secrets, passwords, authorization values, API keys, credentials, and private keys. Each redacted value keeps a deterministic SHA-256 marker for provenance. Durable evidence summaries use the same sanitizer while the full runtime output hash is preserved.

The sanitizer also bounds nesting, collection size, and long string representation. No external DLP service, proxy, database, or paid dependency is required.

## Gateway

Set these environment variables on the gateway service:

- `GITHUB_REPOSITORY` — defaults to `orionrayy/ai-determinism-engine`.
- `GITHUB_GATEWAY_TOKEN` — GitHub token allowed to dispatch the repository event.
- `GATEWAY_SHARED_SECRET` — bearer secret for inbound event authentication.

Endpoints:

- `GET /health`
- `POST /event` with JSON `{"goal":"...","metadata":{...}}`

The gateway is intentionally stateless. Do not store workflow state on its filesystem because Render Free web services have ephemeral filesystems.

## Free-hosting note

Render provides a Free web-service plan suitable for prototypes. Free services can spin down after 15 minutes of inactivity and restart on the next request, so this gateway should be treated as an event ingress rather than an always-on worker.

## Connector Bridge Runtime

The bridge runtime is available at `api/bridge.py` and can run on Vercel Python Functions. Configure `ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET` and `ORCHESTRATOR_CONNECTOR_ROUTES` as deployment secrets/environment variables. Routes are an explicit allowlist mapping connector names to HTTPS upstream endpoints and allowed actions. The bridge enforces protocol validation, HMAC replay protection, HTTPS upstreams, payload bounds, and best-effort idempotency.

The bridge also exposes sanitized connector discovery at `/capabilities` (Render) or `/api/bridge/capabilities` (Vercel). Discovery reports allowlisted actions, declared capabilities, risk, free-tier status, and whether required route credentials are configured; it never returns secret values.

## Reconciliation rearm v16

A confirmed `not_applied` connector reconciliation now settles the prior execution-ledger record before returning the node to `ready`. This allows a fresh side-effect attempt only after an authoritative negative reconciliation; `unknown` remains fail-closed.

## Connector reconciliation v10

When a live connector request becomes uncertain, the bridge can optionally expose a read-only reconciliation endpoint. The orchestrator sends a signed request containing the deterministic request id, connector, and action. The upstream returns `applied`, `not_applied`, or `unknown`. The first two states allow safe continuation without blindly replaying an uncertain side effect; `unknown` remains fail-closed.

## Reconciliation privacy v11

Connector reconciliation persists only a sanitized state record (`applied`, `not_applied`, or `unknown`) plus identifiers and discovery metadata. Arbitrary upstream response bodies are intentionally excluded from persisted workflow state.

## Reliability policy v12

The control plane classifies failures before retrying, applies bounded deterministic jitter to backoff, and rejects oversized connector capability inventories before they enter planner or persisted evidence paths.

## Plan integrity v13

Persisted workflows carry a deterministic plan fingerprint. On resume, the control plane recomputes the fingerprint and fails closed on drift before executing a node. Replanning updates the fingerprint intentionally, while runtime-only fields are excluded.

## Durable state v14

The orchestrator migrates persisted state to its supported schema and verifies completed-node checkpoint digests and bindings before resumed execution. Unsupported future schemas and corrupted checkpoints fail closed.

## Approval intent binding v23

High-risk approval issues now bind to a SHA-256 fingerprint of the exact node definition at approval creation. A labeled approval is accepted only when the current node fingerprint still matches. The approval actor and approval timestamp are recorded for auditability; stale approvals are revoked locally and the node returns to the approval flow.

## Durability barrier recovery v22

A failed pre-side-effect durability barrier is safe to rearm because the barrier invokes no external effect. The worker records the execution as prepared and leaves the node ready for a fresh worker checkout before another live side-effect attempt. This restores liveness without weakening the v19/v20 replay fences.

## Interrupted side-effect recovery v21

On resume, a live side-effecting node is checked against its durable execution ledger. A running node with status prepared is safely rearmed because the external-effect barrier has not been crossed; a running node with status started is converted into an explicit in-doubt failure. Connector nodes then enter the existing reconciliation path; opaque side effects remain fail-closed and are not replayed.

## Post-start side-effect replay fence v20

Once a live side effect has passed the durable START barrier, the worker does not automatically retry or replan a non-idempotent side-effecting operation after an execution failure. The only automatic retry exception is an uncertain connector request whose freshly discovered action contract explicitly declares idempotent: true; that retry reuses the deterministic request id. This keeps a timeout or partial response from turning into a second external write.

## Pre-side-effect durability v19

For live side-effecting nodes, the GitHub Actions worker commits and pushes the execution `START` record to `main` before invoking the external effect. A remote branch drift or push failure blocks the effect. This closes the runner-crash window where an external side effect could succeed before its `started` state became durable. The barrier is intentionally conservative and does not make Git plus an external provider one atomic transaction.

## State durability v15

State JSON writes use same-directory temporary files with flush/fsync followed by atomic replacement. Persisted workflow schemas newer than the supported version fail closed instead of being guessed at.

## General Blueprint Workload Compiler (v63)

The control plane now exposes a deterministic, side-effect-free `blueprint` capability backed by `blueprint_compiler`. Structured workload specifications can be normalized into bounded, traceable execution units without bypassing the authoritative DAG, federation, checkpoint, retry/replan, idempotency, risk, or approval machinery. Markdown/text ingestion only extracts headings and provenance metadata; semantic interpretation remains an explicit research/analysis step.

Bounds are 256 KiB source input, 512 requirements, 64 execution units, 8 requirements per unit, 24 KiB per execution unit/packet, and 480 KiB per compilation manifest. Optional file ingestion is restricted to `ORCHESTRATOR_WORKLOAD_ROOT`, keeping workload repositories separate from this control-plane repository.

## v64 Deep Workload Orchestration

The control plane now supports bounded long-horizon workloads through deterministic context packing, blueprint execution waves, and durable workload lineage. High-fan-in dependency context is globally capped at 48 KiB. Blueprint compilation emits dependency-safe waves capped at 24 units so workloads larger than one runtime DAG can be sliced into resumable boundaries. Blueprint metadata and unit/wave progress can be recorded without coupling the runtime to a specific project.

These primitives are side-effect-free and do not replace the supervisor. Existing risk, human approval, connector idempotency/reconciliation, checkpoint/evidence, federation, free-only routing, and retry/replan policies remain authoritative.
