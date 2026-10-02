## Strict state identity validation v39

State migration now fails closed on malformed explicit schema versions and workflow identity mismatches. Legacy records without an `id` are repaired deterministically from their state key. Worker run-attempt fields are normalized and validated.

## Durable output privacy v38

Connector outputs no longer persist the raw bridge URL. Durable state receives a target fingerprint and sanitized discovery metadata instead, keeping deployment endpoint details out of public workflow artifacts.

## Worker attempt binding v37

Workflow state now binds the latest worker to both GitHub Actions run ID and run attempt. Continuation consumes both fields, preventing an earlier attempt of the same workflow run from satisfying a later continuation event.

## Reconciliation endpoint binding v36

Connector recovery now binds to both the original upstream execution target and the reconciliation endpoint. A change to either target causes reconciliation to fail closed before the lookup is dispatched; the bridge runtime repeats both checks independently.

## Concurrent single-flight failure semantics v35

Bridge-runtime single-flight now coalesces both successful and failed overlapping calls. Waiting duplicates observe the original flight outcome instead of launching a second upstream attempt after a failure, while a later fresh request can retry because failed flights are not stored in the success cache.

## Reconciliation target binding v34

Recovery now binds reconciliation to the same connector target and normalized action contract observed during the failed execution. A target or contract change causes reconciliation to fail closed before the recovery lookup is sent. The bridge runtime repeats the target comparison as an independent boundary.

## Upstream failure boundary v33

The connector bridge no longer treats an upstream non-2xx response as a successful cached operation. Typed upstream failures prevent false `completed` states and preserve the distinction between uncertain 5xx/network failures and non-uncertain 4xx failures. The orchestrator adapter independently rejects nested upstream non-2xx statuses as a second safety boundary.

## Connector target binding v32

Connector requests now bind retry semantics to a hashed upstream target identity in addition to the action contract. If the configured upstream target changes while a request is being retried, the control plane rejects the retry before another POST. The bridge runtime also includes that target identity in its local idempotency fingerprint, preventing an old cached result from being replayed against a different target.

The raw upstream URL is not exposed in discovery output; only its SHA-256 target fingerprint is carried forward.

## Parallel budget admission v31

The parallel executor now admits a safe-node batch against the workflow's remaining durable execution-step budget before persisting any node in `running`. A batch is truncated to available capacity; when capacity is exhausted, the deterministic next ready node is failed before another node can become stranded in an unfinished runtime state.

## Control-plane hardening v30

v30 closes cross-layer correctness gaps found after the v29 audit: workflow creation now persists the configured execution-step budget and uses the current workflow schema constant; connector idempotency is single-flight within a bridge process and cached responses are protected from caller mutation; continuation events bind to the worker run ID and run attempt; and reconciliation keeps the connector request ID distinct from the internal execution ID.

Connector action contracts are fingerprinted and pinned across retry attempts. If the bridge advertises a different action contract after an uncertain connector failure, the retry is rejected before another upstream POST rather than silently executing under changed semantics.

The bridge cache is intentionally process-local and best-effort. Durable duplicate suppression across bridge restarts or multiple bridge instances still depends on the upstream honoring Idempotency-Key.

# AI Orchestrator Core

## Connector intent-bound idempotency v29

Connector bridge request IDs are now derived from the immutable request intent: protocol, workflow, node, connector, action, and payload. Runtime timestamps and goal text do not change the idempotency identity.

The bridge runtime stores the intent fingerprint with cached results and rejects an idempotency-key collision when the same key is reused for a different request intent. Retries of the same intent still replay the cached result; an explicit replan that changes connector/action/payload receives a distinct request identity.

## Continuation chain binding v28

`github_run_id` now tracks the latest worker Actions run that durably persisted the workflow, while `origin_github_run_id` preserves the initial run for provenance. This keeps `workflow_run` continuation correlated across multiple resume cycles.

The continuation worker still resolves the workflow by exact run correlation and dispatches the exact persisted `workflow_id`; it never falls back to the global `last_workflow_id`.
## Route snapshot + policy integrity v27

Every workflow now stores a deterministic route/policy snapshot: persisted tool choice, capability fallback candidates, node risk/action context, relevant tool policy fields, live mode, and the free-only setting.

On resume, the control plane recomputes that policy fingerprint before checkpoint recovery or execution. Registry/policy drift fails closed instead of silently continuing under changed side-effect or free-tier semantics. Explicit replanning refreshes the snapshot because the tool choice has intentionally changed.

Legacy workflows without a policy fingerprint initialize one before their first post-v27 node execution.

## Lifecycle control plane v26

Workflow execution now has a durable logical-step budget (96 by default, bounded to 256) that is consumed before a node activation and persisted across continuation. Budget exhaustion fails closed instead of creating an unbounded replan loop.

Node contracts can declare deterministic postconditions such as field existence/equality, value membership, non-empty fields, or bounded HTTP status. These acceptance checks run before completion is recorded; post-side-effect acceptance failures remain subject to the existing no-duplicate replay fence.

Resume paths preserve the persisted plan/tool selection until an explicit replan, and the manual workflow dispatch exposes `max_steps` for a per-workflow budget.

## Execution preflight + resume plan immutability v25

Persisted workflow tool selection is now immutable across resume until an explicit replan. Initial creation still uses capability routing, while live execution preflights credentials, free-only policy, tool health, and HTTPS prerequisites before a side-effect durability barrier. Preflight failures can safely enter the existing bounded replan path.

The GitHub Actions worker now accepts both source orchestration issues and approval issues at the job filter level, closing the approval-event delivery gap.
Approval labels are fail-closed on recovery: `orchestrator-approved` / `orchestrator-rejected` are accepted only in the authenticated label-event path, while scheduled recovery cannot self-authorize an unlabeled-event context.

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
