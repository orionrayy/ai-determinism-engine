# ORCHESTRATION MEMORY — Canonical Control-Plane Context

## Purpose

This file is the durable, version-controlled memory of the AI orchestration control plane. Before modifying the orchestrator, treat this document and the current source tree as the source of truth. Do not assume older conversation state is still accurate without checking the repository.

## Current baseline

Repository: `orionrayy/ai-determinism-engine`
Primary branch: `main`
Current main baseline: orchestration hardening v66 deterministic trace contract + offline evaluation harness + v65 deterministic context/federation/ingress integrity + v64 deep workload orchestration + v63 general blueprint workload compiler + v62 planner input bound + v61 durable event redaction + v60 attempt-budget accounting + v59 resource-safety and durable diagnostics + v58 checkpoint identity and HTTP resource bounds + v57 concurrent connector idempotency single-flight + v56 connector correctness + v55 durable connector output redaction + v54 connector output contracts/response bounds + v53 terminal workflow state lifecycle/compaction + v52 identity-first ingress deduplication + v51 repository-event target routing + v50 goal-ingress state-load repair + v49 targeted workflow hydration + v48 sharded workflow persistence + v47 continuation/ingress/persistence race closure + v46 discovery snapshot provenance + v45 SSRF-safe artifact verification + v44 free-only reconciliation cost closure + v43 connector upstream cost gate + v42 free Gemini model gate + v41 private structured input boundary + v40 recovery routing/exact Actions run-attempt binding + v39 federation fairness/backpressure + earlier durable control-plane generations.
Execution model: GitHub Actions + stdlib Python
Cost policy: free-first; `ORCHESTRATOR_FREE_ONLY=true` in the production workflow
Current execution-fabric branch: `main`


## Orchestration hardening v42 — free Gemini model gate
- `orchestrator/tools.json` is now authoritative for Gemini model cost policy: it declares the default model plus an explicit `free_models` allowlist.
- `tool_available`, workflow planning, the Gemini planner, and the Gemini executor all enforce the same registry-backed model allowlist when `ORCHESTRATOR_FREE_ONLY=true`.
- `GEMINI_PLANNER_MODEL` can no longer silently bypass the free-only policy. An unlisted model causes the planner to be skipped/fail closed rather than invoking the API.
- Gemini executor requests are rejected before network execution when `GEMINI_MODEL` is outside the registry allowlist.
- Current Google pricing documentation lists Gemini 3.8 Flash, 3.7 Flash, 3.6 Flash, and 3.5 Flash with free-tier standard input/output pricing; the registry is intentionally explicit so future pricing/model changes require a deliberate policy update.
- No additional dependency or paid service was introduced.


## Orchestration hardening v43 — connector upstream cost gate
- `connector_bridge` remains free-hostable, but its upstream vendor cost is not assumed free.
- Bridge discovery now carries `free_tier` at both connector and action level. An action inherits the connector certification unless its action spec explicitly overrides it.
- When `ORCHESTRATOR_FREE_ONLY=true`, live connector execution requires both the connector and selected action to be explicitly certified `free_tier=true`; otherwise the request is rejected before any upstream call.
- Dry-run connector execution remains credential-free and does not require upstream cost certification.
- The connector free-only gate is intentionally performed after discovery and payload validation but before `post_request`, so it cannot create an upstream side effect when certification is absent.
- v47 execution lease was not ported: audit found it lacked provider-side fencing and contained a concrete release signature bug. Attempt counts already have a durable source of truth in Git-backed workflow state, so duplicating them in a remote attempt ledger would add another consistency domain without exactly-once guarantees.
- No new paid service or runtime dependency was added.


## Orchestration hardening v44 — reconciliation cost closure
- Free-only policy now covers the complete connector lifecycle: live invocation and recovery/reconciliation.
- Both the orchestrator connector client and the bridge runtime require connector-level and action-level `free_tier=true` before a reconciliation upstream request is allowed.
- An uncertified reconciliation is rejected before `post_reconciliation`/`dispatch_reconciliation`, so recovery cannot create an unbudgeted vendor API charge.
- Canonical memory layout is header-first and the current baseline explicitly records v43/v42/v41 so historical context cannot be mistaken for the active `main` state.
- No paid dependency or service is introduced.

## Orchestration hardening v45 — artifact verifier SSRF guard
- Artifact URL verification no longer uses the generic redirect-following HTTP helper.
- HTTPS artifact checks resolve the hostname, reject non-public/reserved addresses including IPv4-mapped IPv6, connect to the validated IP directly, preserve TLS SNI/hostname verification, disable redirects, enforce port 443, and bound the response body to 256 KiB.
- URL userinfo and fragments are rejected; path/query are percent-encoded before the HTTP request is built.
- This is a stdlib-only mitigation aligned with OWASP SSRF guidance; arbitrary URL verification remains available for public HTTPS destinations without adding a paid egress/proxy service.
- No new database, broker, queue, or paid dependency was introduced.


## Orchestration hardening v46 — discovery snapshot provenance
- Connector capability discovery snapshots now include a deterministic SHA-256 digest computed from the canonical normalized snapshot.
- The digest is evidence/provenance only; live execution still performs its existing fresh preflight and does not silently freeze a stale connector route.
- This preserves usability while making capability-contract changes observable and auditable across execution, reconciliation, and evidence records.
- No new dependency or service is introduced.


## Orchestration hardening v47 — continuation, ingress, and persistence race closure
- Gateway HMAC binds timestamp, HTTP method, request path, optional `Idempotency-Key`, and raw body; header lookup is case-insensitive.
- Unkeyed unstructured gateway requests receive a deterministic event identity, propagated as `workflow_id` so replay-equivalent repository dispatches share the same workflow concurrency group.
- Newly-created workflows are persisted before their first execution step; execution paths no longer perform stale caller-snapshot `save_state()` writes after durable workflow persistence.
- Source Issue handling explicitly binds `github.event.action` before bash `set -u` executes the labeled-event branch.
- Continuation dispatch explicitly exports the originating `workflow_run.id` and `run_attempt` used by its continuation event identity.
- No paid service, database, broker, queue, or runtime dependency is introduced; the control plane remains GitHub Actions + stdlib Python and free-first.

## Orchestration hardening v48 — sharded workflow persistence
- Workflow snapshots are stored per workflow under `.orchestrator/workflows/` using SHA-256(workflow_id) filenames; the compact `.orchestrator/state.json` is an index/legacy compatibility envelope, not the canonical workflow store.
- `persist_workflow()` writes only the active workflow shard, so unrelated workflows do not share the same JSON write surface during normal execution.
- `load_state()` hydrates legacy records plus canonical shards, lets shards override legacy copies, validates shard filename identity, bounds shard count and aggregate bytes, and recomputes `last_workflow_id` from durable timestamps when sharded.
- The continuation worker, scheduled recovery, and job summary all use the canonical Python `load_state()` path; scheduled recovery no longer parses `state.json` directly.
- `save_state()` remains only as a compatibility/bootstrap migration writer; normal runtime execution uses `persist_workflow()` exclusively.
- Storage format is explicitly tagged `sharded-v1`; the schema version remains unchanged because the workflow contract is unchanged.
- No database, event bus, paid queue, or paid API is introduced. The control plane remains GitHub Actions + stdlib Python and free-first.

## Orchestration hardening v49 — targeted workflow hydration
- `load_workflow(workflow_id)` loads only the canonical workflow shard when the workflow identity is already known, avoiding full-state hydration on ordinary continuation/targeted execution paths.
- The loader falls back to the legacy monolithic state only when the target shard does not yet exist, preserving incremental migration compatibility.
- `main --workflow-id` now uses targeted hydration; global `--resume`, `--list`, and event-id deduplication still use `load_state()` because they require a repository-wide workflow view.
- No new database, cache, service, or runtime dependency is introduced; the optimization reduces read amplification while retaining the existing fail-closed schema and identity checks.

## Orchestration hardening v50 — goal-ingress state-load repair
- Fixed a source-level regression where a literal `\\n` sequence accidentally commented out the `state = load_state()` assignment in the goal-driven ingress path.
- Added a regression that exercises event-id deduplication with a non-empty `ORCHESTRATOR_EVENT_ID`, proving the global state load is actually executed before duplicate workflow creation can occur.
- This is a correctness-only repair; no new service, dependency, or cost surface is introduced.

## Orchestration hardening v51 — repository-event target routing
- `main()` now promotes `ORCHESTRATOR_EVENT_WORKFLOW_ID` into the exact `--workflow-id` execution path when a repository event carries a target workflow.
- This closes the continuation/federation lifecycle gap where `repository_dispatch` supplied a durable workflow ID but the worker previously ignored it and could create a new workflow instead of advancing the stored one.
- Existing approval paths already pass `--workflow-id`; v51 aligns continuation/federation with the same targeted lifecycle path.
- No new service, dependency, or cost surface is introduced.

## Orchestration hardening v52 — identity-first ingress deduplication
- New ingress events derive a deterministic internal workflow ID from `Idempotency-Key` (preferred) or `event_id`, creating a stable canonical shard identity for duplicate requests.
- Goal-driven ingress checks that single canonical shard before falling back to the full `load_state()` scan, so new duplicate detection is O(1)-shard while pre-v52 legacy workflows remain discoverable.
- Reused idempotency/event identities now compare persisted intent/input digests when supplied; a mismatched request fails closed instead of being silently treated as the same operation.
- GitHub Actions repository-dispatch concurrency now falls back through `workflow_id`, `event_id`, and `idempotency_key`, so duplicate ingress instances are serialized before the orchestrator state check.
- This preserves the free-first GitHub Actions + stdlib Python control plane and avoids introducing a centralized paid or remote idempotency ledger.

## Orchestration hardening v53 — terminal state lifecycle/compaction
- Terminal workflow snapshots (`completed`/`cancelled`) older than the configured retention window are compacted rather than deleted; `failed` remains fully recoverable because it can still replan/reconcile.
- Compaction removes bulky goal/node input/output/evidence/reconciliation payloads from the active shard while preserving workflow identity, lifecycle timestamps, status, run lineage, fingerprints, digests, node outcome/error summaries, and hashes of the pre-compaction state/evidence/reconciliation records.
- Default compaction age is 30 days, bounded to 1–365 days. The policy is executed by scheduled recovery after dispatch evaluation and commits only changed workflow shards with the existing bounded rebase/push guard.
- This is a current-state size/lifecycle optimization; Git history remains immutable, so compaction does not rewrite repository history.
- No deletion/purge of workflow identity is performed, preserving targeted lookups and reducing the risk of old event identities becoming silently reusable.
- No new service, database, broker, queue, or paid dependency is introduced; the control plane remains GitHub Actions + stdlib Python and free-first.
## Orchestration hardening v54 — connector output contract and response bounds
- Connector discovery action specs can now optionally declare `result_required` and `result_types` for the sanitized upstream response object; absent fields remain backward-compatible.
- Live connector execution validates the returned response against that output contract after transport success but before the result is returned to the orchestrator for durable node completion.
- Connector invoke and reconciliation response bodies are bounded to 128 KiB, preventing an upstream response from causing unbounded worker memory growth.
- Live connector output contracts are checked after transport success; if a 2xx response is oversized or violates the advertised output contract, the failure is marked uncertain because the upstream side effect may already have occurred, forcing the existing reconciliation/idempotency policy to decide recovery.
- Contract validation remains downstream of the existing free-tier/idempotency/risk gates, so an invalid response cannot be mistaken for a successful side-effecting operation.
- This is stdlib-only and introduces no external schema engine, database, queue, proxy, or paid service.
## Orchestration hardening v55 — durable connector output redaction
- Connector responses can remain available to the current worker for downstream computation, but durable serialization now passes through `sanitize_for_durable()`.
- Sensitive-looking mapping keys such as authorization, password, secret, token, API key, credential, and private key are replaced in durable state/checkpoints with a redaction marker plus a deterministic SHA-256 of the removed value.
- Durable evidence summaries also use the sanitizer while `output_sha256` continues to hash the original runtime output, preserving provenance without persisting the raw secret-bearing representation.
- String, collection, and nesting bounds cap durable representation growth; this is an additional bound beyond the v54 transport response cap.
- The sanitizer is applied at workflow-shard and checkpoint write boundaries, so the runtime object is not mutated mid-execution and same-process downstream nodes retain access to raw connector results.
- No external DLP service, database, proxy, or paid dependency is introduced; the privacy boundary remains stdlib-only and free-first.
## Orchestration hardening v56 — connector idempotency and upstream failure semantics
- Connector bridge idempotency cache entries are now bound to a semantic request digest (connector, action, workflow, node, and input), so reuse of an idempotency key with different intent fails closed instead of returning an unrelated cached result.
- Current free-only policy and action input validation are re-evaluated before a cached replay is returned, preventing policy drift from bypassing a newly stricter gate.
- Bridge upstream invocation now rejects non-2xx upstream responses and propagates 5xx/transport/oversized-response failures as uncertain upstream failures to the HTTP boundary, so the orchestrator's reconciliation policy remains engaged.
- Bridge upstream response size is bounded to 128 KiB and the bounded in-memory idempotency cache has a 24-hour entry lifetime with a 128-entry cap.
- The API and Render bridge handlers distinguish upstream failures from malformed client requests without introducing a new service or dependency.
- Approval fingerprint regression coverage now explicitly verifies that runtime metadata is excluded while semantic payload changes invalidate approval.
- No database, broker, queue, paid dependency, or new runtime service is introduced; the control plane remains GitHub Actions + stdlib Python and free-first.

## Orchestration hardening v57 — concurrent idempotency single-flight
- The connector bridge now serializes concurrent requests sharing the same semantic request identity within a bridge process: one caller owns the upstream execution and concurrent duplicates wait for its completion.
- In-flight requests are bound to the semantic request digest; a concurrent reuse of the same request ID with different input fails closed instead of sharing an unrelated execution.
- A waiting duplicate either receives the completed cached result or returns an uncertain upstream failure after a bounded wait; it never starts a second upstream execution solely because the first request is still in flight.
- The single-flight scope is per request identity rather than a global bridge lock, so unrelated connector actions retain concurrency.
- This reduces thundering-herd duplicate side effects without adding a database, distributed lock service, queue, broker, paid dependency, or new runtime service. Provider-side idempotency/reconciliation remains authoritative across bridge process restarts or multiple bridge replicas.

## Orchestration hardening v58 — checkpoint identity and HTTP resource bounds
- Checkpoint filenames are derived from SHA-256(workflow_id + NUL + node_id) rather than raw identifiers, so even malformed legacy identifiers cannot escape the checkpoint directory through filename construction.
- DAG validation now bounds node identifiers to 100 characters and allows only ASCII letters, digits, dot, underscore, and hyphen. This prevents malformed planner IDs from entering dependency, event, checkpoint, and identity paths.
- The generic stdlib HTTP adapter now enforces a 2 MiB response cap by default; Firecrawl is explicitly allowed 4 MiB because scraped output is a larger but still bounded payload.
- Connector and artifact adapters retain their tighter, tool-specific bounds. No external dependency, database, queue, proxy, or paid service is added.

## Orchestration hardening v59 — resource safety and durable diagnostics
- Added orchestrator/http_safety.py, a stdlib bounded-response reader used across planner/research/bridge call sites to remove duplicated ad-hoc response reads.
- The Gemini planner response is bounded to 512 KiB. Credential-free Wikipedia/Crossref/Arxiv responses are bounded to 512 KiB. Connector reconciliation retains the existing 128 KiB bound.
- Durable serialization now treats trace, traceback, stack, and stack_trace fields as diagnostic material: only a deterministic hash and redaction marker are persisted.
- Durable string sanitization also removes common bearer/API-key/password/secret credential patterns before persistence, while runtime exception objects remain available to the current process.
- Execution callbacks now include Idempotency-Key: execution_id, making the existing three-attempt delivery loop explicitly compatible with idempotent receivers.
- Regression tests cover bounded planner/research responses, the shared reader, durable trace redaction, secret-pattern removal, and callback idempotency headers.
- No database, broker, proxy, paid service, or additional runtime dependency was introduced; the control plane remains GitHub Actions + stdlib Python and free-first.
## Orchestration hardening v60 — attempt-budget accounting
- The workflow-scoped AttemptBudget is now charged only when an execution attempt actually starts. Retry scheduling performs only a non-mutating remaining-capacity check.
- This fixes double charging in execute_with_retries(), which previously could consume two budget units for one retry and reduce later retry/replan capacity.
- Regression tests cover a two-attempt retry with max_attempts=2 and a max_attempts=1 fail-closed retry path.
- No database, broker, queue, paid service, or runtime dependency is introduced; the control plane remains GitHub Actions + stdlib Python and free-first.
## Orchestration hardening v61 — durable event redaction
- `append_event()` now sanitizes its payload before size bounding and append/fsync, so event logs share the workflow-shard/checkpoint secret-redaction boundary.
- This closes the residual durable-state path in which exception traces or credential-bearing error messages could bypass `sanitize_for_durable()` and land in `.orchestrator/events/*.jsonl`.
- Event redaction is fail-closed for diagnostic trace fields and common bearer/API-key/password/secret patterns while preserving event type, identity, and bounded audit metadata.
- No new database, broker, proxy, paid service, or runtime dependency is introduced; GitHub Actions + stdlib Python and free-first remain unchanged.
## Orchestration hardening v62 — planner input bound
- `plan_goal()` rejects goals larger than 48 KiB before reading the connector inventory into the prompt or invoking Gemini.
- The bound aligns with the existing `MAX_CONTEXT_BYTES` control-plane budget and prevents oversized ingress from expanding model prompt size and free-tier quota consumption.
- A regression verifies the external planner adapter is not called when the limit is exceeded.
- No database, broker, proxy, paid API, or runtime dependency is introduced.
## Orchestration hardening v64 — deep workload orchestration
- Added a deterministic aggregate context budget for node handoffs: total context is capped at 48 KiB, dependency count at 24, with bounded dependency bodies, contracts, and repair feedback.
- Added hierarchical execution waves to the general blueprint compiler. Dependency levels are converted into bounded waves with at most 24 units per wave, preserving a deterministic graph while exposing parallel execution opportunities.
- Added durable workload lineage in workflow schema v7: compiled blueprint/manifest digests, unit/wave counts, and bounded completed-unit history survive continuation and recovery.
- The existing supervisor remains authoritative for state and side effects; no new service, queue, broker, database, or paid runtime dependency was introduced.

## Orchestration hardening v65 — deterministic integrity and resource hardening
- Context provenance is now stable: context_digest excludes only its own self-reference and the mutable used_bytes counter, and final size accounting reserves space for digest metadata before returning the context.
- Federation aggregate ZIP members are size-checked from ZipInfo.file_size before decompression, with encrypted members rejected and an exact post-read size check. This closes a decompression-expansion resource path while keeping the existing 512 KiB archive transport cap and 128 KiB aggregate cap.
- Ingress identities now bind a deterministic semantic intent digest derived from the goal, execution mode, external operation identity, and private-input identity. Reuse of an idempotency/event identity with changed semantic intent fails closed instead of silently reusing the workflow.
- Blueprint wave metadata now marks any multi-unit wave as parallel-capable, including waves that depend on earlier levels; the prior metadata understated available parallelism.
- Added regression coverage for all four integrity paths. No new service, dependency, database, broker, queue, or paid API was introduced.

## Orchestration hardening v65 — deterministic integrity, liveness, and free-only hardening
- Context provenance is stable: the context digest excludes its self-reference and mutable byte counter, while final size accounting reserves room for digest metadata.
- Federation aggregate ZIP members are checked against their declared uncompressed and compressed sizes before decompression; encrypted members are rejected and post-read size is verified.
- Ingress identities now include a deterministic semantic intent digest, so reusing an event/idempotency identity for a changed goal or external operation fails closed.
- Safe federated workloads gain a bounded stale-recovery path after 10 minutes; only non-side-effecting delegated nodes can be rearmed and the reserved attempt/federation quota is refunded.
- Interrupted non-side-effecting nodes are durably marked running before execution and are rearmed on worker recovery, preserving the charged attempt budget instead of silently starting an unrecorded attempt.
- Gemini/planner prompts explicitly treat dependency and connector context as untrusted data; Gemini node output is capped at 2,048 tokens and planner output at 4,096 tokens with a single candidate to bound free-tier consumption.
- The free-only supervisor no longer injects disabled paid-adapter credentials into the runtime environment.
- CI push tests are restricted to main while pull-request checks use concurrency cancellation, avoiding redundant feature-branch execution on every intermediate commit.
- These changes remain GitHub Actions + stdlib Python with no database, broker, queue, proxy, paid runtime service, or new package dependency.
## Orchestration hardening v66 — trace contract and offline evaluation
- Durable events now carry a deterministic trace envelope derived from workflow, node, and agent identities. This adds workflow → node → agent correlation without a separate telemetry backend or database.
- The trace layer intentionally remains an event correlation contract rather than a full external observability product: timestamps and lifecycle events already exist in the durable event stream, while trace IDs provide stable hierarchy.
- Added a dependency-free control-plane evaluation harness covering blueprint determinism, aggregate context bounds, semantic ingress identity, free-only routing, execution-wave ordering/parallelism, and trace hierarchy.
- CI executes the evaluation harness with ORCHESTRATOR_FREE_ONLY=true. The harness makes no external model or connector calls.
- No database, broker, queue, telemetry backend, paid API, or new runtime package was introduced.
## Multi-agent coordination
The orchestration model uses a supervised multi-agent fabric without adding a second control plane:
- The orchestrator is the sole supervisor and authoritative state/side-effect writer.
- Nodes are typed agent-cells with a role, capability, stable agent identity, risk ceiling, and contract.
- Supported roles include researcher, skeptic, analyst, architect, implementer, tester, critic, publisher, communicator, operator, and verifier.
- DAG dependencies are explicit agent handoffs through dependency context; independent safe nodes use the existing bounded thread pool as scatter-gather execution.
- Deterministic fallback plans deliberately fan out researcher+sceptic lanes and converge through analysis/critic nodes for research, content, and software workflows.
- Gemini execution receives role-specific instructions and a stable agent identity. Without Gemini credentials, deterministic/free tool routing remains available according to the registry and dry-run remains credential-free.
- agent_fabric.py is stdlib-only and versioned as protocol v1. It can later map to GitHub Actions matrix/reusable-workflow workers using artifacts as ephemeral mailboxes, or to external MCP/A2A bridges, without changing the core task contract.
- Consensus is represented as DAG convergence plus a critic/verifier node, not unbounded peer-to-peer debate; this keeps attempts, retries, and side effects governed by the existing workflow budget and safety gates.

## Architecture

```
GOAL / EVENT
  -> planner (Gemini or deterministic fallback)
  -> validated DAG
  -> risk/policy enforcement
  -> bounded parallel executor for independent safe nodes
  -> serialized executor for side effects / high risk
  -> tool adapter
  -> contract + semantic validation
  -> retry / replan
  -> checkpoint + persisted state
  -> workflow_run continuation
  -> next node
```

State machine:

```
pending -> ready -> running -> validating -> completed
                       |             |
                       +-> retrying -+
                       |
                       +-> waiting_approval
                       |
                       +-> failed -> replanning -> ready
                                    |
                                    +-> reconciling -> validating -> completed
                                                    `-> ready
```

Workflow attempt budget:
- default `max_attempts`: 64
- hard cap: 128
- `attempts_used` is shared by retries and replans and persisted in workflow state

High-risk operations require explicit approval in live mode unless the workflow invocation explicitly supplies approval.

## Trigger model

Supported entry points:
- manual `workflow_dispatch`
- authenticated external ingress through `gateway.py` using `repository_dispatch`
- source GitHub Issues prefixed `[ORCHESTRATOR]`
- scheduled recovery every 15 minutes
- approval/rejection label events on `[ORCHESTRATOR APPROVAL]` issues
- post-run continuation through `workflow_run`

Continuation must target the exact persisted workflow. Never infer the target solely from `last_workflow_id`.

## Free-first policy

Registry is authoritative through `orchestrator/tools.json`.

Free-tier declarations currently include:
- github
- gemini
- wikipedia
- research_bundle
- noop
- local_validator
- artifact_verifier
- connector_bridge

Non-free declarations currently include:
- openai
- firecrawl
- webhook

Live free-only execution must reject a non-free adapter. Dry-run may simulate any registered adapter without requiring credentials, while still preferring free adapters.

Do not add a new tool without an explicit `free_tier` declaration.

## Research layer

`orchestrator/research_bundle.py` provides credential-free:
- Wikipedia
- arXiv
- Crossref

It tolerates partial provider failure as long as at least one provider succeeds.

## GitHub adapter

Allowlisted operations:
- metadata
- read_file
- create_issue
- create_or_update_file
- delete_file
- dispatch_workflow

Write/delete/dispatch operations are high-risk and approval-gated in live mode.

Repository path traversal is rejected. File writes have a 512 KiB payload safety limit.

## Security model

- External HTTP is HTTPS-only.
- Gateway supports bearer or timestamped HMAC authentication with replay window.
- Source Issue triggers are restricted to OWNER/MEMBER/COLLABORATOR.
- Approval labels are additionally authorized against the label actor's repository permission.
- Do not put secrets or private data into GitHub Issues or committed state.
- The current repository is public; persisted state is therefore not a private datastore.
- Keep shell inputs quoted and do not execute goal text as shell code.

## Persistence

Workflow state is committed under `.orchestrator/`.
Each workflow stores:
- workflow id
- created/updated timestamps
- goal
- live/dry-run mode
- trigger issue
- originating GitHub Actions run id
- node state
- retry/replan counters
- outputs/errors/checkpoints

Multiple workflows are scheduled fairly by oldest `updated_at`.

## Recovery semantics

A failed node:
1. retries with bounded exponential backoff when the typed failure policy permits it;
2. returns to `running` before a second execution attempt;
3. for uncertain connector side effects, reconciles before replay;
4. `applied` completes the node without replay;
5. `not_applied` marks the prior execution as settled-not-applied and rearms the node for a new attempt;
6. `unknown` remains fail-closed and blocks replay/replan;
7. known non-uncertain failures may use a registry fallback and replan up to the global limit;
8. otherwise fails the workflow.

Continuation dispatch occurs only after the worker's state persistence, preventing the former pre-persistence race.

## CI gate

`.github/workflows/orchestrator-tests.yml` performs:
- Python compilation
- all orchestrator unit tests
- gateway tests
- workflow configuration invariants

Never merge a control-plane change with a red CI result.

## Important historical defects already fixed

- stale in-worker continuation dispatcher
- approval labels not triggering worker
- approval creating a new workflow instead of resuming exact workflow
- continuation selecting wrong workflow through `last_workflow_id`
- pending workflow starvation
- planner risk understatement
- hard-coded paid-tool blacklist bypass
- dry-run blocked by missing credentials
- retry path `ready -> retrying` illegal transition
- approval actor authorization gap
- missing controlled GitHub artifact operations
- incorrect 404 handling for GitHub file creation

## Known remaining architectural limits

1. GitHub Actions does not directly invoke the ChatGPT-installed connector catalog. External connector bridges still require their own API or gateway boundary. `orchestrator/connector_bridge.py` now defines that vendor-neutral boundary as protocol v1.
2. `.orchestrator/` is repository-backed persistence. Public-repository state is not suitable for secrets or private workflow payloads.
3. Side-effect recovery is fail-closed. Connector bridge executions can be reconciled when the provider advertises a safe read-only reconciliation endpoint; other opaque side effects still require external-state inspection before resume.
4. Connector bridge v1 is vendor-neutral; a real Notion/Figma/Canva/ClickUp/etc. bridge service must implement the protocol and its own vendor OAuth/API policy.
5. Semantic validation now has deterministic output contracts plus an optional Gemini validation worker; domain-specific validators for deployments, published artifacts, and vendor objects remain to be added.
6. Connector action contracts are schema-aware but intentionally bounded to required fields, primitive types, and an idempotency declaration; vendor-specific OAuth semantics and richer JSON Schema are still outside the core.
7. There is no dedicated distributed database or event bus; GitHub Actions + committed state is intentionally the zero-new-service implementation. The v19 pre-side-effect barrier reduces the runner-crash window but does not make Git state and an external provider transactionally atomic.

## Deployment targets

The connector bridge runtime can be hosted as a Vercel Python Function (`api/bridge.py`) or as a Render Web Service (`bridge_server.py`). Both expose the same protocol runtime and require `ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET` plus `ORCHESTRATOR_CONNECTOR_ROUTES`. Deployment is not considered verified until the public `/health` endpoint responds successfully.

## Approval intent binding v23

- High-risk approval issues carry the SHA-256 fingerprint of the exact node definition requested for approval.
- A later approved label is accepted only when the current node fingerprint matches that stored approval fingerprint.
- Stale or missing approval fingerprints are fail-closed: the old approval is cleared, the old issue reference is discarded, and the node returns to `ready` so a fresh approval issue is created.
- The approving GitHub actor and approval timestamp are persisted as audit metadata.
- `approval_fingerprint` is excluded from the plan fingerprint as runtime approval metadata; changing the actual tool/action/payload still changes the plan fingerprint and fails the existing plan-integrity check.

## Orchestration hardening v30

- `AI Orchestrator` concurrency is scoped by workflow identity when available, so independent workflows no longer serialize behind one global lock.
- `repository_dispatch` continuation runs for the same `workflow_id` share the same concurrency group; issue-triggered runs are grouped by issue number; scheduled recovery uses a dedicated global recovery group.
- The concurrency design uses GitHub Actions scheduler-level mutual exclusion rather than a second lease database, preserving the zero-dollar architecture.

## Orchestration hardening v40 (review branch)
- Scheduled recovery no longer executes workflows directly from the global schedule concurrency group; it dispatches the exact persisted workflow ID so the normal workflow-scoped concurrency gate owns execution.
- Running workflows are dispatched by scheduled recovery only after a 5-minute staleness grace window; waiting-approval, active federation, and recoverable uncertain/barrier-failed workflows remain eligible.
- GitHub Actions run identity now persists both github_run_id and github_run_attempt, with origin_github_run_id/origin_github_run_attempt preserving first-worker provenance.
- Continuation matching requires the exact (workflow_run.id, workflow_run.run_attempt) pair. Continuation concurrency is scoped by that same pair and keeps the default single pending slot because duplicate resumes target the same workflow.
- Federation and supervisor workflows retain queue: max because GitHub documents the feature and it is needed to prevent slot starvation. The pinned actionlint v1.7.12 does not recognize the field, so a narrow path-specific ignore is maintained for only those workflows; all other workflow linting remains active.
- Changes add no service, broker, database, or paid dependency; the zero-dollar GitHub Actions execution model remains intact.

## Orchestration hardening v39
- Added `orchestrator/federation_scheduler.py`: deterministic federation slots plus per-workflow batch/task quotas for free-first backpressure.
- Federation uses 3 fixed concurrency slots, each with matrix `max-parallel: 4`, limiting federation worker fan-out to at most 12 agent jobs at once while leaving runner headroom for supervisor/CI jobs.
- Default per-workflow federation quota is 4 batches / 16 tasks, with hard caps of 8 batches / 32 tasks. This quota is separate from the existing logical attempt budget and protects hosted-runner consumption.
- Quota reservation occurs only after task/manifest validation; dispatch failure refunds both attempt and federation reservations.
- Exhausted federation quota causes deterministic fallback to the existing local executor rather than failing the workflow, preserving full usability when federation capacity is unavailable.
- Federation slot selection is deterministic from federation ID and is passed in the repository-dispatch envelope. Different federation IDs may run concurrently across different fixed slots; identical slot IDs serialize at the workflow level.
- Workflow schema is now v6 with persisted federation quota configuration/usage fields.
- Same-tool health shards remain safe at Git persistence boundaries; the next deeper health optimization can move from per-tool last-state overwrite toward per-workflow observations if contention becomes measurable.
- No database, broker, paid queue, or hosted scheduler was introduced.

## Orchestration hardening v38
- Tool health persistence is now sharded per tool under `.orchestrator/tool_health/<sha256(tool)>.json`; the legacy monolithic `tool_health.json` remains readable for migration/backward compatibility.
- Updating one tool's health only reads that tool's shard plus any legacy record for the same tool, reducing unrelated-workflow contention while preserving deterministic routing through the aggregated health view.
- Active federations in `waiting_agents` are now scheduler-recoverable. Recovery queries GitHub Actions artifacts by exact federation result name and ingests a completed artifact without re-executing worker tasks.
- `run_one_step` fails closed on an active `waiting_agents` federation and never treats delegated nodes as ordinary runnable nodes.
- Federation artifact ingestion now uses one downloaded immutable aggregate snapshot rather than downloading the same artifact twice.
- No database, broker, paid queue, or new hosted service was introduced.

## Orchestration hardening v37
- Added `orchestrator/agent_worker.py` and `orchestrator/aggregate_agent_results.py` for isolated safe-agent execution and supervisor-side result aggregation.
- Added `.github/workflows/orchestrator-agent-federation.yml`: manifest validation → bounded matrix fan-out (`max-parallel: 4`, `fail-fast: false`) → one immutable result artifact per task → aggregation → completion dispatch.
- Federated tasks use protocol v2 and are bound to the exact planned tool, role, capability, attempt, workflow ID, agent ID, and input digest. Results are validated against that identity before adoption.
- Federation is opt-in via `ORCHESTRATOR_FEDERATION_ENABLED`; normal operation remains the existing in-process executor, so local/offline/dry-run operation does not depend on GitHub Actions federation.
- Supervisor remains the sole writer to workflow state and the only path permitted to perform side effects. Federated workers are limited to low/medium-risk, parallel-safe, non-side-effecting work.
- Artifact transport uses one-day retention and bounded envelopes. Artifact data is transport/mailbox state, not authoritative workflow state.
- Workflow schema is now v5 with a persisted `federation` execution record.
- Federation delegation reserves attempts only after successful task/manifest validation and refunds them when dispatch fails.
- GitHub Actions matrix output is a JSON matrix object; matrix outputs are not treated as a multi-agent mailbox because matrix/reusable-workflow aggregation semantics are lossy for multiple workers.

## Orchestration hardening v36
- Added `orchestrator/agent_protocol.py`, a bounded protocol for federated workers using immutable task/result envelopes, stable agent identity, input/output SHA-256 digests, protocol versioning, and strict size limits.
- Federated tasks are restricted to low/medium, parallel-safe, non-side-effect agent roles. The supervisor remains the authoritative state and side-effect writer.
- Artifacts are treated as transport/mailbox only; their contents must be validated against the original task identity and input digest before adoption.
- The protocol is compatible with GitHub Actions dynamic matrix fan-out while avoiding matrix-job output aggregation semantics as a source of truth.
- No database, broker, external queue, hosted agent service, or paid dependency was introduced.

## Orchestration hardening v35
- Added orchestrator/agent_fabric.py, a zero-dollar typed agent registry/protocol with stable agent identities, role/capability validation, risk ceilings, team manifests, handoff edges, and collaboration pattern classification.
- Node now carries agent_role; workflow schema is v4 so agent-role assignment is persisted and included in plan integrity.
- Deterministic fallback plans now implement real scatter-gather/parallel-deliberation shapes for research, content, and software tasks instead of forcing every worker into a single linear chain.
- LLM planner output accepts and validates agent_role and is instructed to prefer independent specialist lanes only when they add distinct value.
- Gemini worker prompts are role-specific and include the stable derived agent identity.
- Durable agent.started / agent.completed events make the multi-agent lifecycle auditable.
- No new database, broker, hosted agent runtime, or paid service was introduced.

## Orchestration hardening v34

- Workflow-scoped audit events are now sharded into `.orchestrator/events/<sha256(workflow_id)>.jsonl`; the legacy global `events.jsonl` remains only for events without a workflow identity.
- Sharding reduces cross-workflow Git write contention introduced by v30 while preserving synchronous flush/fsync durability.
- Event shard filenames are derived from a SHA-256 of the workflow ID, preventing path traversal even for externally supplied identifiers.
- The design deliberately does not make the event path asynchronous: side-effect safety still requires durable state before external execution.

## Orchestration hardening v33

- Normalized the committed empty `.orchestrator/state.json` baseline from legacy state version 3 to current state version 4.
- CI now watches `.orchestrator/state.json` and asserts its committed version matches `state_schema.CURRENT_STATE_VERSION`, preventing silent serialization-version drift.
- This does not add a second migration system; the runtime migration remains fail-closed for future versions and idempotent for supported legacy versions.

## Orchestration hardening v32

- Workflow IDs now use UUID4 rather than millisecond timestamps. This removes a collision window introduced by v30's independent workflow concurrency.
- UUID identity is intentionally non-deterministic; replay-sensitive retry inputs continue to derive from the persisted workflow ID and state.
- Added a 256-ID uniqueness regression test.

## Orchestration hardening v31

Audit cross-check outcomes:
- The reported execution-ledger thread race was not confirmed: the thread pool mutates distinct `Node` objects and does not write `workflow["executions"]`; side effects are explicitly serialized. DAG validation also rejects duplicate node IDs, preventing duplicate execution keys within a valid plan.
- `max_attempts` is already centralized in `state_schema.py`; `orchestrator.py` retains compatibility aliases rather than an independent configuration source.
- State migration intentionally fails closed on unsupported future versions rather than performing an unsafe downgrade. Migration is deterministic and now covered by idempotence/future-version tests.
- A real cross-trigger race was identified: approval issue runs used the approval-issue number while continuation runs used workflow ID. Approval handling is now isolated in `.github/workflows/orchestrator-approval.yml`, which validates the actor and dispatches the exact `workflow_id`; the main worker therefore uses one canonical concurrency key for all resumptions.
- The durability barrier still uses a Git commit/push as the hard pre-side-effect fence; it was not replaced by an asynchronous WAL because an async-only local log would weaken crash durability. The remote check is optimized from full `git fetch` to `git ls-remote`, while the final `--force-with-lease` remains the CAS fence.
- Per-node state persistence remains intentionally synchronous for side effects because `started` must reach durable storage before an external effect. Safe parallel nodes are persisted as a batch after execution, avoiding per-thread state writes.

## Orchestration hardening v29

- The `Persist state` workflow step now treats the final state commit as an optimistic-concurrency update: push to `main`, fetch/rebase if `main` advanced, retry at most three times, and fail closed on merge conflicts.
- This recovery is intentionally limited to Git-backed state persistence; it never force-pushes or overwrites a concurrent `main` history.
- The state commit path remains zero-dollar and requires no external database or queue.

## Orchestration hardening v28

- Audit events are bounded to 16 KiB payloads and are written with flush/fsync before the worker proceeds.
- Execution ledger completion records store output SHA-256 and evidence SHA-256 rather than duplicating full node output inside `executions`, reducing persisted state redundancy.
- New workflows are created at workflow schema v3 and new empty state initializes at state version v4.
- Persisted retry seed, attempt counters, and runtime metadata remain excluded from the plan fingerprint where appropriate.

## Orchestration hardening v27

- Retry decisions are centralized in `failure_policy.py`; connector bridges report uncertainty and discovered action idempotency but do not decide whether to retry.
- Every workflow has a bounded `max_attempts` budget (default 64, hard cap 128) shared across nodes and retries/replans. `attempts_used` is persisted with workflow state.
- Retry jitter uses a persisted per-workflow seed. Delays remain reproducible after resume while using a substantially wider jitter window to reduce synchronized retry bursts.
- A confirmed applied connector reconciliation follows `reconciling -> validating -> completed`; if validation fails, the execution ledger remains completed/applied so the external side effect cannot be replayed.
- The side-effect START barrier uses an explicit Git `--force-with-lease` against the observed `refs/heads/main` SHA. A concurrent ref writer makes the barrier fail before external execution.

## Secure structured live boundary v26

- Structured ingress defaults to `dry-run` when no mode is supplied.
- Structured `live` ingress is rejected until a private input transport exists because the public repository dispatch path intentionally carries only bounded execution metadata and digests.
- This prevents the engine from executing an underspecified live operation after the raw input has been withheld from public repository state.

## Execution envelope v25

- `gateway.py` accepts the structured cross-service envelope fields used by the automation core: event id, execution id, workflow id, domain, operation, intent fingerprint, attempt, requested mode, and input digest.
- The gateway validates the supplied intent fingerprint against the structured input before dispatch and rejects malformed execution identities or attempts outside the bounded range.
- Raw structured input is not copied into the public repository-backed metadata; only bounded identity, fingerprint, digest, and mode fields cross the GitHub `repository_dispatch` boundary.
- The orchestrator persists the external execution identity in workflow state and returns it in terminal callback receipts.
- Terminal callbacks are HTTPS-only and HMAC authenticated; repeated terminal callbacks are handled idempotently by the automation-core ledger.
- The engine workflow preserves the requested live/dry-run mode across `repository_dispatch` instead of falling back to local workflow-input defaults.
- Canonical contract: `contracts/execution-envelope.schema.json`.

## Durability barrier recovery v22

- A `barrier_failed` execution is a pre-side-effect failure: the external effect was not invoked by the durability barrier.
- On resume, the failed side-effecting node is safely rearmed to `ready` and its execution ledger returns to `prepared`.
- The worker returns after rearming so the next continuation/scheduled worker checks out the current `main` before attempting the barrier again.
- A remotely accepted barrier commit that was only observed as a client-side push error remains safe: a fresh worker starts from remote `main`, where the durable execution record wins over any stale local state.

## Interrupted side-effect recovery v21

- On worker resume, a live side-effecting node with status running and a durable execution record status prepared is safely returned to ready because the pre-side-effect durability barrier has not been crossed.
- A live side-effecting node with status running and a durable execution record status started is treated as an interrupted in-doubt execution.
- The worker converts that node to failed with execution_uncertain and reconciliation_required metadata before replay is considered.
- connector_bridge nodes enter the existing applied/not_applied/unknown reconciliation path; opaque side effects remain fail-closed without automatic replay.
- This closes the crash case where v19 had durably fenced the side effect but the worker died before persisting a terminal node state.

## Post-start side-effect replay fence v20

- After a live side-effecting node has crossed the durable START barrier, non-idempotent execution failures cannot be automatically retried.
- Started side effects also cannot be automatically replanned to another tool/provider, because the original external operation may already have applied.
- The only automatic retry exception is a connector_bridge ConnectorRequestError that is explicitly uncertain and whose freshly discovered action contract declares idempotent: true; the same deterministic request id is reused.
- Transport-like failures after a started side effect are recorded as uncertain and remain fail-closed. The existing durable started execution record blocks replay on a later worker resume.
- Output/semantic validation failures after an external call also block retry/replan, preventing a successfully applied side effect from being duplicated merely because its returned envelope could not be validated.

## Pre-side-effect durability v19

- In GitHub Actions, a live side-effecting node must have its execution `START` record committed and pushed to `main` before the external effect is invoked.
- The barrier fetches `origin/main` and refuses to proceed when the checked-out commit is stale or the state change cannot be staged and pushed.
- A barrier failure blocks the side effect and records a bounded dependency failure locally.
- The barrier provides an at-most-once attempt fence across runner interruption: if the process dies after the barrier but before completion persistence, the next worker observes the durable `started` record and fails closed instead of blindly replaying.
- This is not a distributed transaction: Git commit/push and the external provider effect remain separate systems.

## Deployment supply-chain hardening v18

- The connector bridge deployment workflow uses Node.js 24 for the CLI execution environment instead of Node.js 20.
- Vercel CLI invocations are pinned to the exact audited package version `59.19.1` instead of mutable `latest`.
- Deployment configuration tests fail if the mutable Vercel CLI tag or the old Node.js 20 CLI runtime reappears.
- This change does not alter the Python function runtime declared in `vercel.json`.

## Actions runtime maintenance v17

- GitHub Actions core dependencies are pinned to immutable commit SHAs for the current Node 24-based releases: checkout v6, setup-python v7, and setup-node v7.
- Workflow configuration tests enforce the expected action references so future tag drift is detected by CI.
- The deployment workflow uses Node.js 24 for the deployment CLI environment; this does not alter the Python function runtime declared in `vercel.json`.

## Reconciliation rearm v16

- A confirmed `not_applied` reconciliation outcome now settles the prior execution-ledger record as `not_applied` before the node is returned to `ready`.
- A subsequent side-effect attempt is therefore allowed only after an authoritative reconciliation says the prior attempt did not apply.
- An `unknown` outcome still leaves the workflow fail-closed and does not rearm the effect.

## State durability v15

- Persisted workflow/state JSON is written to a same-directory temporary file, flushed and fsynced, then atomically replaced; temporary files are cleaned after success or failure.
- Workflow schema versions newer than the supported version are rejected before migration so a newer writer cannot be silently interpreted by an older control plane.
- Atomic persistence protects the state and checkpoint paths used by the control plane from partial JSON writes caused by process interruption.

## Durable state v14

- Persisted state is explicitly migrated to the supported state schema on load/save; future unsupported versions fail closed instead of being guessed at.
- Completed-node checkpoints with a declared SHA-256 digest are verified before resumed execution. Missing checkpoint digests remain `legacy_unverified` for backward compatibility.
- Checkpoint verification binds the stored file to the exact workflow/node and rejects path traversal or checksum drift before downstream execution.

## Plan integrity v13

- Every executable workflow carries a deterministic plan fingerprint derived from node id, capability, tool, dependencies, risk, contract, and non-volatile input.
- Runtime state such as output, retry count, approval issue, context, and repair feedback is excluded from the fingerprint.
- On resume/execution, a fingerprint mismatch fails closed before any node executes and records the expected/actual digests as a plan-drift event.
- Replanning intentionally refreshes the fingerprint after the new tool choice is committed, preserving the same safeguard for later resumes.

## Reliability policy v12

- Failures are normalized into `transient`, `intermittent`, `dependency`, `contract`, `semantic`, `policy`, `uncertain`, or `permanent` classes before retry decisions are made.
- Retry eligibility is bounded by node retry budgets; uncertain connector failures continue to use connector-specific idempotency/reconciliation policy.
- Retry backoff includes deterministic hash-derived jitter rather than wall-clock randomness, reducing synchronized retry bursts while remaining reproducible.
- Connector discovery rejects oversized inventories and caps connector/action/capability counts before data reaches planning or persisted evidence.

## Execution fabric v11

- Reconciliation responses are sanitized at the orchestrator boundary; arbitrary upstream response bodies are never copied into persisted workflow reconciliation state.
- Persisted reconciliation records contain only protocol/request identifiers, connector/action, discovery metadata, explicit state, and check time.

## Execution fabric v10

- Connector bridges may advertise a read-only reconciliation capability via a sanitized `reconciliation` flag and route-level `reconciliation_url`.
- Uncertain connector failures can be reconciled before resume; `applied` completes the node without replaying the side effect, `not_applied` returns the node to `ready`, and `unknown` remains fail-closed.
- Reconciliation requests are signed and carry the deterministic request id as their idempotency key; the upstream reconciliation endpoint is HTTPS-only and must return one of the three explicit states.
- Scheduled resume handles failed uncertain connector workflows through the reconciliation path before any retry.

## Execution fabric v8

- Connector transport failures and HTTP 5xx responses are represented as potentially uncertain request failures.
- The discovered action's `idempotent` declaration controls automatic retry of an uncertain connector request.
- Uncertain non-idempotent connector failures fail closed and are not automatically replanned to another provider.
- An uncertain failure is persisted as requiring reconciliation; live connector recovery can query the bridge for `applied`, `not_applied`, or `unknown` before continuing.

## Execution fabric v7

- Connector discovery now includes sanitized action contracts: required fields, primitive input types, and an idempotency declaration.
- Live planning and live execution validate connector payloads against the discovered action contract before the upstream POST.
- The bridge runtime enforces the same action contract server-side for defense in depth.
- Action schemas are optional for backward compatibility; missing schemas do not fabricate constraints.
- Secret values and unknown/free-form route fields are not propagated into the planner inventory.

## Execution fabric v6

- The live LLM planner may consume sanitized connector capability discovery and must declare `connector`, `action`, and `payload` for connector-bridge nodes.
- Live connector execution performs a fresh capability preflight before upstream invocation. Unknown, unconfigured, or unadvertised actions fail closed before the side effect request.
- Discovery responses are normalized, bounded, briefly cached, and copied into successful connector evidence for auditability.
- Planner discovery is optional and fail-closed: an unavailable bridge inventory does not permit fabricated connector actions in live plans.

## Execution fabric v5

- `orchestrator/capability_graph.py` is the central capability router. Candidate tools are selected using live availability, free-only policy, credential requirements, health state, risk, and deterministic lexical tie-breaking.
- Tool health is persisted as non-secret state in `.orchestrator/tool_health.json`.
- Read-only tools enter a degraded state after one terminal failure and an excluded state after the second; they may become probe candidates after cooldown.
- Side-effecting tool failures enter the excluded state immediately with a longer cooldown and remain subject to existing approval and fail-closed semantics.
- Replanning excludes the failed tool and routes through the same capability graph.
- Connector bridges expose sanitized capability discovery without returning route secrets.

## Evidence and repair layer

- Completed nodes produce deterministic SHA-256 evidence records stored in workflow state and checkpoints.
- Node contracts can require output fields or minimum research source counts.
- Validation nodes evaluate dependency evidence rather than their own validator envelope.
- Failed nodes preserve bounded error/output/next-action feedback for fallback execution.
- Evidence is advisory for orchestration state and is never treated as a substitute for explicit approval of side effects.
- The artifact verifier supports HTTPS URLs, GitHub repository files, and local checked-out files without side effects.

## Orchestration v3 execution policy

- Independent low-risk/non-side-effect nodes may execute concurrently, bounded to 1–8 workers and defaulting to 4.
- Side-effecting and high/critical-risk nodes remain serialized and approval-gated in live mode.
- Dependency outputs are passed forward as bounded context; oversized outputs are truncated.
- Successful nodes receive contract validation and a checkpoint with SHA-256 evidence.
- Replanning returns a failed node to the runnable queue and is bounded by MAX_REPLANS.
- Gateway idempotency keys are propagated as event_id so duplicate ingress events can be suppressed at workflow creation.
- Dry-run remains credential-free; local validation can still execute in dry-run while external adapters are simulated.

## Protocol for future changes

Before changing runtime behavior:
1. Read this file.
2. Inspect current `main`.
3. Identify the exact invariant being changed.
4. Add a regression test before or with the change.
5. Run CI.
6. Merge only after green.
7. Update this memory file when architecture, policy, or a known limitation changes.

## Design principle

The system should fail closed on unsafe tool selection and unknown side-effect outcomes, fail open on optional observability, remain deterministic under duplicate events, and preserve explicit human approval for irreversible external effects.

## Orchestration hardening v63 — general blueprint workload compiler
- Added `orchestrator/blueprint_compiler.py` as a deterministic, side-effect-free workload compiler for structured blueprints and bounded Markdown/text documents.
- The compiler preserves provenance, validates requirement dependencies, rejects cycles, and emits bounded execution units with requirement IDs, workstreams, risk, agent-role/capability hints, artifacts, acceptance criteria, source references, and unit dependencies.
- Resource bounds are explicit: 256 KiB source/structured input, 512 requirements, 64 units, 8 requirements per unit, 24 KiB per unit/packet, and 480 KiB aggregate compilation manifest.
- Plain Markdown/text ingestion is provenance-only and never invents semantic requirements; semantic extraction remains an explicit research/analysis operation.
- `blueprint_compiler` is registered as credential-free and free-tier. Execution goes through the existing orchestrator supervisor and therefore retains current policy, checkpoint, retry/replan, federation, idempotency, and approval behavior.
- File ingestion is restricted to `ORCHESTRATOR_WORKLOAD_ROOT`; path escape is rejected.
- No paid service or runtime dependency was introduced.

## Orchestration hardening v64 — deep general workload execution
- Added deterministic global context packing so high-fan-in DAGs cannot accidentally exceed the 48 KiB orchestration context envelope; dependency evidence and digests remain available when full payload bodies are omitted.
- Blueprint compiler now emits dependency-safe execution waves capped at 24 units, matching the current DAG node cap. This provides a generic slicing boundary for workloads larger than one runtime DAG.
- Workflow state now carries a generic workload envelope for blueprint/version/digest lineage, compilation manifest identity, unit/wave counts, and optional completed unit/wave progress.
- Added regression tests for bounded dependency context and workload-state validation.
- No external side effects or new paid/runtime dependency were added. Existing supervisor policy remains authoritative.
- Main remains the v66 production baseline; the current branch contains v67 boundary hardening plus the adaptive research/epistemic upgrades below.

## Orchestration hardening v68 — adaptive free-first research and epistemic measurement
- Research retrieval now has deterministic fast, balanced, and deep budgets with explicit caps on extended providers and result counts.
- In ORCHESTRATOR_FREE_ONLY=true mode, research defaults to public unauthenticated providers (Semantic Scholar and Europe PMC). Optional metered/credentialed providers such as OpenAlex and CORE are excluded from the default free-only path.
- Extended research stops early when canonical independent evidence already meets the selected target, and the extended fabric is allowed to rescue a complete legacy-provider outage instead of failing before the second stage.
- Research provider calls within an enabled stage execute concurrently, while normalization and final output ordering remain deterministic.
- Canonical evidence counts now use independence_key semantics rather than raw provider-bucket counts.
- Research lanes are no longer query-identical: the skeptic lane is deterministically marked as counterevidence and searches for contradictions, limitations, and alternative findings.
- Epistemic validation now rejects duplicate/malformed evidence identities and duplicate/blank claim identities, while min_coverage is based on evidence linkage rather than forcing contested claims to look supported.
- Workflow state records deterministic epistemic telemetry for material claims, supported/contested/unknown status, evidence linkage, research evidence counts, and independent-source maxima. These are measurements, not truth scores.
- The offline evaluation harness covers epistemic boundary and conditional deliberation behavior in addition to existing control-plane cases.
- Worker/Node validation is path-scoped into a separate CI workflow so ordinary orchestrator changes do not invoke Wrangler/Node checks unnecessarily.
- No paid runtime, database, broker, queue, or Python/Node package dependency was added to the orchestrator runtime.
- Known limitations remain explicit: the deliberation module is still a policy primitive rather than a full automatic second-round federation spawn; empirical epistemic quality still needs labeled ground-truth datasets; open-access route resolution is still metadata-only; and run_one_step/run_workflow retain some execution-path duplication that should be refactored only with additional recovery regression coverage.


## v68.1 free-first retrieval and quota hardening (2026-10-03)

- Dry-run provider adapters now defer provider-specific output validation while retaining deterministic contract checks; this prevents simulation from being mistaken for live evidence retrieval.
- Added bounded local research-response cache under `.orchestrator/research-cache`: SHA-256 keyed by provider/query/result limit, 24-hour default TTL, 128-entry cap, atomic writes, and opt-out via `ORCHESTRATOR_RESEARCH_CACHE=false`. Cache failures never block live retrieval.
- Quota-sensitive adapters (`gemini`, `openai`, `research_bundle`) are serialized in the supervisor batch scheduler to avoid free-tier burst amplification. Independent non-quota-sensitive nodes remain parallelizable.
- Federated worker matrix default is now `max-parallel: 1` because worker LLM calls execute on separate runners and share upstream quotas. Federation remains opt-in and task-count bounded.
- Semantic Scholar documents that its public API is rate-limited and may be further throttled; its documentation recommends keys/bulk endpoints for heavier use. The local cache therefore reduces repeated requests but does not imply unlimited provider access. The project must treat “$0” as a deployment/configuration target, not as an upstream guarantee.


## v68.2 control-plane boundary hardening (2026-10-04)
- Direct GitHub side effects now have deterministic reconciliation semantics where GitHub exposes enough observable state: issue creation uses a durable execution marker in the issue body; file create/update proves an already-applied effect by exact content equality; file deletion proves an already-applied effect by a 404.
- GitHub workflow dispatch remains explicitly non-reconcilable in the generic adapter because the dispatch API does not provide a durable caller idempotency key or returned run identifier. The runtime therefore fails closed instead of guessing.
- Bridge runtime now supports an optional stdlib SQLite idempotency ledger through ORCHESTRATOR_BRIDGE_IDEMPOTENCY_DB. Completed responses survive process restart and multiple bridge processes sharing the same database file. In-flight entries are retained on owner failure so an uncertain upstream side effect cannot become an automatic replay.
- Durable bridge idempotency is deployment-storage dependent: a shared persistent SQLite file improves process/multiprocess safety on one host; it is not a substitute for a distributed database or fencing-aware external provider.
- Added regression coverage for SQLite-backed replay and GitHub-effect reconciliation boundaries.
- Kept all runtime dependencies stdlib-only and did not introduce a paid service.


## v69 distributed control-plane boundary (2026-10-04)
- Live one-step execution can use an optional HTTPS control plane configured through GitHub vars/secrets.
- Workflow leases are per workflow ID and use a monotonically increasing fence_epoch. Side effects use stable effect IDs plus semantic intent digests.
- Live side-effect order is local prepared/started persistence -> durability barrier -> distributed claim -> external execution -> validation -> distributed completion -> local completion.
- Existing inflight/completed claims are never auto-replayed; ambiguous outcomes remain fail-closed and require reconciliation.
- Reconciliation can create a completed control-plane record when a crash happened after the local barrier but before the first distributed claim. A proven not_applied outcome clears the claim before replay.
- Runtime metadata is excluded from semantic effect identity, while private-input digests/fingerprints remain identity-bearing.
- Control-plane failures do not silently fall back to Git-only live side effects; they persist as running/control_plane_blocked so stale-running recovery can retry later.
- Cloudflare Durable Objects are coordination authority only; Git remains source/audit persistence. Cloudflare Queues are intentionally outside the mandatory control path.
