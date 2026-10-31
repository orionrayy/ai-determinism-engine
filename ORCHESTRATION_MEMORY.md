# ORCHESTRATION MEMORY — Canonical Control-Plane Context

## Purpose

This file is the durable, version-controlled memory of the AI orchestration control plane. Before modifying the orchestrator, treat this document and the current source tree as the source of truth. Do not assume older conversation state is still accurate without checking the repository.

## Current baseline

Repository: `orionrayy/ai-determinism-engine`
Primary branch: `main`
Current main baseline: v79.x correctness/evidence hardening + v80 portable enterprise-runtime/control-plane foundation + integrated Telegram control-plane frontend and deployment stack. Main uses deterministic DAG orchestration, bounded parallelism, durable state/recovery, effect contracts/fencing, resource locks, supervised multi-agent coordination, evidence/epistemic gates, workload compilation, GitHub Actions federation, and optional Cloudflare Durable Object coordination authority for distributed live workflows. Security/cost lineage includes v43 connector upstream cost gate, v42 free Gemini model gate, v41 private structured input boundary, and v44 — reconciliation cost closure; v65 deterministic integrity/liveness hardening remains part of the baseline.
Observed main head before this remediation: `c530d9ac542b82c2aefd59ccc65df660e3b55795` (2026-10-06).
The v78/v79 work recorded below is historical implementation context; it is no longer a pending candidate merge on main.
Execution model: GitHub Actions + stdlib Python, with optional Cloudflare Durable Objects and the Telegram Worker control frontend.
Cost policy: free-first; `ORCHESTRATOR_FREE_ONLY=true` in the production workflow.
Current execution-fabric branch: `main`

### Candidate v79.1 efficiency and evidence hardening
- Distributed provider rate gate is implemented for Crossref and uses the public 1 req/s profile without `CROSSREF_MAILTO`, or polite 3 req/s profile when mailto is configured. Local semaphores remain the concurrency backstop.
- Research bundle is no longer globally classified as quota-sensitive; provider-level throttling now controls upstream research, allowing independent research nodes to run in parallel.
- Workflow state CAS and recovery alarm mutation can be sent as one fenced control-plane transaction, reducing one RPC per distributed state persistence and keeping recovery state atomic with the state version.
- Trusted evidence context is explicitly bounded; canonical IDs remain available while abstract excerpts are included only in a bounded lane for exact passage validation.
- Exact normalized evidence-passage validation is available through `evidence_passages`; strict mode can require passages for direct-supported claims.
- OpenAlex remains a free-allowance provider in hard-free mode, and Unpaywall is DOI-only and opt-in.
- CI note: run `37206981912` on the PR merge commit compiled the runtime, passed worker syntax, import smoke, and Actionlint, then failed only because the new `test_v79_boundary_quality` module used a package-qualified import incompatible with the repository's `unittest discover -s orchestrator` layout. Commit `84d7c3a72c312211e0044833ea80bc3984d9bf20` fixes that import; a fresh CI run is required before claiming the branch fully green.

### Candidate v79.2 boundary/rate hardening
- Strict research epistemic contracts now require `evidence_passages` for every material `SUPPORTED_DIRECT` claim.
- Passage references must belong to the claim's own `evidence_refs`; strict mode fails closed when the referenced trusted record has no verifiable abstract/full-text corpus.
- Trusted evidence context now preserves `access_level`, `access_route`, `access_verification`, and `retraction_signal`, preventing adjudication from treating omitted access metadata as verified access.
- Crossref local fallback rate now matches the selected pool: public 1 req/s without `CROSSREF_MAILTO`, polite 3 req/s with it. This remains a fallback only; distributed gating is preferred when configured.
- The 600 KiB distributed control-plane workflow-state ceiling remains intentionally unsolved by lossy remote projection: remote authority is not allowed to discard dependency outputs required for deterministic resume. Future segmentation must preserve a verifiable Git replica binding before implementation.
- Regression coverage was extended for strict passages, corpus verification, trusted access metadata, research passage contracts, and Crossref fallback rates.
### Candidate v79.3 evidence scoring/packing hardening
- Evidence utility ranking now gives full access weight only to explicitly `verified` access; `declared_only` and `metadata_only` remain non-verified.
- Bounded trusted context retains the highest-utility corpus-bearing evidence records for passage excerpts instead of relying only on canonical-ID order.
- When a high-utility record has no abstract, a bounded 700-character `text` excerpt may be taken from available full-text/text content for strict passage validation.
- Evidence passage validation is bounded to 8 passages per claim and 700 characters per passage.
- Latest code commits include `3759a953f06e29ba1115f2bb77dbf55f93c1928` for the corpus-lane correction and subsequent test/consistency hardening; fresh CI must validate the current branch tip before any merge decision.
### Candidate orchestration hardening v79
- PR #142 branch: `feat/v79-boundary-evidence-efficiency`.
- Distributed control-plane v78 remains operational authority for live workflows; Git remains the state/audit replica.
- Evidence boundary is supervisor-trusted: agent-supplied evidence metadata cannot establish authority before canonical binding.
- Claim-level deliberation now requires challenge-scoped evidence references and explicit evidence-delta semantics.
- Truth-lock preserves claim lineage and conservatively rejects material numeric/negation drift when source statements are available.
- Evidence records normalize Europe PMC full-text routes and cross-provider retraction signals; source authority is represented as an explainable heuristic vector/profile.
- Hard free mode includes OpenAlex but never sends `OPENALEX_API_KEY`; Crossref optionally uses `CROSSREF_MAILTO`; Unpaywall is DOI-only and opt-in through `UNPAYWALL_EMAIL`.
- Gemini planning/execution uses bounded `thinkingConfig.thinkingLevel`; free-model policy is registry-pinned and time-aware through `free_until`.
- Evidence cards are ranked by authority, independence, access, integrity, and recency utility.
- Minority candidates with materially stronger trusted evidence trigger adjudication escalation instead of majority acceptance.
- Trace identity is generated before durable payload truncation so correlation survives bounded event storage.
- v79 adds regression coverage for trust binding, evidence access, retraction safety, deliberation deltas, minority escalation, provider policy, and free-tier expiry.


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
## v78 checkpoint — recovery authority, distributed state, evidence, and truth-lock
- fix/v78-recovery-authority-consolidation is based directly on the v77 head and is kept separate from main until explicit merge approval.
- Scheduled recovery imports are now tested from a fresh checkout; the prior orchestrator.private_input / capability_graph import failure is covered by CI smoke checks.
- Distributed workflows treat the Durable Object control plane as coordination/state authority while Git remains a durable/discovery replica.
- Per-workflow recovery alarms, atomic recovery claims, terminal alarm clearing, and authoritative recovery-state refresh are implemented. GitHub cron remains a consumer/backstop rather than the source of recovery truth.
- Research provider consolidation removes duplicate Crossref requests; strict ORCHESTRATOR_FREE_ONLY=true excludes metered OpenAlex from the zero-cost provider set.
- Evidence records expose access verification and post-publication retraction signals; retracted works are excluded from positive independence counts; DOI/arXiv bibliographic crosswalk prevents provider duplication from inflating independence.
- Research drafts use deterministic truth-locking and cannot silently upgrade uncertainty or introduce material claims without adjudicator lineage.
- Consensus policy now rejects single-proposal consensus; provider concurrency is bounded for free-tier stability; epistemic outputs have deterministic claim/evidence size caps.
- Validation checkpoint: GitHub Actions run 1904 passed all gates on commit 9bccd7964537c8869cd9c94725283c5786f0f865.

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

- When the distributed control plane is active, its durable effect claim replaces the Git side-effect durability barrier; Git barrier remains used for Git-only live execution.


## v70 resource-scoped locks and hot state (2026-10-04)
- Nodes support explicit resource keys through Node.resources or input.resource_keys; keys are normalized, deduplicated, sorted, and bounded.
- Resource locks use separate Durable Object identities per resource key and bind ownership to both worker owner and workflow ID.
- Locks are acquired before side-effect preparation/effect claims. Contention refunds both attempt and LLM admission budget, returns the node to ready, and exits the supervisor with a running/resource-waiting state.
- Resource-bearing nodes are excluded from the parallel-safe batch; resource leases renew before retry attempts and release in deterministic reverse order.
- Active live control-plane sessions persist workflow state through a monotonically increasing state_version CAS in Durable Object storage. Targeted load_workflow reads remote hot state first when configured.
- Global recovery hydrates the remote workflow snapshot before executing a Git-discovered candidate, preventing stale Git snapshots from overwriting newer control-plane state.
- Selected lifecycle/control events use a bounded Durable Object outbox (latest 256 events per workflow object); high-volume diagnostics remain local.
- Git remains eventual source/audit snapshot storage; active distributed live execution does not require Git round-trips for each hot-state mutation.


## v71 effect contracts (2026-10-04)
- Side-effect adapters declare action-aware effect contracts in orchestrator/tools.json.
- Contracts specify effect identity, retry mode, reconciliation mode, and fencing model.
- Live execution rejects a side-effect action with no effect identity contract.
- Reconcile-based retry is permitted only when the contract declares an actual reconciliation mechanism.
- Generic webhook effects are intentionally opaque/non-retryable; GitHub create/update/delete effects are deterministic-reconcilable; workflow dispatch remains blocked on uncertain outcome; connector bridge uses a wildcard provider-idempotency/reconciliation contract because the vendor action is dynamic.
- This is a correctness policy layer, not a provider-side fencing guarantee. Providers remain responsible for honoring provider idempotency keys and reconciliation APIs.


## v72 conditional blind deliberation (2026-10-04)
- Research/compare/literature/study/analysis workloads use two bounded reasoning lanes (primary analyst + contrarian skeptic) followed by a critic adjudicator, draft, and final validation.
- The adjudicator activates the existing debate_decision() policy conditionally and receives blind/anonymized candidate views. Candidate ordering is derived from candidate content rather than agent identity; evidence cards are preserved separately from raw dependency outputs.
- Deliberation is capped at two rounds and is not an unbounded peer-to-peer debate. The goal is information gain only when disagreement, uncertainty, or evidence insufficiency warrants adjudication.
- Epistemic outputs request an explicit confidence value. Final validation remains separate from adjudication.
- The critic role is explicitly allowed to perform analyze + validate so the adjudicator keeps critic semantics without introducing a second role taxonomy.
- Multi-agent debate quality remains task/protocol dependent; current design therefore prefers conditional bounded deliberation over always-on debate.


## v73 evidence independence and calibration (2026-10-04)
- "independent_source_count" remains as a compatibility field but is computed as a distinct evidence-work count rather than trusted from an LLM self-report.
- Evidence work identity prefers DOI/arXiv/PMID/PMCID, collapses arXiv versions to one work, then falls back to canonical identifiers, bibliographic fingerprints, or weak provider identity. Each work carries an identity basis and proxy confidence.
- The independence layer explicitly marks itself as a proxy. It does not claim proof of independent authorship, datasets, citations, or study populations.
- Calibration metrics (Brier, ECE, MCE, reliability bins, selective risk) are label-dependent. Samples are accepted only with label_source in ground_truth, benchmark, or human_eval.
- Confidence alone is never treated as calibrated accuracy. Production/runtime metrics remain "unavailable" until explicit outcome labels exist.
- The evaluation harness contains a deterministic calibration/independence case so the epistemic quality layer is CI-gated without consuming live provider calls.

## v74-v77 reliability hardening integration line (2026-10-04)

- v74 selective epistemic admission is now enforced as an actual pass/fail boundary: supported claim coverage is exposed explicitly, and selective abstention cannot coexist with a passing validation result.
- v75 scheduled recovery no longer polls waiting_approval; stale running workflows are checked against their recorded GitHub run before rearming; repository_dispatch uses explicit POST; each candidate dispatch is isolated so one failure does not suppress later candidates; empty shard directories no longer make compaction fail; recovery IDs are derived from stable recovery-relevant state rather than mutable updated_at.
- v75 workflow authority is immutable per workflow. New live workflows are assigned either git_durable or distributed_control_plane authority. Distributed-authority workflows fail closed when the control plane is unavailable or its authoritative state disappears; they never silently downgrade to stale Git. Git-authority workflows do not silently upgrade when control-plane configuration later appears. Ingress idempotency_key is persisted.
- v75 deliberation evidence weakness is derived from structured evidence_records/work identity, never from an LLM-reported independent_source_count. Blind challenge views expose opaque candidate IDs rather than agent/role/voting metadata.
- v76 effect admission uses the active workflow lease rather than reacquiring a second workflow lease inside a supervisor turn. A side effect is admitted only after lease renewal and a bounded expected-duration+safety horizon check. Retry attempts refresh and retain the current lease object so completion uses the latest lease. The same fencing policy is applied to both one-step and full-workflow execution paths.
- v77 free-first evidence routing adds Crossref as a public metadata fallback without an API key requirement. Research responses remain bounded and cached. Evidence records now distinguish L0 identifier, L1 metadata/publisher locator, L2 abstract, and L3 concrete full-text access, plus explicit authority-tier/heuristic fields. These are routing signals, not truth probabilities; provider copies are still collapsed to work-level identity.
- The production runtime remains stdlib-only and free-first. No paid queue, database, workflow engine, or external research subscription was introduced.
- Green verification: v75, v76, and v77 stacked branches passed compile checks, Worker syntax, actionlint, full unit suite, and the control-plane/free-only evaluation. The latest v77 integration-line run also passed the private-input/bridge test workflow.
- Integration state: feat/v77-evidence-access-routing is 313 commits ahead of current main because it includes the full existing v67-v74 lineage plus v75-v77. Integration PR #137 is intentionally draft; main remains unchanged.
- Deferred P1/P2 work: full ExecutionRuntime consolidation, richer claim↔evidence graph/entailment verification, provider-level fencing where external systems support it, contradiction-driven retrieval, risk-coverage calibration, and optional Durable Object Alarm wakeups. These remain separate because adding them before the current invariants are stable would increase failure-domain coupling.

## Integrated Telegram control-plane frontend — 2026-10-05
- Telegram is an interaction/control frontend, not workflow truth, queue, artifact store, or execution authority.
- New branch: feat/telegram-control-plane.
- New Worker: workers/telegram-control-plane. It accepts private-chat webhook updates only, applies an explicit Telegram user allowlist and separate approver allowlist, requires active/revocable consent before orchestration commands, and exposes /privacy as the policy URL endpoint.
- Supported commands: /run (dry-run), /runlive (live request subject to orchestrator policy), /status, /approvals, /resume, /approve (node-scoped), /prompt on|off, /last, /privacy, /revoke, /delete_me, /id, /help.
- Telegram update_id is the inbound replay identity. D1/SQLite metadata includes inbox replay state, consent, encrypted short-lived sessions, and bounded audit records. Raw Telegram user/chat IDs and prompt bodies are not persisted.
- Session encryption uses AES-256-GCM with a key separate from the HMAC key used for derived identifiers.
- Gateway calls use the existing HMAC request boundary. Telegram approval is a node-scoped control action targeting the canonical GitHub approval issue; it does not create a new orchestration workflow or bypass the approval fingerprint/continuation path.
- Generic Telegram repository_dispatch approval-bypass propagation was removed. Telegram approvals now target the existing exact GitHub approval issue for a specific workflow/node, preserving the canonical fingerprint/label/continuation semantics.
- Database reference schema: database/telegram_schema.sql. First target is Cloudflare D1/SQLite-compatible metadata storage; later mapping to the execution-fabric PostgreSQL adapter remains part of E2.
- Dedicated CI: .github/workflows/telegram-control-plane-tests.yml covers Python/Node syntax, database schema tests, worker tests, and a tracked-source bot-token-shaped secret guard.
- Activation: .github/workflows/telegram-control-plane-deploy.yml provisions/resolves D1, applies versioned migrations, deploys the distributed Durable Object control plane first, derives its HTTPS Worker URL, injects that URL as a generated runtime variable plus the remaining secrets through temporary files, deploys the Telegram Worker, configures the Telegram webhook and command menu, and runs /health verification. The control-plane URL is not a secret; the control-plane authentication secret remains secret-only.
- D1 schema is tracked through workers/telegram-control-plane/migrations/0001_telegram_control_plane.sql; database/telegram_schema.sql is the portable snapshot. Base wrangler.jsonc remains deploy-neutral so an unprovisioned D1 UUID is never committed.
- Telegram compliance boundary: no group/channel collection, no public-platform scraping, no Telegram-derived training/ML dataset, no credential requests, no rate-limit bypass, and no Bot-to-Bot automation in v1. User-submitted prompts may be forwarded only for the declared orchestration purpose under explicit consent.
- Bot token is never stored in source control. Any token pasted into chat or another untrusted surface must be rotated before deployment and supplied only as TELEGRAM_BOT_TOKEN secret.

## Remediation hardening — 2026-10-07
- Fixed Telegram D1 inbox claim SQL placeholder/bind arity in `workers/telegram-control-plane/src/db.js`. The claim path now has matching eight-column/eight-value SQL with six bound parameters.
- Added a D1-facing inbox lifecycle regression test covering first claim, completion, duplicate suppression, stale reclaim, and SQL placeholder/bind arity.
- Removed `CONTROL_PLANE_URL` from the static Telegram Worker secret contract. The deployment workflow now injects the derived Durable Object URL through generated `vars`; `CONTROL_PLANE_SECRET` remains the only control-plane credential.
- Added Telegram CI configuration-consistency checks so the static Worker config cannot regress to treating the control-plane URL as a secret.
- This remediation is intentionally limited to correctness/configuration hardening; enterprise queue/state/worker/observability scale phases in issue #149 remain explicit follow-on work until each phase has its own implementation and acceptance evidence.

## Scheduled recovery entrypoint regression — 2026-10-09
- Root cause: `.github/workflows/orchestrator.yml` invokes `python3 -m orchestrator.scheduled_recovery`, but `scheduled_recovery.py` previously defined `main()` without a `__main__` entrypoint. The module therefore exited without writing `/tmp/orchestrator_recovery_failures.json`; the workflow then failed at its recovery-result gate and skipped orchestration continuation.
- Remediation candidate: PR #161 adds the module entrypoint, keeps the production marker path unchanged by default, permits an explicit marker path for isolated tests, and tests the exact `python -m` invocation in a disposable package/state environment.
- Test-safety hardening removes both `GITHUB_TOKEN` and `GH_TOKEN` from the subprocess environment. The fixture uses empty workflow state and removes control-plane credentials, preventing GitHub dispatch or production-state mutation.
- CI evidence before the final test-safety/doc updates: Orchestrator Tests run #2172 (run ID 37924693288) passed compile, schema/worker syntax, import smoke, Actionlint, embedded workflow syntax, unit tests, and the control-plane evaluation harness. The final branch head must pass a fresh CI run before merge.
- Acceptance is not complete at PR CI alone: after merge, verify a scheduled recovery run on `main` reaches the post-recovery gate and that continuation behavior follows the actual recovery result. Do not infer plan/checkpoint/connector integrity solely from marker creation.


## Recovery stderr normalization — 2026-10-14
- Confirmed defect: scheduled recovery normalized only the literal two-character escape sequence `\\n`; actual CR/LF and tabs could remain in dispatch error output. This affects log legibility, not dispatch semantics.
- Candidate fix on `fix/recovery-stderr-normalization-2026-10-14`: add `_normalize_stderr()` to flatten actual whitespace and legacy escaped whitespace, and retain the existing 300-character output bound.
- Added focused regression cases for actual CRLF/tab, legacy escaped whitespace, truncation, and empty/None input. Tests/CI are pending; do not treat this as validated until the PR's actual checks pass.
- Tool-write diagnosis: GitHub read calls and branch-scoped `update_file` calls succeeded during this cycle. Earlier reports did not retain an exact rejection payload, so the earlier “safety gate” cause is unproven. Do not repeat that attribution unless a concrete returned error identifies it.
- Scheduler correction: the ChatGPT automation `Orchestrator Autonomous Engineering Watch` was configured as `condition_watch` despite an hourly RRULE. It was changed to `exact_schedule`, with a local Asia/Jakarta start of 06:00 and hourly execution at minute 00. This changes the audit automation only; the repository's GitHub Actions cron was not altered.

## Aurum Intelligence checkpoint — 2026-10-16

Aurum's project-specific working memory and v1.2.0 implementation checkpoint are kept at `projects/aurum-intelligence/` on draft PR #164 (`feature/aurum-online-dataset-benchmark`). This is a separate product initiative; do not conflate its market-data conclusions with the core orchestrator's epistemic calibration metrics.

- Current draft PR: https://github.com/orionrayy/ai-determinism-engine/pull/164. Latest branch head checked at checkpoint: `ef0dbdf1e0df64e8dff40d6a59d9c21df94ffd7f`. The PR has **not** been merged. The complete v1.2.0 ZIP is a downloadable project artifact, not a binary committed into Git.
- Local v1.2.0 source release: 260,781 bytes, SHA-256 `19912b690e74af802a7170316703d40048a67731a1eb26d806c9929f048d2db3`. Clean release build passed 176 pytest tests, 109 unittest tests, compileall, Mini App inline-JS syntax, 48 unique HTML IDs, ZIP integrity, and extracted-archive retest (176 pytest passed).
- UX increment: explicit Sederhana/Tingkat lanjut mode in the Mini App and Telegram preference flow. Telegram choice persists in SQLite with backward-compatible migration; Mini App choice is browser-local. Display mode changes presentation only and does not relax risk rules or change model probabilities.
- Latest online data workflow on the PR head, run #38885439665, completed successfully: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38885439665. Verified workflow artifact id `12039332998`, SHA-256 `faa9dfd38c9df08e3984d31167422af6154930c67c9da89dd9e791eba58ee993`. Public data input: 35,954 XAUUSD M5 OHLCV rows, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`, with observed timestamps 2026-03-26 through 2026-09-25; this is historical, not a live feed.
- Online diagnostic benchmark: 50 timestamp-spread cases x 5 fixed baseline rules = 250 comparisons; 1,200/1,200 generated counterfactual market-path contracts and 240/240 software robustness contracts passed. Baseline PnL uses assumed spread/slippage, not measured broker costs. One breakout rule made +2.37 price units on only 7 trades, insufficient to establish alpha.
- Data provenance, intended licensing scope, source timezone, quote-side/broker semantics and actual fill costs remain unverified. No validated/calibrated forecast model exists; bundled-source OOS forecast count remains zero for the barrier candidate. Production status is research/UX prototype; decision gate `NO_TRADE`; live order execution remains disabled.
- Next engineering priority: source/rights review and licensed broker-specific Bid/Ask/tick data; multiple non-overlapping chronological regimes and untouched holdout; mature-label leakage controls; calibration diagnostics (Brier/log loss, reliability, ECE/MCE, uncertainty), cost-aware net performance; mobile/accessibility and authorized read-only integration tests. Do not claim “world-leading accuracy” or superiority absent comparable, independently reproducible OOS evidence.

## Aurum Intelligence v1.3.0 checkpoint — full-history benchmark

Updated 2026-10-16 (Asia/Jakarta). This supersedes the earlier v1.2.0 Aurum checkpoint where details differ. Product branch and draft PR: https://github.com/orionrayy/ai-determinism-engine/pull/164. PR title: v1.3.0 full-history replay and probability diagnostics. PR remains open and draft; do not report it as merged. Current branch head when recorded: `cc8ff5edaaa315609dc256e8f9d5a9bb3e5b0e66`.

- Current local source artifact: `aurum-intelligence-v1.3.0.zip`, 278,402 bytes, SHA-256 `53dabb5e86c8ce74691790e4c865d2624667cf531e572818bc571527bd4c2b67`. It is a downloadable project artifact and not a binary tracked under this repository.
- Clean package checks: 181 pytest, 109 application unittest, 11 dedicated online-benchmark tests, compileall, Mini App inline-JavaScript syntax, 48 unique HTML IDs/no duplicates, ZIP integrity and post-extraction pytest (181 passed).
- Hosted current full-history benchmark: GitHub Actions [run #38888666520](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38888666520), successful on source commit `d790435b9f09d623a795aa6ab7ee4d1732515aa5`. Artifact ID `12040248097`, size 14,319 bytes, SHA-256 `8a21fda87e8031b52b7ab2bada70b2dff58b5cefe402728a1afcea121a397dbe`.
- Dataset: GetData Finance community XAUUSD M5 CSV, 35,954 rows, 3,070,226 bytes, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`, timestamp range 2026-03-26 to 2026-09-25. It is historical, not a current live feed. Provider timezone, original provenance and intended-use rights scope are not independently verified.
- Benchmark coverage: 50 selected timestamp cases × 5 rules = 250 comparison rows remains as separate selected-case diagnostic. Full-history replay evaluates 35,893 sequential opportunities per fixed rule after a 60-bar warm-up. Five disjoint chronological fold sizes: 7,175 / 7,176 / 7,176 / 7,176 / 7,190; fold 5 is exactly the final 20% tail holdout. 1,200/1,200 counterfactual market-path checks and 240/240 software robustness checks passed.
- Under assumed total round-trip spread/slippage of 0.50 price units per trade, all four active fixed baselines are net negative in full history and the final tail: last_return_sign -19,531.435 / -3,766.090; ema5_vs_ema10 -17,497.475 / -3,577.310; mean_reversion_z10 -8,710.560 / -1,762.260; prior20_range_breakout -2,290.290 / -480.350. These are simplified next-bar price-unit diagnostics, not broker-executable performance. The earlier selected-case breakout +2.37 from seven trades does not survive full replay.
- Causal prequential direction-only baselines produced 35,893 forecasts: constant-0.5 Brier 0.250000; expanding Laplace Brier 0.250019 (improvement CI [-0.0000639, +0.0000412] spans zero); rolling-288 Brier 0.250749 (worse than constant). This is not a trained model or TP-before-SL calibrated probability. No validated production model exists.
- Project evidence is committed on the PR branch: `projects/aurum-intelligence/benchmarks/full_history_replay_2026-10-16.md` and `.json`. Keep the signal gate `NO_TRADE`, no live trading and no world-leading accuracy/profitability claim. Remaining gates: licensed broker Bid/Ask/tick data and measured costs, independent chronologic regimes, external untouched holdout, exact-target calibration, source-rights review and paper trading/security approval.

## Aurum Intelligence checkpoint override — v1.3.1 temporal calibration (2026-10-16, Asia/Jakarta)

Aurum remains an independent research/UX product stored under `projects/aurum-intelligence/` on the draft PR [#164](https://github.com/orionrayy/ai-determinism-engine/pull/164). It is not merged to `main`; do not confuse these finance-market metrics with the core orchestrator's general epistemic calibration metrics.

- Latest application artifact: `aurum-intelligence-v1.3.1.zip`, 276,128 bytes, SHA-256 `b63af5ab3412e7cb0bac14deffdb0d8cf07e024b0e46364d6fe5e1329d831d5b`. ZIP root path is corrected to `aurum-intelligence-v1.3.1/`. Local clean checks: 185 pytest, 109 unittest, compileall, Mini App inline-JS syntax, 48 unique HTML IDs/no duplicates, ZIP integrity, and extraction/retest (185 pytest passed). The ZIP is a downloadable project artifact, not a binary committed to GitHub.
- v1.3.1 adds Platt calibration diagnostics to the application candidate: fit only from earlier OOS forecast/outcome pairs whose label exit indexes matured before the later chronological holdout; score holdout Brier, log loss, equal-width ECE/MCE, reliability bins, and moving-block bootstrap intervals. The output hard-codes `deployment_calibrated=false`, `validated=false`, and `live_signal_enabled=false`.
- Component source + tests are committed in PR branch `feature/aurum-online-dataset-benchmark` under `projects/aurum-intelligence/temporal-calibration-v1.3/`. Hosted run [#38897894594](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38897894594) passed against code commit `6017eec3141fb803907583abeca006a230e90be4`: 8 calibration tests and 240/240 synthetic contract cases passed, together with the online data acquire/replay assertions and artifact upload. Workflow artifact #12044756186 has SHA-256 `a2561de0161e79dbaac794988b5545cd8f32fe2cf55d38c875e8aed91e99d9b3`, 14,966 bytes, expiry 2026-10-30.
- PR head `41722d74ab7cd9118651b1d577b71d2a2fbc94ee` passed hosted run [#38898121717](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38898121717). Its artifact #12044771562 is 14,966 bytes, SHA-256 `0c4eac0db61562a99f0ae3c2ffc76e4765a8eae54034db5d7ae0a459351d0dbd`, expires 2026-10-30. The current branch tip `f5540f2a47f82e5f2033dd5ea479d7fb8142c458` is a later status-JSON-only commit; no application/calibration/workflow code changed after the successful run.
- Public online dataset remains the GetData community XAUUSD M5 sample: 35,954 rows, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`, period 2026-03-26 to 2026-09-25. This is historical input, not a live feed. Intended licensing scope, original provenance, timezone semantics, broker quote side and actual executable costs/fills remain unverified.
- Hosted full-history replay covers 35,893 opportunities per fixed rule in five chronological folds (7,175 / 7,176 / 7,176 / 7,176 / 7,190, final fold=last 20%). All active fixed-rule baselines lose under assumed 0.50 price-unit round-trip costs; prequential direction baselines do not establish meaningful Brier skill over constant 0.5. The earlier +2.37 breakout result over seven selected trades becomes -2,290.29 over 4,035 full-history trades.
- The 720-row calibration demonstration and its small numeric improvements are synthetic; they are contract/diagnostic demonstrations only, not financial performance evidence. The true bundled market dataset is only 60 bars and the TP-before-SL model gives `insufficient_history` and zero OOS forecasts. Keep `NO_TRADE`; no live trading. Do not make world-leading accuracy or profitability claims.
- Next gates: review data rights/provenance/timezone; obtain licensed broker Bid/Ask/tick data and measured costs/fills; add independent non-overlapping chronological regimes and untouched holdout; validate calibration for exact TP-before-SL matured labels with uncertainty intervals; paper trade and complete separate security/risk approval before any live execution.

## Aurum Intelligence v1.4.0 P0 evaluation and cross-source disagreement checkpoint — 2026-10-17 16:44 WIB

- Aurum application package remains v1.3.1, research/UX prototype. The research evaluator is protocol v1.4.0; it does not imply a v1.4.0 production release. Draft PR #164 remains unmerged. Code/test head verified by hosted CI: `fe6ed83b0ec242e2d8e81317c2c798dac00f93bb`; docs-only checkpoint updates followed. Current decision gate remains `NO_TRADE`.
- Hosted workflow [#39022220573](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39022220573) passed: eight temporal calibration unit tests, 240 synthetic calibration contracts, 26 parser/baseline/evaluation/source-concordance unit tests, online data acquisition, primary replay, cross-source assertions, report generation and artifact upload. Artifact #12096704482, 2,556,935 bytes, SHA-256 `94d432018fa313db265bc01dbeba1e37df5efc0a3c90f65eaec432259877f7de`, expires 2026-10-31.
- Primary GetData community XAUUSD M5: 35,954 rows, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`, represented period 2026-03-26 to 2026-09-25. Secondary `Sai310421/xauusd-data` XAUUSD M5: 17,371 rows, SHA-256 `ee23e18f0ed77160893936d0312c008261e4f2ac16d82365898aba20ea26de9d`, represented period 2026-02-25 to 2026-05-26. The secondary repository labels its repo MIT and describes mid-quote OHLCV; upstream intended-use data rights and provenance remain unverified.
- Cross-source exact-time concordance: 11,509 common timestamps, zero exactly matching OHLC bars; absolute close difference median 134.885383 bps, p95 203.661431 bps, p99 217.616677 bps, max 220.412405 bps. Primary-minus-secondary signed close difference median +63.515 price units (p05 +0.295, p95 +94.445); one-bar return correlation 0.9984495. Cross-source status is `SOURCE_DISAGREEMENT_REVIEW_REQUIRED`. Screening heuristics are median >5 bps or p95 >20 bps; these are triage thresholds, not execution-price truth. The two sources are kept separate and neither is promoted.
- The P0 TP-before-SL evaluator processed 35,954 primary rows and generated 94,604 candidate signal events across five fixed rules; 94,084 outcomes were binary scored and 520 same-bar TP/SL collisions were marked ambiguous and excluded. Partitions are chronological 70/10/20; all predictions use only earlier matured outcomes; the calibrator uses earlier calibration labels whose exits mature by the holdout boundary. Output remains diagnostic only.
- Primary dataset quality profile: median/p95 interval 300 seconds, 136 intervals >2× expected M5, including 132 intervals >1 hour, 97 robust price-jump flags, zero flat bars and zero zero-volume bars after strict parsing. Source-index continuity and intended timezone semantics cannot be proved from this schema.
- Calibration on the exact TP-before-SL target is mixed: some rule/side holdout slices improve after Platt scaling and others degrade. This does not demonstrate predictive alpha or a validated model. No live trading, no world-leading-accuracy claim. `deployment_calibrated=false`, `validated=false`, `live_signal_enabled=false`, `NO_TRADE`.
- Next P0 work: determine whether timestamp/bar labels, price basis, provider methodology or another source issue explains the material cross-source discrepancy; review upstream data-use rights; obtain broker-authorized Bid/Ask/tick data with measured spread/commission/swap/slippage/fill semantics; rerun independent regime and untouched holdout tests. Do not progress to model complexity or order routing until these gates are addressed.


## Aurum Intelligence — P0 cross-source price-basis diagnostics continuation (2026-10-17 17:28 WIB)

Canonical project: `projects/aurum-intelligence`. Application remains v1.3.1 research/UX prototype; evaluator protocol remains v1.4.0 P0. PR [#164](https://github.com/orionrayy/ai-determinism-engine/pull/164) remains open/draft/unmerged. Keep `NO_TRADE`; do not claim global #1 accuracy, validated calibration, profitability or live readiness.

### Code change and test evidence

- Implemented non-mutating cross-source price-basis diagnostics in `projects/aurum-intelligence/online-benchmark/source_concordance_v1_0.py`: signed OHLC differences, OLS level relationship, residual quantiles in price units and bps, signed difference by chronological quartile, and direct consecutive five-minute return differences. The output explicitly records `source_correction_applied=false` and `source_promoted=false`.
- Added regression tests for a controlled additive level offset and multiplicative price scaling. Tests verify diagnostic detection without auto-correction, timestamp shift, merge or source promotion.
- Source implementation commit: `5ec99e2fe4e478ddda60a8ebbef264226c47a076`.
- Code/test head verified by hosted CI: `6b5b2f04a674c81deee84f1a2cdaf5ca442663d5`.
- GitHub Actions [run #39024803960](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39024803960): success. Temporal calibration contracts, parser/baseline/source-concordance unit tests, live online acquisition, full benchmark, count/assertion gates and artifact upload completed. Artifact #12097644871, 2,563,258 bytes, SHA-256 `b2bc267bda4d067ed6baec07d196540f903e84545cf342f9938b17371f607415`, retention through 2026-10-31.
- Documentation updates subsequently committed to the PR branch: progress log `7166b0518febc9b2e7562eda669ab8e63b648569`, status `1cf2c486b315bad80ae683ee3854146931644942`, checkpoint `67a09a6892f80e2fd7bb21408cacbccb75a2cef9`. They do not alter evaluator code; nevertheless, recheck the latest CI before claiming the new branch tip passed.

### Live data diagnostic results

- Primary GetData community XAUUSD M5: 35,954 rows; SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`.
- Secondary Sai310421 community XAUUSD M5: 17,371 rows; SHA-256 `ee23e18f0ed77160893936d0312c008261e4f2ac16d82365898aba20ea26de9d`.
- Exact common timestamps: 11,509; exact OHLC matches: 0; median/p95 absolute close difference: 134.885 / 203.661 bps; signed primary-minus-secondary close difference median: +63.515 price units; one-bar return correlation: 0.9984495.
- OLS descriptive fit (primary close ~= intercept + slope * secondary close): intercept +130.350817 price units; slope 0.984434; level R² 0.940465; residual absolute p95 128.847931 bps. It does not justify an affine correction.
- Median signed close difference across chronological overlap quartiles: +28.55, +72.535, +55.945 and +60.5135 price units, so the difference is not a stable additive offset.
- Consecutive five-minute return absolute difference median/p95: 0.263933 / 0.981664 bps. Return correlation is not proof of executable quote-price equivalence.
- Timestamp offset scan: best candidate +12 bars (+60 minutes) at the configured search boundary; median discrepancy falls by only 2.37%, and return correlation at that candidate is -0.01456. Treat as boundary warning, not an alignment fix.
- Live source state stays `SOURCE_DISAGREEMENT_REVIEW_REQUIRED`. Sources are not merged, normalized, rescaled, time-shifted or promoted.

### Remaining P0 gates / next work

1. Investigate upstream timestamp/bar-label, provider price-basis and candle-construction semantics, and historical revisions; use independent evidence rather than fitting one source to another.
2. Verify intended-use data rights and original provenance for both community sources; repo license labels are not sufficient.
3. Obtain broker-authorized Bid/Ask or tick records plus measured spread, commission, swap, slippage and fill semantics.
4. Keep source-specific replay separate; after the data gate, validate independent regimes and an untouched final holdout, then calibration for the exact matured-label TP-before-SL target with uncertainty intervals.
5. Only after these pass: authorized paper trading, security/risk review, and separate approval for any live execution.

The successful CI proves the acquisition/evaluator/assertion pipeline works and correctly reports disagreement; it does not establish which source is correct, predictive alpha, a calibrated model or profitability. No additional model complexity or order-routing work should proceed while P0 data-quality/provenance gates are open.


## Aurum Intelligence — real tick Bid/Ask + lineage audit (2026-10-17 17:42 WIB)

Canonical project remains in `projects/aurum-intelligence`; application package v1.3.1, P0 research evaluator v1.4.0. PR [#164](https://github.com/orionrayy/ai-determinism-engine/pull/164) remains draft/unmerged. `NO_TRADE` stays active; no predictive model or live execution is validated.

- Added `projects/aurum-intelligence/online-benchmark/tick_bidask_benchmark_v1_0.py` with bounded pull of one CC-BY-4.0 XAUUSD bid/ask tick month from Hugging Face, strict quote validation, UTC M5 aggregation, actual observed feed-spread summaries, and exact timestamp comparison against both community OHLCV inputs. Raw ticks are not redistributed; attribution is recorded.
- Added tests for spread/invalid quote handling, source overlap and lineage diagnostic. Workflow enforces results for both community feeds, no automatic timestamp shift/rescaling/source promotion, and explicit lineage classification.
- Hosted GitHub Actions [run #39025937658](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39025937658) completed **successfully** on code/test head `c6f5062fab2a919b4d3795304c5cbee83b67618a`. All stages completed: temporal calibration contracts; online OHLCV benchmark; tick acquisition/validation; both source comparisons; lineage assertion; artifact upload. Artifact #12098756760, 2,595,322 bytes, SHA-256 `54a3905b3fb3a3909070245da80a9089d2051b029dd9f45bab5332ed260d7aa8`, retention until 2026-10-31.
- Downloaded May 2026 selected tick parts: four Parquet files, 58,806,458 bytes, 6,685,266 ticks, 4,392 UTC M5 quote bars; no missing/non-finite/nonpositive/crossed quote observations after strict parsing. Observed period 2026-05-01 to 2026-05-22 (partial month).
- Spread distribution for this reconstructed feed: median 0.58 price units, P95 0.827, P99 0.96; median/P95 in bps of mid 1.2555/1.7884. These values describe this public dataset, not a user's measured broker costs.
- GetData primary vs tick-derived mid: 4,380 exact timestamp overlaps, absolute close difference median/P95 123.229/138.379 bps, signed primary-minus-mid median +56.325 price units. No exact OHLC matches.
- Sai310421 secondary vs tick Bid: 4,392 exact timestamp overlaps and **4,391 exact OHLC matches (99.9772%)**, despite secondary README description as mid-quote OHLCV. Added diagnostic status `POTENTIAL_SHARED_LINEAGE_OR_PRICE_BASIS_MISLABEL_REVIEW_REQUIRED`. It does not establish whether this is shared lineage or a quote-basis labeling mistake; until verified, these inputs must not count as independent confirmations.
- Online benchmark result has `status=PASS`, 35,954 primary OHLCV rows, 250 direct case/baseline evaluations, 1,200/1,200 synthetic market paths and 240/240 software robustness cases. These are evaluation/code checks, not financial performance validation. `NO_TRADE`, model calibrated/validated = false, live signals disabled.
- Status/progress/checkpoint documents updated on PR branch: progress commit `92c2001beb0c291bed75d791de6faedc9a11b3d2`, checkpoint commit `074d321d9fec5173742aad708b81fba8b665ba14`, status JSON commit `65958a8e68f0c3feed22693996f881b86c9d495f`. Documentation-only changes do not change evaluated code head.
- Remaining P0: source-owner lineage/rights/provenance and timestamp/price-basis semantics; resolve primary-vs-tick price-level discrepancy with broker-authorized evidence; measure broker Bid/Ask/commission/swap/slippage/fill semantics; then validate the exact matured-label TP-before-SL target across truly independent regimes and untouched holdout. No model complexity or order-routing work until the data gate is cleared.


### Follow-up CI closure — 2026-10-17 17:52 WIB

- After saving the latest project status/checkpoint, the full workflow ran again on PR head `65958a8e68f0c3feed22693996f881b86c9d495f`; [GitHub Actions #39026245471](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39026245471) completed successfully. Artifact #12098043414, 2,595,322 bytes, SHA-256 `184ff507423f4902dd8d9f86647d0e6ca5fe3fd73513de4d604f1b3823bc16aa`.
- All workflow stages passed: setup/installation, temporal calibration contracts, OHLCV benchmark, tick Bid/Ask audit, lineage assertion gates and artifact upload. This rerun does not change the data interpretation: secondary mid-labelled OHLCV is 99.9772% exact on Bid bars from the reconstructed tick dataset and remains non-independent until source provenance is resolved.
- PR #164 remains open/draft/unmerged. `NO_TRADE`, model calibrated/validated = false, live signal/order execution disabled. The remaining P0 gates are source ownership/provenance and quote semantics, intended-use rights, primary-vs-tick level discrepancy, broker-authorized quotes/measured costs, and independent chronological validation with untouched holdout.


## Aurum Intelligence — 700k+ M1 Bid/Ask benchmark and fair base-rate result (2026-10-17 18:24 WIB)

Canonical project remains `projects/aurum-intelligence`; app package v1.3.1, evaluator protocol v1.4.0 P0; PR #164 stays draft/unmerged. Keep `NO_TRADE`.

### Latest green full pipeline

- GitHub Actions [run #39028110673](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39028110673) passed on tested head `375aa2e3c225394943f177a5243d5c18ccd04fd9`. Artifact #12099177720, 2,613,340 bytes, SHA-256 `5b5129edebd988ca885e0e06f69b8430a9db01ab807fcdb8e11c6d3b0470891b`, retention through 2026-10-31.
- The hosted workflow completed: temporal-calibration contracts, 46 unit tests including new large-scale cases, current community M5 audit, tick Bid/Ask and lineage audit, 700k+/20k+ gates, large-scale benchmark, final assertions, report generation and artifact upload.
- Later documentation commits on the PR branch do not change benchmark code/workflow files. Workflow path filters don't include all progress/status/checkpoint documents, so the code/test evidence reference is the tested head above.

### Large-scale real dataset

- Source: `kevingtlin/Market-Data-Lab`, dataset ID `XAUUSD-DUKASCOPY-M1-BID-ASK`, metadata reports Dukascopy via `dukascopy-node`; exact broker/source lineage and upstream rights are not independently verified.
- Period 2023-01 to 2025-06, 30 months. Downloaded 131,316,007 bytes of monthly Bid/Ask M1 data; paired valid rows: **1,313,280**.
- Weekday rows: 937,440; weekday active non-flat candles: **870,500**, satisfying the minimum 700,000 without counting calendar/weekend padding. Zero-range Mid OHLC fraction: 32.7459%; reported, not hidden.
- 20,000 unique real timestamp scenario anchors, all valid ATR; 875,611 eligible contiguous windows; 120,000 TP/SL path scenarios (20k anchors × 2 directions × 3 TP distances); 52,883 directional rule events scored; 10,577 final-holdout scored events; 1,218 ambiguous same-bar outcomes excluded.
- Added finite-positive ATR filtering for anchors, a weekday-active non-flat candle gate, a causal global matured-label prevalence comparator, and a moving-block bootstrap interval vs that comparator. Final holdout priors are frozen; no final-test labels are consumed to update prediction priors.

### Calibration result (do not overclaim)

- Aggregate rule/side probabilities Brier: 0.1401016413; global causal matured-label baseline Brier: 0.1400154407; base rate 0.1681258628.
- Brier skill against weak constant-0.5 comparator: +0.439593, **not an edge claim** because the target is imbalanced.
- Against the more appropriate global matured-label baseline, aggregate Brier skill is **-0.000615651**. Moving-block 95% CI for Brier improvement is [-0.000173704, -0.000009063], entirely negative, hence the baseline comparison is a negative/no-edge result.
- Final holdout Brier: 0.163455139 vs global baseline 0.163549419, skill +0.000576467, ECE 0.0433126. The tiny positive holdout delta does not override negative aggregate skill and is not enough to claim stable alpha.
- Code/output stays `deployment_calibrated=false`, `validated_model=false`, `validated_for_profitability=false`, `live_signal_enabled=false`, `decision_output=NO_TRADE`. No “world #1 accuracy” or profitability claim is warranted.

### Open P0 gates / next work

1. Verify source-owner lineage, quote basis, timestamp/bar conventions and data-use rights for GetData, Sai310421 and the reconstructed tick source; secondary OHLCV must not count as independent of tick data until explained.
2. Resolve the primary-vs-tick price-level discrepancy using independent provider/broker evidence.
3. Obtain broker-authorized Bid/Ask/tick data with measured spread, commission, swap, slippage, latency, market-impact and fill semantics.
4. Only after the data gate, validate target-specific calibration on independent chronological regimes and locked untouched holdout with uncertainty, compare against global/constant/prequential baselines, then paper-trade and perform separate security/risk review. Do not enable live execution until then.

Checkpoint files on the PR branch were updated: `projects/aurum-intelligence/PROGRESS_LOG.md` commit `e756edaa52e5cfb97c10dac4852203f58660444e`; `IMPLEMENTATION_STATUS.json` commit `576c1de66e9a7b91da3ba38782235d03732a87dd`; large-scale checkpoint `V1_4_1_IMPLEMENTATION_CHECKPOINT.md` commit `3688a6907f68ae5d80839355e09cf548553785e8`. The connector refused the PR description update because the GitHub account requires a verified email address; repo checkpoint files and this canonical memory are updated.


## Aurum Intelligence — GDELT temporal coverage integrity and P0 rerun closure (2026-10-17)

Canonical project: `projects/aurum-intelligence`; application package v1.3.1, research evaluator v1.4.0 P0. PR [#164](https://github.com/orionrayy/ai-determinism-engine/pull/164) is still draft/unmerged. Decision gate remains `NO_TRADE`; no production, validated profitability, or world-leading accuracy claim is supported.

### Implemented fixes

- GDELT size-budget handling distinguishes an oversized archive from exhaustion of the aggregate budget. Oversized archives are skipped with an explicit warning so later sampled archive windows remain eligible; raw row counts no longer stop acquisition before deduplication.
- GDELT manifests produce UTC 15-minute coverage intervals. A coverage summary merges only truly adjacent/overlapping observed intervals and computes covered minutes/fraction against the requested window. Sample processing completion is separate from full source completeness.
- Point-in-time feature construction receives the acquired archive intervals. A 1h/24h news feature is numeric only if the required window is fully covered by archive batches that were already observable by the decision time. Sparse gaps are not treated as zero-news; future archive metadata cannot make a historical decision eligible.
- Runner reporting separates pipeline execution, unique-observation scale gate, source acquisition, full source completeness, and temporal feature coverage. Latest tested head `f3eaa644a1f73c0d4f9521870e1ba5c460cfaf69` includes the scope fix after the preserved failure [#39035356832](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39035356832) (`NameError` from referencing an acquisition-local variable in report scope).

### Latest end-to-end evidence

- GitHub Actions [#39035732248](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39035732248): **success** on `f3eaa644a1f73c0d4f9521870e1ba5c460cfaf69`; 115 unit/regression tests and 240/240 temporal calibration contracts passed. Artifact #12102745058, 2,624,921 bytes, SHA-256 `b1710712519fd48e3a5f29acf1d532707997821bba1948e49b093004fd7b81f0`.
- Price benchmark in that run: 1,313,280 paired Bid/Ask M1 candles, 20,000 valid ATR anchors, 120,000 TP/SL path scenarios and 19,937 mature unambiguous labels in the context-evaluation cohort. These satisfy engineering scale gates, not trading profitability or independent-market evidence.
- GDELT: 141,020 unique real observations / 100,000 target, no synthetic/duplicated records; 100 sampled archives processed out of 85,809 discoverable archives; 701,820,752 acquired bytes. Observed coverage is only 1,500 of 1,313,279 requested minutes = **0.114218%**. Correct status: acquisition `PASS`, source completeness `PARTIAL`, archive-window complete `false`.
- Temporal feature coverage over 19,937 evaluation rows: COT 19,739 rows (99.0069%, `PARTIAL`), macro 0 (`NOT_MET`), news 0 (`NOT_MET`). BLS returned 119 descriptive observations but no source publication-time calendar; FRED/ALFRED was skipped because `FRED_API_KEY_NOT_CONFIGURED`. News ablations remain ineligible despite passing the scale target.
- Same-row holdout ablations scored 3,987 rows each. Price-only Brier 0.1609893; price+COT/combined Brier 0.1609044. Incremental Brier improvement versus price-only is +0.00008494, with paired moving-block 95% CI [-0.00014777, +0.00032012], which crosses zero. Do not claim robust incremental predictive value; news and macro ablations are `NOT_ELIGIBLE_NO_CONTEXT_FEATURES_IN_DEVELOPMENT`.

### Remaining P0 gates

1. Verify upstream rights, source owner/lineage, timestamp/bar and quote-basis semantics across public OHLCV/tick sources. The secondary community OHLCV series nearly exactly matches tick Bid in 4,391/4,392 bars despite its documented Mid basis; neither source may be promoted as independent/truth until explained.
2. Obtain broker-authorized Bid/Ask/tick and actual fill observations with measured spread, commission, swap, slippage, latency/rejection and fill semantics.
3. Keep news features ineligible until sufficient relevant temporal windows are continuously covered or a missingness policy is explicitly pre-registered and separately validated; supply authoritative BLS publication times and FRED/ALFRED vintages before macro features are eligible.
4. Revalidate against causal base-rate baselines across genuinely independent chronological regimes and an untouched holdout, report calibration/uncertainty and run authorized paper trading plus separate risk/security review.

Checkpoint artifacts updated on the PR branch: `IMPLEMENTATION_STATUS.json` commit `70520020d31e3c5bdffb06f61b9688042987f9e1`; `PROGRESS_LOG.md` commit `0e83623522f52fea8c18abc4f855a5068d256d0f`; `V1_4_1_IMPLEMENTATION_CHECKPOINT.md` commit `4d5a072ffafbf75d5314ae951aca9a3386d3a435`; implementation-plan outcome commit `0afbf90f499b599b18115800e2d6a765ba64d573`; evidence index `BENCHMARKS_AND_EVIDENCE.md` commit `a3427544fed7e5005e2b7f72b0c50ccea1d94a18`. These documentation commits do not merge PR #164. Keep `NO_TRADE`, `deployment_calibrated=false`, `validated_model=false`, `validated_for_profitability=false`, `live_signal_enabled=false`.


## Aurum Intelligence — source qualification gate checkpoint (2026-10-17)

- Follow-on work is on PR #164 branch `feature/aurum-online-dataset-benchmark`; PR remains draft/unmerged. Latest source/test commits: helper `ed6cbd9686bb231bb481b606bc0fbc5ee0e84efb`, helper tests `dffca7b9088ac52a620bdb260f236ce088fdc4d2`, tick report integration `2f442ccdde3b8974f2713ed09763aae87903d7d8`, CI source-gate contract `d2621fcda08456984ec3adef9015f91d3d432ddd`.
- Added `projects/aurum-intelligence/online-benchmark/source_qualification_gate_v1.py`. It separates five research-evidence requirements (independent provenance, intended-use rights, timestamp semantics, bar convention, quote basis) from three execution-evidence requirements (broker authorization, measured execution costs, fill semantics). Every true claim requires a stable referenced evidence ID/URL; truthy strings/numbers and malformed references fail closed.
- Source qualification is not automatic promotion: even complete evidence produces `QUALIFIED_FOR_RESEARCH_REVIEW`, keeps `source_promotion_allowed=false`, and requires manual review. Execution evidence can at most become `ELIGIBLE_FOR_SEPARATE_EXECUTION_REVIEW`; it never turns on live execution. All gates continue to report `NO_TRADE`.
- Integrated the gate into `ci_benchmark.py` for primary and secondary OHLCV sources and into `tick_bidask_benchmark_v1_0.py` for the primary feed, secondary community feed, and reconstructed tick dataset. Existing technical/concordance status remains separate from source qualification.
- Updated hosted workflow assertions to require all three tick-related feeds to remain `REFERENCE_ONLY` / `BLOCKED` until independently referenced evidence is supplied. Added nine unit cases for strict boolean typing, references, manual-review behavior, and execution separation; extended workflow contract tests to verify the benchmark integration.
- Evidence reviewed: GetData public README declares ISO-8601 UTC bar-open timestamps but source rights/provenance still need independent confirmation. Sai310421 README describes its OHLCV as mid-quote and aggregated from public broker feeds; prior CI found 4,391 of 4,392 overlapping M5 bars exactly match the reconstructed tick source's Bid bars. This is a potential shared-lineage or price-basis mislabel signal, not proof of source independence or which feed is correct.
- Workflow run `39037940900` was cancelled when subsequent branch commits arrived and must not be counted as validation. Latest hosted run `39038107206` is pending at this checkpoint; no CI pass is claimed yet.
- The previous verified external-context run `39035732248` remains diagnostic-only. Do not alter `NO_TRADE`, model-calibration, validation, or live-signal flags until the new CI finishes and data/provenance/risk gates are actually satisfied.


### Source qualification gate — final hosted verification (2026-10-17 Asia/Jakarta)

- Resolution of the pending note above: hosted workflow [#39038107206](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39038107206) **completed successfully** on PR code head `d2621fcda08456984ec3adef9015f91d3d432ddd`; the earlier `39037940900` run was cancelled and is not counted as successful evidence.
- Unit/regression suite: **125 passed** in 12.107 seconds. Temporal-calibration contract matrix: **240/240 passed**, zero failures. Hosted stages also passed current community dataset acquisition, tick Bid/Ask audit, the 700k/20k scale gates, external-context point-in-time benchmark, and artifact assertions/upload.
- Run artifact: [#12102563009](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39038107206), `aurum-online-dataset-benchmark-39038107206.zip`, 2,626,545 bytes. GitHub reported SHA-256 `993c27155e00c7d57d87aba5c475c1751d7026fcbfce4345b784996215bc065c`; downloaded local ZIP was hashed and matched.
- Real-data acceptance: 1,313,280 valid paired M1 Bid/Ask candles for 2023-01 through 2025-06; 20,000 unique real anchor timestamps with valid ATR; 875,611 eligible contiguous ATR windows; 120,000 TP/SL path scenarios. These are deterministic research diagnostics and do not imply profitable performance.
- Online public dataset diagnostic: GetData primary XAUUSD M5 35,954 rows; Sai310421 secondary 17,371 rows; 11,509 exact timestamp overlaps, zero exact OHLC matches, close difference median/P95 134.885/203.661 bps. Status remains `SOURCE_DISAGREEMENT_REVIEW_REQUIRED`.
- Tick audit: 6,685,266 rows / 58,806,458 bytes / 4,392 UTC M5 bars. The secondary OHLCV matches reconstructed tick Bid OHLC in 4,391/4,392 overlapping bars while the community README describes mid-quote data. Status remains `POTENTIAL_SHARED_LINEAGE_OR_PRICE_BASIS_MISLABEL_REVIEW_REQUIRED`; no automatic correction, rescale, merge, timestamp shift, independence claim, or promotion.
- New evidence helper `source_qualification_gate_v1.py` gates research on independently referenced provenance, intended-use rights, timestamp semantics, bar convention, and quote basis. Execution additionally requires broker authorization, measured execution cost, and fill semantics. Current primary, secondary and tick sources stay `REFERENCE_ONLY` / `BLOCKED`; even full evidence can only trigger a separate manual review. `source_promotion_allowed=false`, `live_execution_authorized=false`, `decision_output=NO_TRADE`.
- External context: GDELT 141,020 unique observations met the scale target of 100,000 across 100 sampled archive files, but actual covered time remains 1,500 of 1,313,279 requested minutes (0.114218%), so full-window coverage is `PARTIAL` and no news rows qualify. COT qualified 19,739/19,937 rows; macro/news ablations remain ineligible. COT's incremental Brier improvement vs price-only is 0.00008494, paired 95% CI [-0.00014777, +0.00032012] crossing zero; robust incremental value is not established.
- After code/test CI head `d2621fcda08456984ec3adef9015f91d3d432ddd` passed, the PR-branch documentation was updated only: `PROGRESS_LOG.md` commit `ef7c5cecf1f9844fa56035fcf8b02c6c7af14a09`, `BENCHMARKS_AND_EVIDENCE.md` commit `7f391e035b61d4eab81f6143235dae7777a20d64`, `IMPLEMENTATION_STATUS.json` commit `5fa66b942f67965dcc7946c71df67a52156d554a`, and `V1_4_1_IMPLEMENTATION_CHECKPOINT.md` commit `fc2e5cd7625d12cddbd48dc83819fdf14f6a1e00`. These are documentation-only commits after the successful run and were not part of head `d2621fc`. PR #164 remains draft and unmerged. Application remains v1.3.1 research/UX prototype, evaluation protocol v1.4.0 P0. Keep `NO_TRADE` and all live/calibration/validation flags false. Do not claim state-of-the-art accuracy, independent alpha, or profitability until the missing gates are satisfied.


## FFmpeg Motion Studio — v3 animation and narration checkpoint (2026-10-17)

### Outcome and artifact state

- Iteration moved from v2 to a local/generated `ffmpeg-motion-studio-v3` source package. The archive is a current-conversation sandbox artifact at `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v3.zip`; it has **not** been committed as source files in this repository. Do not imply repository source sync has happened.
- A real 20-second visual review reel was rendered from four procedural scenes (IDs 1, 3, 17, 20) at 960×540, 24 fps, H.264. It is silent by design. The 390-second/27-scene film has **not** been rendered.
- Validation completed locally: Python compilation and 7 unit tests passed; narration slot QC passed. This is local engineering evidence, not a hosted CI run and not a quality certification.
- Renderer changes: isolated action layer with restrained camera tracking while captions/progress UI remain fixed; scene-aware human/mascot gaze and articulated arm gestures; staggered product entrance; shelf reflection and scanner sweep; progressive receipt/subtotal/discount/total reveal with total counting to Rp48.600; more progressive tree branches/node pulses, evaluator cards, spreadsheet rows and moving pipeline signals.
- Narration cues in `scenes.json` were rewritten/expanded to approximately 699 estimated spoken words. At an assumed 145 WPM plus 0.28s lead-in, per-scene slot use is estimated at 68.1%–87.1%. The estimate is not a substitute for measured provider duration or listening QC.
- Added `synthesize_indonesian_tts.py`: free-first online Microsoft Edge neural TTS adapter using `id-ID-ArdiNeural` by default and `id-ID-GadisNeural` as a comparison voice; supports selected Indonesian pronunciation adaptations, one cue per scene, fixed 390-second slot alignment, and fail-closed handling rather than silently clipping a long cue. Added `narration_qc.py` and `VOICEOVER_WORKFLOW_ID.md`.
- **Voice gate remains BLOCKED.** This environment had no outbound DNS/network access; installing `edge-tts` failed at DNS resolution, so no actual neural voice track has been generated or listened to. The package only has the prepared implementation. Existing FFmpeg mastering tests use synthetic tones as technical fixtures, not an Indonesian speech quality test.
- Next sequence: run the neural TTS generator in an internet-enabled environment; audition at least opening, C-F-G, arithmetic evaluation, receipt-discount and outro cues; select one consistent voice; fix actual cue durations/lexicon based on listening; then render full 6:30 and perform end-to-end visual/audio QC. Do not state the voice issue or production final is solved until this gate passes.

### Operational note

- Free-first is preserved. Do not spend a paid external voice-generation quota without explicit user approval. The current 20-second preview is visual-only, so it must not be described as demonstrating improved Indonesian audio.


## FFmpeg Motion Studio — v4 focus-first HD QC iteration (2026-10-17)

### User QC feedback and resulting design constraints

- User review of the v3 20-second preview called out stiffness/clutter, unclear and underspecified teaching content, undersized non-prioritized typography, rushed pacing, low apparent HD quality, and absent audio. Treat these as release-blocking QC issues rather than optional styling preferences.
- v4 adds a separate focus-first review reel for the receipt/discount learning segment. Six planned beats: item rows -> subtotal -> 10% discount value -> subtraction -> payable total -> three-step recap. One primary message per beat, larger values/formulas, captions isolated in a high-contrast bottom strip, removal of the tiny persistent six-stage rail, staggered row/card reveals and final-value count-up.
- Rendered artifact: `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v4-preview.mp4`; 41.75 sec, H.264, 1920×1080, 24 fps, 1,002 frames, silent visual master. SHA-256 `7ecf74e4dacc0a318de68c25ff847e79acce00f4561ab3e0afcfffb33139d10f`.
- Contact sheet: `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v4-contact-sheet.png`; SHA-256 `370eb4a4b5e5e13a384d6d0a690153fec8c8697da392a77ab67679c1909a47df`.
- Source/preview/test package: `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v4.zip`; SHA-256 `0b85221299dedef9e099748eeef77493ae7d4b3717d1453dba2767c9f886d797`.
- Seven local delivery tests passed: 1080p H.264 metadata, 24 fps, duration, timing/caption bounds, caption/high-priority typography contract, and honest audio state. Tests did not certify professional animation or natural voice quality.
- Added `renderer_v4.py`, `test_v4.py`, `narration-preview.srt`, script, audio provenance, and `mux_voiceover.sh` in the package. No external image assets used.

### Audio status — partially completed; do not overclaim

- One Indonesian male speech clip was generated with the connected HeyGen speech tool: voice `Renji - Calm & Soothing`, ID `c9116f15487b405cb9cfad3f8a227c0f`, duration 41.7437 sec. Audio URL: `https://resource2.heygen.ai/text_to_speech/c3f80441d3074e0abb3c5077a849e504/c9116f15487b405cb9cfad3f8a227c0f/id=646146b1-0604-4966-af3f-aa39f15805ab.wav`. Provider playback surfaced in the user conversation.
- This execution container could not resolve the provider domain; the returned WAV could not be downloaded/muxed. Therefore `audio_present=false`, speech is **not listening-QC-approved**, and the MP4 remains silent. Package includes `mux_voiceover.sh` to fetch/mux on an internet-enabled machine; verify provider URL still works before using.
- Do not count voice as solved until the audio is integrated into a video and manually checked for Indonesian pronunciation, prosody, intelligibility, volume consistency, and cue alignment. Do not silently substitute eSpeak or claim it is natural.
- This v4 is a focused 41.75-second review reel, not the complete 390-second film. Do not promote the complete film to release status yet.

### Next gated work

1. Listen to the generated voice sample and decide if timbre/prosody is acceptable; if not, select one voice route once authorized and generate a revised clip (avoid blind repeated TTS side effects).
2. In a network-enabled environment, download the voice WAV, run `mux_voiceover.sh`, verify audio stream metadata and actual playback.
3. Conduct mobile-size typography/readability and frame-by-frame motion QC; user review is still required.
4. Expand the focus-first design pattern across the 27-scene, 390-second master, retain topic/detail correctness, and only render release candidate after script, audio, and visual gates pass.


## FFmpeg Motion Studio — v5 story, rig and pipeline checkpoint (2026-10-17)

### User-requested QC outcomes

The user's latest QC criteria are: stronger narrative flow; substantially more visible digits, arithmetic symbols and educational detail; less repetitive compositions and motion; very smooth/stable animation; a complete simple 2D human character with eyes as the only facial features (no nose/mouth), articulated limbs, precise accessories, scalp-anchored hair, and plausible damped motion/gravity; free-first operation; and direct audio/video integration in the preview. Treat absent audio or unverified mux as a release blocker.

### v5 local artefacts and verified measurements

- Source directory: `/mnt/data/ffmpeg_audit/v5_work/ffmpeg-motion-studio-v5/`.
- Source ZIP: `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v5-source.zip`; SHA-256 `e68f4c3fb38444f14b9ed18bf6e80170a10049cf047d4f04f72bd0c97256c045`.
- Preview: `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v5-visual-only.mp4`; SHA-256 `5efaaa3b3f331e67477c2b3c2a4686dcaa935f3a9462bed545dbf3aabf266f9c`.
- Contact sheet: `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v5-contact-sheet.png`; SHA-256 `98f55b2ce8d3423b6ccedbf02164f2546253358d18c9e89e2c7b23b31e17af17`.
- Render report: `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v5-render-report.json`.
- Output is a 41.75-second review reel covering 8 instructional beats: receipt/question → item-level multiplication → subtotal → percentage-to-decimal → discount computation → subtraction → count-up/total → recap. This is **not** the full 390-second/27-scene film.
- Final local `ffprobe`: H.264, 1920×1080, 24/1 fps, 1002-frame expected output, 41.750 s, 6,213,695 bytes. No audio stream; report correctly says `audio_present=false`, `audio_state=PENDING_MUX`.
- Revalidated `py_compile` and all 7 local tests pass. ZIP integrity verified with `unzip -t`. These are local tests only, not hosted CI and not cinematic/speech quality certification.
- Renderer uses Pillow procedural drawings and FFmpeg encoding, deterministic time-based easing/damped springs, distinct scene palettes/layouts, large arithmetic equations and staged reveals. The rig defines eye-only face, head/scalp-anchored hair, articulated arms/legs, IK pointing pose, accessory strap/bag with damped sway. Contact sheet has been manually inspected. Refinements remain possible; the reel is a focused prototype.
- Source deliverables are local sandbox artefacts; they are **not all committed to this repository**. A feature branch `feature/ffmpeg-motion-studio-v5` was created and contains partial code commits for `motion_core.py` and `character_rig.py`; the attempted base64 archive transfer was incomplete and has been removed, as was the dependent incomplete workflow. The branch remains isolated with partial files; no PR opened and no merge. Do not claim source sync or hosted rendering has succeeded.

### Audio: real generated sample, integration still blocked

- Existing generated voice: HeyGen `Renji - Calm & Soothing`, voice ID `c9116f15487b405cb9cfad3f8a227c0f`, duration 41.743673 sec. WAV URL: `https://resource2.heygen.ai/text_to_speech/c3f80441d3074e0abb3c5077a849e504/c9116f15487b405cb9cfad3f8a227c0f/id=646146b1-0604-4966-af3f-aa39f15805ab.wav`.
- No new speech-generation request was made. Download attempts via the local runtime and URL-content fetch both failed (network resolution/access issue); no WAV was locally available to mux. The voice sample remains a separate URL and has not passed user listening QC. The v5 MP4 is silent. This requirement remains OPEN.
- Pipeline source includes a voice-download/mux branch that will try to use the existing WAV when run in a network-enabled environment, verify the input WAV and output audio stream, apply FFmpeg audio mastering, and fail closed rather than marking a silent MP4 as a release. The pipeline is included in the local source package and its contract tests; the audio-download/mux path has not been end-to-end tested due the network block.

### Next execution sequence

1. Restore/commit the complete v5 source files (not a partial or corrupted bundle) on a dedicated branch.
2. In an internet-enabled runner, retrieve the existing WAV URL once and mux into the visual master. Verify with FFprobe that MP4 has AAC audio, 48 kHz, 2 channels, expected duration; audition speech and check Bahasa Indonesia pronunciation, prosody, loudness, and caption timing.
3. If audio fails, report the exact blocker; do not create unapproved alternate TTS jobs or silently substitute robotic local speech.
4. Expand the 41.75-second visual pattern to the 27-scene/390-second master only after the user approves this review reel and audio QC passes. Preserve 1080p/24 fps, scene-level story/lesson checks, font mobile readability, deterministic frame timing, character identity, animation stability, and artifact checksum.
5. Keep free-first; no paid voice service or resource purchase without explicit authorization.


## FFmpeg Motion Studio — v6 screenshot-driven geometry and layout fix (2026-10-17)

### User's screenshot-specific defects and root causes

- User reported that numbers overflowed their panel, pointing hand/arm proportions were wrong, hair/depth layering was wrong, and the character was partially off-screen before the reveal.
- **Confirmed accessory bug:** `rotate_points(points, cx, cy, angle)` treats input points as absolute screen/rig coordinates, but the v5 bag corners and seam were passed as local offsets. This transformed the accessory to the top-left origin and produced the stray edge fragment/incorrect layering. Fixed by converting each bag-local corner/seam point to rig-space before rotation; regression test checks both transforms.
- **Confirmed numeric overflow:** opening formula `2 × 15.000 + 3 × 8.000` was drawn at fixed 48px in a 700px panel. Replaced with two separate fit-checked item chips, currency labels, an independent plus operator, and a concise causal caption. Added font-metric test to ensure the item strings remain within their available 250px inner width.
- **Confirmed off-screen/stretch issue:** v5 root began at x=2025 while the arm was asked to reach an on-screen target. That made an isolated limb cross the scene and the elbow solver clamped elbow geometry while leaving the wrist outside feasible reach. v6 root is held within the frame; character is drawn on a transparent isolated layer and revealed as one group; wrist reach is bounded before the elbow solve; upper/lower limb widths and hand details were reduced.
- Hair layering is explicit: rear hair silhouette → ears/face → front cap/bangs/side lock → eyes. Facial marks remain eyes only. Accessory strap and bag are composited in the torso rig in intentional depth order.

### v6 local implementation, rendered visual and QC

- Local code directory: `/mnt/data/ffmpeg_audit/v6_work/ffmpeg-motion-studio-v6/`.
- Preview: `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v6-visual-only.mp4`, SHA-256 `1e1fa6264f71527b58616fb3a56d9019b1ecd0db8a9ef1b8a87cfebe75aa7b44`, 2,187,675 bytes; H.264, 1920×1080, 24/1 fps, 1,002 frames, 41.750 seconds.
- Contact sheet: `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v6-contact-sheet.png`, SHA-256 `a1f518a4aa531d516b7db644f41910a1810fea146abe90fbf94d810a327a6163`.
- Full handoff ZIP: `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v6-source.zip`, SHA-256 `2b322507d4edfb8442032616213c3d046b2387cf4c420fa98e3921d2eb5401d2`. Source-only ZIP: `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v6-source-only.zip`, SHA-256 `b1a0f9ebd4294be95a3d5096a6a77f1c87fd2afd475428856d59f43256cfc3aa`.
- Validation report: `/mnt/data/ffmpeg_audit/ffmpeg-motion-studio-v6-validation-report.json`. A fresh `ffprobe` confirmed the expected H.264 video metadata. Output intentionally remains marked visual-only.
- **12/12 local unit/geometry/layout tests passed**, including: safe character bounds at 10 timeline points, group reveal at t=0 (no pixels changed in base frame), bounded arm reach, no old x=2025 off-screen root, formula text fit, arithmetic, time/easing invariants, and bag/seam coordinate-space regression. These are local tests; no hosted CI ran and no professional animation certification is claimed.
- Frame/contact review sampled 14 timestamps (0.0, 0.35, 0.7, 1.1, 1.7, 2.2, 4.2, 8.2, 11.8, 14.5, 20.5, 27.5, 34.5, 39.5 sec). All inspected character parts stay inside frame-safe margins; opening expression is split within its chips; no bag polygon is drawn at the origin.

### Audio and source-control boundary

- Reattempted downloading the existing HeyGen Indonesian narration WAV once; local runtime returned DNS resolution failure (`Temporary failure in name resolution`). No new TTS generation was submitted. Thus v6 MP4 is **silent** with `audio_state=PENDING_MUX_NETWORK_DNS_BLOCKED`. The speech URL remains documented by the v5 checkpoint. Audio integration and listening QC are still open.
- Source files and video are available as local sandbox deliverables. The v6 code package has **not** been committed to the repository and no hosted CI run is claimed. A scratch feature branch `feature/ffmpeg-motion-studio-v6-qc-fix` was created during a sync attempt, but source files were not transferred and no PR/merge occurred; do not present it as a completed implementation branch.
- Next: run the checked-in source ZIP from a network-enabled CI runner or after complete source sync, fetch the existing WAV (no new TTS call), mux through `pipeline.py`, validate audio stream via FFprobe and audition speech/caption alignment. Full 390-second film remains a separate milestone.


## Motion Studio — canonical product goals, scope contract, and current checkpoint (2026-10-20)

This is a persistent product-goal memory, not a claim that all features exist. Read the full subsystem-by-subsystem acceptance contract before any Motion Studio task:
https://github.com/orionrayy/project-memory-and-database/blob/main/projects/video-production-pipeline/MOTION_STUDIO_MASTER_TARGETS.md

### Full requested feature groups — do not narrow

1. **Character authoring and rig:** bone rigging and character setup; hierarchical skeleton; bone constraints/limits; weight painting and mesh deformation; IK and FK plus blending; pivot/joint placement; bone mirroring and symmetry.
2. **Animation:** walk and run cycles; overlapping action; motion smoothing; anticipation/follow-through; secondary motion; background/property calibration and integration; interaction/physics; pose-to-pose, keyframe interpolation and smart keyframing; per-body-part timing, contact and constraints.
3. **Effects and image compositing:** particle system; speed effects; mask/matte; outline, overlay, blend, blur; color grading and filters; custom LUTs; chroma key; reusable effect stacks and transitions.
4. **Captions and typography:** smart/auto captions; attention-catching subtitle transitions including fade/glitch where appropriate; word emphasis/color; kinetic typography; lower thirds; configurable fonts, font size/weight, outline/shadow, safe zones and mobile readability.
5. **Camera and editing:** camera angles, tracking, pan and zoom; motion tracking; match cuts/match-on-action; compound clips/nesting; multitrack timeline; speed ramps with transition combinations; social-video formats such as 9:16/4:5 and 16:9.
6. **Audio and sound design:** hook SFX, ambience/idle/motion SFX, noise treatment, pacing and intentional pause; volume balance/keyframes; surgical EQ, equalizer and frequency-masking avoidance; crossfades; compression/limiting; reverb and per-clip effects; beat sync; layered BGM/sound; spectral/multiband or sidechain ducking; independent voice/music/SFX/ambience tracks.
7. **Project/export:** one canonical editable project graph, timeline, scene/clip/effect/keyframe/asset schemas, save/reload, undo/redo, renderer, codecs/export presets, manifests, caching, resumable jobs, deterministic tests, render logs and checksums.
8. **Rights and free-first operation:** verify actual copyright/royalty-free/commercial license terms for each font, image, video, music, SFX, model and other asset; retain origin, license version, attribution, restrictions, checksum and acquisition metadata. “Royalty-free” must not be treated as unrestricted or copyright-free.
9. **Quality gates:** report source vs render vs human visual/audio acceptance independently; inspect bulk frames and full-speed playback; verify self-intersections, foot skating, mesh/joint flips, visual layout, audio/video sync, audibility, source provenance and export decoding.

### Evidence/status rules
- FFmpeg filters are implementation primitives, not proof that a Motion Studio product subsystem exists. A feature is implemented only if its source/model/API or UI, persistence, tests and demonstrable export path are available.
- Geometry tests cannot substitute for natural movement or visual review. An MP4 that decodes is not automatically acceptable.
- Never mark a legacy feature verified before recovering its missing dependencies and re-running end-to-end validation.
- Keep unimplemented features as explicit roadmap goals; do not imply a production-grade editor exists today.

### Motion Studio checkpoint v0.8.0
- User's screenshot of v0.7.0 showed clear defects (crossed leg projections, stiff upper body, arm/prop pose quality); the screenshot is evidence that v0.7.0 failed artistic acceptance.
- Implemented candidate: Hermite foot swing with stance-velocity matching; stable pole-target two-bone IK; depth-projected bilateral leg tracks; heel/toe roll; separately configured basket-carry and free-arm profiles; measured carry elbow angles; explicit prop-contact checks.
- Current code review branch in orionrayy/projects: motion-studio-v0.8.0-character-quality, head 048a472e9171d491bd3bf8695e78c4520f5a86f3. Hosted run https://github.com/orionrayy/projects/actions/runs/39311955265 succeeded with 9/9 tests for the committed solver/scene slice. It does not render the complete source package or assess artistry.
- Local full bundle tests: Node 62/62; focused v0.8 Node 10/10; Python 13/13; scene/evidence schemas, synthetic motion benchmark, rendered-pose check 96/96 and reference keyframe-clip validation passed.
- Carry preview: H.264 1280x720, 24fps, 192 frames / 8s, no audio, SHA-256 8a89e4d26114fa341782fc7d6b5b4a6c8f1bd7c0a823c09b1349f4bf7dd71d01.
- Free-arm preview: H.264 1280x720, 24fps, 192 frames / 8s, no audio, SHA-256 9f1e5137ef2bce54e25c8007e2561960014b59cb70e490795dea02d0de9ebb7b.
- Contact sheets for both clips were inspected; elbow flexion and leg path order improved, but the character still reads as schematic. Status: PROTOTYPE / VISUAL ACCEPTANCE PENDING. Full-speed artistic review has not approved naturalness; browser-vs-FFmpeg raster parity remains unresolved; these studies contain no audio.
- Main branch of orionrayy/projects remains unchanged at 39415a2cb71d99bddcd84c10805c0ff0b8995cca. Do not merge automatically.

### Ordered next work
A. Character quality first: frame-level limb/mesh intersection and ground/contact checks, leg/foot phase art direction, hand/prop grip, body-part sync, and full-speed human acceptance.
B. Resolve reproducible browser-vs-FFmpeg raster parity; do not silently convert timeouts into passes.
C. Recover and vendor/version the missing legacy renderer dependency before reusing V3 caption/VFX/SFX features.
D. Build the real timeline/project graph and generic effect/caption UI after the character/source gates; next implement audio multitrack and objective sync/listening checks.
E. Finish rights ledger, commercial-license validation, export/package checksums and end-to-end hosted CI before any production-ready claim.

## Motion Studio durable product target (2026-10-20)

- Motion Studio canonical requirements are version-controlled at https://github.com/orionrayy/projects/blob/main/docs/MOTION_STUDIO_CANONICAL_TARGETS.md. This is the persistent target contract for follow-up work; reconcile it before declaring any subsystem complete.
- Required character/rigging subsystems: hierarchical bone rigging and character setup; Bone Constraints & Limits; Weight Painting & Mesh Deformation; IK vs FK and authoring UI; Pivot Point & Joint Placement; Bone Mirroring & Symmetry.
- Required animation subsystems: walk and run cycles; stable planted-foot/contact handling; full-body phase coupling across pelvis, torso, shoulders, arms, neck and head; carry/free-arm profiles; overlapping action, anticipation/follow-through, secondary motion, gravity/inertia/weight transfer; smooth transitions and no silhouette crossing/foot sliding.
- Required production subsystems: editable timeline/keyframes/layered tracks; scene/camera/compositing; FFmpeg preview and full render; audio mixer/narration/music/SFX/ducking; smart captions/SRT/transcript; asset rights/provenance; manifests, checksums, contact sheets, logs and reproducible validation.
- Separate Full Motion Analysis research goal: continuous space-time motion evaluation (not screenshots alone), broad-domain benchmarks (daily life, POV, third person, film, anime/cartoon, abstract/nirmana and fractal zoom), held-out video-level evaluation, model integrity and no promotion on insufficient data.
- Current verified distinction as of this checkpoint: v0.8.x remains a procedural engineering prototype. A green geometry/CI suite is not human visual acceptance. v0.8.1 rendered-validation branch adds carry/free-arm SVG preview generation, JSON metrics and hosted FFmpeg artifact flow; actual artifact review, naturalness, renderer parity, full 240-second film integration and editor features remain open until independently verified.
- Execution policy: free-first, hosted Linux when useful, do not assume local Windows/GPU; use explicit authorization/idempotency for external side effects; never bypass host/tool safety; report repository/CI/artifact state with evidence and do not infer production readiness from version numbers.

## Motion Studio v0.8.2 checkpoint — 2026-10-20

- Current review branch in `orionrayy/projects`: `motion-studio-v0.8.2-natural-motion`, head `e36fd96e8a30c1699216a02e585d57b0332afd5b`. Do not merge to `main` without review.
- Hosted GitHub Actions run `39318601556` passed, with artifact `motion-studio-v0.8.2-rendered-review` (ID `12218302959`, retention through 2026-11-02 UTC). Both carry/free previews independently probe as H.264, 1280×720, 24 fps, 192 frames, 8 seconds.
- Carry metrics: minimum projected foot separation 31.82 px; max bone-length error approximately 8.53e-14 px; prop-contact error 0; ground penetration 0; torso angular range 11.47 degrees; head angular range 5.97 degrees.
- Free-arm metrics: wrist travel relative to root 63.29 px near / 104.72 px far; near-elbow angle 114.67–153.01 degrees. Quantitative gates pass, but these metrics do not establish natural motion.
- Contact sheets were inspected. Compared with v0.8.1 the feet are better separated and torso/head motion reads more clearly; arm/character motion is still schematic and is not accepted as natural or production-ready. Human full-speed review, remaining foot/contact/synchronization defects, browser/FFmpeg parity and integration into the full 240-second audio film remain open.
- Product target contract: `https://github.com/orionrayy/projects/blob/main/docs/MOTION_STUDIO_CANONICAL_TARGETS.md`. This remains the durable feature/subsystem checklist: rigging and character setup; hierarchical bones and constraints/limits; weight painting and deformation; IK/FK; pivots/joint placement; mirroring/symmetry; walk/run; overlapping/secondary motion; timeline/editor; camera/compositing; FFmpeg rendering; audio mixer/narration/SFX/ducking; captions/SRT/transcript; asset rights/provenance; full-motion-analysis QA across varied domains.
- Operating rules: free-first; hosted Linux instead of assuming local Windows/GPU; evidence before claims; CI geometry and human visual acceptance are separate gates; external mutations require clear scope, idempotency and post-action verification; never bypass host/tool safety.

## Motion Studio v0.8.3 checkpoint — 2026-10-20

- Current PR candidate: `orionrayy/projects#9`, draft only, head branch `motion-studio-v0.8.2-natural-motion`, head commit `7a6eaa28659a148593bf6d46198cf95407de737e`. Do not merge until review.
- Three hosted workflows passed on the latest PR head: v0.8.1 rendered validation run `39319572230`; v0.8.2 rendered validation run `39319572231`; v0.8.3 temporal-quality run `39319567013`.
- v0.8.3 artifact `motion-studio-v0.8.3-temporal-review`, artifact ID `12218243866`, expires 2026-11-02 UTC. Both previews verified H.264 1280×720, 24fps, 192 frames, 8 seconds.
- v0.8.3 metrics (carry/free): minimum projected foot separation 31.82 px; maximum bone-length error about 8.53e-14 px; ground penetration 0; carry-hand/prop contact error 0; torso angle range 11.47 degrees; head angle range 5.97 degrees; maximum joint acceleration 7.58 px/frame²; maximum knee step 11.35 px; stance-foot drift per consecutive stance frame about 1.14e-13 px. Free-arm wrist range relative to root is 63.29 px near / 104.72 px far; near elbow angle 114.67–153.01 degrees.
- Contact sheets at 4fps were inspected. Leg spacing and torso/head readability are improved over v0.8.1, but animation remains stylized/schematic; metrics cannot approve naturalness. Full-speed human acceptance, motion weight/balance, non-leg synchronization, carry interactions beyond a simple basket, editor functionality, browser/FFmpeg parity, and integration into the 240-second narrated film remain open.
- Persistent subsystem contract remains `https://github.com/orionrayy/projects/blob/main/docs/MOTION_STUDIO_CANONICAL_TARGETS.md`. Continue using it as the product scope checklist rather than assuming absent editor features have shipped.

## Motion Studio v0.8.4 checkpoint — 2026-10-20

- Current PR is https://github.com/orionrayy/projects/pull/9, draft only, base main, current head `f5a95311e442356490555ab200bbe28c8f49f4b6` on branch `motion-studio-v0.8.2-natural-motion`. Do not merge before visual review.
- Latest current-head hosted workflows all passed: v0.8.1 run `39320194284`, v0.8.2 run `39320194273`, v0.8.3 run `39320194222`, v0.8.4 run `39320194280`. Dedicated v0.8.4 run `39320187966` passed too.
- v0.8.4 artifact `motion-studio-v0.8.4-interaction-review`, ID `12219419215`, expires 2026-11-02 UTC. Both carry/free MP4s verified H.264, 1280×720, 24 fps, 192 frames, 8 seconds.
- v0.8.4 behavior: free-arm render omits the unrelated/unheld basket; carry mode applies a cycle-phased basket tilt whose measured range is 5.1566 degrees. Wrist/handle contact error remains 0, maximum bone-length error about 8.53e-14 px, ground penetration 0. Temporal metrics still show stance drift near 1.14e-13 px/frame; max joint acceleration 7.58 px/frame².
- Review contact sheets show cleaner scene semantics: no floating prop in free-arm mode, visible restrained tilt in carry mode. Character movement remains schematic and not accepted as natural/production-ready. Full-speed acceptance, proper weight transfer, shoulder/arm/head synchronization, grip/finger articulation, foot-crossing review, general rigid-body dynamics, editor/rigging subsystems, browser-vs-FFmpeg parity and full 240-second film integration remain open.
- The durable requirements contract is https://github.com/orionrayy/projects/blob/main/docs/MOTION_STUDIO_CANONICAL_TARGETS.md. Keep its entire rigging, IK/FK, weight painting/deformation, mirroring/pivots, walk/run, overlapping/secondary action, timeline, audio, subtitles, asset-rights, and Full Motion Analysis QA checklist as the canonical follow-up scope.


## Hallmark workspace design skill integration (2026-10-29)

Imported the upstream `Nutlope/hallmark` skill package into `.agents/skills/hallmark/` for repository-scoped Codex/agent use. The import is pinned to upstream commit `8f6969f8922caabde9de1784959eae11ddf68073` (2026-10-28) and includes all 182 files from upstream `skills/hallmark/`, including `SKILL.md`, references, theme previews/catalogue, scripts, fixtures, and the MIT license. Repository documentation is in `.agents/skills/README.md` and `.agents/skills/hallmark/UPSTREAM.md`.

Validation is wired through `.github/workflows/hallmark-skill-validation.yml`: the upstream installer is invoked in `--dry-run` mode to validate references/themes without installing into the runner's home directory, and the Node scripts receive syntax checks. No npm dependencies, API keys, paid services, or browser binaries are installed by this integration. The Together AI image-generation path remains optional and is not called automatically.

This package is a workspace-level coding-agent skill, not an automatic registration in `orchestrator/enterprise_runtime/skills.py`. Runtime execution must continue to use the existing `SkillManifest` identity/digest, trust, network, billing, and permission authorization boundary.


## Repository-local skill routing and Document Craft (2026-10-29)

The repository now has an explicit skill allowlist at `.agents/skills/catalog.json`, containing Hallmark v2.0.0 and repository-authored Document Craft v1.0.0. Hallmark's source is pinned to upstream commit `8f6969f8922caabde9de1784959eae11ddf68073`. Before this integration, the repository-local skill catalogue contained Hallmark only; it had no separate academic/document-writing skill. Hallmark is for UI design and is not a document-writing engine.

Runtime integration is connected to the main orchestrator: `orchestrator/llm_planner.py` provides compact skill-routing hints to the planning LLM, then `orchestrator/orchestrator.py` invokes `orchestrator/skill_runtime.py` before DAG validation. The router selects skills by Indonesian/English artifact intent, verifies prompt content against the allowlisted Git blob SHA, builds a `SkillManifest`, registers it in `enterprise_runtime.SkillRegistry`, and authorizes it against trust, network-class and billing policy before injection. Verified prompt text is attached only to relevant LLM-node instructions; node input also records skill ID/version, manifest digest, content blob SHA and prompt path. A `skills.routed` event records the selected routes and a SHA-256 of the goal instead of duplicating the raw goal. Integrity or authorization failures stop routing instead of using unverified prompt content.

Both skills are declared `free_public`, non-side-effecting and network-free for context routing. The router respects `ORCHESTRATOR_FREE_ONLY`; a paid skill cannot be routed when hard-free mode is enabled. Hallmark's optional image-generation API is not called by this runtime integration.

Document Craft covers academic papers, coursework, essays, practicum/lab reports, internship/PKL reports, proposals and professional documents. It targets natural, specific prose and authentic author voice when supplied, while preserving factual accuracy, exact source support, real citations, and the distinction between observations and analysis. It explicitly rejects fabricated lab/field evidence and AI-detector-evasion tactics. When a goal combines UI work and document writing, node-specific routing prevents the two skill contexts from being indiscriminately mixed.

Validation is covered by `orchestrator/test_skill_runtime.py`; the repository skill workflow runs it, and the main orchestrator CI is triggered by changes under `.agents/skills/**`. Skill guidance is context only, not executable code, and does not replace the established DAG, evidence, approval, side-effect, or privacy controls.

## Document Craft corpus-based human-authored benchmark roadmap (2026-10-29)

This is a required next-phase acceptance contract from the user. It is a roadmap/checkpoint, **not evidence that the corpus or benchmark has already been built**.

### Current verified baseline
- Hallmark import PR #175 and runtime/Document Craft integration PR #177 are merged into `main`.
- Latest integration commit: `4d207fe3c420afe27fe66a4c3e5159d46bdf2640`.
- At the time this checkpoint was recorded, GitHub Actions `Orchestrator Tests` and `Hallmark skill validation` passed for that commit.
- Existing `document-craft` guidance already addresses evidence/citation integrity, natural prose, report-specific sections, and checks against repetitive stock transitions and overformatted prose.
- The existing seven runtime tests validate routing, authorization, allowlist/blob integrity, mixed-node routing, and free-only behavior. They do **not** establish empirical writing quality against a benchmark of original human-authored documents.
- No verified multi-genre corpus, generated-vs-original cross-comparison report, format-specific visual-layout benchmark, or human-rating dataset is claimed complete at this checkpoint.

### Required methodology before claiming the skill is empirically improved
1. **Build a provenance-tracked reference corpus from genuine original documents.** Cover multiple examples and real structural variants for (at minimum) LP/asuhan keperawatan, laporan PKL/magang, skripsi/tugas akhir, laporan kegiatan, makalah, laporan kerja/profesional, proposal, lab/practicum reports, and Excel workbooks. Add further formats when evidence supports them. Prefer primary institutional/professional sources, official repositories, publicly released exemplars, and lawfully supplied files. Record source URL or supplied-file origin, creator/organization when known, date/version, document type, license/permission, provenance confidence, and any anonymization.
2. **Do not invent authenticity or provenance.** A generated example must never be labelled original or human-authored. Mark uncertain authorship as uncertain. Exclude private, confidential, or identifiable personal records without proper authorization; use public/de-identified clinical examples for LP askep. Retain citation, attribution, licensing and removal notes.
3. **Stratify by document genre and purpose rather than force one universal template.** Inspect section order, heading levels, paragraph functions, sentence/paragraph variation, terminology/register, use of first person, evidence placement, citation patterns, table/figure locations, captions, page breaks, cover/approval pages where expected, and the visual hierarchy. For spreadsheets, separately inspect sheet purpose, row/column layout, formulas, number formats, merged cells, data validation, conditional formatting, chart placement, and visual density. Learn a format's conventions from multiple examples; do not assume one institution's template is universal.
4. **Cross-compare against outputs produced by the current writing system.** Generate a representative baseline for every document category, preserve inputs/prompts/model/version and artifacts, then compare original and generated documents using a documented rubric. Review recurring model-like habits including repeated rhetorical contrast frames (e.g. “bukan A, melainkan B”), formulaic transitions and openings, prompt restatement, generic grand claims, predictable section symmetry, excessive colons/semicolons/em-dashes, repetitive conclusions, compressed paragraphs, unnecessary headings, and generic visual composition. Treat these as context-dependent warning signs, not banned strings in every situation.
5. **Use multiple quality signals.** Combine source-grounded structural/linguistic analysis, document-layout checks, factual and citation validation, and blind human review by suitable reviewers where available. Report the sample size, category coverage, rubric, inter-reviewer disagreements if applicable, limitations, and baseline-to-candidate differences. Do not invent comparison results or claim statistical significance from inadequate samples.
6. **Convert findings into format-specific skill guidance and regression tests.** Keep a traceable link from observed failure pattern → reference evidence → revised rule/template/check → regression case. Review DOCX/PDF visual renderings and Excel workbook behavior when relevant, not only the prompt text. Make rules conditional on document intent, supplied template and evidence; avoid replacing one rigid “human style” with another rigid template.
7. **Success criterion.** Claim “corpus-based improvement” only after the reference inventory is inspectable, source permissions/provenance are recorded, cross-comparison artifacts exist, regression tests run, and quality differences are honestly reported. Until then, describe Document Craft as a prompt-guidance/routing implementation, with corpus-based evaluation still pending.

### Style and integrity constraints
- The target is document-specific authenticity, readability, accurate structure, useful visual arrangement and an evidence-grounded voice, based on the user's real inputs and genuine reference patterns.
- Do not add intentional mistakes, fabricate a personal experience, imitate a named living author, or claim the document was authored by a human when that is not established. This work is about quality and contextual fit, not evading AI-detection systems or promising detector outcomes.
- Continue using the required repository contracts for privacy, provenance, trustworthy citations, free-first operation, authorization, idempotency and honest verification status.



## SOL Sentinel execution checkpoint — 2026-10-30 (session-2026-10-30-sol-sentinel)

### Repository and integrity baseline
- Main SHA at checkpoint: \`d5672c84a0ae184c3a95d3f925925887ece71aa4\`.
- Implementation branch: \`feat/sol-sentinel-paper-trader-v1\`, audited head \`7d74015ecb99c0f1768baddd40b1643399c5bea2\`; PR #178 remains draft/open/unmerged: https://github.com/orionrayy/ai-determinism-engine/pull/178.
- This checkpoint and the detailed research report are maintained on a separate docs-only branch based on current main; no main modification or merge of the implementation was performed.
- Audit report: \`projects/sol-sentinel/RESEARCH_AUDIT_2026-10-30.md\`, review PR #180: https://github.com/orionrayy/ai-determinism-engine/pull/180.

### Current verified CI and research
- Tests workflow run #268: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928041385; success on candidate SHA \`7d74015ecb99c0f1768baddd40b1643399c5bea2\`; job log confirms 70 tests passed.
- Research workflow run #25: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032346; success on the same SHA. Artifact 12921441578: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032346/artifacts/12921441578; 1,223,457 bytes when inspected, 7-day retention.
- Dataset SHA-256 \`ccc89cec753644f11248bc75ac8b7b13fd9042ef4eb1d5a72074a75cdb9e8b26\`, 43,200 one-minute public Binance SOLUSDT candles spanning 2026-09-29 18:08 UTC to 2026-10-29 18:07:59.999 UTC.
- FX conversion: Frankfurter USD/IDR reference 18,158.0 dated 2026-10-29. It is a denomination proxy for USDT, not an intraday conversion price or Reku execution reference.
- Chronological 60/20/20 splits: development 25,920 bars; validation 8,640; out-of-sample 8,640. All produced zero candidates, entry attempts/fills, open/closed positions, and net P&L. At zero trades, expectancy, win rate and profit factor are undefined/null; drawdown of zero is not evidence of safety.
- Full-window filter reason counts overlap: 1m breakout rejection 40,141; missing 1m relative-volume confirmation 34,800; 5m momentum not confirmed 28,248; 15m trend not confirmed 25,313; 5m RSI outside entry range 23,648; 1m momentum exhaustion/missing 1,787; cost-gate rejection 198. These are rejection events, not unique bars.
- Joint rejection pattern requiring all five conditions (15m trend, 5m momentum, 5m RSI, 1m breakout, 1m volume) failed on 11,543 development, 6,959 validation, and 2,079 OOS observations. This identifies a conjunctive filter bottleneck but does not isolate one single root cause or justify changing thresholds.
- Split cost-gate rejections: 119 development, 43 validation, 34 OOS. Difference from full-window 198 is due to independent split boundaries/warm-up, not a profitability result.
- Scenario comparisons are predeclared, not auto-selected. Reku Pro assumptions: fee components 42.11 bps + 10 bps assumed round-trip spread + 20 bps assumed round-trip slippage = 72.11 bps total, or 97.11 bps including the configured 25 bps net-edge buffer. Current single public Reku quote spread sensitivity was 27.359781 bps; with same fees/slippage modeled cost 89.469781 bps, 114.469781 bps including buffer. It is one snapshot held constant for sensitivity, not a historical spread series.
- The theoretical Lightning fee-only floor (zero spread/slippage) produced 10 trades, net P&L \`-IDR 3,893.219\`, net return \`-1.29774%\`, profit factor \`0.2760\`, maximum drawdown \`1.62433%\`. It is a non-executable mathematical floor, not a realistic trading scenario.

### Spread diagnosis
- Conventional midpoint formula: \`((ask - bid) / ((ask + bid) / 2)) * 10000\`.
- Bid 100 / ask 102 gives 198.01980198 bps; bid 99 / ask 101 gives exactly 200 bps. Historical test mismatch is mathematically consistent with fixture/expectation mismatch. The exact pre-fix failing fixture was not recovered for independent reproduction; avoid claiming that specific pre-fix run was replayed.
- Candidate \`7d74015\` validates missing/nonnumeric/boolean/nonfinite/nonpositive and crossed quotes, accepts numeric strings for API compatibility, uses half-sum midpoint to reduce overflow risk, and tests zero spread plus extremely small/large values. Current CI validates this change.

### Venue-data limits and readiness
- Public Reku snapshot reported bid IDR 1,825,000, ask IDR 1,830,000, 27.359781 bps midpoint spread, order-book status OK and chart probe OK. Snapshot is historical to its capture time, not a live current quote.
- Legacy public Reku \`/v2/activitytrans\` returned HTTP 404; collector fails fast and records \`endpoint_not_found\`. Native Reku trade history remains unavailable/unverified; no substitute or synthetic trades are fabricated.
- Official fee references checked 2026-10-30: https://reku.id/fees/crypto and CFX update effective 2026-10-01 https://reku.id/information/informasi-penyesuaian-biaya-cfx. Binance docs: https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md and https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md. USD/IDR source: https://api.frankfurter.dev/v2/rate/USD/IDR.
- Status is \`RESEARCH_ONLY\` and \`INSUFFICIENT_EVIDENCE\`, not \`PAPER_READY\`. No live orders, account changes, deposits, withdrawals, transfers, private trading API calls or credentials were used. Real-trading adapter must remain disabled.
- Local clone and local test rerun were blocked because this session container could not resolve \`github.com\`. GitHub Actions logs are the test evidence. Android/Termux was not accessible or tested; execute \`projects/sol-sentinel/RUNBOOK_TERMUX.md\` on the real device before claiming compatibility.
- Target ekuitas Rp3.500.000 is aspirational only; no actual balance or target attainment is verified.
- Acceptance gates: reconcile PR #178 with current main without force-push; verify native Reku data/price compatibility or retain explicit limitations; freeze strategy before independent OOS; positive net expectancy and PF at least 1.2 when defined with enough trades (100 trades is a research sample target, not a forcing function); cost-sensitivity resilience; invariant/backtest/SQLite idempotency checks; actual Termux testing; no live mode promotion without separate authorization.


## SOL Sentinel follow-up checkpoint — 2026-10-30 (session-2026-10-30-sol-sentinel-final-followup)

### Implementation corrections after the initial audit
- Ledger hardening commit: \`2cac2368e2a6fa992335a07926a4994770be1e7f\`. \`persist_ledger\` no longer silently ignores conflicting \`run_id\` or trade collisions. Identical retries are no-ops only when schema, dataset hash, complete serialized report, and stored trade rows match; mismatched reports or missing/extra/changed trade rows raise \`ValueError\`. Duplicate/blank trade IDs and non-finite P&L/report fields are rejected; SQLite foreign-key enforcement is enabled.
- Candle-time/finite-input guard commit: \`833e57a13022c3efe6ff5115df2b2e0697072817\`. CSV candles and direct \`backtest()\` input now require a one-minute candle close time equal to \`open_time_ms + 59,999\`; direct bars also reject non-finite OHLCV, non-positive prices, negative volume, and impossible high/low.
- Independent reference-fixture tests commit: \`8fb11f7081a355958add549052b1ecec90464423\`. Added hand-calculated EMA, RSI and ATR checks to complement earlier spread and risk tests.
- Latest branch-head test workflow #278: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40930666859 succeeded; log says 79 tests passed. Two duplicate triggered runs (#277 and #278) both succeeded on the same test commit. Research workflow #27 was executed on production-code commit \`833e57a...\` (the test-only commit came after it): https://github.com/orionrayy/ai-determinism-engine/actions/runs/40930361106; succeeded and uploaded artifact #12922686248 (1,223,413 bytes).

### Latest research evidence (run #27; supersedes run #25 as latest capture)
- Data: 43,200 one-minute Binance SOLUSDT candles, 2026-09-29 18:23:00 UTC through 2026-10-29 18:22:59.999 UTC; dataset SHA-256 \`6e140e14be2e05d30819b96e90e37372ee13df55bb7d4f31fc9c9382bef86c14\`.
- Dated conversion reference: USD/IDR 18,158.0 dated 2026-10-29 from Frankfurter; this is a USDT unit proxy, not an executable price.
- Full-window baseline: 43,199 signal evaluations, 154,149 multi-label rejection events, zero candidates, zero entry attempts/fills, zero closed trades, initial/final equity IDR 300,000, net P&L/return zero due to no positions, maximum drawdown zero, and undefined/null win rate, expectancy and profit factor.
- Fixed chronological split: development 25,920 bars / 92,282 rejection events; validation 8,640 bars / 29,263 events; out-of-sample 8,640 bars / 29,513 events. Each segment again has zero candidates/trades/P&L; this cannot validate expected return.
- Full-window overlapping counters: breakout rejection 40,141; missing relative-volume confirmation 34,801; 5m momentum not confirmed 28,248; 15m trend not confirmed 25,328; 5m RSI out of range 23,648; 1m momentum exhaustion/missing 1,787; estimated net edge below threshold 196. Cost-gate split counters: development 117, validation 43, OOS 34.
- Largest joint filter failure (15m trend + 5m momentum + 5m RSI + 1m breakout + 1m volume) was 11,544 development; 2,215 validation; 2,081 OOS observations. Counters remain overlapping. These show filter conjunctions, not a proven single-threshold defect.
- Current public Reku snapshot again showed bid IDR 1,825,000 / ask IDR 1,830,000, midpoint spread 27.359781 bps; orderbook/chart probes OK. Native transaction history remains unavailable: legacy endpoint returns \`endpoint_not_found\` (HTTP 404).
- Cost sensitivity remains as previously defined: Reku Pro 72.11 bps modeled fees + assumed spread/slippage (97.11 bps including 25 bps edge buffer); one-snapshot spread scenario 89.469781 bps (114.469781 including buffer). The theoretical Lightning fee-only floor (22.11 bps with zero spread/slippage) still yields 10 trades, net \`-IDR 3,893.219\`, return \`-1.29774%\`, PF \`0.2760\`, max drawdown \`1.62433%\`. No cost scenario establishes a tradable edge.
- Operational state remains \`RESEARCH_ONLY\` / \`INSUFFICIENT_EVIDENCE\`. PR #178 remains draft/open/unmerged; its recorded base SHA \`eaa864a...\` differs from current main. No live order/account activity occurred; Termux/Android remains untested. No merge, force-push or main branch modification was performed by this session.
- Current implementation head and 79-test evidence must be kept distinct from research run #27's code SHA. Always include both commit and dataset identities when citing these metrics.


## Final reconciliation and validation — 2026-10-30 (session-2026-10-30-sol-sentinel-final-verification)

### Latest commits and CI status
- Current \`main\`: \`d5672c84a0ae184c3a95d3f925925887ece71aa4\`. The SOL Sentinel candidate PR #178 branch is \`feat/sol-sentinel-paper-trader-v1\`; its head before the last documentation-only append was \`5f18753d4f5f059195b9b2e815627652e6564ca8\`, followed by final reconciliation journal commit \`3d2054da51ebb34f98380c50257e61735e650475\`.
- Latest completed tests at the source/code branch head before that last documentation-only append: [run #280](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40931067383), 79 tests passed, 0 failed (log: 0.115s). Duplicate run #279 also passed. The last append changes only project documentation; inspect its automatically triggered check before stating the newest journal-only commit is green.
- Research [run #27](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40930361106) completed successfully on source implementation SHA \`833e57a13022c3efe6ff5115df2b2e0697072817\`; artifact #12922686248, 1,223,413 bytes. The subsequent independent-indicator test commit is \`8fb11f7081a355958add549052b1ecec90464423\`; the report explicitly separates the research code SHA from the latest test-only/docs commits.
- Docs branch \`docs/sol-sentinel-continuity-20261030\` was based on main SHA \`d5672c84...\`; current audit PR #180 remains draft/open/unmerged. Its latest verified Orchestrator Tests run #2250 succeeded: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40930987014.
- Final canonical root file now has multiple idempotently keyed follow-up markers, and the audit report has an appended section for each later dataset/code correction. Each write was followed by a fresh GitHub file read.

### Branch reconciliation finding (not merge-authorized)
- GitHub compare of current main and PR #178 head reports \`status=diverged\`, merge base \`eaa864a554c2c995ce73bd9f6b4d07206ecc56c5\`, \`ahead_by=167\`, \`behind_by=2\`, and 27 changed files. PR metadata says \`mergeable=true\`, but that is not evidence of safe or reviewed history. The implementation branch is not merged; no rebase, force-push, protection-rule change, or main modification occurred. Perform human review of the complete 167-commit delta and its relationship with the 2 newer main commits before any integration.
- PR #178 stays draft/open; the separate documentation report/memory update is PR #180, also draft/open and based on current main. Neither was merged.

### Ledger and candle integrity
- SQLite previously used \`INSERT OR IGNORE\`, meaning a reused \`run_id\` with altered report data could hide conflicts while trade rows were independently considered for insertion. Commit \`2cac2368e2a6fa992335a07926a4994770be1e7f\` now uses exact report/trade-row comparison: identical replays are no-ops; conflicting report or damaged/missing trade row sets fail closed. It validates duplicate/blank IDs and finite P&L/JSON and enables foreign keys.
- Commit \`833e57a13022c3efe6ff5115df2b2e0697072817\` validates completed 1m candle close-time boundaries and direct OHLCV input, preventing malformed timestamps from exposing later timeframe data during indicator selection.
- Test-only commit \`8fb11f7081a355958add549052b1ecec90464423\` adds hand-calculated EMA/RSI/ATR tests. Latest fully verified test log at this point: 79 passed; exact current journal-only head check is tracked separately.

### Persistence incident and recovery
- The first memory append attempt accidentally retained the old blob due an idempotency check against the already-appended string. A post-write read detected the issue; the checkpoint was rewritten with the new blob, re-read, and checked for markers. This note records the failed attempt and correction so future continuation does not mistake the first tree creation for successful persistence.
- Current verified readiness: \`RESEARCH_ONLY\` / \`INSUFFICIENT_EVIDENCE\`. Zero OOS trades; no valid profitability claim. Native Reku trade history unavailable via legacy HTTP 404; Termux not run; zero live orders or account actions in this session.


## Final compare refresh — 2026-10-30

The last fresh GitHub compare, performed after all current documentation commits were pushed, reports current \`main\` \`d5672c84a0ae184c3a95d3f925925887ece71aa4\` versus feature head \`3d2054da51ebb34f98380c50257e61735e650475\`: \`status=diverged\`, merge base \`eaa864a554c2c995ce73bd9f6b4d07206ecc56c5\`, \`ahead_by=168\`, \`behind_by=2\`, 27 files in PR #178. This supersedes the earlier point-in-time count of 167 commits ahead recorded in previous notes. \`mergeable=true\` is only API metadata and does not authorize merge; keep PR #178 draft and obtain review of the full history delta before integration. No rebase, force-push, merge, or change to main was performed.


## Final fixed-head compare — 2026-10-30 (session-2026-10-30-final-compare-head-7e1b62d)

This was measured after finalizing the feature branch journal at \`7e1b62df19a950f79f58eea2a0a1a118fdb03d2d\`. A fresh GitHub compare against current \`main\` \`d5672c84a0ae184c3a95d3f925925887ece71aa4\` reports \`status=diverged\`, merge base \`eaa864a554c2c995ce73bd9f6b4d07206ecc56c5\`, \`ahead_by=169\`, \`behind_by=2\`, 27 PR-diff files. This supersedes earlier point-in-time counts 167/168 recorded above. Freeze feature branch at this head for this session; do not add more commits while citing this compare snapshot. \`mergeable=true\` is not authorization or a complete review of the 169-commit delta. PR #178 remains draft/unmerged; no rebase, force-push, merge, or main mutation.


---

## SOL Sentinel session checkpoint — current audit, Jupiter quote probe, and infrastructure decision (2026-10-30)
<!-- SOL_SENTINEL_SESSION_2026-10-30_CURRENT_AUDIT_JUPITER_V2 -->

### Verified GitHub state at checkpoint
- Main: `d5672c84a0ae184c3a95d3f925925887ece71aa4`.
- Implementation branch `feat/sol-sentinel-paper-trader-v1`: `0b5dac9f3ac64c1753437fcc4afe96b8171cc079`.
- PR #178 remains open/draft/unmerged: https://github.com/orionrayy/ai-determinism-engine/pull/178. `main...feat` compare is `diverged`, 175 ahead / 2 behind, merge-base `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`, 29 files. Do not merge/rebase/force-push without a deliberate reviewed integration plan.
- Feature CI runs #294 and #295 succeeded on head `0b5dac9f3ac64c1753437fcc4afe96b8171cc079`: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40936815027 and https://github.com/orionrayy/ai-determinism-engine/actions/runs/40936825849, latest relevant head `0b5dac9f...`. Tests source count at that head is 88 across 8 `test_*.py` modules, including 9 new quote-probe tests. The Actions compile and offline unittest-discovery steps succeeded. Decoded Actions job log fetched from GitHub confirms `Ran 88 tests in 0.181s`, `OK` (0 failures) on head `0b5dac9f...`; source count independently matches 88 methods across eight test modules.
- Documentation branch after infrastructure report and research-audit addendum: latest audited commit before this canonical-memory reconciliation: `d7d5b9a1cc1563e4c3e65804c3ee41ee70f35cb0` (verify branch head again before next write). PR #180 remains open/draft/unmerged: https://github.com/orionrayy/ai-determinism-engine/pull/180. It targets current `main`, six older commits before this audit addendum; check latest compare and CI before any merge consideration.
- Docs PR's previous current-head run #2264 / #40936927144 passed on `c1229ea4436074394c1501d0a0bbc552896bf157`. Docs CI #2282 / run ID `40937095243` succeeded on `a8d72ea6b84e577dee18a5fd5a0725b14db3b036`; the later audit CI run #2283 / `40937201903` succeeded on `cf64bce6bd42886418be20fc306f2ccd86606a6d`. Audit run-number correction was then committed at `d7d5b9a1...`; its own CI result must be checked after the current canonical-memory update.
- Repo clone/local full-suite execution was blocked by container DNS resolution to github.com. CI status is read from GitHub Actions API; no claim that the full repository suite was run locally.

### Code change in this session
- Added `projects/sol-sentinel/jupiter_quote_probe.py` (`ed203fcae1ed0176c27959330865e69e2fb282a7`) and `projects/sol-sentinel/test_jupiter_quote_probe.py` (`a0849839638c509cc5289c1c3d84f16c95eb08d7`).
- Added explicit compile coverage in `.github/workflows/sol-sentinel-tests.yml` (`88918e8fe666e26406b1939bda6ad94c9ea9a16d`) and Termux runbook guidance (`0b5dac9f3ac64c1753437fcc4afe96b8171cc079`).
- Probe uses standard library, calls Jupiter Swap V2 `/swap/v2/order` without `taker`, has no signing/submit code, whitelists fields and explicitly returns `execution_enabled=false`, rejects unexpected assembled transaction payload, checks response amount/mint identity, bounds timeout/body, and does not print an API key supplied from environment. It uses raw units; conversion requires verified decimals. The actual live Jupiter quote API was not called, so external endpoint/schema currentness still needs a controlled read-only runtime check.
- Official Jupiter refs: https://developers.jup.ag/docs/swap/order-and-execute ; https://developers.jup.ag/pricing ; https://developers.jup.ag/changelog/developer-platform. Current docs list Free plan at $0 / 1 request per second with API key; keyless is available with lower rate limits. Verify at use time.
- Latest relevant test workflow succeeded on quote-probe + runbook head. This is implementation-test evidence only and not evidence the strategy is profitable or ready for production.

### Research and strategy readiness (historical evidence remains unchanged)
- Last research run #27 passed on earlier code SHA `833e57a13022c3efe6ff5115df2b2e0697072817`, not current head. Dataset SHA `6e140e14be2e05d30819b96e90e37372ee13df55bb7d4f31fc9c9382bef86c14` (43,200 1m Binance SOLUSDT candles, 2026-09-29 through 2026-10-29).
- Results: baseline zero candidates/trades; chronological development/validation/OOS also zero trades. Rejection counters overlap; common failures include 1m breakout and relative volume, 5m momentum/RSI and 15m trend. Do not lower multiple filters just to create transactions. Fee-only and lower-friction sensitivity did not establish a positive edge; the theoretical Lightning floor with zero spread/slippage made 10 trades but net about -Rp3,893 and PF 0.276.
- Existing paper-engine fixes remain: SQLite idempotency checks full report and trade rows, conflicts fail closed; strict candle interval/OHLCV validation; independent EMA/RSI/ATR reference tests. Reviewed signal alignment maps higher-timeframe bars only when their close timestamps are <= decision close. No live trading code was added.
- Current mode stays `RESEARCH_ONLY`; strategy status stays `INSUFFICIENT_EVIDENCE`. No live orders, transfers, deposits, withdrawals, signing, production credentials or private keys used.

### Infrastructure finding and decision
- New decision report: `projects/sol-sentinel/INFRASTRUCTURE_DECISION_2026-10-30.md`, created at `c1229ea4436074394c1501d0a0bbc552896bf157`; it records sources, assumptions, gate criteria and capital math.
- Candidate architecture is **local IDR on-ramp (Reku as currently known candidate) -> dedicated Solana self-custody wallet -> Jupiter V2 quote adapter -> paper execution**. It is not authorized to sign/submit swaps. A wallet provides custody/signing, not market liquidity or a strategy. Avoid asserting Reku is the only or necessarily cheapest venue.
- Current source refs: OJK AKD/crypto trader list (position 22 Sep 2026): https://ojk.go.id/id/Fungsi-Utama/ITSK/Perizinan-ITSK-Aset-Keuangan-Digital-Aset-Kripto/Pages/Daftar-Penyelenggara-Perdagangan-Aset-Keuangan-Digital-Posisi-22-September-2026.aspx; Reku fees https://reku.id/fees/crypto; Pintu Pro fees https://www.pintu.co.id/en/faq/spot-trading-at-pintu-pro; Jupiter docs/pricing above; Solana network fees https://solana.com/docs/core/fees .
- Simplified source-published fees, same nominal Rp300,000 buy and sell: Reku Pro components 0.4211% (~Rp1,263.30), Reku Lightning 0.2211% (~Rp663.30), Pintu Pro maker 0.4611% (~Rp1,383.30), taker 0.5611% (~Rp1,683.30). These exclude spread, slippage, transfer/withdrawal fees, route fees, priority fees and off-ramp effects. Reku/Pintu SOL external withdrawal fee and supported exit path remain unverified live values; do not invent them. Pintu’s official trading API support was not confirmed. Phantom help reports 0.85% app swap fee on most swaps; do not assume UI swaps are free or conflate it with Jupiter direct API costs.
- Solana base transaction fee source: 5,000 lamports/signature; optional priority fees; fees are charged even if transaction fails.
- Goal math for equity (not profit) Rp300,000 -> Rp3,500,000: 11.6667x final capital, requiring +1,066.67% cumulative return before added capital and costs; aspirational only, no timeline/return promise. 10% drawdown at Rp300,000 is Rp30,000 and needs 11.11% recovery; 20% = Rp60,000 and 25% recovery; 30% = Rp90,000 and 42.86% recovery; 50% = Rp150,000 and 100% recovery.
- The full decision memo preregisters gates: chronological OOS unaltered; at least 100 closed trades before profitability claim; positive net expectancy with 95% bootstrap lower confidence bound > 0, PF >= 1.2, max DD within pre-set risk budget; >=30 days forward-paper with timestamped actual quotes and reconciled ledger; operational security/kill-switch checks. A low sample or 0 trades is `INSUFFICIENT_EVIDENCE`. These are prudential project criteria, not a future guarantee.

### Next checkpoint
1. Re-read latest branch refs/PR heads and compare before any code integration.
2. Verify the current Jupiter docs/API with the read-only probe in a user's local Termux environment. Do not put `JUPITER_API_KEY` or wallet secret into repo, Actions, logs, or chat.
3. Capture synchronized quote samples and full path fees including current SOL withdrawal/return compatibility before comparing local CEX to wallet/DEX economics.
4. Diagnose zero-trade strategy via fixed one-filter-at-a-time ablations on development data and freeze candidate before validation/OOS.
5. Keep all execution disabled until strategy, cost, forward paper and security gates are met; any real-money pilot requires separate explicit authorization.


### Follow-up verification after the current-memory checkpoint
<!-- SOL_SENTINEL_SESSION_2026-10-30_POSTWRITE_VERIFY_E9F037A -->

- At the post-write verification instant, docs branch head `e9f037a3855168e707c2f5667aef2c6e954f20a5` had passed current Docs CI run #2290: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40937496241.
- The docs branch compared `ahead` of current main `d5672c84a0ae184c3a95d3f925925887ece71aa4`, 12 ahead / 0 behind, three changed paths (canonical memory, infrastructure decision, research audit). PR #180 remained draft/open/unmerged. This note is a point-in-time verification; check branch/head and CI again after future documentation changes.
- Implementation branch head `0b5dac9f3ac64c1753437fcc4afe96b8171cc079` remained diverged 175 ahead / 2 behind, 29 changed files. PR #178 remained draft/open/unmerged. SOL Sentinel CI #294/#295 succeeded; the latest research run is still #27 on earlier code SHA `833e57a13022c3efe6ff5115df2b2e0697072817`, with dataset SHA `6e140e14be2e05d30819b96e90e37372ee13df55bb7d4f31fc9c9382bef86c14`.
- The 88 unit-test methods are confirmed by source enumeration across 8 test modules; both code compile and unittest-discovery steps succeeded on the current implementation head. No new research backtest on `0b5dac9f...` is claimed.
- The explicit infrastructure decision remains Reku as an on-ramp candidate, isolated self-custody wallet, Jupiter V2 quote-only probe, and no automated signing/submission. Exact current SOL withdrawal fee/limits and an end-to-end IDR off-ramp cost remain unverified.
- Readiness remains `RESEARCH_ONLY` / `INSUFFICIENT_EVIDENCE`. No real orders, transfers, deposits, withdrawals, private keys, or production credentials were used.



### Additional official Reku transfer-availability finding (2026-10-30)
<!-- SOL_SENTINEL_SESSION_2026-10-30_REKU_TRANSFER_STATUS -->

- Official Reku update dated 2 October 2026 says Send/Receive Crypto was restored: https://reku.id/en/information/send-receive-crypto-sudah-dapat-dilakukan.
- Official Reku notice on 2 October 2026 says IDR withdrawals/deposits were reopened: https://reku.id/en/information/layanan-withdrawal-idr-telah-dibuka. Reku reiterated on 3 October that these services were active: https://reku.id/information/buy-sell-direct-deposit-usd-staking-unstaking-kini-sudah-tersedia.
- Thus the published service notices found do not show a blanket suspension preventing the intended Reku -> self-custody wallet route. This does **not** verify the user's current account, SOL-specific chain/network support, current withdrawal minimum/fee, processing time, or return-path deposit acceptance. The fee page does not expose a current SOL-specific value in the parsed view: https://reku.id/fees/crypto. User must confirm on the official app screens before any separately authorized transfer.
- No SOL was bought, no wallet was connected, and no transfer/deposit/withdrawal was made in this session.
- Infrastructure decision memo appendendum saved in `projects/sol-sentinel/INFRASTRUCTURE_DECISION_2026-10-30.md` at commit `d1803c3dd658b1eb957927d34cd6b9c0743c404d`.


## SOL Sentinel adapter hardening checkpoint — 2026-10-30 (session follow-up)
<!-- SOL_SENTINEL_SESSION_2026-10-30_ADAPTER_HARDENING_V3 -->

### Verified code state
- Implementation branch: `feat/sol-sentinel-paper-trader-v1`; verified code head `70b1734f260202fad70a60b31730c5097fb8bb95`. PR #178 remains draft/open/unmerged: https://github.com/orionrayy/ai-determinism-engine/pull/178.
- Latest CI for that exact head: SOL Sentinel Tests run #40942942848, conclusion `success`; runner log: `Ran 116 tests in 0.121s`, `OK`. https://github.com/orionrayy/ai-determinism-engine/actions/runs/40942942848
- Changes include `projects/sol-sentinel/execution_adapters.py`, `test_execution_adapters.py`, `PLATFORM_ADAPTER_DECISION_2026-10-30.md`, scanner/history guards, expanded CI compile coverage, and updated Termux runbook.
- A first run of the initial adapter patch failed two newly authored tests; root causes were an incorrect test fee arithmetic expectation (10 bps on Rp180,100 notional = Rp18.01, not Rp1.801) and a transfer fixture that supplied a deposit minimum while expecting it missing. Test fixture/math were corrected; no validation was weakened. Subsequent run #40941545790 passed 112 tests, and final head run #40942942848 passed 116.
- GitHub compare remains `diverged`, 196 commits ahead / 2 behind, 32 changed files at the latest reconciliation. Do not merge/rebase/force-push this branch blindly. Main has not been changed.

### Implemented adapter contracts
- Platform-neutral protocols now separate market data, capital source, custody/balance lookup, transfer planning, transaction signer, chain submitter, execution and ledger.
- Implemented `IndodaxPublicMarketDataAdapter` for public-only depth and pair catalogue reads; no API key is sent, no account balance endpoint is called, no orders are sent. It validates public metadata, exact market ID, bid/ask levels, bounded response size, network timeout, retry policy and a 0.35s minimum spacing (max theoretical about 171 calls/minute, under INDODAX's documented public cap of 180/minute). Current dynamic pair fields still need a direct real read in the deployment environment before treating their current values as operationally verified.
- Venue metadata maps `base_currency` as quote currency and `traded_currency` as base asset, reading `price_precision`/`pricescale`, `quantity_increment`, `trade_min_base_currency`, and `trade_min_traded_currency`; unit tests cover missing/ambiguous metadata. Both base quantity minimum and quote notional minimum are gated.
- Paper execution uses Decimal amounts, venue/symbol identity, tick/step, minimum quantity/notional, quote identity/age, future clock skew, spread cap, and marketability. Non-marketable limits return `NO_FILL`; repeated identical idempotency intents return the first receipt; changed payload under a reused key fails closed.
- Transfer assessment only checks source/destination network name, explicit verification flags, fee/minimum knowledge, net amount and stale/missing cost evidence. It always reports `transfer_sent=false`. `ExecutionPolicy(LIVE)` continues to reject even when approval flags are set because no live executor exists.
- Explicitly no private key/seed handling, no wallet signer, no signed exchange API, no production credentials, no transfer, no exchange order, and no mainnet execution.

### Input validation fixes
- `sol_sentinel.py`: identical duplicate candles deduplicate; same timestamp with conflicting OHLCV now raises `ValueError` instead of silent last-write-wins.
- `capture_history.py`: malformed kline rows, non-finite quote volume, negative quote volume, fractional/negative trade count, misaligned interval timestamps, nonstandard close time and conflicting duplicates are rejected. Capture still fails closed on insufficient/gapped history.
- `reku_history.py`: activity rows reject fractional side codes and zero/negative notional or quantity, in addition to existing finite/range checks. The legacy Reku activity history endpoint remains known to return HTTP 404; this code path remains unavailable until an official currently supported endpoint is verified.
- These changes distinguish malformed-data errors from strategy no-signal outcomes; they do not alter strategy thresholds.

### Strategy and venue decision — still blocked
- Research result #27 remains the latest verified full backtest, on engine SHA `833e57a13022c3efe6ff5115df2b2e0697072817`, dataset SHA `6e140e14be2e05d30819b96e90e37372ee13df55bb7d4f31fc9c9382bef86c14`, 43,200 Binance SOLUSDT 1m bars; all chronological 60/20/20 segments have 0 trades. New adapter tests are not new evidence of strategy profitability.
- The rejection counter totals are overlapping rejection events, not independent blocked candles. Root cause therefore remains multi-filter rejection plus economic gate, and no single threshold is proven to be the sole culprit. Do not lower thresholds to manufacture entries. Costs remain scenario assumptions, not measured fill friction.
- INDODAX SOL/IDR is the first exchange-adapter candidate because its official market page lists an active SOL/IDR pair, and official Trade API docs distinguish View, Spot Trading, and Deposit & Withdraw permissions. Any eventual automation should use the least permission required; never combine withdrawal permission into the execution key.
- Official INDODAX API PDF (copyright 2026) documents public `/api/pairs`, `/api/depth/$pair_id`, `/api/server_time`, and a public rate cap of 180 requests/minute; examples list pair increment/minimum fields. We reviewed the rendered Public REST page (PDF page 84) and the documented schema. Direct API JSON for `/api/pairs` and `/api/depth/solidr` could not be accessed through the current web connector, and the GitHub CI used mocks, so actual endpoint JSON compatibility is not yet verified in this session.
- Official INDODAX SOL/IDR page snapshot at audit time showed price Rp1,822,124 and 24h volume about Rp5.9bn; this is volatile display data, not a quote used for the backtest or proof of executable spread. https://indodax.com/trade/SOLIDR
- Reku remains an on-ramp candidate only. Its official fee page lists Pro buy 0.1% + 0.00555% CFX; Pro sell 0.1% + 0.21% PPh + 0.00555% CFX; Lightning buy 0% + 0.00555% CFX and sell 0% + 0.21% PPh + 0.00555% CFX. External send fees/minimums vary by network; the current SOL-specific withdrawal amount/minimum was not verified in parsed official content. https://reku.id/fees/crypto
- Jupiter Swap V2 remains a self-custody alternative with quote-only probe implemented, but not an execution adapter. The official Developer Platform states keyless is 0.5 RPS and free API key is 1 RPS; programmatic signing, swap submission, wallet custody, priority-fee and IDR off-ramp costs remain unimplemented/unverified. https://developers.jup.ag/
- Current official Solana docs state base fee 5,000 lamports/signature plus optional priority fees; failed transactions still pay fees. This is not the complete cost of a routed swap. https://solana.com/docs/core/fees
- Indonesian platform legal eligibility and current SOL network deposit/withdrawal availability must be checked per current official account/app before a separately approved transfer. OJK's September 2026 update notes two crypto exchanges and 34 licensed crypto-asset ecosystem entities; do not infer every exchange, token or network route is supported. https://iru.ojk.go.id/iru/WebSite/ArticleList/View/1084_September_2026_Board-of-Commissioners-Meeting-Financial-Services-Sector-Stability-Remains-Resilient-Amid-Global-Economic-Pressures

### Readiness
- Technical contract readiness: improved, with 116 offline tests passing on latest implementation head.
- Strategy readiness: `INSUFFICIENT_EVIDENCE` (zero historical trades).
- Operational readiness: `RESEARCH_ONLY`; public INDODAX endpoints not directly verified from this session; Termux still not tested.
- Live readiness: blocked. No live executor, no approval gate completed, no actual order/transfer or wallet signing.
- Next: run an isolated real *public read-only* INDODAX metadata/depth smoke test from an allowed network; inspect response fields and server-time freshness; then add read-only contract check. In parallel, diagnose signal rejection combinations and preserve fixed parameters for validation/OOS. Update the two PRs only after checking their current head/base and CI; no merge or base rewrite without reviewed merge plan.


## SOL Sentinel continuation checkpoint — adapter hardening and cross-venue boundary (2026-10-30)

- Current implementation head checked at the end of this pass: `5f2acac4bddfa3dc72c4dea3bc429e14de9d6172` on `feat/sol-sentinel-paper-trader-v1`. Current SOL Sentinel Tests run #40945972566 succeeded on that head: 120 offline tests, 0 failures. CI URL: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40945972566. Compile and offline unit-test steps passed. The suite does not execute live API calls.
- New `projects/sol-sentinel/execution_adapters.py` defines venue-neutral contracts for market data, capital source metadata, custody balance lookup, transfer planning, order execution, signer boundary, chain submission boundary, and ledger. Only mock market/custody fixtures, a read-only public INDODAX market-data adapter, an in-memory idempotent ledger and deterministic paper execution are implemented. There is no exchange order client, no credentialed account adapter, no wallet signer implementation, and no transaction submitter.
- INDODAX adapter loads the official public `/api/pairs` metadata and public `/api/depth/{pair_id}` order book using bounded timeouts/retries, response-size checking, request pacing, quote parsing and venue/quote identity checks. Official INDODAX documentation describes public rate limit 180 requests/min and the pair metadata fields; see https://indodax.com/downloads/INDODAXCOM-API-DOCUMENTATION.pdf. The adapter is validated only with mocked transport in CI: a real read-only endpoint smoke test could not be completed through this session's browser connector. Do not call the platform data integration production-verified.
- Regression hardening added: conflicting duplicate Binance scanner/history candles fail closed instead of silently overwriting; capture validates OHLCV, candle boundaries, quote volume and integer trade count; Reku activity parser rejects fractional transaction side values and zero/negative price/notional/quantity rows; transfer planner rejects non-boolean "network verified" flags, stale or absent fee evidence, mismatched networks, unknown costs, and plans leaving no transfer value. Adapter tests cover tick/lot/minimum checks, stale/future quotes, spread ceilings, idempotency conflicts, non-marketable no-fill and the live-mode kill gate. Do not treat mock fills as actual venue fills.
- Two initial adapter test runs exposed fixture expectation failures; the first was a fee arithmetic fixture (IDR 18.01 at 10 bps on IDR 18,010 not IDR 1.801), and the next reflected the planner's actual explicit zero-remainder reason/value. The fixtures were corrected to match the documented Decimal arithmetic and fail-closed planner contract; final run #40945972566 then passed 120/120. No production condition was removed to make tests green.
- Current official fee/venue check: INDODAX's support article says new all-in fee structure effective 1 October 2026; current exact fee detail is shown in the user's Profile/Trading Fees screen. The same help page states INDODAX Lite minimum IDR 10,000 and direct INDODAX Pro minimum IDR 25,000, with Pro orders IDR 10,000–24,999 routed through Lite: https://help.indodax.com/hc/en-us/articles/4416646599705-Details-of-Transaction-Fees-on-INDODAX. This is separate from the per-market `trade_min_base_currency` field; the eventual mode-specific floor must be applied explicitly in the simulator. Withdrawal charge/minimum remains network-specific and unverified in this run.
- Reku's current public coin-status page lists SOL deposit and withdrawal as Active, and Reku's official notice historically specifies SOL withdrawals through the Solana network. Current per-account eligibility, exact SOL fee/minimum, and INDODAX's current Solana network deposit acceptance still need confirmation in official screens. The intended route remains a candidate only; no transfer was made.
- Jupiter docs confirm its Free API key tier at 1 request/second and keyless access at 0.5 request/second: https://developers.jup.ag/pricing and https://developers.jup.ag/changelog/developer-platform. Solana's official fee documentation lists 5,000 lamports per signature plus optional priority fees, and fees may still be charged for failed transactions: https://solana.com/docs/core/fees. Sentinel's Jupiter probe remains quote-only.
- Latest verified strategy result is still research run #27 on earlier strategy code SHA `833e57a13022c3efe6ff5115df2b2e0697072817`, dataset SHA-256 `6e140e14be2e05d30819b96e90e37372ee13df55bb7d4f31fc9c9382bef86c14`, 43,200 SOLUSDT one-minute candles, and zero trades in development/validation/out-of-sample. No new full research run on the current branch head is claimed. Keep strategy `INSUFFICIENT_EVIDENCE` / `RESEARCH_ONLY`.
- PR #178 remains draft/open/unmerged. Current compare to main is diverged: 207 commits ahead / 2 behind, 32 changed files, merge base `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`. PR #180 remains draft/open/unmerged and is the canonical-memory/audit handoff. No merge, rebase, reset, force-push, transfer, signing, account action, or live order was performed.
- Next gates: run a real public read-only INDODAX `/api/pairs` and `/api/depth/solidr` smoke test in an allowed network; confirm actual pair tick/quantity step/minimum and current all-in fees from official app display; verify SOL network on the INDODAX deposit screen and Reku withdrawal preview without sending; run Termux compatibility test; diagnose zero trades separately from bugs; and design mode-specific venue costs without selecting parameters from OOS. Do not implement live order or wallet signing until these gates and a separate explicit approval are satisfied.


## SOL Sentinel checkpoint — verified native IDR capture and currency-safe paper gate (2026-10-30)

- Current implementation branch head at checkpoint: `39d093b6e44635431475ea635f018327310793fd` (`feat/sol-sentinel-paper-trader-v1`). Exact-head push CI [#40949432502](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40949432502) succeeded; decoded runner output: **135 offline unit tests, 0 failures**. PR-triggered duplicate run #40949443270 was still in progress at the point checked. Compile and offline tests do not prove trading profitability or live execution.
- Public INDODAX data access was smoke-tested through GitHub Actions in successful research run [#40948530609](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40948530609), on code/workflow SHA `c088d6bdf2e387ee267149db44a393da3f26ef08`: 30 days / 43,200 contiguous 1-minute native SOL/IDR candles. Dataset SHA-256 `4297c660ed1c3485bb5641babb5a286b8738290e88f8bb18c9f58bb903fa687b`; interval 2026-09-29 20:24:00 UTC through 2026-10-29 20:23:59.999 UTC; `data_status=OK`, quote currency IDR, `fx_conversion_applied=false`, pair ID `solidr`, source `https://indodax.com/tradingview/history_v2`. An earlier smoke run #40948205968 also collected 43,200 bars successfully on an earlier 30-day window. This is a real public-readonly smoke test from GitHub's runner, not from Android/Termux or the user's account. It does not verify any SOL deposit network on the account.
- The contemporaneous public pair metadata showed price tick IDR 1, quantity step 1e-8 SOL, per-market quote minimum IDR 10,000 and minimum base quantity 0.00559952 SOL. These are captured public market filters, not a current account/mode confirmation. Official INDODAX help states Lite minimum IDR 10,000 and direct Pro minimum IDR 25,000, with Pro-interface orders IDR 10,000–24,999 routed via Lite; current all-in fee components must be read from the account's Profile/Trading Fees screen. Do not hard-code an assumed INDODAX fee tier.
- Cross-venue currency bug addressed: Binance SOLUSDT research input is explicitly labeled USDT and gets a dated USD/IDR proxy; native INDODAX SOL/IDR input is labeled IDR and uses identity rate 1.0 (no FX conversion). Backtest reports record venue, pair, currency, and fee model. Capture sidecars with non-OK `data_status` or mismatched `csv_sha256`/`dataset_sha256` are rejected. Cost-sensitivity presets using Reku Pro/Lightning assumptions refuse native INDODAX data. Native INDODAX backtests require `--fee-model indodax-custom`, explicit finite non-negative entry/exit fee, spread/slippage inputs, a positive minimum-notional value and a non-empty dated source label. Zero/NaN costs or zero order minimums fail closed. This prevents accidental reuse of Reku costs but does not verify the fee inputs supplied by a user.
- Native history capture file is `projects/sol-sentinel/capture_indodax_history.py`; adapter and regressions are `projects/sol-sentinel/execution_adapters.py`, `test_execution_adapters.py`, and `test_capture_indodax_history.py`. The research workflow includes native IDR capture as a read-only artifact step with 7-day retention; it does not automatically backtest INDODAX until venue-specific fees are verified and explicitly supplied.
- Latest full 30-day completed research evidence: run [#40948530609](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40948530609), Binance SOLUSDT dataset SHA-256 `08cb2e654c42e198d71817d2e92d7c354f4ef0206a7e9a352a2eff4b486395ae`, 43,200 one-minute bars. Baseline and all fixed chronological 60/20/20 segments yielded zero candidates and zero closed trades; the hypothetical Reku-Lightning fee-only floor yielded 10 simulated trades and net P&L about IDR -3,893.22 while assuming zero spread/slippage, so it is a mathematical floor only. Reku native `/v2/activitytrans` still returned HTTP 404 during that run; the separate public market snapshot was `data_status=OK`. A new full research run on current commit `39d093b6...` was pending at this checkpoint; do not claim results until its exact run and artifact complete and are inspected.
- Safety/readiness remains `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`. There is no live order adapter, no signer/submitter, no account-auth adapter, and no live approval path; `ExecutionPolicy(LIVE)` deliberately blocks. No order, transfer, wallet connection, signing, private key, or account credentials were used. Termux has not been tested.
- PR #178 remains draft/open/unmerged. Its branch history is intentionally not reconciled destructively with `main`; the last refreshed compare before this newest fee-guard test/docs-only writes was diverged 235 ahead / 2 behind, merge base `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`, 34 files. Refresh compare before any eventual merge. PR #180 remains draft/open/unmerged and is the canonical memory/audit handoff.
- Next gates: inspect completion/artifacts for current-head research run; confirm INDODAX account-mode fees, mode minimum and current market filters from official display; confirm the exact Solana network on both Reku withdrawal preview and destination deposit screen without sending anything; run the documented tests on real Termux; run venue-specific native IDR backtest/evaluation with measured cost evidence; diagnose signal-rejection combinations without optimizing on OOS; require a forward-paper sample before any live approval. No live order or wallet signing until separate explicit approval and all readiness gates pass.


## Reconciliation update — full 30-day research run #57 completed (2026-10-30 WIB)

- The previously pending research run on pre-test-update commit `cb3ebab0b22a82f5ed1c9a6e748cfb5930d1439a` has now completed successfully: [SOL Sentinel Research #40949409224](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40949409224). The strategy/backtest core in this commit is unchanged at the current implementation head `39d093b6e44635431475ea635f018327310793fd`; the intervening change was a regression-test adjustment. Exact-head offline test CI at the latter SHA also succeeded in [#40949432502](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40949432502), 135 tests, 0 failures.
- Binance SOLUSDT source: 43,200 1-minute bars; dataset SHA-256 `cb685aa993bbde9e2abec8c9c350e98097ba97d6b7c50fa323de1921ed1a7a47`; range 2026-09-29 20:29:00 UTC to 2026-10-29 20:28:59.999 UTC. Baseline: `signal_candidates=0`, entry attempts/fills=0, closed trades=0, net P&L 0 only because no position was opened; profit factor/expectancy/win rate are undefined. Fixed chronological 60/20/20 development/validation/OOS also each yielded 0 trades.
- Full-window overlapping rejection counters: 25,424 `15m_trend_not_confirmed`; 40,161 `1m_breakout_not_confirmed`; 1,786 `1m_momentum_exhaustion_or_missing`; 34,801 `1m_volume_confirmation_missing`; 28,329 `5m_momentum_not_confirmed`; 23,759 `5m_rsi_outside_entry_range`; and 195 `estimated_net_edge_below_threshold`. These are overlapping failure events, not independent counts and not evidence of a single root cause. The strategy had zero candidates before order sizing/fill, so zero trades are a strategy/filter outcome in this dataset, not a broken fill ledger; do not relax thresholds solely to force trades.
- Native INDODAX SOL/IDR capture in the same successful run: 43,200 bars, `data_status=OK`, 30-day range 2026-09-29 20:30:00 UTC to 2026-10-29 20:29:59.999 UTC, dataset SHA-256 `08f20992fed30c0921dee547e4cb4623c4dbf3b3c74ef732ce895ebc1fcf5633`, pair ID `solidr`, quote currency IDR and no FX conversion. The native data were captured but not backtested with INDODAX-specific costs because fee/mode assumptions must be explicitly provided; do not substitute Reku fee presets.
- This same run again confirms the legacy Reku `/v2/activitytrans` endpoint returns HTTP 404 while the separate Reku public market snapshot returns `data_status=OK`. This does not prove withdrawal is unavailable; it means this history endpoint is not a valid source.
- The new research workflow on code/test head `39d093b6...` is run #58 [#40949432867](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40949432867), still in progress at the last check. Do not claim its results until completion. Its backtest core is the same as the successful run #57; the latest code change between those two commits is test-only.


## Latest native INDODAX depth + history proof — current research run #60 (2026-10-30 WIB)

- Latest full successful research run on current engine logic: [run #40950218320](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950218320), workflow/collector head `89a8d7181954a1524379d3b03fa77e703eb9331d`, completed successfully. It includes the best-effort public INDODAX depth snapshot in the same read-only native-history collector. Latest test CI on current feature branch head at the time of this checkpoint: [run #40950649418](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950649418) succeeded with 137 tests / 0 failures.
- Exact native IDR dataset for run #60: 43,200 contiguous 1-minute SOL/IDR candles over 30 days; SHA-256 `c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30`; range 2026-09-29 20:36:00 UTC through 2026-10-29 20:35:59.999 UTC. Pair ID solidr; quote currency IDR; data_status OK; FX conversion false.
- Live public INDODAX depth snapshot in that same run: market_quote_status OK, market_quote_error null, request-start timestamp 2026-10-29T20:37:47.062073Z; bid IDR 1,777,113; ask IDR 1,779,923; midpoint IDR 1,778,518; spread 15.799671411815905 bps. orders_enabled=false. This is a single point-in-time public quote, not a historical spread series or actual fill.
- A separate earlier snapshot from run #40950150440, about two minutes earlier, recorded bid IDR 1,775,886 and ask IDR 1,780,450 (spread about 25.6669 bps). The difference reinforces why a single quote must not be extrapolated across the 30-day backtest.
- Public pair metadata in run #60 showed API minimum quote notional IDR 10,000, minimum base quantity 0.00563136 SOL, price tick IDR 1, quantity step 1e-8 SOL. These are point-in-time market filters, not a user/account-specific mode fee rule. Official help still distinguishes Lite minimum IDR 10,000 and direct Pro minimum IDR 25,000; all-in fee components are in the account's Profile/Trading Fees view.
- Binance SOLUSDT research in the same successful run used 43,200 1m bars, dataset SHA-256 `bcc67c412280453fc5c499f9e0179ab8a118b331766470f0b59a8a18f62305e7`; baseline and all fixed 60/20/20 splits still had zero candidates and zero closed trades. The theoretical Reku Lightning fee-only floor again yielded 10 simulated trades and net P&L about IDR -3,893.22 with zero spread/slippage assumed. No strategy threshold was relaxed and this is not an executable cost estimate.
- The Reku public market snapshot returned data_status OK in this run; the separate legacy public activity-history collector still returned HTTP 404 and was deliberately handled as a non-fatal collector failure by continue-on-error. This is a history endpoint problem, not evidence that all Reku transfers are unavailable.
- Correction to prior checkpoint wording: before run #40950150440, INDODAX /api/depth had not been smoke-tested live; the earlier live proof covered /api/pairs + TradingView history. Runs #40950150440 and #40950218320 now independently exercised the public depth endpoint successfully from GitHub Actions. This still does not test Android/Termux or private order endpoints.
- Current source code remains on branch feat/sol-sentinel-paper-trader-v1; PR #178 is draft/open/unmerged, with history diverged 239 ahead / 2 behind, merge base eaa864a554c2c995ce73bd9f6b4d07206ecc56c5. PR #180 remains draft/open/unmerged and carries the memory/audit handoff. No merge/rebase/reset/force-push, wallet access, transfer, signing, or live order was performed.
- Readiness remains RESEARCH_ONLY / INSUFFICIENT_EVIDENCE. Before native INDODAX backtest, verify actual account fee components and mode minimum, then supply explicit native venue fee/spread/slippage inputs. The native data itself was captured; it was not backtested with unverified INDODAX costs. Termux remains untested.


## Latest audit refresh — 2026-10-30 WIB (current branch snapshot)

- Implementation PR #178 current head checked: `479f2adc5bc4d401ca107ce1c044610e0a45c0d2`. Its latest exact-head CI [#40951133963](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40951133963) completed successfully: **137 offline unit tests, 0 failures**. This is software regression evidence only.
- Current source branch compare to `main`: `status=diverged`, 246 commits ahead / 2 behind, 34 changed files, merge base `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`. PR #178 is still draft/open/unmerged. Do not merge, rebase, reset or force-push without reviewing the complete history delta.
- Successful 30-day research capture [run #40950218320](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950218320), at code checkpoint `89a8d7181954a1524379d3b03fa77e703eb9331d`, produced artifact [#12931446637](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950218320/artifacts/12931446637). Native INDODAX SOL/IDR capture: 43,200 one-minute candles, `data_status=OK`, SHA-256 `c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30`, range 2026-09-29 20:36:00 UTC to 2026-10-29 20:35:59.999 UTC, no FX conversion. Capture is public read-only; no orders or transfer.
- Same research run's public depth quote reported `market_quote_status=OK`, bid Rp1,777,113, ask Rp1,779,923, midpoint Rp1,778,518, spread 15.7996714 bps, at 2026-10-29 20:37:47.062 UTC. A preceding capture [#40950150440](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950150440) measured about 25.6669 bps. These are two point observations, not an intraday/historical spread distribution; do not treat either as a stable cost assumption.
- Latest full 30-day strategy run still applied the Binance SOLUSDT/USDT reference data to the frozen strategy (dataset SHA-256 `bcc67c412280453fc5c499f9e0179ab8a118b331766470f0b59a8a18f62305e7`). Baseline and all chronological 60/20/20 segments: zero signal candidates, zero closed trades, net P&L zero due to no positions. A theoretical Reku Lightning fee-only floor had 10 simulated trades but lost approximately Rp3,893.22 while assuming zero spread/slippage. No profitability evidence; do not relax filters to manufacture trades.
- Native INDODAX SOL/IDR data was captured, but it was **not yet backtested using verified INDODAX account/mode fee, spread and slippage inputs**. Reku legacy native history endpoint continues returning HTTP 404; the optional collector is non-fatal. Reku public quote snapshot remains separate and reports OK.
- PR #180 documentation handoff remains draft/open/unmerged; docs branch is 31 commits ahead / 0 behind `main`. The Sentinel continuity entry is in PR #180's branch, not canonical `main` memory yet. Termux remains untested. Strategy and live readiness stay `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`.

## SOL SENTINEL CHECKPOINT UPDATE — 2026-10-30 (native INDODAX replay and Termux hardening)

Checkpoint verified during the 2026-10-30 continuation session. This entry supplements (does not delete or rewrite) earlier Sentinel notes below. Re-read current PR head and workflows before the next write.

### Repository and pull request state

- Repository: orionrayy/ai-determinism-engine.
- Main SHA at audit: d5672c84a0ae184c3a95d3f925925887ece71aa4. The canonical memory in main did not yet contain this Sentinel checkpoint.
- PR #178, branch feat/sol-sentinel-paper-trader-v1, remains open/draft/unmerged. The feature branch head at final audit was 91e421e8fa9460e7d86bbdf760b928ce0cb3ab34. An earlier ancestry comparison with head 8ceeb6a2feb96691d9d323ced053c3a294869174 showed diverged history, 261 commits ahead / 2 behind main, 37 changed files. Recompute against the latest head before any reconciliation; do not merge, rebase, reset or force-push blindly.
- PR #180, branch docs/sol-sentinel-continuity-20261030, remains open/draft/unmerged, based on main d5672c84a0ae184c3a95d3f925925887ece71aa4. This root-memory entry is on the documentation PR branch; it is not yet in main.
- No main branch update, merge, rebase, force-push or destructive history reconciliation occurred in this session.

### Native INDODAX history and backtest

- Original native dataset artifact: workflow #40950218320, artifact #12931446637. CSV SHA-256 c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30. It contains 43,200 contiguous one-minute candles from 2026-09-29T20:36:00Z through 2026-10-29T20:35:59.999Z; venue INDODAX, pair SOL/IDR, quote currency IDR, fx_conversion_applied=false. Local verification checked file hash, timestamp continuity, duplicates/gaps, close-time boundaries, finite OHLCV and OHLC invariants.
- Important correction: the historical artifact's original backtest/evaluation JSON was Binance SOLUSDT plus Reku fee assumptions. It was not an INDODAX native-cost evaluation. The dedicated INDODAX replay was added and completed in workflow #40993527618, commit b1ff4fe57bfb3118e31bc5314b21c5baa91732da, artifact #12949927917.
- Frozen native replay 60/20/20: all three Pro sensitivities had zero candidates/fills/trades across development, validation and OOS. Total modeled friction was 87.909671 bps for Pro taker with 15.799671-bps spread and 5-bps/side slippage; 107.7769 bps for taker stress; 67.909671 bps for maker sensitivity. Formula: entry fee + exit fee + round-trip spread + 2 × per-side slippage. Prior conversational summary mistakenly gave 82.11 and 72.11 bps; those totals are wrong and the correct values are recorded here and in INDODAX_FEE_MODEL_2026-10-30.md.
- The research workflow was then made self-contained: it uses the native CSV captured in the same run rather than re-downloading a fixed artifact that expires after seven days. Run #40994319663 succeeded on workflow commit 92810145ac60487a87730c977f3b26c6bb3a4ecb, artifact #12950538188.
- Same-run dataset SHA-256 909b24977537110d2bd0ca114d5715d29646eb36b6ed9a268bf9470b64a05c59; 43,200 contiguous bars from 2026-09-30T02:54:00Z to 2026-10-30T02:53:59.999Z; native SOL/IDR IDR prices, no FX conversion. Quote snapshot 2026-10-30T02:55:48.533456Z: bid Rp1,729,198, ask Rp1,729,703, spread 2.920003 bps. This is one snapshot, not historical spread evidence.
- Same-run Pro sensitivities: taker current snapshot 75.030003 bps; taker stress 107.7769 bps; maker current-snapshot sensitivity 55.030003 bps. Taker cases produced zero signals/trades across all splits. Maker produced one development-only simulated closed trade and zero validation/OOS trades. The single development trade lasted nine 1m bars, net +Rp466.818 after Rp574.698 exchange fees and Rp175.991 modeled friction. One trade cannot establish an edge; 100% win rate and infinite PF are non-informative with n=1. No scenario produced an OOS trade; expectancy/win rate/PF for OOS remain undefined.
- Current-run taker rejection-event counters overlap: breakout 40,510; relative-volume confirmation 35,882; 5m momentum 28,299; 15m trend 25,989; 5m RSI 23,415; 1m exhaustion/missing 3,142; net-cost edge gate 143. These are reason events, not unique rejected bars; no filters were weakened.

### Venue fees and execution assumptions

- Official INDODAX Pro IDR all-in rates checked 2026-10-30 and recorded in projects/sol-sentinel/INDODAX_FEE_MODEL_2026-10-30.md: maker buy 10.555 bps, taker buy 20.555 bps, maker sell 31.555 bps, taker sell 41.555 bps. All-in includes CFX and sell-side tax; do not add those components twice. Direct Pro minimum Rp25,000; Lite minimum Rp10,000; Pro UI orders Rp10,000–24,999 route via Lite. Lite exact all-in account fee was not verified.
- Public INDODAX pair metadata is time-varying: tick Rp1, quantity step 1e-8 SOL, API notional Rp10,000; minimum base quantity differs between captures. Pair API limits do not replace selected app mode minimum.
- Spread evidence is sparse snapshots only (15.799671 bps, 25.6669 bps, 2.920003 bps). Slippage 5/10 bps per side is an explicit scenario assumption, not calibrated. Maker path lacks queue/post-only/maker-vs-taker fill model. Cost model remains unvalidated empirically.
- INDODAX official fee source: https://help.indodax.com/hc/id/articles/4416646599705-Rincian-Biaya-Transaksi-di-INDODAX . More details and effective-date context are in the fee audit file. Profile → Trading Fees and order preview must be checked for the actual user's account and mode.

### Code, testing and operations

- Added projects/sol-sentinel/termux_doctor.py and test_termux_doctor.py (stdlib only): runtime/Termux detection, Python version, standard library, memory/disk, core compile, SQLite integrity/close-reopen/idempotency, optional public INDODAX pair metadata GET, and streaming CSV/hash/sidecar/timestamp/OHLCV validation. Runbook: projects/sol-sentinel/RUNBOOK_TERMUX.md.
- Added CI compilation and 6 regression cases for the doctor; tests #40994692715 passed 146 tests. Test workflow #40994852707 passed the explicit offline doctor smoke step (--skip-network). This validates the GitHub runner only. The user's Android/Termux handset has not been accessed or tested.
- Paper adapter now includes optional best-level bid/ask quantities, aggregates quantity at the best price, and simulates capped partial fills as PARTIALLY_FILLED_SIMULATED. Tests cover partial, zero and invalid sizes. It does not walk multiple book levels, persist/resume residual quantity or model maker queues.
- Previously, paper execution was full-or-no-fill; this top-of-book partial-fill hardening is now in code but needs further simulation and integration with persistent order lifecycle.
- Scanner watch mode logs read-only JSONL signal records; it is not yet full forward-paper monitoring with venue-aware quotes, portfolio state, persistent partial orders, risk recovery and reconciled P&L. SQLite paper ledger exists but durable end-to-end forward replay/monitor integration still requires proof.
- Jupiter Swap V2 probe remains quote-only, without taker/sign/submit. Existing fields and token mint/amount checks are partial; expiry, slippageBps, routePlan/liquidity, priority fee, rent and complete impact/cost validation are outstanding. A quote is neither an execution nor a guaranteed fill.
- Reku remains an initial capital acquisition/on-ramp role only, not the required bot trading venue. Its public quote endpoint returned data, but legacy public activity history returned HTTP 404; no authenticated purchase/order integration is verified. Transfer planner remains non-sending. Source/destination asset-network compatibility, withdrawal fee/minimum, destination deposit minimum and arrival requirements are unverified.
- No signer, wallet key, API secret, order API, real order, purchase, transfer, deposit, withdrawal or blockchain transaction was used. Live policy remains fail-closed.

### Current readiness decision and next gates

- Decision: RESEARCH_ONLY / INSUFFICIENT_EVIDENCE. No OOS trades; single development-only maker trade is not validation. Rp3,500,000 target remains aspirational, not a return forecast.
- Next: inspect latest PR head and workflow states; run repeated time-stamped INDODAX depth capture; model multi-level depth walk and residual partial orders; validate account-specific mode fees and current pair filters; finish Jupiter V2 response validation; verify both-side Solana network/deposit/withdrawal constraints without moving funds; run doctor on actual Termux; implement persistent forward-paper account/ledger/monitor and gather a meaningful sample; rerun OOS only against frozen parameters.
- All new reports/commits and workflow links are recorded in projects/sol-sentinel/EXECUTION_LOG.md and INDODAX_FEE_MODEL_2026-10-30.md on the feature branch. Start the next room by reading this entry and verifying branch heads/workflows rather than repeating old capture/replay work.

## SOL SENTINEL FOLLOW-UP — latest repeated native INDODAX window #40994685989 (2026-10-30)

This later completed run supersedes the prior one-trade maker-development sensitivity as the latest rolling-window result. It is an important stability check: the one trade in workflow #40994319663 did not recur in the following 30-day window.

- Research workflow #40994685989 succeeded on feature code commit 50c0c00dc4bc5a78ab212dc20bd4894fb99783e6; artifact #12951295388.
- Dataset SHA-256 d76434d4c9019249e924e72c5b05d0b200c0dfe3c4709c02a6d821fa0b22327c; 43,200 contiguous 1m native INDODAX SOL/IDR candles from 2026-09-30T03:01:00Z to 2026-10-30T03:00:59.999Z; no FX conversion.
- Public depth snapshot at 2026-10-30T03:02:06.143046Z: bid Rp1,737,289, ask Rp1,739,906, mid Rp1,738,597.5, spread 15.0523626084 bps. Public filters: tick Rp1, quantity step 1e-8 SOL, API min notional Rp10,000, minimum base quantity 0.00576894 SOL.
- Pro taker snapshot total modeled friction 87.162363 bps; taker stress 107.7769 bps; maker snapshot 67.162363 bps. Formula remains entry fee + exit fee + spread + twice per-side slippage. Spread still a single point and slippage is an assumed 5/10 bps per side.
- All three scenarios (taker current snapshot, taker stress, maker current snapshot) produced zero candidates, zero attempts/fills, zero positions/trades in development, validation and OOS. No estimable expectancy, win rate or profit factor. P&L zero solely reflects no trades.
- Overlapping taker-current rejection reasons: breakout 40,506; volume confirmation 35,873; 5m momentum 28,306; 15m trend 25,981; 5m RSI 23,415; 1m exhaustion/missing 3,135; net-edge cost gate 143.
- In #40994319663 the maker sensitivity had one development-only simulated trade (+Rp466.818 net) and none in validation/OOS. It was not reproduced on the newest rolling window, reinforcing that this isolated n=1 result was unstable, not evidence of a reproducible edge.
- Latest public Reku snapshot: bid Rp1,725,000 / ask Rp1,730,000, spread 28.94356 bps, 40 book levels each side. Five-minute chart probe returned 75 points; interval/timestamp interpretation remains unverified. Legacy activity-history route remained HTTP 404.
- Latest CI suite #40994692715 passed 146 offline tests. Termux doctor smoke step #40994852707 passed with network disabled. Both ran on GitHub-hosted CI, not the user's Android handset.
- Project implementation log has the same latest update. Strategy state remains RESEARCH_ONLY / INSUFFICIENT_EVIDENCE. The single earlier dev trade is superseded; no OOS trade exists. No real order, purchase, signer, transfer, deposit or withdrawal occurred.
## Final repository snapshot — 2026-10-30 10:07 WIB

- Feature PR #178 latest head after the final project README/continuity update: 459eb008e1331c464f8a0d3ca2b6431a7eb7564b. Latest compare against main d5672c84a0ae184c3a95d3f925925887ece71aa4: diverged, 265 commits ahead / 2 behind, 37 changed files. PR remains draft/open/unmerged. Do not merge or rebase without a full ancestry and file-by-file conflict/loss review.
- Docs PR #180 current head before this append: ae1d50f00c2ca690bc74bc450d9141943e4e4408; base main d5672c84a0ae184c3a95d3f925925887ece71aa4; compare shows 38 commits ahead / 0 behind, 3 changed files (ORCHESTRATION_MEMORY.md, INFRASTRUCTURE_DECISION_2026-10-30.md, RESEARCH_AUDIT_2026-10-30.md). It remains draft/open/unmerged.
- Feature head 459eb008e1331c464f8a0d3ca2b6431a7eb7564b passed latest SOL Sentinel CI in workflow #40995460729. The latest 146-test log at commit eccf68dc8d182a2c241fb025e396dc50aacbdfc0 explicitly reports Ran 146 tests, OK. The Termux doctor's explicit no-network smoke step passed in #40994852707.
- The newest completed native rolling-window research remains #40994685989, artifact #12951295388, dataset SHA-256 d76434d4c9019249e924e72c5b05d0b200c0dfe3c4709c02a6d821fa0b22327c. All Pro scenarios are zero-trade across 60/20/20; this supersedes the single development-only maker trade in #40994319663.
- The local diagnostic archive is /mnt/data/sol-sentinel-research-40994685989.zip. It was downloaded and inspected; its per-scenario JSON confirms the zero-trade result above.
- Main was unchanged. PR #178 and #180 remain unmerged. The status remains RESEARCH_ONLY / INSUFFICIENT_EVIDENCE; no device-level Termux run or real financial action occurred.
## Final handoff correction — feature head 40c00977 (2026-10-30)

- PR #178's exact latest head is 40c00977c74baa619ac7c32f25e2d2ad07c17766. Latest compare to main d5672c84a0ae184c3a95d3f925925887ece71aa4: diverged, 266 commits ahead / 2 behind, 37 changed files. CI #40995563402 and duplicate #40995568088 both passed the offline suite and the no-network Termux doctor smoke step. The unit-test log reported 146 tests, 0 failures.
- PR #180's previous verified head is 030a55e69848be8b1d53335275f13c72d51d5226, 39 commits ahead / 0 behind main, 3 files. Root memory and research audit are in this branch, not merged to main.
- Current native rolling-window research evidence remains #40994685989 (dataset d76434d4c9019249e924e72c5b05d0b200c0dfe3c4709c02a6d821fa0b22327c), zero candidates/trades in every split and all three Pro sensitivities. See preceding latest-window section and project EXECUTION_LOG.md.
- Current readiness remains RESEARCH_ONLY / INSUFFICIENT_EVIDENCE; no real order, signer, transfer, deposit, withdrawal, or device-level Termux execution occurred.
## Additional code-path audit — data leakage and signal execution timing (2026-10-30)

- Reviewed evaluate_splits.py and paper_engine.py against the no-look-ahead requirement. Splits are chronological, disjoint 60/20/20; each segment is checked for contiguous one-minute bars and evaluated independently; there is no automatic parameter optimization.
- Higher-timeframe 5m/15m candles are aggregated only from complete contiguous minute buckets. At a 1m decision close, bisect_right selects only higher-timeframe candles whose close_time is <= the decision close. Prior-high and relative-volume baselines are built from the preceding 20 bars. A signal is recorded at the bar close and entry is deferred to the next 1m open.
- Intrabar stop/target collisions use conservative stop-first handling; end-of-data open positions are force-closed at the last close as a simulation convention. No look-ahead defect was identified in the inspected paths; this is a code review, not a formal proof.
- Each split resets its own indicator warmup, dropping the segment's initial warmup period instead of carrying prior segment context. OHLC cannot establish exact intrabar path; spread/slippage, maker queue and fills remain uncalibrated. See project EXECUTION_LOG.md and RESEARCH_AUDIT_2026-10-30.md.
- Feature branch subsequently advanced to code/log commit a0cb4aa901678a8f3aac9d26a2cc6be828e36a81 and its current compare must be refreshed before integration. No merge/rebase occurred.

## SOL SENTINEL IMPLEMENTATION CHECKPOINT — INDODAX DEPTH COLLECTOR (2026-10-30)

Verified remote state at this checkpoint:
- Repository: `orionrayy/ai-determinism-engine`.
- `main`: `d5672c84a0ae184c3a95d3f925925887ece71aa4` (unchanged).
- Feature branch `feat/sol-sentinel-paper-trader-v1`: `708fef5d043113e68a9d0b2d43e21e1586167a92`.
- PR #178: open, draft, unmerged; 268 commits ahead / 2 behind `main`, diverged, 39 changed files. No merge/rebase performed.
- Feature CI run #40999102295 and duplicate push/PR run #40999106258 both succeeded on exact commit `708fef5d043113e68a9d0b2d43e21e1586167a92`. Logs show 168 tests passed, 0 failures. Termux doctor smoke reported 6 PASS, 1 WARN (runner is Ubuntu rather than Termux), 0 failures; public network probe was skipped.
- Implementation commit: https://github.com/orionrayy/ai-determinism-engine/commit/708fef5d043113e68a9d0b2d43e21e1586167a92
- CI evidence: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40999102295 and https://github.com/orionrayy/ai-determinism-engine/actions/runs/40999106258

### Added INDODAX microstructure collector
- Added `projects/sol-sentinel/indodax_microstructure.py` and `test_indodax_microstructure.py`.
- Uses only Python standard library and unauthenticated public GET `https://indodax.com/api/depth/solidr`; no balance, trading, wallet, signing, deposit, withdrawal, transfer or transaction interface.
- Parses up to 500 ordered bid/ask price/quantity levels. Rejects missing/empty sides, malformed values, non-finite/non-positive prices or quantities, duplicate/unsorted levels and crossed books. A locked book is retained as a valid zero-spread observation.
- Calculates quoted spread `(ask-bid)/midpoint × 10,000`, preserves exchange timestamp only if present as an integer millisecond field, and writes top-of-book plus all observed levels into a local SQLite journal. Observations are content/timestamp keyed and idempotent for an identical duplicate.
- Request attempts, success/error and latency are journaled. Summary computes mean, median, nearest-rank P95 and population standard deviation for spread and IDR depth inside a midpoint band; aggregation uses SQL rather than loading the entire archive into memory.
- Collector defaults to 30 samples separated by 10 seconds, enforces a minimum 1-second interval (60 requests/minute), caps response body at 1 MB, bounds retries and level count, and stops on consecutive failures. CLI also supports offline `--summary-only`.
- New module's 22 deterministic fixture tests passed locally; full GitHub suite passed 168/168. The fixture tests validate schema, calculations, failure handling, SQLite durability/idempotency and finite sampling.
- **No actual live order-book sample was collected in this coding session.** Therefore spread/depth distribution statistics from the collector are still unavailable (zero recorded observations until first real collection). Do not infer historical depth from candle CSVs or previous isolated quote snapshots.
- Updated Termux doctor compile-module list, CI compile list, `RUNBOOK_TERMUX.md`, `SPEC.md`, `README.md`, `SOURCES.md` and project `EXECUTION_LOG.md`.

### Prior evidence still in force
- Latest native 1m SOL/IDR dataset evidence remains workflow #40994685989, artifact #12951295388, SHA-256 `d76434d4c9019249e924e72c5b05d0b200c0dfe3c4709c02a6d821fa0b22327c`, 43,200 contiguous bars from 2026-09-30T03:01:00Z through 2026-10-30T03:00:59.999Z, no FX conversion.
- That replay has zero candidates, attempts, fills, positions and closed trades across dev/validation/OOS under all reported Pro cost sensitivities. The previous single development maker trade was not reproduced. No positive expectancy evidence exists.
- Cost observations and assumptions remain sparse/uncalibrated; Pro fee rates used in the previous report are not account-specific confirmation. No new spread/depth empirical distribution was produced here.
- Strategy status remains `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`. The collector's CI green does not change financial or operational readiness.

### Open work and exact next dependencies
1. Collect a real finite depth sample from the Android Termux device or another explicitly documented host; preserve DB summary, environment, exact collector version and timestamp. Review rate limits and response schema before longer sampling.
2. Build the multi-level paper simulator as a separate checkpoint: deterministic depth walk, weighted execution price, partial-fill/residual state, cancel/expire, fill journal and SQLite replay. Preserve the existing top-level adapter contract until the new simulator has dedicated regression coverage.
3. Add per-filter signal telemetry and bounded ablation/sensitivity runs without tuning on OOS.
4. Integrate the forward-paper loop only after the execution journal, portfolio accounting and risk state can recover idempotently.
5. Complete Jupiter V2 quote validation and actual handset portability checks.
6. Reconcile PR #178's 268/2 divergence with file-by-file and ancestry review. Do not merge/rebase blindly.
7. Keep PR #180 draft/open and canonical memory on its branch until review/merge approval.

### Safety and portability
No live order, account query, transfer, wallet signature, deposit, withdrawal or chain submission occurred. No private key, seed phrase or secret API credential was requested or stored. The CI runner was Linux/Ubuntu with Python 3.11.17; Android/Termux hardware remains untested.


## SOL SENTINEL v1.2 EXECUTION ENGINE CHECKPOINT — 2026-10-30

### Verified remote repository state
- Repository: `orionrayy/ai-determinism-engine`.
- `main`: `d5672c84a0ae184c3a95d3f925925887ece71aa4`; unchanged by this work.
- Feature branch: `feat/sol-sentinel-paper-trader-v1`.
- Latest feature commit: `6e9424e9b8ad322e94f898d9de86324680ca0ae1`.
- PR #178 remains open/draft/unmerged; compare to main: `diverged`, 270 commits ahead / 2 behind, 41 changed paths. No merge, rebase or force-push was performed.
- Docs branch: `docs/sol-sentinel-continuity-20261030`, current head at the start of this checkpoint `94a5f1b319befd48b5bf56473175c6fc3aedb335`; this append is committed separately below.
- PR #180 remains open/draft/unmerged; compare to main at last verification: 43 commits ahead / 0 behind, 3 changed files. Its contents are not yet in `main`.

### Implementation delivered on the feature branch
- `projects/sol-sentinel/depth_execution.py`: a new isolated, Python-standard-library, Decimal and SQLite paper-only depth execution engine. The legacy `MarketQuote` and `PaperExecutionAdapter` were not modified.
- `projects/sol-sentinel/test_depth_execution.py`: regression and fault-injection tests.
- Updated `termux_doctor.py`, `RUNBOOK_TERMUX.md`, `SPEC.md`, `README.md`, `EXECUTION_LOG.md`, and the SOL Sentinel CI compile list.

### Execution semantics and invariants
- Immutable timestamped snapshots preserve observed bid/ask levels and have content-derived snapshot identity.
- Buy orders walk ask levels; sell orders walk bid levels. Every fill records level index, price, quantity, gross IDR and fee IDR. VWAP is computed from cumulative gross and cumulative filled quantity using Decimal.
- Limit orders stop before a price level that violates their limit. Cash and fees constrain buys; reconstructed SOL inventory constrains sells. Non-zero starting SOL inventory is refused until a reliable acquisition cost basis is available.
- GTC orders retain residual quantity for later eligible snapshots; IOC remainder is cancelled after its attempt. Cancellation and expiration preserve all earlier fills.
- Snapshot liquidity is shared across simulated orders. Further hardening added independent bid/ask liquidity pools keyed by venue, symbol, side, and the complete ordered price/quantity sequence. Different observation timestamps with an identical side-book do not reset consumed liquidity. A bid-side change does not reset an unchanged ask-side pool; a changed side-book fingerprint creates a new observed pool. This is a conservative approximation because trade-flow/queue data are unavailable.
- SQLite persists orders, snapshots/levels, per-snapshot consumption, side-book pools and per-level pool consumption, fills, events, cycle results and the paper account. Each order/snapshot matching cycle uses `BEGIN IMMEDIATE` and commits or rolls back liquidity consumption, fills, events, order state and portfolio together.
- Fill/event/snapshot records are append-only. Order creation and order/snapshot cycles are idempotent. An after-commit lost-response retry returns the committed cycle; pre-commit failure injection proves rollback.
- Recovery backfills shared side-book pool mappings from existing snapshot/fill records and reconciles both per-snapshot and aggregate pool consumption. If aggregate historical consumption exceeds observed depth, migration fails closed rather than hiding the discrepancy.
- Portfolio replay derives cash, SOL quantity, average entry cost, realized P&L and fees from immutable fills. Reconciliation fails closed on tampering or mismatch; it does not silently force balances.

### Evidence
- Local validation on final source: `py_compile` passed; `python -W error::ResourceWarning -m unittest -v` reported **38 tests, OK**.
- GitHub Actions run [#41002769671](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41002769671) succeeded on exact feature commit `6e9424e9b8ad322e94f898d9de86324680ca0ae1`. Runner log: `Ran 206 tests in 4.966s`, `OK`, no failures. Compile step includes depth engine and tests.
- The same CI job ran Termux doctor smoke: 6 PASS, 1 WARN, 0 FAIL. Warn is expected because host is Ubuntu Linux rather than Android/Termux; public network probe was skipped. This is not handset validation.
- Earlier feature commit `c1e008dbbcc27dc922c3dd2162ade232fc7178dd` passed 202 tests in runs #41002332334 and #41002337823; it is superseded by #41002769671.

### Boundaries and outstanding work
- No order-book live samples were collected here. Historical native INDODAX OHLC remains the previously verified 43,200 1m candles, SHA-256 `d76434d4c9019249e924e72c5b05d0b200c0dfe3c4709c02a6d821fa0b22327c`. Candle history is not order-book depth history.
- The depth engine is not an exchange matching-engine emulator: maker queue, post-only behavior, queue priority, adverse selection, hidden liquidity, cancellations, trade flow and latency races are not modeled. Side-book fingerprint changes are an observed-state approximation, not proof of refill.
- The engine currently receives parsed snapshots in memory via `DepthSnapshot.from_indodax`; it does not automatically stream the collector SQLite archive or schedule an unattended observation loop.
- This is not yet a complete forward-paper orchestrator. Strategy invocation, strategy rejection telemetry, daily-loss/cooldown/circuit-breaker risk controls, multi-cycle monitoring and fully integrated paper-run recovery remain outstanding.
- Jupiter V2 remains quote-only and is not part of this execution-engine change.
- Android Termux remains untested on the user's handset.
- Latest strategy evidence remains unchanged: no OOS trade / no demonstrated positive net expectancy. The one earlier development maker trade was not reproducible on the subsequent rolling dataset.
- Overall status remains `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`; passing engineering tests does not establish trading edge or operational readiness.

### Next dependencies
1. Capture actual repeated public INDODAX depth samples and inspect the collector's SQLite report; maintain the read-only boundary.
2. Review the side-book pool policy against observed snapshot sequences and add trade-flow evidence only if a documented public source is available.
3. Integrate the persistent engine with the existing strategy and risk gate in a separate checkpoint; prevent residual orders from bypassing risk limits.
4. Add full forward-paper cycle idempotency, operator halt, daily loss/cooldown and restart monitoring tests.
5. Complete Jupiter quote schema validation and actual Termux handset validation.
6. Review PR #178 divergence file-by-file before any branch integration. No automatic merge/rebase.

### Safety
No live order, authenticated account action, signing, wallet, deposit, withdrawal, transfer, SOL movement or on-chain submission occurred. No secret API key, private key or seed phrase was requested or stored.

 
## SOL_SENTINEL_V1_3_FORWARD_PAPER_ACTIVE_2026_10_30 — integrated forward-paper activation (2026-10-30)

- Implemented an integrated paper-only cycle in `projects/sol-sentinel/forward_paper.py`. Existing `paper_engine.signal_at`, native INDODAX adapter, `indodax_microstructure.DepthClient/DepthStore`, `DepthSnapshot` and v1.2 `SQLitePaperExecutionLedger` are reused; no replacement execution engine or live-order route was added.
- The worker uses one file-backed SQLite database for forward cycle state, signal intents, daily risk, strategy position metadata, the existing execution ledger, and collector depth observations. Startup performs ledger recovery and incomplete-cycle recovery. Signal/client-order IDs are deterministic; conflicting signal replay, order re-binding, and active-position overwrite fail closed. Pending order-intent recovery searches the existing execution ledger; expired unsubmitted intents are closed instead of remaining pending forever.
- Paper risk supervisor marks equity in native IDR using cash plus SOL inventory at the captured bid. Default paper controls: IDR 300,000 initial cash, 0.5% risk budget per entry, 50% maximum marked exposure, 3% UTC-day loss circuit breaker, 10% maximum drawdown, 2% warning level, 1 open order, 10 daily entry intents, 50 bps entry spread cap, 10-second depth age and 120-second candle age. Entry residuals are cancelled on a breaker/spread gate; protective simulated SELL residuals remain eligible. Manual resume requires ledger consistency, no recovery-required order, and attributed inventory.
- Native history fetch aligns its start to a UTC 15-minute boundary while retaining the newest completed 1m candle. It uses 600 minimum history bars (occasionally up to 14 additional bars for alignment), the frozen 1m/5m/15m strategy and native SOL/IDR with no FX conversion. No strategy threshold was loosened.
- Added public operator status, halt and guarded resume operations. A bounded cycle emits market, signal, risk, order, portfolio, reconciliation, and microstructure status. Default CLI invocation runs one cycle; multi-cycle local operation is bounded at 100.
- Added `.github/workflows/sol-sentinel-forward-paper.yml` for one-cycle five-minute schedule plus manual run/status/halt/resume. It restores and checkpoints a dedicated `sol-sentinel-paper-state` branch under serialized concurrency. The recurring schedule is configured but only becomes active once this workflow is merged into the repository default branch. The push/PR test workflow already attempts a one-cycle activation after offline tests pass on the feature branch.
- Latest verified implementation branch: `feat/sol-sentinel-paper-trader-v1`, exact tested head at the second activation checkpoint: `e827a1c2008c5cb0cabf04d79a53188646460c84`.
- Exact-head GitHub Actions [run #41006225151](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006225151) passed **216 offline unit/regression tests**, 0 failed. Termux Doctor in the Ubuntu runner reported 6 PASS, 1 WARN, 0 FAIL. The warning is expected for a non-Android host; no handset validation is claimed.
- Immediate forward-paper activation completed in the same test workflow after the tests passed. Artifact [#12955632037](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006225151/artifacts/12955632037); cycle `92f2cb5c750ddf6d648407cb75368fd14aedf9ad`, start `2026-10-30T05:03:43.767120Z`, completion `2026-10-30T05:03:47.325018Z`, 603 native 1m candles, collector observation `e5fbfc98f14a2ed57ac75a7888a4d4982dbe26d3d41b4fa543f92bb61ba79570`, execution snapshot `f72481320b8b56c512ffbdc238e7e51de90944339915072e6faf44241f35d18a`.
- The latest strategy result is `NO_TRADE` with reasons `15m_trend_not_confirmed` and `1m_momentum_exhaustion_or_missing`. Risk was `NORMAL`, new entries allowed, marked equity IDR 300,000, exposure 0, daily loss 0, drawdown 0. Portfolio remained IDR 300,000 cash / 0 SOL; 0 orders, 0 fills, 0 fees, 0 realized P&L; ledger recovery and portfolio reconciliation were consistent. This is an active monitoring cycle, not a fabricated trade.
- Collector summary at that checkpoint: 12 read-only depth observations, 12 successful attempts / 0 errors, mean quoted spread 8.999621884961035 bps, median 9.171850874576316 bps, nearest-rank P95 12.324036763976032 bps. This is a very small, newly collected sample, not a calibrated spread distribution or proof of fill quality.
- Earlier immediate cycles #41005908768 and #41005904577 on the same candle also completed with `NO_TRADE`; the repeated signal was recognized as `signal_slot_already_processed`. No duplicate order was created. The state database was persisted to the dedicated `sol-sentinel-paper-state` branch; the latest recorded state-branch checkpoint at this review was commit `3f0f5c1511eb0b602a6281c12330ff00a0a89fd9` (subsequent activation checkpoints may advance it).
- Economic readiness remains `RESEARCH_EVIDENCE_INSUFFICIENT`: the prior 30-day chronological historical baseline had zero trades in dev/validation/OOS; the forward cycle has not produced a qualifying trade. This does not block continued paper observation. Live execution remains disabled and separately unauthorized.
- Remaining deployment items: integrate the tested changes to default `main` through a clean ancestry branch (do not merge the 296-ahead/2-behind PR #178 branch blindly); make the five-minute scheduled workflow available on default branch; verify the user's actual INDODAX fee mode and minimum in the official account UI; test Termux on the physical Android device; continue collecting order-book snapshots and calibrate execution assumptions; do not promote profitability without sufficient OOS evidence.

 
## SOL_SENTINEL_V1_3_MAIN_MERGED_AND_SCHEDULED_2026_10_30 — clean integration merged to main (2026-10-30)

- Clean integration PR [#181](https://github.com/orionrayy/ai-determinism-engine/pull/181) was merged via squash into `main`: merge commit `58ec3df095c60ccc6868bf0cbccb3c18384419eb`. The branch had 1 commit ahead / 0 behind with merge base equal to main, 48 changed files. Draft PR #178 was deliberately not merged because its history remained diverged (296 ahead / 2 behind at the pre-integration review). Draft PR #180's canonical memory/audit/infrastructure files were copied into the clean integration branch. Original PRs are retained as historical draft references.
- `.github/workflows/sol-sentinel-forward-paper.yml` and `projects/sol-sentinel/forward_paper.py` are now present on the default branch. The forward-paper workflow defines one bounded SOL/IDR paper cycle every five minutes, with serialized state writes, one SQLite checkpoint file on `sol-sentinel-paper-state`, integrity checking and run artifact upload. GitHub's scheduled-workflow restriction is now satisfied because the workflow is on `main`; the first post-merge scheduled invocation is not yet separately observed in this checkpoint.
- PR [#181](https://github.com/orionrayy/ai-determinism-engine/pull/181) exact-head checks succeeded: Orchestrator Tests [#41006890245](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006890245) success; SOL Sentinel Tests [#41006890361](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006890361) success. The Sentinel workflow ran **216 offline unit/regression tests, 0 failed**, Termux Doctor 6 PASS / 1 WARN / 0 FAIL on Ubuntu, and the same PR run's forward-paper activation job succeeded.
- Latest integrated activation artifact [#12955034631](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006890361/artifacts/12955034631): exact `source_commit_sha=5bf6c2cc9598d49c4ecec4458373538b3afa18f5`, cycle `7e08efadca684497f5341b3bdb760ad18cb62857`, native INDODAX `SOL/IDR`, 610 completed 1m candles, public depth observation `62ffba97d99af94e33ad7b8d7d154dbce11a22d583c4cdf24859a5597b605c8f`, execution snapshot `e7f3dd4c7860e65be38cec14a32171c57741714a997b349ce8d51a97b145fb38`.
- Result: `FORWARD_PAPER_ACTIVE`; signal `NO_TRADE` with `15m_trend_not_confirmed` and `1m_momentum_exhaustion_or_missing`; risk `NORMAL`, new entries allowed, marked equity IDR 300,000, exposure/daily loss/drawdown 0. Portfolio remains IDR 300,000 cash / 0 SOL; 0 orders, 0 fills, 0 fees, 0 realized P&L. Ledger recovery and portfolio reconciliation are consistent. No simulated trade was invented.
- Collector had 17 actual public depth observations, 17 successes / 0 errors; mean quoted spread 9.634764 bps, median 11.105374 bps, P95 16.071871 bps. This is a small sample, not a calibrated cost distribution, queue estimate, or fill guarantee.
- The paper-state branch persisted a new SQLite checkpoint at `c3f259242d450d64b4baeb01ae083c178fd4d8b0`. The database branch remains intentionally separate from source code and must not be merged into `main`.
- Main push verification: [Orchestrator Tests #41007000945](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41007000945) succeeded on merge commit `58ec3df095c60ccc6868bf0cbccb3c18384419eb`.
- Research remains `INSUFFICIENT_EVIDENCE`: no positive OOS expectancy has been established; the current forward signal is NO_TRADE. The 5-minute schedule can continue sampling with no 30-day activation delay. Actual physical Android/Termux validation, account-specific fee-tier verification and market-microstructure calibration remain outstanding. Live order execution, wallet signing and fund movement remain disabled and unauthorized.


## SOL_SENTINEL_V1_4_SCHEDULER_AND_PERSISTENCE_HARDENING_2026_10_30

Checkpoint time: 2026-10-30. Current source main before this memory entry: `e5d12cfd1671d52cb926d156a81dd749d1645bee`.

### Repository and integration
- PR #181 is merged as `58ec3df095c60ccc6868bf0cbccb3c18384419eb`.
- Follow-up reliability PR [#182](https://github.com/orionrayy/ai-determinism-engine/pull/182) is also merged as `8c8e0c0362161ec6644ba44fa538d55dcb9fe18c`.
- PR #178 and #180 remain closed/unmerged historical drafts. Do not merge them.
- Main push runs triggered by the reliability fix and documentation updates succeeded: [#41010295827](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41010295827) on `8c8e0c...` and [#41010619804](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41010619804) on `cfdbaa...`; the latest audit-document push run [#41010670822](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41010670822) also completed successfully. Each Sentinel test run reports 216 tests passing, 0 failures, and Ubuntu Termux Doctor 6 PASS / 1 WARN / 0 FAIL. The activation job was skipped, intentionally, so these runs did not create duplicate paper cycles.
- The run triggered by #182 on PR event, [#41010214471](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41010214471), also passed 216 tests; activation skipped.

### Scheduler status
- Forward-paper workflow exists on the default branch and retains a one-cycle five-minute cadence, now offset to `2-57/5 * * * *` UTC. State-changing operations are guarded to `refs/heads/main`; only the activation job in the test workflow gets contents write permission. Unit tests use contents read. Step Summary contains run/event/ref/commit/cycle/strategy/risk/portfolio/reconciliation details.
- GitHub guidance confirms scheduled events run from the default branch, use UTC unless timezone is set, have a five-minute minimum interval, and can be delayed or dropped during load spikes near the top of an hour. The cron was moved off minute zero accordingly.
- Last actual `event=schedule` audit (through 2026-10-30T05:50:44Z) found only unrelated `AI Orchestrator` scheduled runs; no `SOL Sentinel Forward Paper` schedule event after #181/#182. Do not mark the acceptance gate as passed. Status remains `FIRST_CRON_PENDING`. No manual dispatch evidence is claimed.
- The GitHub Action history must be checked again after this checkpoint. A real scheduled run must match workflow name `SOL Sentinel Forward Paper`, event `schedule`, branch `main`, and the actual merged SHA; inspect its artifact/Step Summary and state branch commit before changing the status.

### Persistent state and recovery
- Persistence architecture: SQLite `state/forward-paper.sqlite3` is committed to a separate dedicated Git branch `sol-sentinel-paper-state` at the end of each bounded invocation, after a WAL checkpoint and `PRAGMA integrity_check`. The next runner checks out that branch before opening the DB. This is Git-versioned durable checkpointing, not a managed external database or runner-local file survival.
- Last verified state branch head: `c3f259242d450d64b4baeb01ae083c178fd4d8b0`. Its linear history showed at least 12 checkpoint commits corresponding to distinct Actions run IDs (#41005882837 through #41006890361), with increasing timestamps and parent links. This supports cross-run persistence of the saved database.
- Actual portfolio is still flat in the last paper artifact: IDR 300,000 cash, 0 SOL, no position, no open orders, no fills, no residual quantity, zero fees and zero P&L. Ledger/portfolio reconcile and restart recovery are consistent. A real non-flat fill/residual recovery path has not occurred in operation because no simulated trade has been produced; unit tests exercise synthetic pending-intent, order recovery, duplicate suppression, and daily loss breaker persistence.
- Do not reset this state branch or merge its DB into source main.

### Latest paper result and market observations
- Latest recorded paper cycle remains from activation run [#41006890361](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006890361), artifact [#12955034631](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006890361/artifacts/12955034631), code source SHA `5bf6c2cc9598d49c4ecec4458373538b3afa18f5`. Cycle `7e08efadca684497f5341b3bdb760ad18cb62857`; native INDODAX SOL/IDR; 610 completed 1m candles; 2026-10-30T05:10:22.869190Z through 05:10:23.853315Z.
- Output `FORWARD_PAPER_ACTIVE`, signal `NO_TRADE` (15m trend not confirmed; 1m momentum exhaustion/missing), risk `NORMAL`, new entries allowed, equity/cash Rp300,000, SOL 0, no orders/fills/fees/P&L, consistent ledger/portfolio, duplicate signal suppressed.
- Collector: 17 public depth observations, 17 success/0 errors, mean spread 9.634764 bps, median 11.105374 bps, P95 16.071871 bps. This short sample is descriptive, not calibrated slippage/queue evidence. Mean/median/P95 visible bid quote notional: about Rp179.9m/Rp57.9m/Rp617.2m; ask: about Rp44.8m/Rp49.7m/Rp57.0m.
- Slippage budget 10 bps per side remains a conservative assumption, not empirical execution slippage. Book-walk impact is modeled against displayed levels. Maker queue probability is unvalidated.

### Strategy research and fees
- The newest native 30-day INDODAX SOL/IDR rolling dataset recorded in the research audit contains 43,200 contiguous 1m bars (dataset SHA `d76434d4c9019249e924e72c5b05d0b200c0dfe3c4709c02a6d821fa0b22327c`, no FX conversion). The latest Pro taker/maker cost scenarios produced zero trades in development, validation and OOS. Expectancy, win rate and profit factor remain undefined.
- The recorded overlapping reason counters were: 40,506 1m breakout failures; 35,873 relative-volume failures; 28,306 5m momentum failures; 25,981 15m trend failures; 23,415 5m RSI failures; 3,135 1m momentum exhaustion/missing; 143 cost-edge failures. They overlap and are not unique candle counts. Technical/timeframe/volume filters dominate observed rejection counts; cost gate alone does not explain NO_TRADE.
- Earlier one-off maker-only development trade (+Rp466.818) was not repeated and had no validation/OOS trade; no edge claim. Baseline thresholds unchanged.
- Official INDODAX Pro IDR all-in fee schedule effective 2026-10-01: maker BUY 10.555 bps / SELL 31.555 bps; taker BUY 20.555 bps / SELL 41.555 bps. CFX component alone is 0.555 bps; direct Pro minimum is Rp25,000; Rp10,000–24,999 orders route through Lite. Account-level fee display still needs confirmation in Profile → Settings → Trading Fees. A crossing limit order can be charged as taker. Official sources are documented in `RESEARCH_AUDIT_2026-10-30.md` and `FORWARD_PAPER_OPERATIONS.md`.

### Termux, security and current status
- Android/Termux physical handset test is not available in this run. Ubuntu CI 6 PASS / 1 WARN / 0 FAIL is not device compatibility evidence. The operations guide now has a one-command JSON report procedure with offline tests and bounded public GET.
- All market requests are unauthenticated public GETs; no private key, secret, wallet signer, exchange order, deposit, withdrawal or transfer was used. Live remains disabled.
- Current classification: `ENGINEERING_VALIDATED` for the tested main commits; `FORWARD_PAPER_ACTIVE`; `FIRST_CRON_PENDING`; persistence checkpointing works cross-run for the flat portfolio; research `NO_TRADE / INSUFFICIENT_EVIDENCE`; physical Termux `NOT_TESTED`; live `DISABLED`.
- Next checkpoint: capture a true main-branch scheduled event and verify its artifact plus next state-branch commit. Then continue bounded monitoring; test non-flat ledger/residual recovery in a synthetic persistence integration harness without mutating the active portfolio. Keep collecting independently timestamped depth and trade-print data before calibrating slippage/maker probability.


## SOL_SENTINEL_V1_4_DURABLE_STATUS_MONITORING_2026_10_30

- PR [#183](https://github.com/orionrayy/ai-determinism-engine/pull/183) merged to `main` as `0bcfe96c47c6d566051a48d1b031e993d96062d0`.
- The forward-paper workflow now runs the read-only CLI status operation against the same SQLite DB after WAL checkpoint/integrity validation. It emits `forward-output/status.json` in the artifact and expanded GitHub Step Summary metrics: event/run/ref/main SHA, last cycle/finish/error, last observed depth timestamp, strategy/reasons, circuit breaker, daily loss, drawdown, cash/SOL, realized P&L, fees, position, open orders and residual remaining quantities, ledger counts/consistency, recovery and actionable warning.
- This added status query does not call `run_once`, collect depth, create a signal or alter portfolio/order state. In PR testing, activation job is skipped; the monitoring logic will be exercised by the next real run of the forward-paper workflow.
- PR #183's exact-head workflow test run #41010917135 passed 216 tests, 0 failures; Termux Doctor Ubuntu 6 PASS / 1 WARN / 0 FAIL; activation skipped. Post-merge main push run [#41010966379](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41010966379) succeeded on `0bcfe96c47c6d566051a48d1b031e993d96062d0`, with the same 216-test result and activation skipped.
- Operations guide updated in commit `f6d192cdf08ed10b1b2f6197c701d76369b7781b` to document `status.json` and the durable run summary. The v1.4 research audit remains in `projects/sol-sentinel/RESEARCH_AUDIT_2026-10-30.md`.
- Latest state-branch head remained `c3f259242d450d64b4baeb01ae083c178fd4d8b0`; no new paper checkpoint is expected from PR/test CI because activation is intentionally skipped.
- Scheduler remains `FIRST_CRON_PENDING`: the latest Actions schedule query through 2026-10-30T05:54:12Z returned only the unrelated AI Orchestrator scheduled history, with no post-merge run for workflow `SOL Sentinel Forward Paper`. No manual dispatch or duplicate forward-paper cycle was invoked. Continue to check only for actual `event=schedule` with the SOL Sentinel workflow name and main SHA.
- Current classifications: `ENGINEERING_VALIDATED` (main source/test workflows pass); `FORWARD_PAPER_ACTIVE` (last operational artifact is still the verified flat paper account); `FIRST_CRON_PENDING`; durable checkpoint persistence across successive flat-state runs observed; non-flat residual production recovery not yet exercised live; `NO_TRADE / INSUFFICIENT_EVIDENCE`; physical Android/Termux `NOT_TESTED`; live `DISABLED`.


## SOL_SENTINEL_V1_4_OPERATOR_CONTROL_REGRESSION_2026_10_30

- PR [#184](https://github.com/orionrayy/ai-determinism-engine/pull/184) merged via squash into `main` as `161a3909438b95de8485af001613481588f94594`.
- Added three temporary-SQLite regression tests for operator controls:
  1. HALT persists after reopening the database and makes RiskSupervisor deny new entries.
  2. HALT cancels an existing simulated BUY residual and preserves cash/ledger consistency.
  3. RESUME reactivates only after ledger recovery/reconciliation and preserves all portfolio values.
- Exact PR test run #41011193758: **219 tests passed, 0 failed**; Termux Doctor Ubuntu 6 PASS / 1 WARN / 0 FAIL; activation skipped. Post-merge exact-main run [#41011238183](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41011238183) also succeeded with **219 tests**, 0 failures, Termux Doctor 6 PASS / 1 WARN / 0 FAIL, and activation skipped. Tests used temporary DB/fake market/depth only; the active paper-state branch was not accessed.
- Main HEAD at this checkpoint: `161a3909438b95de8485af001613481588f94594`. Canonical memory update itself is on the next documentation commit.
- Latest actual schedule audit at 2026-10-30T05:57:06Z still showed no `SOL Sentinel Forward Paper` run with event `schedule`. This is not satisfied by unrelated AI Orchestrator scheduled events or Sentinel push/PR tests. Status remains `FIRST_CRON_PENDING`.
- The next true scheduler acceptance requires a run whose name is `SOL Sentinel Forward Paper`, event is `schedule`, branch `main`, whose run artifact includes both `result.json` and durable `status.json`, and whose state-branch commit advances after a completed cycle. No manual dispatch or duplicate cycle was launched.

## SOL_SENTINEL_V1_5_NEW_ROOM_CONTINUITY_2026_10_30

Audit checkpoint: 2026-10-30T06:36:51Z. This addendum was prepared on review branch \`fix/sol-sentinel-ci-isolated-recovery-20261030\`; it is not yet on \`main\` until PR #185 is reviewed and merged.

### Repository truth
- Verified default branch \`main\` HEAD: \`ba2a8d323b2231422367aea09436fcc1ab463dd2\`. Its parent is \`161a3909438b95de8485af001613481588f94594\`; the only change between these commits is a 13-line canonical-memory addition. No code delta separates the latest main HEAD from the exact Sentinel-tested parent.
- PR #181 merged as \`58ec3df095c60ccc6868bf0cbccb3c18384419eb\`; #182 as \`8c8e0c0362161ec6644ba44fa538d55dcb9fe18c\`; #183 as \`0bcfe96c47c6d566051a48d1b031e993d96062d0\`; #184 as \`161a3909438b95de8485af001613481588f94594\`. PRs #178 and #180 are closed and not merged. Do not revive them.
- Exact post-merge Sentinel CI run #41011238183 passed 219 tests (0 failures) at SHA \`161a3909438b95de8485af001613481588f94594\`; Termux Doctor on Ubuntu reported 6 PASS / 1 WARN / 0 FAIL. Main commit \`ba2a8d3...\` is documentation-only after that tested code commit. Orchestrator CI #41011317438 passed on the exact latest main SHA; it is not a replacement for the Sentinel suite.
- Current main forward-paper workflow blob: \`3d0cf0eb1d32e66f2a82bc0f1e95b81df919c616\`. Current main test workflow blob before PR #185: \`590f866de0ff7cdbcee62f51140cb92e9b9ba00c\`.

### Scheduler proof — still pending
- \`.github/workflows/sol-sentinel-forward-paper.yml\` is on default branch \`main\`; its schedule is \`2-57/5 * * * *\` UTC (minutes 02, 07, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57). It has a main-ref guard, eight-minute timeout, one shared concurrency group with \`cancel-in-progress: false\`, a state restore, bounded one-cycle invocation, WAL checkpoint/integrity check, state persistence, artifact upload and Step Summary/status output.
- Requeried repository Actions with \`event=schedule\` and \`branch=main\`: API total 152 schedule events across pages 1 and 2; all returned runs were the separate \`AI Orchestrator\` workflow. No result matched \`SOL Sentinel Forward Paper\` or its workflow path. No \`workflow_dispatch\`, push, or PR event is counted as cron proof.
- Current status remains \`FIRST_CRON_PENDING\`. The workflow file/default-branch/cron syntax conditions are satisfied, but the workflow's enabled/disabled setting could not be read through the available GitHub connector actions. Do not infer that it is enabled or disabled. Do not trigger another paper cycle just to test the scheduler.
- GitHub's current Actions documentation states schedules run on the latest default-branch commit, default to UTC, and have a five-minute minimum; it also notes high-load delays/dropped events and that a scheduled workflow may stop when its associated actor is suspended/deleted. Recheck workflow enablement and its associated actor through repository Actions settings before changing cron syntax.

### State branch provenance and flat portfolio
- State branch HEAD remains \`c3f259242d450d64b4baeb01ae083c178fd4d8b0\`, parent \`49966ce0530458854e640da4d491dda106f5e53b\`. The SQLite file \`state/forward-paper.sqlite3\` is 1,323,008 bytes; Git blob SHA is \`a9bbe1ae63aa1aa496fe000f23d30d4259de1c09\` (Git object SHA, not a raw-file SHA-256).
- The state branch has 17 linear SQLite checkpoint commits, from 2026-10-30T04:56:41Z through 05:10:23Z. Every corresponding run ID (\`41005623914\`, \`41005726897\`, \`41005771703\`, \`41005811889\`, \`41005832769\`, \`41005882837\`, \`41005908768\`, \`41005904577\`, \`41006187202\`, \`41006193535\`, \`41006220391\`, \`41006225151\`, \`41006524572\`, \`41006535344\`, \`41006530350\`, \`41006854612\`, \`41006890361\`) maps to \`SOL Sentinel Tests\` with event \`push\` or \`pull_request\` on a feature branch. None is a \`schedule\` run on \`main\`.
- This history is evidence that independent hosted runner invocations restored the prior state branch and committed subsequent SQLite checkpoints. It is not proof of scheduled operation. It is also not non-flat recovery: the latest inspected artifact is flat.
- Latest actual activation artifact: run #41006890361, artifact #12955034631; source SHA \`5bf6c2cc9598d49c4ecec4458373538b3afa18f5\`; cycle \`7e08efadca684497f5341b3bdb760ad18cb62857\`; 610 completed native INDODAX SOL/IDR 1m candles. Result JSON exit code 0, stderr empty; \`FORWARD_PAPER_ACTIVE\`, \`NO_TRADE\`, risk NORMAL, entries allowed; Rp300,000 cash, 0 SOL, no position/orders/fills/residuals, fees/P&L zero; recovery and reconciliation consistent; no live order route. This artifact predates the durable \`status.json\` monitoring update and therefore does not validate the later status artifact.
- The latest cycle reasons remain \`15m_trend_not_confirmed\` and \`1m_momentum_exhaustion_or_missing\`. The artifact reports 17 real public INDODAX depth observations, 17 successful attempts, 0 errors, first 2026-10-30T04:56:41Z and last 05:10:23Z. Spread mean 9.634764 bps, median 11.105374 bps, nearest-rank P95 16.071871 bps. These describe a short sample only. Slippage remains assumed; realized exchange slippage and maker queue probability remain unvalidated.

### Non-flat recovery regression — PR #185, not yet merged
- Open review PR: #185, \`fix/sol-sentinel-ci-isolated-recovery-20261030\`, head \`2093fcbc7286f5b988947e3dbd8db92a029111ff\`, base \`ba2a8d323b2231422367aea09436fcc1ab463dd2\`.
- Removed the state-mutating \`forward-paper-activation\` job from the general \`SOL Sentinel Tests\` workflow. The old job checked out a hard-coded feature branch, loaded the shared state branch and ran/wrote a one-cycle paper execution on push/PR. It caused the 17 checkpoint commits above. Regression CI should use temporary state only; recurring forward-paper state writes belong to the dedicated main-branch scheduled workflow.
- Added \`projects/sol-sentinel/test_cross_process_recovery.py\`: three genuinely separate Python processes share only a temporary SQLite file. Process A creates a partially filled BUY (2/5) and SELL (1/2), leaving two residuals and SOL inventory 1. Process B independently restores/reconciles then fills the remaining BUY 3 and SELL 1 against a changed snapshot. Process C independently restores again, verifies SOL inventory 3, four fill rows, zero open orders/residuals, fees, \`PRAGMA integrity_check=ok\`, no foreign-key violations, and idempotent replay of the same snapshot without duplicate fills.
- PR #185 CI run #41015083173 (event \`pull_request\`, branch \`fix/sol-sentinel-ci-isolated-recovery-20261030\`, exact SHA \`2093fcbc7286f5b988947e3dbd8db92a029111ff\`) succeeded: 220 tests passed, 0 failed; Ubuntu Termux Doctor 6 PASS / 1 WARN / 0 FAIL. The test job had no activation job and did not advance the state branch. This proves the isolated synthetic non-flat scenario on the PR branch; it is not yet integrated into main and is not evidence of an actual non-flat operational fill.
- PR #185 remains open and must not be auto-merged.

### Controls, Termux, research and boundaries
- Main's 219-test suite includes temporary-DB tests for daily-loss breaker persistence across state reopen; HALT persistence/blocking new entries; HALT cancellation of pending BUY residual while preserving ledger; and RESUME without portfolio reset after reconciliation. No test was performed against the active paper DB.
- Ubuntu Termux Doctor evidence is 6 PASS / 1 WARN / 0 FAIL. The warning correctly states this is not the user's Android host. Physical Android/Termux remains \`NOT_TESTED\`.
- The current NO_TRADE rejection counters are overlapping observations: 40,506 1m breakout, 35,873 relative-volume, 28,306 5m momentum, 25,981 15m trend, 23,415 5m RSI, 3,135 1m momentum exhaustion/missing, 143 cost-edge. Current native IDR scenarios recorded zero trades in development, validation and OOS. Do not calculate expectancy, win rate or profit factor from zero trades. Research remains \`NO_TRADE / INSUFFICIENT_EVIDENCE\`.
- Slippage 10 bps per side remains an assumption; book-walk is a displayed-depth model, not measured execution. Maker queue remains \`UNVALIDATED\`. No live trading, exchange order, wallet signing or transfer is authorized.
- State branch should not be reset. The five-minute schedule implies up to 288 state commits/day if every invocation changes the DB; monitor Git history growth and Actions artifact/resource use before any retention redesign. Do not rewrite state-branch history or alter cron cadence without measured need.
- Current classifications: repository \`MAIN_VERIFIED\`; engineering main test baseline \`ENGINEERING_VALIDATED\` with PR #185 pending; scheduled operation \`FIRST_CRON_PENDING\`; state persistence \`FLAT_STATE_PERSISTENCE_VERIFIED\`; synthetic non-flat recovery passed on PR #185 but not merged; Android \`NOT_TESTED\`; strategy \`NO_TRADE / INSUFFICIENT_EVIDENCE\`; live \`DISABLED\`.

### Next continuation
1. Review PR #185 and merge only after its 220-test run and diff remain valid; do not merge automatically.
2. Verify the workflow enabled state and associated actor in GitHub Actions settings; investigate why no Sentinel schedule appears despite the valid default-branch workflow. Do not use a manual cycle as cron proof.
3. Capture one genuine \`schedule\` run on \`main\`; verify artifact, \`status.json\`, summary, cycle, state restore, and resulting state-branch commit.
4. Keep Android validation pending until the user runs the documented diagnostic on the handset. Continue accumulating depth/trade data before changing slippage/queue assumptions.

## SOL_SENTINEL_V1_5_FOLLOWUP_DURABLE_ARTIFACT_ORDER_2026_10_30

- Additional audit of main workflow \`.github/workflows/sol-sentinel-forward-paper.yml\` found a production evidence packaging defect: \`Upload forward-paper evidence\` ran before \`Capture durable operational status\`. Thus the workflow could produce \`status.json\` for the Step Summary but upload an artifact ZIP that omitted it.
- Opened separate review PR [#186](https://github.com/orionrayy/ai-determinism-engine/pull/186), branch \`fix/sol-sentinel-status-artifact-order-20261030\`, head \`49ae6fd78faf72185c0bfb331d65de5b5f6b1e5a\`. The job order is now: fail unsuccessful operation, capture durable status, upload \`forward-output/\`, publish summary.
- Added \`projects/sol-sentinel/test_workflow_contract.py\` to assert status generation precedes artifact upload and that the upload action includes \`forward-output/\` with 30-day retention.
- Exact PR CI [#41015701518](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41015701518), event \`pull_request\`, SHA \`49ae6fd78faf72185c0bfb331d65de5b5f6b1e5a\`, succeeded: 221 tests passed, 0 failed; Termux Doctor Ubuntu 6 PASS / 1 WARN / 0 FAIL. The forward-paper activation job was skipped. PR #186 remains open, not merged.
- Main still remains \`ba2a8d323b2231422367aea09436fcc1ab463dd2\`; state branch still \`c3f259242d450d64b4baeb01ae083c178fd4d8b0\`. No new paper cycle or state mutation was triggered.
- Current GitHub documentation confirms scheduled workflows run on the latest default-branch commit, use UTC by default, have a five-minute minimum interval, and may be delayed during high load. A manually disabled workflow does not respond to triggers. Workflow state is controllable through Actions UI, REST API, or GitHub CLI, but the connected GitHub actions used for this audit did not expose the workflow state, so it remains unknown. Inspect repository Actions → SOL Sentinel Forward Paper and verify it is enabled, then inspect the workflow's schedule actor. Sources: GitHub Docs, [workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax), [troubleshooting workflows](https://docs.github.com/en/actions/how-tos/troubleshoot-workflows), [manage workflow runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs), and [events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).
- Neither PR #185 nor #186 is merged. Canonical continuity changes in #185 are still review-branch checkpoints pending integration. Do not claim they are on main.
- Updated next actions: review #185 for shared-state test isolation + deterministic non-flat recovery; review #186 for operational artifact ordering; inspect/enable scheduled workflow via GitHub UI if disabled; then wait for a genuine scheduled run and verify event/ref/SHA/artifact/state advancement.

## SOL_SENTINEL_V1_5_FOLLOWUP_STATE_INTEGRITY_GUARD_2026_10_30

- Further review of main forward-paper workflow found a second state-safety defect: both \`Checkpoint and verify SQLite state\` and \`Persist paper state\` used \`if: always()\`. Thus a failed WAL checkpoint or \`PRAGMA integrity_check\` did not prevent the later state-branch commit attempt.
- PR #186 was extended to gate \`Persist paper state\` on \`steps.state_integrity.outcome == 'success'\`. When validation does not pass, the status capture step avoids querying the bad database and writes a structured \`STATE_INTEGRITY_FAILED\` status with \`ledger.consistent=false\`; the Step Summary marks state integrity, persistence, and status capture outcomes and raises ACTION REQUIRED if any are not successful.
- The evidence artifact upload remains after status capture, so the artifact includes \`status.json\`. The summary is published last.
- Four workflow-contract tests now assert: status before upload; upload path/retention; persistence is integrity-gated; failed integrity produces safe status and an actionable summary.
- Final exact CI run for PR #186: [#41016258548](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41016258548), event \`pull_request\`, SHA \`988aea096b90d3e0e7a07960fbd547eb982e40ee\`, succeeded with **223 tests passed, 0 failed**, Termux Doctor Ubuntu 6 PASS / 1 WARN / 0 FAIL. The forward-paper activation job was skipped.
- An earlier run #41016205926 on the intermediate test commit failed one static assertion because the test encoded a literal workflow expression with a stray backslash. The assertion was corrected; all four contract tests pass on the final SHA. No operational paper job ran.
- PR #186 latest head: \`988aea096b90d3e0e7a07960fbd547eb982e40ee\`; open, mergeable, not merged. Main remains \`ba2a8d323b2231422367aea09436fcc1ab463dd2\`; active paper-state branch remains \`c3f259242d450d64b4baeb01ae083c178fd4d8b0\`.
- Keep both PRs open for review; never merge automatically. PR #185 covers CI isolation and non-flat recovery; PR #186 covers safe state persistence and artifact/monitoring correctness.

## SOL_SENTINEL_V1_5_FINAL_WORKFLOW_PR186_VALIDATION_2026_10_30

- PR #186 additionally sets \`concurrency.queue: max\` while retaining \`cancel-in-progress: false\` and the same single-writer group. Current GitHub Actions docs specify the default concurrency behavior allows one pending run and replaces the existing pending run when a new run queues; \`queue: max\` permits up to 100 pending runs, after which additional queued runs are canceled. This protects scheduled cycles from the default one-pending replacement behavior during brief backlog. It is bounded, not infinite. Sources: GitHub Docs workflow syntax and concurrency.
- Added a fifth static workflow contract test to guard \`queue: max\`, the shared single-writer concurrency group, and \`cancel-in-progress: false\`.
- Final exact PR #186 CI: run [#41016440890](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41016440890), event \`pull_request\`, branch \`fix/sol-sentinel-status-artifact-order-20261030\`, SHA \`86c50783e1e6f850c74aa9f24bc75615773d4f12\`, conclusion success. **224 tests passed, 0 failed**; Ubuntu Termux Doctor 6 PASS / 1 WARN / 0 FAIL; activation job skipped.
- This supersedes the earlier intermediate 221- and 223-test PR #186 check records. The earlier test-string assertion failure was corrected before the final SHA.
- PR #186 head is now \`86c50783e1e6f850c74aa9f24bc75615773d4f12\`, open and mergeable, not merged. The five workflow contract guards are: durable status before artifact upload, correct artifact path/retention, state persist only after successful SQLite integrity validation, safe status fallback and actionable summary on failure, and bounded pending concurrency queue.
- Latest verified main remains \`ba2a8d323b2231422367aea09436fcc1ab463dd2\`; latest state remains \`c3f259242d450d64b4baeb01ae083c178fd4d8b0\`. No scheduled run was found as of 2026-10-30T06:50:40Z; do not promote the first-cron gate.

## SOL_SENTINEL_V1_5_READ_ONLY_ACTIVE_STATE_FORENSICS_2026_10_30

- Added a one-time read-only CI audit job to PR #185, guarded strictly to pull_request number 185. It checks out \`sol-sentinel-paper-state\` using \`actions/checkout\` with \`persist-credentials: false\`, opens the database using SQLite URI \`mode=ro\` plus \`PRAGMA query_only=ON\`, runs integrity/FK checks and row-level provenance/reconciliation, and does not commit or write the state branch.
- Exact successful audit/test run: [#41017189745](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41017189745), event \`pull_request\`, head SHA \`83d0f22d0d2b6013352e4d0c4fd052ad02b082a3\`. Both \`active-state-read-only-audit\` and \`unit-tests\` succeeded. Sentinel suite: **220 passed, 0 failed**; Ubuntu Termux Doctor 6 PASS / 1 WARN / 0 FAIL. The non-flat three-process recovery test passed. Matching Orchestrator run [#41017189742](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41017189742) also passed on the same SHA.
- Raw SQLite file metadata: \`state/forward-paper.sqlite3\`, 1,323,008 bytes, raw-file SHA-256 \`149ea9184abb9977cdba032820e7bc57fa33526349530633906008d49d33427c\`. This is now an actual SHA-256 of file bytes, not a Git blob ID.
- Database checks: \`PRAGMA integrity_check=ok\`; foreign-key violations 0; journal mode \`wal\`; ledger schema version \`1\`; \`PRAGMA user_version=0\`; 0 foreign key errors. SQLite was opened in \`mode=ro\`, \`query_only=1\`, no writes made.
- Relevant row counts: 17 \`fp_cycles\`, 10 \`fp_signals\`, 1 \`fp_daily_risk\`, 2 \`fp_control\`, 1 \`paper_accounts\`; 17 \`observations\`, 5,100 \`levels\`, 17 \`attempts\`; 0 \`orders\`, 0 \`fills\`, 0 \`events\`, 0 \`execution_cycles\`, 0 \`snapshots\`, 0 \`fp_position\`.
- Portfolio: Rp300,000 cash; 0 SOL; average entry 0; realized P&L 0; fees 0; open orders 0, fills 0. Cash and SOL balances reconcile to the fill journal (both equations true; no fill records exist). Thus this is definitively flat-state persistence, not non-flat operational recovery.
- Market-depth provenance: 17 observations and 17 unique observation IDs, 17 unique book hashes, 0 duplicate book hashes; collector version \`indodax-depth-collector-v1\`, endpoint \`https://indodax.com/api/depth/solidr\`. First \`2026-10-30T04:56:41.272158Z\`; last \`2026-10-30T05:10:23.793208Z\`. Each snapshot has 150 bid and 150 ask levels (5,100 rows total). Attempts: 17 OK, 0 ERROR; mean request duration 612.18 ms. No record had an exchange timestamp (\`exchange_ts_ms=null\` for all 17). UTC hour distribution: 6 observations during 04:00–04:59 and 11 during 05:00–05:59.
- Recomputed spread distribution directly from stored rows: mean 9.6347641883 bps; median 11.1053738673; nearest-rank P90 16.0662374831; P95 16.0718713822; population standard deviation 3.7951068970.
- Visible quote notional depth within ±25 bps of each snapshot midpoint, computed directly from the 150x150 level rows:
  - Bid: mean Rp179,924,343.79; median Rp57,933,685.96; P90 Rp338,854,082.90; P95 Rp617,159,539.84; population standard deviation Rp164,507,936.70.
  - Ask: mean Rp44,832,636.84; median Rp49,739,722.33; P90 Rp55,129,463.32; P95 Rp56,951,424.57; population standard deviation Rp15,786,136.49.
- The highly skewed 17-snapshot sample spans only about 13m42s; it has no exchange-side timestamps, trade prints, queue position or queue-depletion linkage. Visible book notional is not a guaranteed executable fill. The 10 bps-per-side slippage setting remains an assumption, not measured realized slippage.
- Current PR #185 head at this audit was \`83d0f22d0d2b6013352e4d0c4fd052ad02b082a3\`; PR open/not merged. This audit is a temporary CI read-only job and does not alter the paper-state branch.
- The previous attempted version of the second depth-band script had a trivial Python syntax error (assignment inside dict literal); it failed before any database writes. It was corrected; the final audit on #41017189745 passed. Do not treat the failed intermediate run as the final evidence.

## SOL_SENTINEL_V1_5_OFFICIAL_INDODAX_FEE_MODEL_RECONCILIATION_2026_10_30

- Verified official INDODAX Pro IDR all-in fee table effective 2026-10-01. The official table lists:
  - Taker BUY 0.20555% = 20.555 bps; Taker SELL 0.41555% = 41.555 bps; total round-trip all-in fees 62.11 bps.
  - Maker BUY 0.10555% = 10.555 bps; Maker SELL 0.31555% = 31.555 bps; total maker round-trip all-in fees 42.11 bps.
- Official INDODAX notice dated 2026-09-29 says IDR-market CFX was reduced to 0.00555% on both buy and sell, effective 2026-10-01. The official fee image breaks out service fee, sell-side tax and CFX to those exact all-in totals. Official page instructs members to check their own all-in/tax details in Profile > Settings > Trading Fees; account-level values should be confirmed there before any future real-money use.
- Main \`ForwardPaperConfig\` has \`fee_buy_bps=20.555\` and \`fee_sell_bps=41.555\`, matching the official **Pro taker IDR all-in** schedule. The active executor uses market orders, so this is the correct side-specific fee class for the simulated path. No fee code correction is indicated by this comparison.
- Reconcile terminology: the historical 87.162363 bps "Pro taker current snapshot total modeled friction" is **not fee-only**. It is 62.11 bps taker all-in fees + 15.0523626084 bps single quoted spread + 10 bps assumed round-trip slippage (5 bps per side). Historical maker 67.162363 bps similarly equals 42.11 bps maker fees + 15.0523626084 spread + 10 bps assumed round-trip slippage. The prompt's shorthand "Pro taker current" can otherwise be misread as pure fees.
- The active forward engine config now uses 10 bps assumed slippage per side. Using the 17-observation sample spread mean gives a modeled total of 62.11 + 9.634764 = 71.744764 bps before slippage, or approximately 91.744764 bps after adding 20 bps of assumed round-trip slippage. At the sample P95 spread, that becomes approximately 98.181871 bps. These are estimates, not measured fills or realized slippage.
- The 10 bps-per-side setting remains an assumption; the 17 static snapshots cannot infer actual execution impact or maker queue probability. No strategy parameters or fees were changed and no trades were forced.
- Official sources checked 2026-10-30: INDODAX Help Center "Details of Transaction Fees on INDODAX" and its official fee-table image; INDODAX announcement "Penurunan Biaya CFX", 2026-09-29, effective 2026-10-01.

## SOL_SENTINEL_V1_5_READONLY_ACTIVE_STATE_AUDIT_2026_10_30

Verified from exact PR #185 workflow run #41017557539 at 2026-10-30T07:02:16Z (14:02 WIB), branch \`fix/sol-sentinel-ci-isolated-recovery-20261030\`, head \`438923fb356945357b58fadbb600a2044aa804e9\`. Both unit-tests and active-state-read-only-audit jobs passed. This is a read-only CI inspection of \`sol-sentinel-paper-state\`, not a paper cycle.

### Exact SQLite file and integrity

- File: \`.paper-state/state/forward-paper.sqlite3\`
- Size: 1,323,008 bytes.
- Raw SHA-256: \`149ea9184abb9977cdba032820e7bc57fa33526349530633906008d49d33427c\`.
- \`PRAGMA integrity_check\`: \`ok\`.
- \`PRAGMA foreign_key_check\`: 0 violations.
- SQLite ledger schema version 1; journal mode WAL; PRAGMA user_version 0.
- Connection was opened using SQLite URI \`mode=ro\`, \`PRAGMA query_only=ON\`; state checkout had \`persist-credentials: false\`, workflow permission \`contents: read\`. No state write, checkpoint, mutation, or commit was performed.
- Table counts: observations 17, levels 5,100, attempts 17, fp_cycles 17, fp_signals 10, fp_daily_risk 1, fp_control 2, fp_position 0, fp_metadata 4, paper_accounts 1, orders 0, fills 0, events 0, execution_cycles 0, snapshots 0, snapshot_levels 0, snapshot_consumption 0, liquidity_pools 0, liquidity_pool_levels 0, snapshot_liquidity_pools 0; metadata 1.
- Account: initial cash Rp300,000; current cash Rp300,000; initial/current SOL 0; avg entry cost 0; realized P&L 0; fees 0; open orders 0; fills 0. Cash and SOL both reconcile exactly to fill rows. Execution event journal is empty (0 events), no active position.
- This is strong integrity/provenance evidence for the actual latest flat paper state, not non-flat operational recovery.

### Depth collector provenance

- Endpoint: \`https://indodax.com/api/depth/solidr\`.
- Collector: \`indodax-depth-collector-v1\`.
- 17 observations, 17 unique observation IDs and 17 unique book hashes; 0 duplicate book hashes.
- Observation window: 2026-10-30T04:56:41.272158Z–05:10:23.793208Z (13m42s). UTC hours: 04 = 6 snapshots; 05 = 11 snapshots.
- 5,100 level rows: 150 bid + 150 ask levels per observation.
- 17 collection attempts: OK=17, errors=0, mean duration 612.176 ms.
- Exchange timestamp presence: 0/17. Snapshot timing is collector-local; no exchange-side timestamp was supplied in these records. Do not present the observation timestamps as exchange event timestamps.

### Reproducible spread statistics

Computed directly from the stored 17 \`observations.spread_bps\` values:
- Mean 9.6347641883 bps.
- Median 11.1053738673 bps.
- P90 nearest-rank 16.0662374831 bps.
- P95 nearest-rank 16.0718713822 bps.
- Population standard deviation 3.7951068970 bps.

### Visible depth within ±25 bps of snapshot midpoint

Computed from immutable \`levels\` rows, summing price × quantity per side where bid >= midpoint × 0.9975 and ask <= midpoint × 1.0025. This is visible quote notional, not guaranteed executable liquidity.

Bid quote depth IDR (n=17): mean Rp179,924,344; median Rp57,933,686; P90 Rp338,854,083; P95 Rp617,159,540; population SD Rp164,507,937.

Ask quote depth IDR (n=17): mean Rp44,832,637; median Rp49,739,722; P90 Rp55,129,463; P95 Rp56,951,425; population SD Rp15,786,136.

Bid observations are highly skewed relative to median; do not size orders from the mean or extrapolate beyond this small 13m42s window. Static depth does not encode trade prints, queue position, replenishment/depletion, or realized fills. Queue probability and realized execution slippage remain unmeasured; the 10 bps per-side slippage is still an assumption.

### Latest validation and status

- PR #185 head \`438923fb356945357b58fadbb600a2044aa804e9\`: Sentinel run #41017557539 passed 220 tests, 0 failures; read-only active-state audit passed; Termux Doctor Ubuntu 6 PASS / 1 WARN / 0 FAIL; Orchestrator run #41017557515 passed.
- PR #186 head \`86c50783e1e6f850c74aa9f24bc75615773d4f12\`: Sentinel run #41016440890 passed 224 tests, 0 failures, including five workflow contract tests. PRs remain open, unmerged.
- Main HEAD remains \`ba2a8d323b2231422367aea09436fcc1ab463dd2\`; state HEAD remains \`c3f259242d450d64b4baeb01ae083c178fd4d8b0\`.
- At 2026-10-30T07:13:28Z (14:13 WIB), schedule history query \`event=schedule&branch=main\` still showed no run matching the Sentinel workflow; public workflow page showed 0 runs. \`FIRST_CRON_PENDING\` remains.

## v1.6 verified integration and scheduler checkpoint — 2026-10-30T08:21:42Z (15:21:42 WIB)

### Repository and integration

- Current main HEAD: 6e2231c84399687c43b644056b0d5aaec57f64f8.
- PR #181 merged as 58ec3df095c60ccc6868bf0cbccb3c18384419eb; PR #182 as 8c8e0c0362161ec6644ba44fa538d55dcb9fe18c; PR #183 as 0bcfe96c47c6d566051a48d1b031e993d96062d0; PR #184 as 161a3909438b95de8485af001613481588f94594; PR #185 as b7f9b190ec61502c23fe1e1721db6ab329795636; PR #186 as 6e2231c84399687c43b644056b0d5aaec57f64f8.
- PR #185 and #186 are both closed/merged. PR #178 and #180 are closed, not merged; do not revive them.
- Post-merge SOL Sentinel Tests run [#41020901299](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41020901299) is a push run on main at the exact current SHA, conclusion success. The unit job reports 225 tests passed, 0 failed. This is CI regression evidence, not a scheduled paper execution.

### PR reliability changes on main

- PR #185 removes the old push/PR forward-paper-activation state-mutating job from the general tests workflow. CI is now regression-only and must not restore, run or commit the active paper database. It also adds an isolated three-process SQLite recovery test using a temporary database.
- PR #186 adds status.json before artifact upload, gates paper-state persistence on successful SQLite WAL checkpoint/integrity validation, emits a safe structured status on integrity failure, and configures bounded queue: max in the single-writer concurrency group.
- Main regression log confirms the three-process BUY/SELL partial-fill/residual recovery test passes. Operator tests pass for persistent daily-loss breaker, HALT persistence, pending BUY residual cancellation, RESUME after ledger reconciliation, and rejection of unattributed inventory.

### First actual cron status

- Workflow: .github/workflows/sol-sentinel-forward-paper.yml; cron: 2-57/5 * * * * UTC.
- A fresh query of Actions runs filtered by event=schedule&branch=main returned 153 runs across pages 1 and 2. All 153 were AI Orchestrator at .github/workflows/orchestrator.yml; zero matched SOL Sentinel Forward Paper or the Sentinel workflow path. Latest returned scheduled run was #41019077386 at 2026-10-30T07:16:28Z. Workflow file's latest commit is current main SHA at 07:34:20Z, so multiple five-minute slots after that commit elapsed without a Sentinel schedule run.
- FIRST_CRON_PENDING remains. Workflow enabled/disabled state and exact schedule actor are NOT VERIFIED through the read-only GitHub API connection. Do not guess, do not count push/PR/manual runs, and do not dispatch a mutating cycle to simulate cron proof. Repository Actions overall is active because AI Orchestrator scheduled runs exist; that fact does not prove this workflow is enabled.

### Active paper state — read-only evidence

- State branch HEAD remains c3f259242d450d64b4baeb01ae083c178fd4d8b0; database blob SHA a9bbe1ae63aa1aa496fe000f23d30d4259de1c09; file size 1,323,008 bytes.
- Exact raw database SHA-256 from the successful read-only audit: 149ea9184abb9977cdba032820e7bc57fa33526349530633906008d49d33427c.
- Audit run [#41018963015](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41018963015), job active-state-read-only-audit, passed on PR #185 with contents: read, checkout persist-credentials: false, SQLite URI mode=ro, and PRAGMA query_only=ON. integrity_check=ok, foreign-key violations 0, WAL journal, ledger schema 1.
- Counts: 17 forward cycles, 10 signals, 1 daily risk row, 2 control rows, 1 account, 17 depth observations, 5,100 depth levels, 17 attempts; 0 execution orders, fills, events, execution cycles, position rows or residuals.
- Paper account remains flat: Rp300,000 cash, 0 SOL, 0 fees, 0 realized P&L, zero open orders and fills. Cash/SOL reconcile with the empty fill journal. This is flat-state persistence only; it does not demonstrate non-flat operational recovery.

### Synthetic recovery and operator controls

- The integrated test runs A/B/C in three distinct Python processes sharing only a temporary SQLite DB. A creates a partial BUY (2/5) and partial SELL (1/2), two residuals and 1 SOL inventory; B restores and fills the remaining quantities; C restores again and verifies both orders filled, 4 fill rows, 0 residuals, SOL inventory 3, positive fees, consistent replay, duplicate replay suppression and SQLite/FK integrity.
- Classification: NON_FLAT_SYNTHETIC_RECOVERY_VERIFIED; NON_FLAT_OPERATIONAL_RECOVERY_VERIFIED remains unproven because the live paper account has no fills or non-flat position.
- HALT/RESUME and daily-loss circuit-breaker persistence tests pass in the 225-test main suite. These are isolated integration tests, not operator actions against the active account.

### Termux and research

- Ubuntu CI Termux Doctor: 6 PASS / 1 WARN / 0 FAIL; network branch is skipped in the --skip-network CI smoke. Physical Android/Termux remains NOT_TESTED.
- The persisted 17-book dataset remains a 13m42s collector-time sample with 0 exchange timestamps, 0 trade-print linkage and 0 queue depletion linkage. Spread mean 9.634764 bps, median 11.105374 bps, P90 16.066237 bps, P95 16.071871 bps. Visible quote depth within ±25 bps: bid mean Rp179.924m / median Rp57.934m; ask mean Rp44.833m / median Rp49.740m. These are descriptive sample statistics only.
- An additional single public INDODAX depth read during this audit returned 150 bid and 150 ask levels, spread 15.694144 bps, and visible quote notional within ±25 bps of midpoint of about Rp270.716m bid / Rp93.035m ask. It was not written to the paper DB, contains no exchange-side timestamp, and is not a fill/slippage observation.
- Current strategy research classification remains NO_TRADE / INSUFFICIENT_EVIDENCE. Prior recorded rejection reasons include 15m_trend_not_confirmed and 1m_momentum_exhaustion_or_missing; no new signal evaluation was produced by this read-only audit. Preserve frozen baseline parameters and do not force trades.
- LIVE_TRADING=DISABLED. No order endpoint, signer, credential, wallet or financial transfer path was invoked.

### Immediate remediation

1. Inspect the workflow page's exact enablement state and schedule actor using an authenticated browser session; the read-only API cannot expose either in the current connector path.
2. If disabled, enable only this workflow. If enabled, investigate its actor/schedule registration using the actual visible configuration and GitHub's documented recovery path; do not alter the cron interval without evidence.
3. Wait for and verify a genuine schedule run on main, with result.json, status.json, cycle ID, matching state checkpoint and hash; then update this checkpoint.
4. Execute the on-device Termux Doctor command in RUNBOOK_TERMUX.md; keep Android status unverified until device output is captured.


## v1.7 authenticated scheduler and CI checkpoint — 2026-10-30T15:38:36+07:00

### Repository and PR reconciliation

- Current default branch remains `main`, HEAD `6e2231c84399687c43b644056b0d5aaec57f64f8`; repository metadata confirms default_branch=main and repository is not disabled.
- PRs #181–#186 remain merged: #181 `58ec3df095c60ccc6868bf0cbccb3c18384419eb`; #182 `8c8e0c0362161ec6644ba44fa538d55dcb9fe18c`; #183 `0bcfe96c47c6d566051a48d1b031e993d96062d0`; #184 `161a3909438b95de8485af001613481588f94594`; #185 `b7f9b190ec61502c23fe1e1721db6ab329795636`; #186 `6e2231c84399687c43b644056b0d5aaec57f64f8`. PR #178 and #180 remain closed/unmerged.
- PR #187 remains open and unmerged, mergeable, base main at `6e2231c84399687c43b644056b0d5aaec57f64f8`. Its v1.7 branch update now contains four files: the three continuity documents plus a narrowly scoped Actionlint configuration exception. Current branch head at the time of this append is `f1de2a142a8530489677fe47d9cbb8e8c6810734`.
- Exact-head checks on `f1de2a142a8530489677fe47d9cbb8e8c6810734`: SOL Sentinel Tests run #41027680475 succeeded; Orchestrator Tests run #41027680469 succeeded. Earlier Orchestrator Tests #41026206804 failed only at Actionlint because pinned actionlint v1.7.12 did not recognize GitHub's `concurrency.queue` key in the Sentinel workflow.
- The root cause of that CI failure was a linter compatibility mismatch, not invalid GitHub workflow syntax. GitHub's documented `queue: max` behavior is retained. `.github/actionlint.yaml` now scopes the exact known diagnostic suppression only to `.github/workflows/sol-sentinel-forward-paper.yml`; the pre-existing path-scoped suppressions for the two orchestrator workflows remain. The Sentinel workflow contract tests continue to assert `queue: max` and `cancel-in-progress: false`. This is a lint-only change, not a runtime/scheduler change.

### Authenticated workflow inspection

- Read-only inspection used the authenticated GitHub browser profile. On the dedicated workflow page the overflow menu showed `Disable workflow` (not `Enable workflow`), with an active `Run workflow` control, and no disabled/suspended banner. This establishes the workflow is enabled at inspection time. The workflow page reported zero workflow runs.
- Repository Actions settings showed “Allow all actions and reusable workflows”; the full-SHA pin requirement was unchecked; first-time-contributor approval was selected; default GITHUB_TOKEN permissions were read-only for contents/packages; artifact/log retention was 90 days. No schedule-specific repository restriction or notice was visible. The workflow itself explicitly requests `contents: write` in YAML for the dedicated state-commit job; the repository default read-only token permission is not by itself evidence of a blocker.
- Default branch `main` and the workflow file on that branch are verified. The cron is `2-57/5 * * * *` UTC. The dedicated runner is main-ref guarded, bounded to one cycle, uses the shared state concurrency group and performs a SQLite integrity gate before state persistence.
- The commit which changed the cron expression to its current staggered form was PR #182 merge commit `8c8e0c0362161ec6644ba44fa538d55dcb9fe18c`, whose author login is `orionrayy`. This identifies the user associated with the most recent cron syntax change. The account is the repository owner/admin and remains active: other repository scheduled events carry actor and triggering_actor `orionrayy`, including AI Orchestrator run #41019077386. No Sentinel schedule run exists from which to read a Sentinel-specific actor field; do not represent the actor metadata as directly observed on a Sentinel run.
- Two pages of `event=schedule&branch=main` history contain 153 events total; all 153 belong to `AI Orchestrator` at `.github/workflows/orchestrator.yml`. None belongs to `SOL Sentinel Forward Paper` or its workflow path. Latest returned unrelated run #41019077386 was created at `2026-10-30T07:16:28Z`, actor `orionrayy`, conclusion success. The workflow's own page still reports zero runs.
- Scheduler classification remains `FIRST_CRON_PENDING`. Enablement is verified as enabled; default-branch placement, cron syntax and repository Actions policy have no visible blocker; the associated cron editor/actor is identified and appears active. The exact reason GitHub has not emitted a Sentinel schedule event remains unresolved. Do not label this a confirmed GitHub platform defect, disabled workflow, or actor suspension. No workflow enable/disable action, manual dispatch, rerun, or paper cycle was performed.

### Persistent state and operational safety

- State branch remains `sol-sentinel-paper-state` at `c3f259242d450d64b4baeb01ae083c178fd4d8b0`. A fresh Git tree read still identifies `state/forward-paper.sqlite3` as blob `a9bbe1ae63aa1aa496fe000f23d30d4259de1c09`, size 1,323,008 bytes. This is byte-identical at the Git object level to the last SQL-audited DB.
- Last raw SHA-256 remains `149ea9184abb9977cdba032820e7bc57fa33526349530633906008d49d33427c`. The prior read-only audit #41018963015 reported SQLite `integrity_check=ok`, 0 FK violations, WAL mode, schema 1, 17 cycles, 17 depth observations / 5,100 levels, 17 attempts, zero execution orders/fills/residuals and a flat reconciled portfolio (Rp300,000 cash; 0 SOL; no fees or realized P&L). Binary contents were not re-opened for a new SQL pass in this checkpoint; the unchanged Git blob SHA and size establish the inspected file is the same artifact as the prior audit. Do not describe this as a newly executed SQL query.
- `NON_FLAT_SYNTHETIC_RECOVERY_VERIFIED` remains true from the merged three-process temporary-DB test. `NON_FLAT_OPERATIONAL_RECOVERY_VERIFIED` remains unproven; the active ledger is flat.
- No active state branch or database write was performed.

### Termux and research boundary

- Current `termux_doctor.py` is stdlib-only, has read-only bounded public INDODAX pair metadata GET, checks Termux/Python/runtime/memory/disk/module compilation and exercises SQLite close/reopen/idempotent replay on a temporary DB. Optional flags include `--json`, `--skip-network`, `--run-tests`, `--data`, `--expected-sha256`, and bounded `--timeout`.
- Prior Ubuntu CI Termux Doctor result remains 6 PASS / 1 WARN / 0 FAIL with the public network probe skipped in that smoke. Physical Android/Termux remains `NOT_TESTED`; do not infer device success.
- The 17 persisted depth observations remain a short 13m42s sample with no exchange timestamps, trade prints, queue-depletion linkage or realized fills. The earlier spread and ±25 bps depth statistics remain descriptive only; 10 bps per-side slippage remains uncalibrated.
- Strategy remains `NO_TRADE / INSUFFICIENT_EVIDENCE`. Existing rejection telemetry and the historical zero-trade OOS result do not establish positive expectancy. No strategy settings or OOS boundaries were changed. `LIVE_TRADING=DISABLED`.

### Outstanding root cause and next controlled action

1. Preserve the current enabled setting; no toggle is justified by the evidence.
2. Investigate a same-cadence schedule registration refresh as a separate reviewable change only after explicit approval. Do not change cadence, disable/re-enable the workflow, or start manual cycles to simulate cron proof.
3. After any approved registration change, require a genuine `event=schedule`, branch `main`, workflow identity match, successful run, both `result.json` and `status.json`, cycle ID, matching state-branch checkpoint and reconciled ledger before promoting to `SCHEDULED_OPERATION_VERIFIED`.
4. Capture an actual on-device Termux Doctor JSON report. Keep strategy, microstructure and Android statuses separate from scheduler status.


## v1.7 final PR-head CI verification — 2026-10-30T15:42:11+07:00

- PR #187's latest documentation/configuration tree at head `afec3e0f0deba9ad21b91897d7043908cfacc020` has four files and seven commits ahead of `main`, zero behind. The PR remains open, mergeable and unmerged.
- Exact-head SOL Sentinel Tests run #41027953386 succeeded: 225 unit tests passed, zero failed; Ubuntu Termux Doctor 6 PASS / 1 WARN / 0 FAIL with network skipped; `active-state-read-only-audit` was skipped because this PR does not change the active state or runtime.
- Exact-head Orchestrator Tests run #41027953355 succeeded after the scoped Actionlint compatibility exception. The job reports 463 orchestrator tests passed, then 86 shared tests passed, and the evaluation harness passed.
- The canonical memory and supporting operations/research documents were verified remotely on the PR branch. Main remains `6e2231c84399687c43b644056b0d5aaec57f64f8`; paper-state branch remains `c3f259242d450d64b4baeb01ae083c178fd4d8b0`.
- The enabled workflow still has zero Sentinel runs. Schedule history remains 153 unrelated AI Orchestrator runs and zero Sentinel schedule events. Preserve `FIRST_CRON_PENDING`; no schedule registration change, activation toggle, dispatch or paper cycle has been performed.


## v1.8 cron registration refresh checkpoint — 2026-10-30T16:04:46+07:00

### Repository and PR integration

- Default branch is `main`; latest verified HEAD after the PR #188 squash merge is `1ec2a6a11dbe8bcf84d910fc3017c9d0173ad01a`.
- PR #187 is closed/merged, squash merge commit `1326dccff5c286eabfbe810b8060702b10469f48`. It integrates the v1.7 authenticated scheduler findings, operations and research continuity, and a path-scoped Actionlint compatibility exception.
- PR #188 is closed/merged, squash merge commit `1ec2a6a11dbe8bcf84d910fc3017c9d0173ad01a`. It changes only the cron expression representation and adds one workflow-contract assertion. Exact-head CI run #41030654487 (`pull_request`, head `c9209e92a36c14f5fb1471b07e441c6e8c61435a`) passed 226 tests, 0 failed; active-state audit was skipped by design for the schedule-only PR.
- PRs #181–#186 remain merged. No duplicate integration was performed.
- PR #188 changes cron from `2-57/5 * * * *` to `2,7,12,17,22,27,32,37,42,47,52,57 * * * *`. These are semantically equivalent 12 five-minute slots per hour. No job condition, executor, permissions, concurrency, state, artifact or risk logic was changed.

### Exact workflow API and authenticated inspection

- GitHub REST workflow object: ID `401397705`, name `SOL Sentinel Forward Paper`, path `.github/workflows/sol-sentinel-forward-paper.yml`, `state=active`. Workflow creation timestamp returned as `2026-10-30T05:11:10Z`; the workflow API `updated_at` field also returned this timestamp.
- Authenticated GitHub Actions workflow page continued to show `Disable workflow` in its overflow menu, no disabled/suspended banner, and `0 workflow runs`. The API state and UI agree that the workflow is enabled.
- Repository default branch is `main`; workflow is in `main`. Repository Actions settings showed all actions allowed, default GITHUB_TOKEN contents/packages read-only, and no schedule-specific repository restriction visible. Branches settings showed no existing branch-protection rules/rulesets on main.
- Schedule-actor evidence: the prior cron expression was last changed by commit/PR authored by `orionrayy`; PR #188 is also authored/merged by `orionrayy`. The account is repository owner/admin and remains active, with unrelated AI Orchestrator schedule events listing actor `orionrayy`. No Sentinel event exists for direct run-level actor metadata. Actor suspension is not supported by available evidence.
- Exact workflow-ID API query for `workflow_id=401397705`, `event=schedule`, `branch=main` returned `total_count=0` and no pagination link before the next scheduled slot after PR #188 was merged. Do not count unrelated repository-wide AI Orchestrator schedule events as Sentinel execution.
- Workflow remains `WORKFLOW_ENABLED_FIRST_RUN_PENDING`; project acceptance remains `FIRST_CRON_PENDING`. The equivalent cron refresh is a controlled re-registration attempt, not a proven root-cause fix. The actual platform cause remains unresolved until a schedule event appears or GitHub returns a specific error.

### Scheduler safety and runtime boundary

- Current main workflow cron: `2,7,12,17,22,27,32,37,42,47,52,57 * * * *` UTC, twelve invocations per hour. Main-ref guard, one bounded cycle, `sol-sentinel-forward-paper-state` concurrency group, `cancel-in-progress: false`, `queue: max`, timeout, SQLite integrity gate, persistence, `result.json`/ `status.json` and artifact order remain unchanged.
- GitHub documentation: schedules run from latest default-branch commit, use UTC by default and require at least five-minute spacing; a workflow can stop if its associated actor is suspended/deleted; committing a cron syntax change can reactivate a deactivated scheduled workflow and transfer actor association. Here the workflow API currently says `active` and the actor appears active, so these facts explain the supported refresh procedure but do not prove a disabled workflow/actor as the original cause.
- No workflow enable/disable toggle, `workflow_dispatch`, manual paper cycle, rerun, order, wallet, signer, deposit, withdrawal, transfer, or live action was executed. `LIVE_TRADING=DISABLED`.

### Active paper-state identity and audit limitation

- State branch remains `sol-sentinel-paper-state` at `c3f259242d450d64b4baeb01ae083c178fd4d8b0`.
- Database Git blob remains `a9bbe1ae63aa1aa496fe000f23d30d4259de1c09`, 1,323,008 bytes. It is byte-identical by Git object identity to the database examined by the last read-only SQL audit.
- Last known raw SHA-256 remains `149ea9184abb9977cdba032820e7bc57fa33526349530633906008d49d33427c`; last recorded SQL audit #41018963015 reported SQLite integrity `ok`, 0 FK violations, WAL, schema 1, 17 forward cycles, 17 depth observations / 5,100 levels, 17 attempts, no orders/fills/residuals, cash Rp300,000, 0 SOL, fees/P&L Rp0. This session verified the unchanged Git blob and size but could not download the binary for a fresh SQL query; do not claim a new SQL pass.
- Flat-state persistence remains the only active-state persistence evidence. Synthetic non-flat recovery remains `NON_FLAT_SYNTHETIC_RECOVERY_VERIFIED`; operational non-flat recovery remains `NOT_YET_OBSERVED`.

### Research and device status

- Strategy remains `NO_TRADE / INSUFFICIENT_EVIDENCE`; no new forward cycle or signal evaluation was produced. Prior OOS segments have zero trades; no performance ratios are estimable.
- Persisted depth dataset remains 17 observations over 13m42s, 5,100 levels, no exchange-side timestamps, no print linkage and no queue-depletion linkage. Spread mean 9.634764 bps, median 11.105374 bps, P90 16.066237 bps, P95 16.071871 bps. 10 bps per-side slippage remains an unvalidated assumption.
- Physical Android/Termux remains `NOT_TESTED`. Ubuntu Termux Doctor from Sentinel CI reports 6 PASS / 1 WARN / 0 FAIL with network skipped; the v1.8 schedule PR run had 226 tests passed, zero failed.
- No profitability claim, trade-forcing parameter change, OOS retuning, or live execution was performed.

### Acceptance state and next step

- Repository/PR integration: PR #187 and #188 merged and verified on `main`.
- Workflow activation: `active`, workflow ID `401397705`.
- First scheduled run: still `FIRST_CRON_PENDING`; no actual `schedule` event, run ID, result/status artifact, cycle ID or schedule-generated state checkpoint has been observed.
- Next verification: inspect exact workflow-ID scheduled history after the next eligible UTC slot, then inspect run metadata/jobs/artifact and state-branch advancement. If still zero, capture GitHub-side evidence and escalate through supported GitHub diagnostics; do not dispatch a manual paper cycle as a substitute.


## SOL_SENTINEL_V1_9_SCHEDULER_CANARY_AND_NATIVE_MULTI_STRATEGY_RESEARCH_2026_10_30

- Main checkpoint: `5dcfccf0138620328a9850c892c36327d7715d5d`. PR #190 merged as `dd8764993861cd1799f7f9968a05f8430387ff8b` (isolated scheduler canary); PR #191 as `d1bd8fbc6baf9fce4ed1ed327e4e0f7a9a3e3cb3` (native strategy-family research); PR #192 as `39289436ead3ec66774e2c3c76812233d40fed89` (explicit opt-in read-only state audit).
- Canary workflow ID401581054, path `.github/workflows/sol-sentinel-schedule-canary.yml`, active, cron `0,30 * * * *`; Forward Paper ID401397705 remains active with 12 five-minute slots/hour. Exact API run queries filtered `event=schedule&branch=main` returned `total_count=0` for both after the 10:00 UTC slot and again at approximately 10:06 UTC. No actual scheduled run yet; only one canary slot had elapsed. Root cause unresolved; don't diagnose repository-level scheduler fault from a single missed slot. Keep `FIRST_CRON_PENDING`.
- Manual read-only state audit run #41035585382 succeeded. State branch c3f259242d450d64b4baeb01ae083c178fd4d8b0; DB Git blob a9bbe1ae63aa1aa496fe000f23d30d4259de1c09; raw SHA-256 149ea9184abb9977cdba032820e7bc57fa33526349530633906008d49d33427c; size1,323,008 bytes. SQLite integrity ok, FK0, WAL, schema1, mode ro and query_only true. 17 cycles/10 signals; 17 observations/5,100 levels/17 OK attempts; no orders/fills/events/execution cycles/positions. Rp300k cash, zero SOL, fees and realized P&L; reconciliation true. Flat-state persistence only.
- Manual 30-day Research workflow #41037517595 succeeded on main head 5dcfccf0138620328a9850c892c36327d7715d5d. Artifact #12967394940 (3,113,839 bytes, ZIP digest sha256:0f668a4a1380f5c13364d2a99061b155985966d156de05c201e87905d16944cb; expiry 2026-11-06T10:03:08Z). Native INDODAX SOL/IDR CSV 43,200 contiguous 1m bars, 2026-09-30T10:01:00Z..2026-10-30T10:00:59.999Z, SHA-256 95e95e5600df8a2a21f7de34393321cf98226ace42d485d1da12c480b667656d, 2,988,266 bytes, no FX conversion, 0 gaps/duplicates/invalid OHLC/negative volume; zero-volume rows13,789.
- baseline_v1_locked fingerprint 757d5c049a82dca4656ab45e0bcc4e652a51043010063c7e4d391e56fa270023 produced 0 candidates/trades across 43,199 evaluations. Overlapping reasons: breakout40,499; relative volume35,890; 5m momentum28,311; 15m trend26,101; 5m RSI23,381; 1m RSI exhaustion/missing3,187; net edge gate139.
- Five predeclared families under Pro taker base modeled cost107.7769 bps (fees62.11 + spread25.6669 + 10 bps/side assumed slippage) all failed: OOS trade count16-23, zero wins, PF0, negative net expectancy (approx -Rp1,307 to -Rp1,455/trade). OOS block bootstrap CIs remain INSUFFICIENT_SAMPLE (<30). All remain RESEARCH_ONLY_INSUFFICIENT_EVIDENCE; no candidate promoted.
- Capital hurdle Rp300k->Rp3.5m is 11.6667x/+1,066.67% in 24h; probability deliberately INSUFFICIENT_SAMPLE because no candidate passed validation/OOS gate. Required compound return/trade:27.85% over10 trades;10.33% over25;5.04% over50;2.49% over100. No Monte Carlo probability was manufactured.
- INDODAX fee precedence: post-2026-10-01 Pro all-in schedule remains maker BUY10.555/SELL31.555 bps and taker BUY20.555/SELL41.555 bps. The static help-page screenshot dated 2026-09-28 shows the pre-cutover CFX value; official 2026-09-29 announcement sets IDR CFX0.00555% from 2026-10-01. Account-specific mode still needs Profile > Settings > Trading Fees check. No code fee default was changed.
- Research artifact dataset/report are evidence for offline replay only. Do not call the manual research run a scheduled run. Do not write active state, dispatch production forward paper to test cron, loosen strategy filters, promote a paper candidate, or enable live trading.
- Physical Termux remains NOT_TESTED. Ubuntu CI Termux Doctor output does not establish Android device compatibility. Live trading remains DISABLED.


### Shaolin Monks LAN Co-op — checkpoint 2026-10-30

- Private build/CI repository: `orionrayy/project-memory-and-database`. Target Android port: `RanggaCasper/ShaolinMonks-Android`, pinned source commit `359f3a9051e96a3ed1251920a55aa1df0a3e473c`. Verified XBE SHA-1 `20e283c44fdb6062a760e4732d765036bac18067`; overlay ZIP SHA-256 `fdccb6bc8aad0f91807aea175645def60377087f626910b4c0231af73ce4c1ee`.
- Latest completed diagnostic APK: workflow run `40384749932`, artifact `ShaolinMonks-Coop-Diagnostic-APK` ID `12680753128`; APK SHA-256 `6bda9fc191ba05f73bb911173037db73784da595f344d4db6ec05525182247cc`, 32,185,948 bytes, signature verified. Extracted `lib/arm64-v8a/libmain.so` independently contains the observation trace literal `smcoop: candidate tick hook hit=` (one occurrence); binary SHA-256 `d7bad64e2cc78289dc739c71b370469da41f30a8150a5ebd1e45a4f4adcaa265`. The earlier note that the marker was absent was incorrect and is superseded by this direct archive inspection.
- The artifact manifest explicitly remains `multiplayer_playable: false`: the current LAN page tests UDP handshake only; it is not wired to submit per-tick gameplay input, no simulation barrier/live game state probe is integrated, and no two-device gameplay test has passed.
- Fresh local validation of the overlay source archive on 2026-10-30: GCC CTest 5/5 passed; Python unittest 26/26 passed with `PYTHONPATH` pointing at the extracted overlay. These validate the independent protocol/worker/coordinator/state-sync modules and installer/test tools, not the full Android game runtime.
- New build guard commit in the private build repository: `034aae2caa26318724d44fac570c574e1a48241e`, requiring the trace marker in the generated APK before artifact publication. Workflow `41077301908`: preflight and overlay/hardening regression step passed; native APK build was still in progress at checkpoint. No artifact from this run is confirmed until all steps finish and the artifact is listed.
- Static XBE disassembly maps `0xCB800` to raw offset `0xBB800`. The site follows game frame-counter/skip-render selection and enters a large update block, making it a useful observation candidate. It is not certified as a safe simulation barrier: runtime trace, controlled pause/step evidence, and proof that all relevant simulation work is after this site are mandatory. Keep `SetLockstepGateOpen(false)` until those proofs exist.
- Next gates, in order: (1) obtain native runtime logs from title/menu/gameplay/transition/pause and analyze with `tools/analyze_frame_trace.py`; (2) experimentally verify thread/step behavior without enabling network lockstep; (3) validate original local P2 gameplay first; (4) map stable read-only state fields and active-profile save write-set from runtime evidence; (5) bind worker/coordinator/input capture to the proven tick edge; (6) run fault, packet-loss, transition, and rejoin tests; (7) test two physical Android devices over hotspot with identical save/profile and compare final canonical state/save digests. Do not infer that any gate passed from the standalone test suite or APK signature.



### Shaolin Monks LAN Co-op — paired-input shadow experiment checkpoint 2026-10-30

- Private implementation repository: `orionrayy/project-memory-and-database`. Draft PR #6: https://github.com/orionrayy/project-memory-and-database/pull/6. Branch `feat/shaolin-coop-shadow-input-experiment`, latest head `c0bd4fffe0df0caf7c16fc969c93b45826f4b8c2`. Keep it as a draft candidate; it is not merged into main.
- The LAN diagnostic page now has a one-shot **EXPERIMENTAL PAIRED INPUT EXCHANGE (SHADOW)**. Each peer sends a synthetic tick-1 `InputFrame` through `LanSessionWorker`; host slot 0 uses bits `0x1001` and axes `[-12000,0,0,0]`; client slot 1 uses bits `0x2002` and axes `[12000,0,0,0]`; both require all analog channels zero and validate the complete combined P1/P2 tick. The exchange runs on its own background thread and stops the diagnostic session after PASS/FAIL.
- This experiment does not read physical gamepad/touch inputs, inject into XInput/gameplay, fabricate a `StateProbe`, or enable the lockstep/simulation gate. The workflow manifest must continue to set `gameplay_input_injection_enabled=false`, `simulation_barrier_verified=false`, and `multiplayer_playable=false`.
- Added `shaolinmonks-build-input/tests/test_patch_lan_transport_ui.py` and `shaolinmonks-build-input/docs/EXPERIMENTAL_PAIRED_INPUT_EXCHANGE.md`. PR workflow triggers on pull_request, validates the patch Git blob, runs patcher tests/idempotence and native CTest, builds Android APK, verifies the trace and shadow UI strings inside `libmain.so`, and uploads `ShaolinMonks-Coop-Shadow-Experiment-APK`.
- First PR CI #41084295151 failed only because one test expected the phrase `not injected into the game` while the source explicitly said `did not inject input into the game`; the implementation and native CTest 5/5 passed. The assertion wording was corrected (commit `46eb9b9938ea09c4853b64783dd7387896edb792`). Subsequent run #41084807662 passed preflight, extraction, overlay hardening, three shadow-specific Python tests, and native CTest 5/5; Android APK compilation was still in progress at checkpoint. This run represents the earlier shadow source before the later full-axis/analog validation tightening.
- Latest branch head `c0bd4fffe0df0caf7c16fc969c93b45826f4b8c2` tightens full array validation and updates the workflow's pinned patch blob. Its run #41085177191 was pending at checkpoint; use only its artifact (or a later successful run from the latest head) as the current APK candidate, not a stale earlier build. Do not merge the PR automatically.
- Fresh local validation on 2026-10-30: Python overlay tests 26/26; CTest 5/5; `smcoop_lan_worker_tests` repeated independently 10/10, including localhost host/client handshake and paired P1/P2 tick; diagnostic Python UDP loopback 63/63 round-trips, 0% loss, 0 duplicates/out-of-order. These are local/localhost evidence, not Wi-Fi/hotspot or physical-device acceptance.
- Still unverified: actual two-phone shadow exchange result; runtime `0xCB800` trace from title/menu/gameplay/pause/transitions; proof of safe guest-thread simulation boundary; live game-state probe address map; original local P2 acceptance; packet-loss/desync/rejoin acceptance; real two-device gameplay. Keep the simulation gate closed until device evidence proves the complete tick boundary. The user's uploaded older diagnostic APK is not the shadow-experiment build.



### Shaolin Monks LAN Co-op — expanded acceptance + calibration checkpoint 2026-10-30 (late)

- Implementation repository: `orionrayy/project-memory-and-database`. Draft PR #6: https://github.com/orionrayy/project-memory-and-database/pull/6. Branch `feat/shaolin-coop-shadow-input-experiment`; latest verified head at checkpoint: `6a6364a078953e11d8f1194b5773985910808d3f`. PR is open/draft/unmerged.
- Acceptance framework now has 21 gates and 10 mandatory pre-playable gates. Machine-readable matrix: `shaolinmonks-build-input/acceptance/shaolin-coop-acceptance-v1.json`. Required gates include runtime hook proof, baseline local P2, active save allowlist, two-device shadow, actual gameplay, network calibration, lifecycle, release, fault-injection and security.
- Fault matrix: `shaolinmonks-build-input/acceptance/fault-injection-matrix-v1.json`, 31 unique scenarios. `FAULT-01` now traces 22 unique test references from all fault scenarios. Known statuses include the explicit unverified device security state; do not reclassify device acceptance as automated PASS.
- Static traceability validator: `shaolinmonks-build-input/scripts/validate_acceptance_contract.py`; Python tests: `shaolinmonks-build-input/tests/test_acceptance_contract.py`. Tests enforce closed release interlocks and resolve referenced test cases against repo/overlay sources.
- Added dependency-free Termux raw UDP quality probe: `shaolinmonks-build-input/acceptance/lan_quality_probe.py`, tests in `test_lan_quality_probe.py`, procedure `LAN-QUALITY-PROBE.md`. Probe measures raw link only, not game input/tick sync. Calibration requires >=5 minutes, 18,000 samples at 60 Hz and 3 repetitions; record p50/p95/p99/max, jitter, loss, duplicate/reordered replies and scheduler skips; 60/30 Hz must be compared against corresponding frame budgets. A raw link PASS is not gameplay acceptance.
- Physical-device evidence template: `acceptance/device-acceptance-template.json`. A two-phone Android device is still required to run the shadow test and collect host/client logs.
- Watchdog test-only timing margin increased from 180 ms to 750 ms; production state-confirmation watchdog remains 2,000 ms. CI native CTest is configured for `--repeat until-fail:5`.
- CI run #41092057854 on prior head `36d2f03a4403a23725389b78e7e10e45fbfe4c5e` failed in the acceptance Python layer after overlay install and native repeat CTest passed 5/5. Root cause identified from logs: `FAULT-031` status was not admitted by validator and FAULT-01 did not enumerate every fault-matrix test reference. Both have been corrected on latest head.
- Latest CI run #41093663872 targets head `6a6364a078953e11d8f1194b5773985910808d3f`; status was in_progress at checkpoint (preflight running/clean so far). Do not claim CI success or provide current APK until this latest run finishes, the acceptance tests and static validator pass, APK signing/provenance pass, trace marker verifies, and artifact is downloaded/inspected.
- Release interlocks remain false: gameplay injection disabled, simulation barrier unverified, multiplayer not playable, external acceptance incomplete, WAN/public Wi-Fi disabled, trusted private LAN only. No actual game runtime evidence, local P2 baseline device acceptance, active save path proof, two-device shadow, actual gameplay scenarios, sustained physical network calibration or lifecycle fault acceptance have passed yet.


### Shaolin Monks LAN Co-op — targets 3–6 execution checkpoint 2026-10-30 late

- Implementation repo: `orionrayy/project-memory-and-database`; draft PR #6 https://github.com/orionrayy/project-memory-and-database/pull/6. Branch `feat/shaolin-coop-shadow-input-experiment`. Current head at checkpoint `2c9d9e006128e75f529ff59bae7358d7982aa56b`; 24 changed files, 2,906 additions, 14 deletions at last PR fetch. Still draft/unmerged.
- Current head workflow #41099158372 was queued when last inspected. No artifact exists for current head at checkpoint. Older fully successful run #41093663872 on head `6a6364a078953e11d8f1194b5773985910808d3f` produced artifact #12996437964, but that APK predates save-I/O tracer, widened paired shadow timeout, host-FPS-aware 30Hz probe and quality aggregator. Do not use it as the current target3–6 build.
- Target 3 (save/profile identity): pinned upstream commit `359f3a9051e96a3ed1251920a55aa1df0a3e473c` maps Xbox E: to `<app::ExeDir>/hdd/E`; game image resides separately in shared `ShaolinMonks` directory. Added `scripts/patch_save_io_trace.py` (blob `b0936473dbaa8bb6bf4199295785f53d382d98f9`) and analyzer `scripts/analyze_save_io_trace.py`. Tracer emits E: OPEN/OPENFILE/WRITE/SETINFO/RENAME/DELETE path metadata only; it never logs save bytes. Tests `tests/test_patch_save_io_trace.py`, `tests/test_analyze_save_io_trace.py`; documentation `acceptance/ACTIVE-SAVE-DISCOVERY.md`. Workflow applies tracer twice, compiles it, and requires compiled marker strings. Manifest has `save_io_trace_enabled=true`, `save_contents_logged=false`, `active_save_allowlist_verified=false`. Physical runtime save/load traces and before/after snapshots have NOT been collected; no active profile is certified.
- Target 4 (two-device paired shadow): UI patch blob `0d56e5a80d2a4169b529dbc4cf61d5294e41b940`; one synthetic tick validates P1 bits 0x1001 / axis -12000 and P2 bits 0x2002 / axis +12000; all other axes and eight analog channels are zero. Submit wait 2.5s, pair tick wait 5s; prompt says both peers should tap within four seconds; each side logs `smcoop: paired input shadow result=`. It remains synthetic shadow only, no real controller capture, gameplay injection, state probe or gate enablement. Physical two-phone run NOT DONE; this environment has no adb binary or connected device.
- Target 5 (physical link calibration): `acceptance/lan_quality_probe.py` emits host/client JSON for raw UDP sequence echo. Five-minute floors now track FPS: 18,000 at 60Hz and 9,000 at 30Hz, with host/client FPS consistency. `acceptance/aggregate_lan_quality.py` requires exactly 3 host/client pairs and uses worst RTT p95/p99/max/loss; any failed run blocks topology verdict. Tests `test_lan_quality_probe.py`, `test_aggregate_lan_quality.py`. Usage docs `LAN-QUALITY-PROBE.md`, `LAN-QUALITY-AGGREGATION.md`. Raw link metrics are not game protocol or tick-pair sync. Physical hotspot runs at both 60/30Hz not done.
- Target 6 (gameplay): added `acceptance/GAMEPLAY-ACCEPTANCE-RUNBOOK.md` with 21 scenario IDs for P1/P2 movement, combat, combos, guard/specials, damage, knockback, enemy AI, interactions, objectives, death/revive, cinematics/checkpoints, pause/resume, app background/foreground, process/Wi-Fi disconnect, explicit rejoin and save integrity. Requires 3 repetitions and full host/client state evidence. Remote gameplay remains BLOCKED until local P2 baseline, runtime simulation tick/thread proof and real state probes are established. No gameplay tests have been run or claimed.
- Acceptance matrix has 21 gates, 10 mandatory pre-playable gates; fault matrix has 31 unique cases and 22 unique referenced test functions. Save gate still explicitly active-profile allowlist unverified; fault/security/device/performance gates not promoted without evidence.
- Workflow changes pin current UI/save patches, run patcher idempotency, Python tests, native CTest repeat=5, matrix validator, APK build/signing/provenance and compiled markers. CI previously passed #41093663872 on earlier head; that APK is stale for targets3–6. Current latest-head run #41099158372 queued; check before saying passed, download only an artifact tied to latest branch SHA.
- All release interlocks remain false: gameplay input injection disabled, simulation barrier unverified, multiplayer not playable, external acceptance incomplete, trusted private LAN only, WAN/public Wi-Fi disabled. PR #6 updated with target3–6 details and remains draft.


### Shaolin Monks LAN Co-op — target 3–6 scope refinement 2026-10-30 final checkpoint

- Latest implementation branch head remains `2c9d9e006128e75f529ff59bae7358d7982aa56b`; PR #6 remains draft/unmerged and its description was updated to cover targets 3–6.
- Current latest-head workflow #41099158372 is still `pending` with no jobs and no artifacts at last query; do not treat as a successful build. The prior successful APK #12996437964 belongs to old head `6a6364a...` and is stale for targets 3–6.
- Target 3 source work is complete for an *instrumented evidence build*: Android E: path tracer plus analyzer and tests are in the PR. Active save/profile is not yet identified because no phone has produced gameplay save/load logs. The APK can emit path/write metadata only; it does not log save contents.
- Target 4 UI rendezvous improved to 2.5s submit / 5s wait, both players asked to tap within four seconds, native per-peer PASS/FAIL log added. Real two-phone shadow exchange not run; this runtime has no ADB/device connection.
- Target 5 raw UDP probe + three-pair aggregator supports 60Hz (18,000 samples / five minutes) and 30Hz (9,000 / five minutes) with host/client FPS consistency. Latest matrix now traces the 30Hz host sample-floor test. Host/client physical hotspot runs were not performed.
- Target 6 runbook defines 21 scenarios and 3 repeats; actual remote gameplay is blocked by unproven simulation barrier, no real state probes, and no local P2 device acceptance. No gameplay test is claimed passed.
- Test caveat: native CTest repeat-five and acceptance tests passed at previous head #41093663872 before recent save tracer and quality probe changes; latest head requires a fresh green CI before APK can be distributed. All gameplay/release interlocks remain false.


### Shaolin Monks LAN Co-op — calibration report-integrity hardening checkpoint 2026-10-31

- Implementation repository: `orionrayy/project-memory-and-database`; PR #6 remains OPEN / DRAFT / UNMERGED. Latest branch head after the current hardening commits: `8a3be2c9aa6464c1a219864a76a072b2edd4a556`.
- Target 5 raw UDP probe hardened:
  - `acceptance/lan_quality_probe.py` now emits a 32-hex SHA-256 session fingerprint in both host and client reports, derived from the session token. The raw token is never written to reports.
  - Host and client report duration is derived from monotonic elapsed time; both emit `duration_pass` rather than echoing requested duration.
  - Recommended physical test duration is 305 seconds per run to leave margin above the 300-second / 18,000-at-60Hz / 9,000-at-30Hz floors, while keeping the 0.5% sender-schedule skip gate meaningful.
- Aggregator `acceptance/aggregate_lan_quality.py` upgraded to schema v2:
  - requires matching host/client session fingerprints and three unique fingerprints across three repetitions;
  - recomputes round-trip loss from sent/unique-received counters and sender skip percentage from raw counters;
  - validates finite non-negative RTT p50/p95/p99/max, percentile ordering, and p95/p99 against selected FPS budget;
  - requires measured client and host duration >=300 seconds, FPS-specific sample floors, matching FPS and host receive-order/count checks;
  - retains `gameplay_sync_accepted=false`. Session fingerprint is a pairing aid only, not a signature or proof of device identity.
- Regression tests added for cross-session mismatch, reuse of one session across repetitions, inconsistent raw counters/latency, non-finite metrics, measured duration, and safe shared session fingerprint. Acceptance matrix `NETWORK-01` traces these tests and declares both 60Hz and 30Hz sample floors; runbook docs updated.
- PR #6 description updated with this hardening. Latest head's CI run is not yet confirmed and no APK artifact is available for commit `8a3be2c...`. Workflow #41102474352 was associated with an earlier commit `36f7547...` and must not be represented as validation of the latest head.
- Known code-test review correction: the worst-tail-latency fixture now raises p95, p99 and max together so it respects percentile ordering.
- Physical Android acceptance is still NOT DONE: save identity requires on-device trace/snapshot review; shadow exchange requires two devices; network calibration needs 3 x 305-second runs at 60Hz and 30Hz; remote gameplay needs local P2 baseline, proven simulation barrier, guest-thread mapping and real state probes.
- All release interlocks remain closed: `gameplay_input_injection_enabled=false`, `simulation_barrier_verified=false`, `multiplayer_playable=false`, `external_acceptance_gates_complete=false`, `trusted_private_lan_only=true`, `wan_or_public_wifi_enabled=false`. Do not merge or claim playable based on source-level tests, a shadow PASS, or raw UDP PASS.


### Shaolin Monks LAN Co-op — active-sampling timing correction checkpoint 2026-10-31

- Implementation repo `orionrayy/project-memory-and-database`; draft PR #6 remains open and unmerged. Exact current branch head: `7ee17c15e7029e24e97bb843f3e5acd4ebeebb4a`.
- Target 5 timing correction:
  - Both host and client now distinguish `duration_seconds` (active measurement/send interval) from `completion_duration_seconds` (active interval plus response-drain completion).
  - Client stamps `send_end` when the sampling deadline is first reached; later PONG drain cannot inflate active sample duration.
  - Host stamps `measurement_end` at active deadline; idle grace used to close out final packets does not inflate measured active duration.
  - Aggregator requires active duration >=300s on both sides, `duration_pass=true`, and completion duration finite and >= active duration.
  - Added loopback assertions showing drain-inclusive completion exceeds the active interval, and aggregation test proves an active 299s run with 305s completion still fails.
- Target 5 session/counter hardening remains in place: non-secret shared 32-hex SHA-256 session fingerprint; host/client pair must match; all 3 repetitions must be unique; raw counters must agree with loss and skipped-send percentage; finite ordered RTT p50/p95/p99/max are recomputed against 30/60Hz frame budgets; host/client FPS, host count/order, and 5-minute sample floors are enforced.
- Acceptance matrix NETWORK-01 traces the added tests and records both 60Hz 18,000 and 30Hz 9,000 sample floors. Runbooks recommend 305s and explain active vs completion duration and fingerprint limitations.
- Latest workflow associated with current head is #41103768982, queued at checkpoint, no completed checks/artifacts confirmed. Do not use earlier workflow runs or old artifacts as validation of this head.
- Target 3 active save/profile allowlist remains unverified on device; target 4 two-phone paired shadow remains not run; target 5 no physical hotspot calibration yet; target 6 actual gameplay remains blocked by missing local-P2 baseline, verified simulation boundary, guest-thread mapping and canonical state probes.
- All safety interlocks remain false/closed: gameplay injection off, simulation barrier unverified, multiplayer not playable, external acceptance incomplete, WAN/public Wi-Fi off, trusted private LAN only. No PR merge or playable claim.



### Shaolin Monks LAN Co-op — NtOpenFile patcher CI failure and correction checkpoint 2026-10-31

- Implementation repo: `orionrayy/project-memory-and-database`; PR #6 remains OPEN / DRAFT / UNMERGED. Exact verified head after the current correction: `77e511e5c1c35a46fc0ba20b3d1c3734e407dc3c`; branch `feat/shaolin-coop-shadow-input-experiment`.
- CI run #41103768982 on prior head `7ee17c15e7029e24e97bb843f3e5acd4ebeebb4a` failed in the APK build job before Android compilation. Evidence from the job log:
  - private preflight passed;
  - overlay Python tests 27/27 passed;
  - native CMake build passed and CTest repeated 5 times: 5/5 tests passed each repetition;
  - fail-closed overlay, UI patching and idempotence steps passed;
  - save-I/O tracer refused to patch `sdk/runtime/kernel/files.cpp`: `NtOpenFile trace: expected exactly one anchor, found 0`.
- Root cause confirmed by comparing pinned upstream `RanggaCasper/ShaolinMonks-Android@359f3a9051e96a3ed1251920a55aa1df0a3e473c`: the real `NtOpenFile` function declares `std::error_code ec;` between `const std::string& host = t.host;` and the `if (host.empty() || !fs::exists(host, ec))` condition. The prior anchor omitted that line. It was a patch anchor defect, not a C++ compile failure.
- Corrected `scripts/patch_save_io_trace.py` to include the exact upstream declaration and preserve it when inserting the trace. New patch Git blob: `e96af55f0747b46eb6ec95fc9943561eba49d39a`. Added `test_openfile_anchor_matches_pinned_upstream_source_shape` to `tests/test_patch_save_io_trace.py`. Updated workflow blob pin to the corrected patch. Commits: patch `0f05b12c05a3e433bbe335f37c80ab26d18825e5`, regression test `ebc1fe5cee9db86d54f392bcef4a4711b938aa38`, workflow `77e511e5c1c35a46fc0ba20b3d1c3734e407dc3c`.
- Latest workflow triggered for head `77e511e5c1c35a46fc0ba20b3d1c3734e407dc3c`: run #41108542168, currently QUEUED when checked; no jobs and no artifacts yet. Do not claim the save tracer or Android APK build passed until this exact-head workflow completes. Old artifacts/runs do not validate this head.
- Target 5 hardening remains present: host/client share a non-secret 32-hex SHA-256 session fingerprint; reports distinguish active measurement duration from drain-inclusive completion duration; aggregator validates session pairing and uniqueness, recomputes loss and scheduler skips from raw counters, validates finite ordered RTT percentiles and 30/60Hz budgets, duration, FPS and sample floors. Three distinct sessions are required per batch. Fingerprint is not a signed report/device identity.
- Acceptance matrix NETWORK-01 references session, raw-counter, non-finite, and drain-time regression tests; 60Hz and 30Hz sample floors are 18,000 and 9,000 respectively. Physical hotspot calibration remains NOT RUN.
- Target 3: save I/O tracing is source-patched but has not yet produced a successful APK after the corrected anchor; active profile allowlist remains unverified until device logs and stable before/after save snapshots are reviewed.
- Target 4: paired shadow UI/source exists; two-physical-device shadow test remains NOT RUN.
- Target 6: 21-scenario gameplay runbook exists; no remote gameplay acceptance. Local P2 baseline, simulation barrier/guest-thread proof and stable read-only gameplay state probes remain outstanding.
- All release interlocks remain closed: `gameplay_input_injection_enabled=false`, `simulation_barrier_verified=false`, `multiplayer_playable=false`, `external_acceptance_gates_complete=false`, `trusted_private_lan_only=true`, `wan_or_public_wifi_enabled=false`. Keep PR draft; no merge or playable claim.


### Shaolin Monks LAN Co-op — automatic Android/native log integration checkpoint 2026-10-31

- Implementation repo: `orionrayy/project-memory-and-database`; PR #6 remains OPEN / DRAFT / UNMERGED. Exact latest branch head: `d7f0281ee27bbfdfa64dfc935e44c2ce817f7d35`; branch `feat/shaolin-coop-shadow-input-experiment`.
- Added `scripts/patch_coop_diagnostic_logging.py` (Git blob `5accb0d8a7101c01967021947f5e02fd8be3efc9`) to patch the generated native LAN diagnostic UI and pinned Android `ShaolinMonksActivity.java`.
- Automatic files in `/storage/emulated/0/ShaolinMonks/` when shared/all-files storage permission is granted:
  - `co-op-session.jsonl`: LAN page open, host/join attempt/outcome, session state snapshots, peer endpoint, one-second connected heartbeat, session stop, synthetic shadow P1/P2 submit/pair/result and worker errors.
  - `co-op-android.jsonl`: process start, activity resume/pause/destroy, focus changes, storage access checks, app/device/API/ABI metadata and network available/lost/capability events. Site-local/link-local/loopback addresses only; no Wi-Fi SSID/password, public IP or session token.
  - Each stream rotates at 8 MiB, active plus four numbered archives (.1–.4). The two writers use separate files to avoid concurrent append races.
- LAN page now displays the two log paths. Documentation: `acceptance/COOP-DIAGNOSTIC-LOGGING.md`; acceptance README updated. Unit test file `tests/test_patch_coop_diagnostic_logging.py` covers writer/lifecycle instrumentation, idempotence and fail-closed anchor handling. SHADOW-01 matrix now traces these test functions.
- Workflow applies the logging patch twice, checks its pinned Git blob, runs regression tests, then verifies native `libmain.so` contains session logging markers and every DEX contains Android lifecycle/network logging markers. APK manifest lists logging output paths/rotation/scope.
- CI troubleshooting:
  - Run #41110956870 on head `630cb899004a2dadfa813c202c637e01f1276eaa` completed failure with zero jobs; workflow name fell back to the file path, and inspection found stale duplicated PowerShell/Dex verification text appended at the end of the YAML. This was a workflow syntax problem, not an APK/compiler result.
  - Reconstructed the entire workflow tail with exactly one APK verification step, one APK artifact upload and one failure-log upload. The latest valid workflow #41111231442 targets current head `d7f0281...`, event `pull_request`, status QUEUED at checkpoint. No jobs/artifacts yet. Do not send any older artifact as the latest logging-enabled APK.
- In-app user test checklist once the correct APK is produced: open Options → LAN CO-OP; host on phone A; join from phone B by local IPv4 on a trusted private Wi-Fi/hotspot; check RUNNING and peer state; both tap RUN PAIRED INPUT EXCHANGE within four seconds; repeat three fresh sessions; STOP/RESET; optionally observe disconnect/reconnect. Shadow uses only one synthetic tick and never injects into the game.
- Current logging does not fabricate raw UDP RTT/loss/retransmission data because the in-app worker does not expose those counters. Termux raw UDP probe remains the independent link calibration source.
- Target3–6 device gates remain incomplete: active save/profile allowlist unverified, two-phone shadow acceptance not yet run, physical 60/30Hz calibration not run, actual 21-scenario gameplay not run. All release interlocks remain closed: gameplay injection false, simulation barrier false, multiplayer playable false, external gates false, trusted-private-LAN true, WAN/public-Wi-Fi false.


### Shaolin Monks LAN Co-op — shared-storage diagnostics permission correction checkpoint 2026-10-31

- Latest PR head after the permission-aware logging changes and workflow pin update: `1bc80907d973a32761fb1d1a6283a9e08bcd8959`. PR #6 still open/draft/unmerged.
- Logging patch blob now `e57ee20b6bcb1a5368ac06a7c315f9617cd9f354`; workflow `$expectedDiagnosticLoggingPatchBlob` pin updated to the same SHA. Regression test fixture expanded to match upstream's actual `askForGame` and `askAllFilesAccess` source anchors and test now asserts the permission rationale and prompt behavior.
- Important Android 11 behavior: previously, if the game image was in app-private `getExternalFilesDir()/assets`, the app could bypass the all-files-access prompt. The patch now requests the existing shared-storage permission (unless the user chose Not now) even when game assets are private, because the user explicitly requires logs in `/storage/emulated/0/ShaolinMonks`. The permission dialog text says it is used for game files and automatic co-op diagnostics. User may decline; when declined, no public log files are promised or written.
- Logs: native `co-op-session.jsonl`; Android `co-op-android.jsonl`; each 8 MiB, active plus four numbered archives; details and privacy limits in `COOP-DIAGNOSTIC-LOGGING.md`.
- Latest workflow on the new head: #41111651843, pull_request, status QUEUED at last check, no jobs/artifacts yet. Several previous queued PR runs exist because each branch commit triggered a workflow and concurrency uses cancel-in-progress=false. The exact latest head must produce a successful workflow and artifact before sending the APK.
- The prior run #41110956870 was malformed workflow YAML (zero jobs) caused by an earlier duplicated old PowerShell fragment. The workflow tail has since been rebuilt with exactly one APK verification step, one APK artifact upload, and one failure-log upload. Do not cite that failure as application build failure.
- No current-head APK available yet. Never provide previous shadow APK as containing auto-log integration.
- In-app diagnostic tests available after current APK is built: open Options → LAN CO-OP, HOST on phone A, JOIN from phone B with private IPv4, confirm RUNNING/peer, both press RUN PAIRED INPUT EXCHANGE within four seconds, repeat as three fresh sessions, STOP/RESET, then observe controlled Wi-Fi disconnect/reconnect. Only synthetic tick; no gameplay injection or shared simulation. Raw UDP RTT/loss still comes from Termux probe, not fabricated in-app metrics.
- All release/gameplay interlocks remain closed and no device-side acceptance has been performed.


### Shaolin Monks LAN Co-op — native diagnostic logger C++ compile fix checkpoint 2026-10-31

- Implementation repo: `orionrayy/project-memory-and-database`; PR #6 remains OPEN / DRAFT / UNMERGED. Branch `feat/shaolin-coop-shadow-input-experiment`; current head at checkpoint: `36d16044dc6821f2fe933214f64e2cd0a7b2680d`.
- The previous exact-head build run #41111651843 failed in native compilation at generated `src/common/smcoop_lan_diagnostic_ui.cpp:222`: logger concatenated `std::string("result=")` with `ToString(submitted)`, whose return type is `std::string_view`; libc++ rejects `std::string + std::string_view`.
- Fixed the generator `shaolinmonks-build-input/scripts/patch_coop_diagnostic_logging.py` to explicitly construct `std::string(ToString(submitted))` before concatenation. Patcher blob: `83f100cb91ae17d401a95d8d162a234e5cec5b34`.
- Added regression assertions in `shaolinmonks-build-input/tests/test_patch_coop_diagnostic_logging.py` to require the explicit string conversion and reject the invalid expression. Test blob: `7509062a84b4c2c5ba475fdd8b39b1f5b2b7787a`.
- Updated `.github/workflows/shaolinmonks-diagnostic-apk.yml` exact patch-blob pin to `83f100cb91ae17d401a95d8d162a234e5cec5b34`. Workflow blob: `59c734db7e812da7bfabd0d85c676d935ca074f3`.
- Atomic branch commit: `36d16044dc6821f2fe933214f64e2cd0a7b2680d`, message `fix(shaolinmonks): stringify shadow submit status for native build`. PR #6 head verified to match this commit.
- New exact-head workflow #41154653873 (`pull_request`, run number 111) was created for head `36d1604...`. At latest checkpoint, `preflight` is SUCCESS; `build-diagnostic-apk` remains IN_PROGRESS at step `Build native Android APK`. Checkout, pinned SDK install, private bundle extraction, fail-closed patching, Python regression suite, repeated native CTest (5x), and acceptance traceability all completed successfully. Job log endpoint did not provide logs while the job was still running.
- No APK artifact is available for head `36d1604...` yet. Only consider this build distributable after the same run completes SUCCESS, APK signature check passes, native `libmain.so` contains the shadow/session/save/diagnostic markers, DEX contains Android lifecycle/network markers, and the artifact is confirmed to belong to the exact head SHA. Do not send any older APK as containing automatic logs.
- Automatic log design is unchanged: `/storage/emulated/0/ShaolinMonks/co-op-session.jsonl` and `co-op-android.jsonl`, each 8 MiB with active plus four rotated archives; requires shared/all-files storage permission. No save/ISO contents, SSID/password, session token or public IP are logged; local/private IPs are logged for same-LAN diagnosis.
- Physical testing still required after APK build: host/join two devices on trusted private Wi-Fi/hotspot; both execute the one-tick synthetic paired-input shadow test within four seconds; repeat across three fresh sessions; STOP/RESET; retain both devices' logs. Raw RTT/loss still requires the Termux UDP calibration probe.
- Do not confuse shadow transport PASS with gameplay co-op. Real gameplay input injection remains disabled; simulation barrier remains unverified; `multiplayer_playable=false`; external acceptance gates incomplete; WAN/public Wi-Fi disabled; trusted-private-LAN-only. Keep PR #6 in draft and do not merge/claim playable.
