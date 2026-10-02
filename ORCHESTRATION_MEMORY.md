# ORCHESTRATION MEMORY — Canonical Control-Plane Context

## Purpose

This file is the durable, version-controlled memory of the AI orchestration control plane. Before modifying the orchestrator, treat this document and the current source tree as the source of truth. Do not assume older conversation state is still accurate without checking the repository.

## Execution preflight + resume plan immutability v25
- Terminal `completed` and `cancelled` workflows are idempotent on resume: the worker does not re-enter execution or emit a duplicate side effect.

- Persisted workflow plans are no longer re-routed during resume before plan-integrity verification. Tool selection is treated as part of the durable intent and changes only during explicit replanning or initial plan construction.
- Initial workflow creation still routes each node through the capability graph so free-tier, credential availability, risk, and health are considered before the plan is persisted.
- Live execution performs a local preflight before approval/barrier handling: tool registration, free-only policy, required credentials/secrets, quarantined health state, and HTTPS endpoint prerequisites are checked before any external side-effect START fence.
- A preflight failure can re-enter the existing bounded replan path because no side-effect durability barrier has been crossed.
- The ready state permits a policy/dependency failure transition to failed, making pre-execution failures explicit without pretending that an external action started.
- The AI Orchestrator workflow job explicitly accepts both [ORCHESTRATOR] source issues and [ORCHESTRATOR APPROVAL] issues so approval labels can reach the worker.
- Approval labels are accepted only during an authenticated `issues` `labeled` event carrying an allowlisted approval/rejection label; scheduled recovery cannot bootstrap approval from an unverified label.
## Lifecycle control plane v26

- Both run_one_step() and run_workflow() preserve the persisted node tool choice during resume; capability routing is reserved for initial plan construction or explicit replanning.
- Every logical node activation consumes one durable workflow execution-budget step before entering running or crossing the side-effect barrier.
- Default workflow budget is 96 logical steps with a hard maximum of 256; exhausted budgets fail closed and do not trigger another replan.
- The budget is persisted and schema-validated so crash recovery cannot silently reset the runaway-loop guard.
- Node contracts may declare deterministic postconditions: field_exists, field_equals, field_in, non_empty, or bounded http_status. These checks execute before a node is marked completed.
- Postcondition failure remains inside the existing failure policy, so a post-side-effect acceptance failure cannot silently trigger duplicate execution.
- Durable non-side-effect nodes interrupted while running are rearmed to ready on the next worker; their re-execution remains bounded by the same workflow execution-step budget.
- Manual workflow dispatch exposes max_steps; repository-triggered workflows use the production default unless explicitly configured.
## State-machine bounds hardening v41

- Workflow `replan_count` is now schema-bounded to 0–2, matching the runtime replan ceiling; malformed or excessive values fail closed instead of enabling unbounded replanning.
- Persisted node `retry_count`/`max_retries` are validated and bounded (`max_retries` <= 8) so corrupted state cannot create an unbounded retry loop.
- Node status/risk/id/capability/dependency and runtime payload container types are validated during migration before recovery can execute them.

## Private input signature hardening v44

- Private-input request signatures now bind HTTP method, request path, timestamp, and body. This prevents reusing a valid signature for a different reference/path within the replay window.
- Private-input execution identities are required to be 64-character lowercase hexadecimal values before an opaque reference is derived.
- The versioned private-input envelope remains vendor-neutral. The receiver is responsible for rejecting stale timestamps, verifying HMAC in constant time, enforcing expiration, and keeping the raw payload out of logs and public stores.

## Private input transport v43

- Structured live ingress can now accept live requests only when an authenticated HTTPS private-input store is configured; otherwise it fails closed with `private_input_unavailable`.
- The gateway stores the raw structured input in the private store and sends GitHub Actions only an opaque HMAC-derived `private_input_ref`, execution identity, intent fingerprint, and digest. Raw structured input is not copied into repository-backed state.
- The worker passes the private reference into the persisted workflow. For structured live connector operations, the planner is bypassed so private payload content is never sent to Gemini/OpenAI; the worker executes `external_domain.external_operation` through the connector bridge and fetches the payload just-in-time.
- Private connector payloads are injected only into the in-memory node during the upstream call and are removed from node input in a `finally` block before validation/persistence. The connector request ID and intent fingerprint therefore still bind to the actual payload while durable state retains only the reference/digest.
- The private-input protocol uses HTTPS, HMAC request signing, a deterministic secret-bound reference, bounded envelope/input size, digest verification, execution-identity binding, and a bounded TTL (24 hours by default, configurable up to 7 days).
- `contracts/private-input.schema.json` is the versioned envelope contract. The protocol is vendor-neutral; the backend must implement `POST /v1/inputs` and `GET /v1/inputs/{input_ref}`.
- No private-input backend is deployed or health-verified in this repository yet. Structured live execution remains operationally unavailable until a private store is provisioned, configured in the gateway/worker secrets, and verified end-to-end.

## Executor activation + dependency boundary hardening v42

- Fixed a regression in the two-phase parallel executor where the Phase 2 activation block was accidentally nested under the execution-budget exception path; successful budget admission now actually activates and executes the admitted batch.
- A preflight replan now returns to the worker/continuation boundary instead of reusing the stale in-memory admission batch. The durable replanned plan is therefore revalidated before another execution batch is built.
- DAG validation rejects malformed node runtime containers before intent serialization so invalid objects fail with an explicit contract error rather than an incidental attribute error.
- Test fixtures now match the strict schema, exact approval-issue binding, sanitized connector-response contract, and current workflow schema.
- The root dependency manifest is reserved for the stdlib control plane; legacy ML dependencies are isolated in `requirements-legacy.txt` so deployment/runtime environments do not inherit the unrelated ML stack.
- Render is pinned to Python 3.12 to match the control-plane CI runtime.

## Approval trust-boundary hardening v40

- Approval labels are now accepted only when the actor has `admin`, `maintain`, or `push` repository permission; `triage` is excluded even though it can manage issues/labels.
- This keeps issue/label moderation power separate from the authority to authorize high-risk orchestrator actions.

## Strict state identity validation v39

- State/workflow schema version fields now fail closed when explicitly malformed instead of being silently coerced to legacy version `1`.
- Each workflow dictionary key is bound to `workflow.id`; mismatches are rejected, while genuinely legacy records missing `id` are deterministically backfilled from the key.
- Worker run-attempt fields are schema-validated as positive integers or null.

## Durable output privacy v38

- Connector success output no longer persists the raw bridge URL; it carries only a hashed bridge target identity and sanitized discovery data.
- Raw connector endpoint configuration remains deployment-only data and is not written into public workflow checkpoints/evidence.

## Worker attempt binding v37

- Durable workflow worker identity now stores both `github_run_id` and `github_run_attempt`.
- Continuation matching requires an exact run ID and run attempt, so a rerun of an Actions workflow cannot satisfy a continuation event from an earlier attempt.
- `origin_github_run_id` and `origin_github_run_attempt` preserve first-worker provenance while `github_run_id`/`github_run_attempt` track the latest durable worker execution.

## Reconciliation endpoint binding v36

- Reconciliation requests now carry a fingerprint of the configured reconciliation endpoint in addition to the original connector execution target fingerprint.
- The orchestrator records that reconciliation-endpoint fingerprint on the failed connector node and refuses recovery when the endpoint changes before reconciliation.
- The bridge runtime independently verifies both execution target and reconciliation endpoint fingerprints before dispatching the recovery lookup.
- This keeps recovery fail-closed across connector migrations or routing changes instead of interpreting a new backend's answer as the outcome of the original side effect.

## Concurrent single-flight failure semantics v35

- Bridge-runtime single-flight now propagates the first flight's result or failure to requests that actually overlapped that flight; a waiting duplicate never starts an implicit second upstream attempt.
- After a failed flight is released, a later fresh request may retry normally; failures are not stored as completed idempotency results.
- This prevents concurrent duplicate callers from turning an uncertain/non-idempotent upstream failure into a second external attempt while retaining normal retry behavior for a new request cycle.

## Reconciliation target binding v34

- Connector reconciliation now re-discovers the current connector action contract and target fingerprint before contacting the reconciliation endpoint.
- If the saved execution contract fingerprint differs from the current action contract/target, reconciliation fails closed and does not query or mutate the connector target.
- The reconciliation request carries the target fingerprint; the bridge runtime independently compares it with the currently configured upstream target before dispatching reconciliation.
- Legacy workflows without a saved contract fingerprint remain compatible, while newly hardened executions are protected against target drift during recovery.

## Upstream failure boundary v33

- Bridge upstream HTTP responses outside 2xx are now raised as typed upstream failures instead of being cached as successful bridge responses.
- Upstream 5xx/network failures propagate as HTTP 502 from the bridge server so the orchestrator retains an explicit uncertain-transport signal; upstream 4xx remains a non-uncertain client/dependency failure.
- The orchestrator connector adapter also rejects a nested upstream non-2xx status as defense-in-depth, even if a bridge implementation incorrectly returns HTTP 200 around it.
- Failed upstream attempts release the local idempotency slot without populating the completed cache, allowing a later bounded retry to reach the upstream again.

## Connector target binding v32

- Connector discovery now exposes only a SHA-256 `target_fingerprint` for each configured upstream URL; the URL itself is not persisted in discovery output.
- The connector action contract fingerprint now binds both the normalized action contract and the discovered target fingerprint, so a route-target change after an uncertain connector failure fails closed before another POST.
- Bridge-runtime idempotency fingerprints also bind the target fingerprint, preventing a process-local cached response from being replayed against a different configured upstream target under the same request ID.
- This remains bounded by the upstream/provider idempotency boundary across bridge process restarts: the bridge itself has no durable cross-process cache.

## Parallel budget admission v31

- Parallel safe-node batches are now admitted against the remaining durable execution-step budget before any node in the batch is persisted as `running`.
- When fewer budget slots remain than the proposed safe parallel batch, the batch is deterministically truncated to the available capacity instead of partially reserving nodes and failing a later reservation.
- When no execution slots remain, the first deterministically selected ready node is failed at the admission boundary; no other node is left stranded in `running`.
- This keeps budget exhaustion compatible with scheduled recovery and preserves the durable redrive invariant that completed work is retained while only executable unfinished work is retried.

## Control-plane hardening v30

- Workflow creation now persists execution_budget.max_steps from ORCHESTRATOR_MAX_EXECUTION_STEPS, bounded to 1–256, instead of wiring the input only through YAML.
- New workflows use CURRENT_WORKFLOW_SCHEMA_VERSION (currently v6) rather than a stale hard-coded schema number.
- Connector bridge idempotency is single-flight per request ID inside a bridge process; identical concurrent requests wait for the first result instead of launching duplicate upstream calls.
- Idempotency cache entries are intent-bound and returned as defensive copies. Same-key/different-intent and same-key/different-in-flight-intent collisions fail closed.
- Continuation events are keyed as continuation:<workflow_run_id>:<run_attempt> and dispatched with the exact workflow ID. The continuation workflow uses a per-run/per-attempt concurrency group. The pinned actionlint version does not understand an explicit queue key, so the default single-pending behavior is retained.
- Reconciliation output now exposes the connector request ID separately from the durable internal execution ID.
- Discovered connector action specs are fingerprinted and pinned across retry attempts. Contract drift after an uncertain failure stops before the next upstream POST.
- The bridge idempotency cache remains process-local and best-effort. Upstream/provider-side idempotency remains the durable boundary across process restarts or multiple bridge replicas.
- No control-plane change in v30 executes a production side effect automatically.
- Scheduled recovery now treats a recently persisted running workflow with a current worker run ID as fresh: for five minutes it defers that workflow instead of competing with continuation. Manual `--resume` remains able to process a fresh running workflow. Missing/malformed worker timestamps fail open to recovery rather than silently suppressing it.

## Connector intent-bound idempotency v29

- Connector bridge `request_id` is now derived from protocol, workflow id, node id, connector, action, and payload, while transient goal/timestamp fields remain outside the identity.
- This separates the durable internal `execution_id` used for workflow recovery from the external connector idempotency key used for request deduplication.
- The bridge runtime records the request-intent fingerprint with its cached result and rejects reuse of the same idempotency key for a different intent instead of returning the wrong cached response.
- Retries of the exact same connector intent reuse the same request id; a replan that changes connector/action/payload gets a different request id.
- This follows the durable-redrive pattern where an execution identity and a request/client token have distinct purposes. citeturn897311search2turn897311search6
## Continuation chain binding v28

- `github_run_id` now means the latest worker Actions run that durably touched the workflow; it is updated on workflow persistence.
- `origin_github_run_id` preserves the first worker run for provenance and audit history.
- This closes the chained-continuation gap where a resumed workflow kept its original run id and the next `workflow_run` event could no longer correlate to the exact persisted workflow.
- The continuation workflow still matches the triggering run id against the exact persisted workflow and dispatches the exact `workflow_id`; it does not fall back to `last_workflow_id`.
- Workflow schema is now v6; top-level state schema remains v4.
## Route snapshot + policy integrity v27

- Each workflow now persists a deterministic route/policy snapshot covering the selected node tool, capability fallback candidates, risk/action context, the relevant tool policy fields, live mode, and the free-only setting.
- A SHA-256 `policy_fingerprint` binds that snapshot to the workflow. Resume checks the current registry/policy against the durable fingerprint before checkpoint recovery or node execution.
- A policy mismatch fails closed as `workflow.policy_drift`; the runner does not silently re-route a persisted plan or continue under changed side-effect/free-tier semantics.
- Legacy workflows without a policy fingerprint initialize one on their first post-v27 resume before any node execution; subsequent resumes are protected by the fingerprint.
- Explicit replanning refreshes both the route snapshot and policy fingerprint because the tool choice intentionally changes.
- Historical note: at v27, the durable route/policy fields were introduced before the workflow schema advanced to v6 in this hardening line; the top-level state schema remains v4.
## Current baseline

Repository: `orionrayy/ai-determinism-engine`
Primary branch: `main`
Main remains the production merge baseline at v26; the active hardening branch has advanced through v42. Always verify the current `main` ref before modifying.
Execution model: GitHub Actions + stdlib Python control plane.
Cost policy: free-first; `ORCHESTRATOR_FREE_ONLY=true` in the production workflow.
Current execution-fabric branch: `hardening/connector-intent-idempotency-v29` (draft; main remains separate until the full hardening line is reviewed/merged).

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
                                    +-> reconciling -> completed
                                                    `-> ready
```

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
