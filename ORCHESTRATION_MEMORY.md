# ORCHESTRATION MEMORY — Canonical Control-Plane Context

## Purpose

This file is the durable, version-controlled memory of the AI orchestration control plane. Before modifying the orchestrator, treat this document and the current source tree as the source of truth. Do not assume older conversation state is still accurate without checking the repository.

## Current baseline

Repository: `orionrayy/ai-determinism-engine`
Primary branch: `main`
Current main baseline: orchestration hardening v38 health-state sharding + federation recovery on top of v37 GitHub Actions matrix federation, v36 bounded federated protocol, v35 supervised multi-agent fabric, v34 workflow-sharded audit events, v33 committed state schema CI contract, v32 collision-resistant workflow identities, v31 canonical approval ingress/barrier optimization, v30 workflow-scoped concurrency, v29 state-persistence recovery, v28 bounded/fsynced audit state, v27 deterministic retry/reconciliation hardening, and the earlier secure live-boundary/durability-barrier generations.
Execution model: GitHub Actions + stdlib Python
Cost policy: free-first; `ORCHESTRATOR_FREE_ONLY=true` in the production workflow
Current execution-fabric branch: `main`

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
