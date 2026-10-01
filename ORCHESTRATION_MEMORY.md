# ORCHESTRATION MEMORY — Canonical Control-Plane Context

## Purpose

This file is the durable, version-controlled memory of the AI orchestration control plane. Before modifying the orchestrator, treat this document and the current source tree as the source of truth. Do not assume older conversation state is still accurate without checking the repository.

## Current baseline

Repository: `orionrayy/ai-determinism-engine`
Primary branch: `main`
Current main baseline: execution-fabric v10 reconciliation support is merged; always verify the current `main` ref before modifying.
Execution model: GitHub Actions + stdlib Python
Cost policy: free-first; `ORCHESTRATOR_FREE_ONLY=true` in the production workflow
Current execution-fabric branch: `main`

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
1. retries with bounded exponential backoff;
2. returns to `running` before a second execution attempt;
3. may use a registry fallback tool;
4. may replan up to the global limit;
5. otherwise fails the workflow.

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
3. Side-effect recovery is fail-closed, not automatically reconciled. An `execution_uncertain` workflow needs external-state inspection before it can safely be resumed.
4. Connector bridge v1 is vendor-neutral; a real Notion/Figma/Canva/ClickUp/etc. bridge service must implement the protocol and its own vendor OAuth/API policy.
5. Semantic validation now has deterministic output contracts plus an optional Gemini validation worker; domain-specific validators for deployments, published artifacts, and vendor objects remain to be added.
6. Connector bridge v1 is still an execution boundary rather than direct access to the ChatGPT-installed connector catalog.
7. Connector action contracts are schema-aware but intentionally bounded to required fields, primitive types, and an idempotency declaration; vendor-specific OAuth semantics and richer JSON Schema are still outside the core.
8. There is no dedicated distributed database or event bus; GitHub Actions + committed state is intentionally the zero-new-service implementation.

## Deployment targets

The connector bridge runtime can be hosted as a Vercel Python Function (`api/bridge.py`) or as a Render Web Service (`bridge_server.py`). Both expose the same protocol runtime and require `ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET` plus `ORCHESTRATOR_CONNECTOR_ROUTES`. Deployment is not considered verified until the public `/health` endpoint responds successfully.

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
7. Merge only after green.
8. Update this memory file when architecture, policy, or a known limitation changes.

## Design principle

The system should fail closed on unsafe tool selection, fail open on optional observability, remain deterministic under duplicate events, and preserve explicit human approval for irreversible external effects.
