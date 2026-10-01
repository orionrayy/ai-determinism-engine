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

## Connector reconciliation v10

When a live connector request becomes uncertain, the bridge can optionally expose a read-only reconciliation endpoint. The orchestrator sends a signed request containing the deterministic request id, connector, and action. The upstream returns `applied`, `not_applied`, or `unknown`. The first two states allow safe continuation without blindly replaying an uncertain side effect; `unknown` remains fail-closed.

## Reconciliation privacy v11

Connector reconciliation persists only a sanitized state record (`applied`, `not_applied`, or `unknown`) plus identifiers and discovery metadata. Arbitrary upstream response bodies are intentionally excluded from persisted workflow state.
