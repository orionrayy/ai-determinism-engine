# AI Orchestrator Core

Canonical technical memory and operating protocol: `ORCHESTRATION_MEMORY.md`.

Free execution path: GitHub Actions + stdlib Python control plane.

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
