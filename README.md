# AI Orchestrator Core

Free execution path: GitHub Actions + stdlib Python control plane.

## Components

- `orchestrator/orchestrator.py`: DAG/state/retry/replan engine.
- `orchestrator/tools.json`: capability and tool registry.
- `gateway.py`: authenticated event ingress that emits a GitHub repository dispatch.
- `.github/workflows/orchestrator.yml`: execution worker and scheduled resume.
- `.github/workflows/orchestrator-tests.yml`: compile + unit-test gate.

## LLM routing

Gemini is the primary LLM adapter and OpenAI is an optional fallback. Google currently lists Gemini 3.7 Flash as free at the standard API tier. API-key authentication is still required.

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
