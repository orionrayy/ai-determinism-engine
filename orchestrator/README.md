# AI Orchestrator Core — Free Execution Path

This is a no-builder-quota control plane.

Architecture: **goal → DAG planner → state machine → execution adapters → validation → retry → checkpoint → persistent state**

The GitHub Actions runner is the execution worker. State is kept in `.orchestrator/state.json`, so a separate workflow SaaS is not required. Long DAGs run one node per worker run; `.github/workflows/orchestrator-continuation.yml` dispatches the next node only after the previous worker has completed and pushed state.

## Modes

- Dry-run is the default and has no external side effects.
- Live execution is persisted in the workflow state; scheduled runs can therefore resume a previously-live workflow.
- `ORCHESTRATOR_FREE_ONLY=true` is the default safety mode. It blocks OpenAI, Firecrawl, and generic webhook adapters during live execution so the runner cannot create an unexpected API bill.
- Free live execution can use Gemini (Free Tier), Wikipedia, GitHub Actions, and GitHub APIs. Deployment/publishing that requires an external paid API remains blocked until a free-compatible adapter is configured.
- High-risk nodes (for example deploy/publish) pause for explicit approval unless the run is invoked with `approve_high_risk=true`.

## LLM backends

- `GEMINI_API_KEY` enables the primary Gemini adapter. Google currently lists Gemini 3.7 Flash as free-of-charge at the standard API tier.
- `OPENAI_API_KEY` is an optional fallback.
- `ORCHESTRATOR_WEBHOOK_URL` for a generic HTTPS tool gateway.
- `ORCHESTRATOR_WEBHOOK_SECRET` for bearer authentication to that gateway.

A ChatGPT subscription does not itself provide an OpenAI API key or API billing.

## Triggers

- Manual: Actions → AI Orchestrator → Run workflow.
- Event-driven: `repository_dispatch` type `orchestrator.event` with payload `{ "goal": "..." }`.
- Scheduled: every 15 minutes for recovery of stalled/running workflows.

## Connector boundary

GitHub Actions cannot directly invoke the ChatGPT-installed connector catalog (Firecrawl, Notion, Figma, Canva, etc.). Those services need their own API credentials or a webhook gateway. The state machine is connector-agnostic and the adapters can be expanded without changing orchestration semantics.

## Alternative visual engines

Activepieces and Windmill both have free/open-source paths; n8n also publishes a self-hosted Community Edition. For a zero-new-builder-dependency route, this repository uses GitHub Actions instead.
