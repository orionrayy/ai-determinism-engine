# Agent Instructions

## GitHub + Cloudflare operating rule

This repository participates in a cross-provider development environment. For every task that reads, searches, edits, tests, reviews, merges, or deploys this repository through GitHub:

1. Read the platform contract before substantive changes: [GitHub–Cloudflare Integration Contract](https://github.com/orionrayy/ai-determinism-engine/blob/main/CLOUDFLARE_GITHUB_INTEGRATION.md).
2. Use the connected GitHub account (`orionrayy`) for source, history, issues, PRs, and CI evidence.
3. Also inspect the connected Cloudflare account when the task can affect an app, API, deployment, database, storage, schedule, domain, or runtime. Verify current state; do not rely on stale chat context.
4. Map the repository to exact Cloudflare resources before deploying or changing them. Similar names are not proof of ownership.
5. Treat GitHub CI and Cloudflare runtime verification as separate acceptance gates.
6. Prefer free-first Cloudflare capabilities; check current limits and potential billing before provisioning.
7. Never put secrets in source, PRs, logs, or chat. Use provider-managed secrets and do not echo secret values.
8. Do not trigger production deployment, change DNS/routing, delete resources, revoke/rotate credentials, enable paid features, or run destructive migrations without explicit authorization for that operation.
9. Use preview/staging first. Keep production deployment gated until its own validation and authorization.
10. If no relevant Cloudflare resource exists, document that result rather than inventing a mapping or provisioning blindly.

## Repository context

- Repository: `orionrayy/ai-determinism-engine`.
- For `orionrayy/ai-determinism-engine`, `ORCHESTRATION_MEMORY.md` remains the canonical technical/project memory.
- GitHub remains the canonical source-control system. Cloudflare complements it as the runtime and infrastructure plane; do not replace Git history or PR review with Cloudflare Artifacts.
