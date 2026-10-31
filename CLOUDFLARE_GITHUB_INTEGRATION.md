# GitHub–Cloudflare Integration Contract

Last audited: 2026-11-01 (UTC)
Status: GitHub connector and Cloudflare connector are authenticated, but Cloudflare's Git provider authorization is not yet connected.

## Canonical ownership

- GitHub account: `orionrayy` (GitHub numeric user ID `155287930`).
- Cloudflare account: `bedf587d0956d6b4a622b3aa5511c396`.
- GitHub remains the canonical source-control and review system: repositories, branches, commits, PRs, issues, and code history.
- Cloudflare is the runtime/infrastructure plane: Workers, Pages, D1, KV, R2, Queues, Cron, DNS/security, deployments, and runtime observability where applicable.
- Do not migrate Git hosting to Cloudflare Artifacts merely to enable this integration. Keep GitHub as the source of truth unless the user explicitly changes that policy.

## Mandatory paired workflow

Whenever work uses GitHub for one of these repositories:

1. Identify the exact repository and branch/commit in GitHub before acting.
2. Check the connected Cloudflare account identity and inspect the relevant Cloudflare resources for that project. Do not rely on a stale inventory.
3. Treat GitHub CI as code/build evidence and Cloudflare runtime/deployment state as a separate evidence source. A passing GitHub workflow is not proof of a successful Cloudflare deployment.
4. Map each repository to an exact Worker/Pages project, D1 database, KV namespace, bucket, queue, domain, or other resource before using it. Never infer ownership solely from a similar name.
5. Keep GitHub as the source of code changes and review. Use Cloudflare for deploy/runtime validation when the project is actually hosted there.
6. Prefer free-tier-compatible services and check current limits/pricing before provisioning or increasing usage.
7. Keep secrets out of Git history, PRs, logs, and chat. Use the provider's secret store; never echo secret values.
8. Do not automatically deploy production, alter DNS/routing, delete resources, rotate/revoke credentials, enable paid features, or run destructive migrations without explicit user authorization.
9. For deployments, use preview/staging first, validate health and logs, then require a separate production approval unless the user clearly authorizes that specific production deployment.
10. If no Cloudflare resource is mapped to a repository, state that fact and do not provision resources blindly.

## Account inventory observed on 2026-11-01

### GitHub repositories visible to the connected account

- `orionrayy/ai-determinism-engine` — public
- `orionrayy/ace-math-puzzle` — private
- `orionrayy/project-memory-and-database` — private
- `orionrayy/projects` — public
- `orionrayy/sugarnymous-platform` — private
- `orionrayy/black-iso-android` — private

### Cloudflare resources

Account ID: `bedf587d0956d6b4a622b3aa5511c396`

Workers:
- `sol-scalper-monitor` — tag `565044f240d74ce4b845d9f20853f871`
- `sugarnymous-native-edge` — tag `055c1a9a870342a083f9499334dac731`
- `sugarnymous-staging-edge` — tag `8c138fe817774cdfbb34c07be6acdebf`
- `sugarnymous-static-site` — tag `8d8f4dd34ce44130b03ff058d8177e87`
- `sugarnymous-web-staging` — tag `a198c9aa8a3d412cb367427cdd0a3d02`

Other resources:
- D1: `dari-kasir-production` (UUID `6f6d4dc3-9a32-41f5-bdfe-5f32d0956508`; do not assume it belongs to another project).
- KV: `SOL_SCALPER_STATE` (namespace `18b40cca789447048d2c07e2ecd20f38`).
- Pages projects: none observed.
- Queues: none observed.
- R2: API reports it must first be enabled in the Cloudflare dashboard; do not assume R2 is provisioned.
- Workers Builds: no build configurations, triggers, or build tokens were found for the five listed Workers.
- All five Workers reported `last_deployed_from: api`, not a connected Git repository.

### Repository mapping state

- `sol-scalper-monitor` is a Cloudflare Worker whose name suggests SOL/Sentinel use, but its source-repository mapping has not been verified. Do not connect it to `ai-determinism-engine` until the source and deployment root are inspected.
- The four `sugarnymous-*` Workers appear project-related by naming, but exact source roots and deployment commands have not yet been verified. Inspect each repository's Wrangler config and worker source before creating triggers.
- No mapping is yet established for the other repositories.

## Git provider authorization blocker

A read-only audit found no Workers Builds trigger/configuration for the existing Workers. An attempted repository-connection upsert for `orionrayy/ai-determinism-engine` was rejected with Cloudflare error `8000008`: the Cloudflare project is disconnected from the Git account.

Cloudflare's documented prerequisite is to authorize/install the Cloudflare GitHub App from the Cloudflare dashboard before configuring Workers Builds. This is a user OAuth/installation step and must not be bypassed.

Dashboard route: https://dash.cloudflare.com/?to=/:account/workers-and-pages

Suggested safe one-time action:
1. Open Workers & Pages, select an existing Worker, then Settings → Builds → Connect → GitHub.
2. Complete the Cloudflare GitHub App authorization for the `orionrayy` account and grant access only to repositories intended for integration.
3. Stop before saving a Worker/repository build configuration. Saving that configuration can enable automatic deployment on pushes; no production auto-deploy was authorized by this integration request.
4. Return to the agent. Re-check Git provider authorization, then create repository connections and preview/staging build configurations only after validating the source mapping and deploy command.

## Workload placement policy

- Keep Android/Java/Kotlin APK compilation, native reverse engineering, heavy Python backtests, and other full Linux/VM workloads in GitHub Actions or an appropriate runner. Cloudflare Workers are not a general-purpose Linux build host.
- Use Workers for edge APIs, webhooks, lightweight scheduled work, auth gateways, and lightweight orchestration.
- Use D1 only for relational metadata and workloads compatible with SQLite/D1 semantics; do not move an existing SQLite database without schema, transaction, and concurrency review.
- Use KV for small, eventually consistent lookup/state data, not as a transactional database.
- Use R2 for object/artifact storage only after the account feature is enabled and cost/retention/access policies are validated.
- Use Queues/Workflows/Durable Objects when their execution and state semantics are appropriate; verify plan limits before adoption.

## Completion criteria

The integration is considered fully wired only after:
- Cloudflare GitHub App authorization is confirmed;
- the correct repository-to-Worker mappings are verified from source configuration;
- preview/staging build configurations exist and are tested;
- build status and preview links are visible in GitHub;
- production deployment remains gated until separately approved;
- a real test commit produces a verifiable Cloudflare build and expected runtime behavior.
