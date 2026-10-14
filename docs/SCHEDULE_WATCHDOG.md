# Scheduled-recovery watchdog

The repository's GitHub Actions `schedule` trigger remains the primary low-cost scheduler. A Cloudflare Worker Cron Trigger is the independent fallback wake-up source for recovery only; it never enables live tool execution or approves high-risk effects.

## Operation

- Runs every five minutes and reads the ten most recent `schedule` runs for `main`.
- Does nothing when the latest scheduled run is younger than 30 minutes, when a recent scheduled run is queued/in progress, or when the run history is unavailable.
- When stale, emits `repository_dispatch` with event type `orchestrator.recovery`. The GitHub workflow routes that event only to `schedule-recovery`, not to the planner/executor job.
- A Durable Object claim suppresses parallel/duplicate dispatches, allows only one successful dispatch per stale-run identity per cooldown, and applies a five-minute retry backoff to dispatch failures.
- It fails closed when GitHub credentials, run history, or the watchdog binding are missing.

## Deployment prerequisite

The Cloudflare script is not active merely because this file exists. After reviewing/merging the PR, deploy `control-plane/cloudflare` with Wrangler and set `GITHUB_RECOVERY_TOKEN` as a Worker secret (never store it in the repository):

```sh
cd control-plane/cloudflare
npx wrangler secret put GITHUB_RECOVERY_TOKEN
npx wrangler deploy
```

Use a dedicated, minimum-scope GitHub credential that can read Actions workflow runs and create repository dispatches for `orionrayy/ai-determinism-engine` (fine-grained repository access with Contents write is a typical required permission). Do not paste the token into command arguments, source code, CI logs, or this PR. Validate the deployed Cron Trigger and the first watchdog logs before treating the fallback as operational.

This does not replace/alter the canonical orchestration state authority, perform a blind workflow rerun, mutate `main` directly, or retry an external connector side effect. The existing workflow-scoped recovery and idempotency rules remain authoritative.
