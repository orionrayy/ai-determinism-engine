# Scheduled-recovery watchdog

The repository's GitHub Actions `schedule` trigger remains the primary low-cost scheduler. A Cloudflare Worker Cron Trigger is the independent fallback wake-up source for recovery only; it never enables live tool execution or approves high-risk effects.

## Operation

- Runs every five minutes and reads the ten most recent `schedule` runs for `main`.
- Sends a stable `User-Agent` on GitHub API calls. Schedule age is measured against actual invocation time (`Date.now()`), not the intended `controller.scheduledTime`, so delayed Cron invocations do not distort the stale threshold.
- Does nothing when the latest scheduled run is younger than 30 minutes, when a recent scheduled run is queued/in progress, or when the run history is unavailable.
- When stale, emits `repository_dispatch` with event type `orchestrator.recovery`. The GitHub workflow routes that event only to `schedule-recovery`, not to the planner/executor job.
- A Durable Object claim suppresses parallel/duplicate dispatches, allows only one successful dispatch per stale-run identity per cooldown, and applies a five-minute retry backoff to dispatch failures.
- It fails closed when GitHub credentials, run history, or the watchdog binding are missing.

## Deployment prerequisite

The Worker must not be deployed without its GitHub credential. `wrangler.toml` declares `GITHUB_RECOVERY_TOKEN` as required, so a deployment without it should fail validation.

The repository's existing `.github/workflows/telegram-control-plane-deploy.yml` currently places only `CONTROL_PLANE_SECRET` in the Control Plane `--secrets-file`; it does not inject `GITHUB_RECOVERY_TOKEN`. Do not use that automated deployment path for this configuration until the credential flow is explicitly approved, wired, and covered by a deployment contract test. The private secrets-file procedure below remains the documented path; no credential has been created or deployed by this change.

Create a small JSON secrets file **outside the repository** and restrict access to it. Its content must have this shape; replace the placeholder locally and never paste the real token into ChatGPT or commit the file:

```json
{"GITHUB_RECOVERY_TOKEN":"<set-this-locally-in-a-private-file>"}
```

Then deploy code and secret together:

```sh
cd control-plane/cloudflare
npx wrangler deploy --secrets-file "$HOME/.config/ai-determinism-engine/watchdog-secrets.json"
```

Use a dedicated, minimum-scope fine-grained GitHub credential with **Actions: read** and **Contents: write** permissions for `orionrayy/ai-determinism-engine`. The watchdog does not read or log the token value. The documented `wrangler secret put GITHUB_RECOVERY_TOKEN` command is an alternative after the Worker already exists; it deploys a new version immediately.

After deployment, validate that the Cron Trigger is listed and inspect the first watchdog logs. The connected Cloudflare account previously listed only `sugarnymous-staging-edge`, so this Worker has not been deployed by this task; the PR intentionally does not create production resources.

This does not replace/alter the canonical orchestration state authority, perform a blind workflow rerun, mutate `main` directly, or retry an external connector side effect. The existing workflow-scoped recovery and idempotency rules remain authoritative.
