# Telegram deployment bootstrap

This repository includes a reproducible activation workflow: `.github/workflows/telegram-control-plane-deploy.yml`.

## Security prerequisite
Rotate/revoke any bot token previously pasted into chat and use only the replacement as the GitHub repository secret `TELEGRAM_BOT_TOKEN`. Never place the token in source, workflow inputs, issue bodies, logs, or this document.

## Required GitHub repository secrets
`CLOUDFLARE_ACCOUNT_ID`
`CLOUDFLARE_API_TOKEN`
`TELEGRAM_BOT_TOKEN`
`TELEGRAM_WEBHOOK_SECRET`
`TELEGRAM_ALLOWED_USER_IDS`
`TELEGRAM_APPROVER_USER_IDS`
`TELEGRAM_DATA_ENCRYPTION_KEY`
`TELEGRAM_DATA_HMAC_KEY`
`GATEWAY_URL`
`GATEWAY_SHARED_SECRET`
`CONTROL_PLANE_URL`
`CONTROL_PLANE_SECRET`

`TELEGRAM_D1_DATABASE_ID` is optional. Omit it when the activation workflow should discover the existing D1 database by name. Set `provision_d1=true` when a new D1 database should be created.

`TELEGRAM_APPROVER_USER_IDS` must be a subset of `TELEGRAM_ALLOWED_USER_IDS`.

## Generate cryptographic secrets
The encryption key must represent 32 random bytes. Keep the encryption key and HMAC key independent.

`python -c "import secrets,base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('='))"`

Use independent values for `TELEGRAM_DATA_ENCRYPTION_KEY`, `TELEGRAM_DATA_HMAC_KEY`, and `TELEGRAM_WEBHOOK_SECRET`.

## Run activation
After this PR is merged into the repository's default branch, open GitHub Actions → `Deploy Telegram Control Plane` and run it manually. GitHub requires a `workflow_dispatch` workflow to exist on the default branch before the manual Run workflow control is available. You can then select the desired branch/ref for the run. citeturn550058search0turn550058search2

`provision_d1=false` is the safe default. Use `true` only when D1 has not been provisioned and the Cloudflare token has permission to create databases.

`drop_pending_updates=false` is the safe default. Set it to `true` only when deliberately discarding updates accumulated before activation.

The workflow resolves/provisions D1, applies the versioned migration `workers/telegram-control-plane/migrations/0001_telegram_control_plane.sql`, generates an ephemeral Wrangler configuration, injects secrets through a temporary file, deploys the Worker, configures the Telegram webhook and command menu, sets the command menu button, and verifies `/health`.

## BotFather configuration
Recommended v1:
- Restrict bot usage: ON.
- Allow Groups: OFF.
- Group Privacy: ON.
- Group Admin Rights: 0.
- Channel Admin Rights: 0.
- Inline Mode: OFF.
- Bot Management Mode: OFF.
- Guest Chat Mode: OFF.
- Guard Mode: OFF.
- Secretary Mode: OFF.
- Bot-to-Bot Communication Mode: OFF.
- Threaded Mode: OFF.
- Privacy Policy: deployed Worker `/privacy` URL.

Do not enable Bot-to-Bot or Bot Management Mode until their separate policy, secret-vault, loop-control, and ownership design is implemented.

## First smoke test
Use a private Telegram chat only:

`/start` → Authorize → `/run verify Telegram control plane` → `/status <workflow_id>`.

Then:

`/approvals <workflow_id>`

Only after Gateway, D1, control plane, ownership, and approval credentials are confirmed should `/runlive` be tested.

## Operational boundary
Telegram is the frontend/control plane. Workflow truth, queueing, worker execution, effect reconciliation, artifacts, and scheduling remain in the VORENYX orchestration architecture.