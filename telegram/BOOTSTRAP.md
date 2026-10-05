# Telegram deployment bootstrap

This is an activation runbook only. No bot token is stored here.

## 1. Rotate the exposed token
The token pasted into a chat must be revoked in @BotFather and replaced. Telegram credentials are confidential and must not be public.

## 2. Provision D1
From `workers/telegram-control-plane/`:

`npx wrangler d1 create ai-orchestrator-telegram --binding DB --use-remote --update-config`

Copy the generated binding into `wrangler.jsonc` if Wrangler does not update the intended config.

Apply the schema from the repository root:

`npx wrangler d1 execute ai-orchestrator-telegram --remote --file=database/telegram_schema.sql`

Cloudflare documents `wrangler d1 create` for provisioning and `wrangler d1 execute --file` for applying a SQL file.

## 3. Generate secrets
Use independent random values. The data encryption key must represent 32 random bytes; do not reuse the HMAC key.

`python -c "import secrets,base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('='))"`

Use the result for `TELEGRAM_DATA_ENCRYPTION_KEY`, and generate a separate 32-byte value for `TELEGRAM_DATA_HMAC_KEY`.

Use a separate random URL-safe value for `TELEGRAM_WEBHOOK_SECRET`.

## 4. Load Worker secrets
Wrangler supports `wrangler secret put`; required secrets declared in `wrangler.jsonc` can be validated during deployment.

Set these values interactively, never commit them:

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

`TELEGRAM_APPROVER_USER_IDS` should be a subset of the main allowlist.

## 5. Deploy
`npx wrangler deploy`

## 6. Configure Telegram webhook
Telegram's Bot API supports an HTTPS webhook plus a `secret_token`; the resulting request contains `X-Telegram-Bot-Api-Secret-Token`. It also allows limiting update types.

Use the rotated replacement token configured as the Worker secret; do not paste it into source control or workflow input.

`curl -sS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" -d "url=${TELEGRAM_WORKER_URL}/telegram/webhook" -d "secret_token=${TELEGRAM_WEBHOOK_SECRET}" -d 'allowed_updates=["message","callback_query"]' -d 'drop_pending_updates=true'`

Keep `drop_pending_updates=true` only when intentionally discarding updates accumulated before activation.

## 7. Configure the command menu
Telegram supports `setMyCommands`; set the public menu to `/start`, `/help`, `/menu`, `/run`, `/runlive`, `/status`, `/resume`, `/approve`, `/prompt`, `/last`, `/privacy`, `/revoke`, and `/delete_me`. The Bot API also supports a private-chat menu button for commands or a Web App.

Set the privacy-policy URL for the bot in @BotFather to the deployed `/privacy` endpoint.

## 8. First smoke test
Use the bot in a private chat only:
`/start` -> Authorize -> `/run verify Telegram control plane` -> `/status <workflow_id>`.

Do not test `/runlive` until Gateway, control-plane, database, ownership, and approval configuration are confirmed.