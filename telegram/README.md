# Telegram control-plane frontend

This component turns Telegram into the orchestration frontend and command surface while keeping execution semantics outside Telegram.

Architecture:
Telegram Bot API -> Cloudflare Worker webhook -> allowlist and private-chat policy -> consent and encrypted session/idempotency/audit metadata -> HMAC gateway request -> existing VORENYX gateway -> GitHub repository_dispatch and orchestrator -> existing control-plane and worker fabric.

The database is a metadata plane. It does not replace workflow truth, effect fencing, checkpoints, or orchestration state.

## Interface
/start shows the privacy notice and authorization controls.
/run <goal> creates a dry-run orchestration.
/runlive <goal> requests live execution. It does not bypass the orchestrator's approval policy.
/status <workflow_id> returns a bounded workflow summary without returning the stored goal or input payload. It also lists a bounded set of pending approval node IDs.
/resume <workflow_id> requests continuation.
/approvals <workflow_id> lists pending approval nodes. `/approve <workflow_id> [node_id]` applies approval to exactly one existing approval issue; when node_id is omitted, there must be exactly one pending approval.
/prompt on|off enables or disables direct plain-text prompting. With prompting enabled, ordinary text is treated as a dry-run goal.
/last returns the last workflow recorded for the Telegram chat.
/privacy, /revoke, /delete_me, /id, and /help provide privacy, consent, deletion, identity, and help controls.

## Security model
Only private chats are accepted. The bot does not read groups or channels.
The bot token is loaded only from TELEGRAM_BOT_TOKEN. Never commit it.
The webhook uses Telegram's X-Telegram-Bot-Api-Secret-Token mechanism.
The user allowlist and approver allowlist are environment secrets. Approval authority is not inferred from usernames. The approval gateway targets the exact open GitHub approval issue for the requested workflow/node and rejects zero or multiple matches.
Direct prompting requires active consent. The prompt body is not written to the Telegram metadata database.
Gateway requests are HMAC-authenticated and use Telegram update_id as the webhook replay identity.
No Telegram data is used by this component as a training corpus or other unrelated machine-learning dataset.

## Database
database/telegram_schema.sql is the portable D1 and SQLite schema.
The first deployment should bind a Cloudflare D1 database as DB and apply the versioned migrations under workers/telegram-control-plane/migrations. database/telegram_schema.sql remains the portable schema snapshot used by tests and documentation. The logical contract can later be mapped to the PostgreSQL adapter planned by the execution-fabric program.
Do not place raw Telegram identifiers into the database schema, logs, audit fields, or workflow state.

## Deployment checklist
1. Rotate the bot token that was pasted into chat and use the replacement only as TELEGRAM_BOT_TOKEN.
2. Provision the D1 database and bind it as DB.
3. Load database/telegram_schema.sql.
4. Create separate secrets for the bot token, webhook secret, data encryption key, data HMAC key, gateway URL and secret, and control-plane URL and secret.
5. Set TELEGRAM_ALLOWED_USER_IDS to the exact numeric IDs allowed to operate the bot.
6. Set TELEGRAM_APPROVER_USER_IDS to a subset of the allowlist.
7. Deploy the worker.
8. Configure the Bot API webhook to /telegram/webhook with the same secret token used in TELEGRAM_WEBHOOK_SECRET.
9. Set the BotFather privacy-policy URL to the deployed /privacy endpoint.
10. Test /start -> Authorize -> /run a harmless dry-run goal -> /status workflow-id before attempting /runlive.
11. Keep the bot private-chat-only. Do not add Bot-to-Bot communication or group automation until a separate terms and policy review exists.

For production operations, keep Telegram as the interaction plane. Queueing, scheduling, workers, artifacts, effect reconciliation, and workflow truth stay in the existing orchestration architecture.