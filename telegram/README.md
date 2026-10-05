# Telegram control-plane frontend

This component turns Telegram into the orchestration frontend and command surface while keeping execution semantics outside Telegram.

Architecture:
Telegram Bot API -> Cloudflare Worker webhook -> allowlist and private-chat policy -> consent and encrypted session/idempotency/audit metadata -> HMAC gateway request -> existing VORENYX gateway -> GitHub repository_dispatch and orchestrator -> existing control-plane and worker fabric.

The database is a metadata plane. It does not replace workflow truth, effect fencing, checkpoints, or orchestration state.

## Interface
/start shows the privacy notice and authorization controls.
/run <goal> creates a dry-run orchestration.
/runlive <goal> requests live execution. It does not bypass the orchestrator's approval policy.
/status <workflow_id> returns a bounded workflow summary without returning the stored goal or input payload. It also lists a bounded set of pending approval node IDs. Git-durable workflows are read through the authenticated Gateway; distributed-control-plane workflows are read only from the control plane, preserving the workflow's immutable authority.
/resume <workflow_id> requests continuation.
/approvals <workflow_id> lists pending approval nodes. `/approve <workflow_id> [node_id]` applies approval to exactly one existing approval issue; when node_id is omitted, there must be exactly one pending approval.
/prompt on|off enables or disables direct plain-text prompting. With prompting enabled, ordinary text is treated as a dry-run goal.
/last returns the last workflow recorded for the Telegram chat.
/privacy, /revoke, /delete_me, /id, and /help provide privacy, consent, deletion, identity, and help controls.

## Security model
Only private chats are accepted. The bot does not read groups or channels. Status access is also bounded by workflow ownership: the owner can inspect their workflow, while configured approvers may inspect workflows relevant to approval handling.
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
1. Revoke/rotate any bot token previously pasted into chat. Store only the replacement in GitHub secret `TELEGRAM_BOT_TOKEN`.
2. Add GitHub repository secrets `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET`, `TELEGRAM_ALLOWED_USER_IDS`, `TELEGRAM_APPROVER_USER_IDS`, `TELEGRAM_DATA_ENCRYPTION_KEY`, `TELEGRAM_DATA_HMAC_KEY`, `GATEWAY_URL`, `GATEWAY_SHARED_SECRET`, `CONTROL_PLANE_URL`, and `CONTROL_PLANE_SECRET`. `TELEGRAM_D1_DATABASE_ID` is optional.
3. Ensure `TELEGRAM_APPROVER_USER_IDS` is a subset of `TELEGRAM_ALLOWED_USER_IDS`.
4. Run GitHub Actions → `Deploy Telegram Control Plane`. Leave `provision_d1=false` unless a new D1 database must be created.
5. Review the deployment summary and `/health` result before using the bot.
6. Configure BotFather exactly as documented in `telegram/BOOTSTRAP.md`: usage restricted, groups disabled, admin rights zero, and advanced modes disabled for v1.
7. Set the BotFather privacy-policy URL to the deployed Worker `/privacy` endpoint.
8. Smoke-test `/start` → Authorize → `/run verify Telegram control plane` → `/status <workflow_id>`.
9. Test `/approvals <workflow_id>` and node-scoped `/approve <workflow_id> <node_id>` only with a deliberately harmless approval-gated workflow.
10. Keep Telegram private-chat-only. Do not enable Bot-to-Bot, Secretary, Guest, Group, or Bot Management modes until their dedicated policy and control-plane contracts exist.
For production operations, keep Telegram as the interaction plane. Queueing, scheduling, workers, artifacts, effect reconciliation, and workflow truth stay in the existing orchestration architecture.