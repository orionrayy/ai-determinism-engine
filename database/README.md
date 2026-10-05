# Telegram control-plane database

This directory is the free-first metadata-store contract for the Telegram frontend.

The database is deliberately not the orchestration source of truth. Workflow state, effect fencing and execution semantics remain in the orchestration/control-plane layers. This database contains only Telegram-facing authorization, session, webhook-idempotency and audit metadata.

Privacy boundary:
- Raw Telegram user IDs and chat IDs are not persisted.
- Prompt/message bodies are not persisted.
- Telegram-derived keys are HMAC-derived with an application-held key.
- Session blobs are encrypted with a separate AES-256-GCM application key.
- Audit rows contain action names, workflow references and bounded digests only.
- The bot provides /delete_me to remove stored records associated with the account.
- No public group/channel content is collected or processed.

Recommended first deployment target: Cloudflare D1. The schema is SQLite-compatible and keeps the Telegram metadata plane separate from hot workflow/effect state.

The enterprise runtime can later map the same logical contract to the PostgreSQL adapter planned in the execution-fabric program. Provisioning an external database remains an explicit deployment operation.