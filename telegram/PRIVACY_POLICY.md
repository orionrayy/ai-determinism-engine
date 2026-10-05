# VORENYX Orchestrator Core — Privacy Policy

Effective date: 2026-10-05

This Telegram bot is a third-party application and is not operated by Telegram.

## Purpose
The bot provides a command and user-interface layer for the VORENYX orchestration system. It receives commands and prompts that a user explicitly sends to the bot, routes approved requests to the orchestration gateway, and returns bounded workflow information.

## Data minimization
The bot accepts private-chat messages only. Group and channel content is not collected or processed.
Raw Telegram user IDs and chat IDs are not persisted. Persistent identifiers are HMAC-derived keys.
The Telegram database stores only consent state, short-lived encrypted session state, webhook replay/idempotency state, and bounded audit metadata. Prompt/message bodies are not stored in the Telegram database.
Session data is encrypted at rest with an application-managed AES-256-GCM key. The HMAC key used for derived identifiers is separate from the encryption key.

## AI and downstream processing
Telegram-derived data is not collected for model training, fine-tuning, dataset construction, or unrelated analytics.
A prompt or command that a user explicitly submits may be forwarded to the orchestration engine. For live execution, the orchestrator may route the requested operation to external providers according to its own policy, approval, cost, privacy, and reconciliation rules. The Telegram bot does not silently broaden a request into unrelated data collection.

## Consent
Before orchestration commands are enabled, the user must actively authorize the stated processing purpose. Authorization is revocable.
Use /revoke to revoke consent. Use /delete_me to request deletion of bot-owned persistent records associated with the account.

## Retention and deletion
Webhook replay records, sessions, and audit metadata are bounded operational records with explicit expiration/cleanup paths. Deletion requests remove bot-owned records associated with the user's HMAC-derived identity.
The orchestrator's own workflow state or external provider records are outside the Telegram metadata database and remain subject to their respective retention/deletion policies.

## Security
Secrets are server-side only and are not embedded in source control. Access is allowlisted by Telegram user ID. Webhook requests require Telegram's secret-token header. Gateway requests use an HMAC signature bound to method, path, idempotency key, timestamp, and body.
The bot is private-chat-only by default, does not request passwords or one-time codes, and does not attempt to circumvent Telegram rate limits, moderation, or account restrictions.

## Contact
The deployment exposes this policy at the worker /privacy endpoint. That URL should be used as the bot's public privacy-policy URL in BotFather.