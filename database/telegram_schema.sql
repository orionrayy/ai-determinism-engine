-- VORENYX Orchestrator Telegram metadata store
-- SQLite/D1-compatible reference schema. Raw Telegram IDs and prompt bodies are intentionally absent.
-- Application-layer encryption/HMAC is required for values stored in encrypted_session.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS telegram_inbox (
  event_id TEXT PRIMARY KEY,
  principal_key TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('processing','completed','failed')),
  workflow_id TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS telegram_inbox_expiry_idx ON telegram_inbox(expires_at);
CREATE INDEX IF NOT EXISTS telegram_inbox_principal_idx ON telegram_inbox(principal_key, created_at);

CREATE TABLE IF NOT EXISTS telegram_consents (
  principal_key TEXT PRIMARY KEY,
  policy_version TEXT NOT NULL,
  consented_at INTEGER NOT NULL,
  revoked_at INTEGER
);

CREATE INDEX IF NOT EXISTS telegram_consents_revoked_idx ON telegram_consents(revoked_at);

CREATE TABLE IF NOT EXISTS telegram_sessions (
  chat_key TEXT PRIMARY KEY,
  encrypted_session TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS telegram_sessions_expiry_idx ON telegram_sessions(expires_at);

CREATE TABLE IF NOT EXISTS telegram_audit (
  event_id TEXT PRIMARY KEY,
  principal_key TEXT NOT NULL,
  chat_key TEXT NOT NULL,
  action TEXT NOT NULL,
  workflow_id TEXT,
  intent_digest TEXT,
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS telegram_audit_principal_idx ON telegram_audit(principal_key, created_at);
CREATE INDEX IF NOT EXISTS telegram_audit_workflow_idx ON telegram_audit(workflow_id, created_at);
