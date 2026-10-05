import {decryptJson, encryptJson, keyedIdentity} from "./crypto.js";

export function hasDatabase(env) {
  return Boolean(env.DB);
}

function requireDatabase(env) {
  if (!env.DB) throw new Error("database_not_configured");
  if (!env.TELEGRAM_DATA_HMAC_KEY) throw new Error("data_hmac_key_missing");
  if (!env.TELEGRAM_DATA_ENCRYPTION_KEY) throw new Error("data_encryption_key_missing");
}

export async function principalKey(env, telegramUserId) {
  if (!env.TELEGRAM_DATA_HMAC_KEY) throw new Error("data_hmac_key_missing");
  return keyedIdentity(env.TELEGRAM_DATA_HMAC_KEY, "telegram:user", telegramUserId);
}

export async function chatKey(env, telegramChatId) {
  if (!env.TELEGRAM_DATA_HMAC_KEY) throw new Error("data_hmac_key_missing");
  return keyedIdentity(env.TELEGRAM_DATA_HMAC_KEY, "telegram:chat", telegramChatId);
}

export async function getConsent(env, pKey) {
  requireDatabase(env);
  const row = await env.DB
    .prepare("SELECT policy_version, consented_at, revoked_at FROM telegram_consents WHERE principal_key=?")
    .bind(pKey)
    .first();
  return row && row.revoked_at == null ? row : null;
}

export async function setConsent(env, pKey, policyVersion) {
  requireDatabase(env);
  const now = Math.floor(Date.now() / 1000);
  await env.DB
    .prepare(
      "INSERT INTO telegram_consents(principal_key,policy_version,consented_at,revoked_at) VALUES(?,?,?,NULL) " +
      "ON CONFLICT(principal_key) DO UPDATE SET policy_version=excluded.policy_version,consented_at=excluded.consented_at,revoked_at=NULL",
    )
    .bind(pKey, policyVersion, now)
    .run();
}

export async function revokeConsent(env, pKey) {
  requireDatabase(env);
  const now = Math.floor(Date.now() / 1000);
  await env.DB
    .prepare("UPDATE telegram_consents SET revoked_at=? WHERE principal_key=?")
    .bind(now, pKey)
    .run();
}

export async function getSession(env, cKey) {
  requireDatabase(env);
  const row = await env.DB
    .prepare("SELECT encrypted_session, expires_at FROM telegram_sessions WHERE chat_key=?")
    .bind(cKey)
    .first();
  if (!row) return null;
  if (Number(row.expires_at) <= Math.floor(Date.now() / 1000)) {
    await env.DB.prepare("DELETE FROM telegram_sessions WHERE chat_key=?").bind(cKey).run();
    return null;
  }
  return decryptJson(env.TELEGRAM_DATA_ENCRYPTION_KEY, row.encrypted_session);
}

export async function saveSession(env, cKey, session, ttlSeconds = 7 * 24 * 60 * 60) {
  requireDatabase(env);
  const now = Math.floor(Date.now() / 1000);
  const encrypted = await encryptJson(env.TELEGRAM_DATA_ENCRYPTION_KEY, session);
  await env.DB
    .prepare(
      "INSERT INTO telegram_sessions(chat_key,encrypted_session,created_at,updated_at,expires_at) VALUES(?,?,?,?,?) " +
      "ON CONFLICT(chat_key) DO UPDATE SET encrypted_session=excluded.encrypted_session,updated_at=excluded.updated_at,expires_at=excluded.expires_at",
    )
    .bind(cKey, encrypted, now, now, now + ttlSeconds)
    .run();
}

export async function audit(env, {eventId, pKey, cKey, action, workflowId = null, intentDigest = null}) {
  requireDatabase(env);
  const now = Math.floor(Date.now() / 1000);
  await env.DB
    .prepare(
      "INSERT OR IGNORE INTO telegram_audit(event_id,principal_key,chat_key,action,workflow_id,intent_digest,created_at) VALUES(?,?,?,?,?,?,?)",
    )
    .bind(eventId, pKey, cKey, action, workflowId, intentDigest, now)
    .run();
}

export async function deleteUserData(env, pKey, cKey) {
  requireDatabase(env);
  await env.DB.batch([
    env.DB.prepare("DELETE FROM telegram_audit WHERE principal_key=?").bind(pKey),
    env.DB.prepare("DELETE FROM telegram_consents WHERE principal_key=?").bind(pKey),
    env.DB.prepare("DELETE FROM telegram_sessions WHERE chat_key=?").bind(cKey),
  ]);
}
