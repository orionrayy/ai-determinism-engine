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

export async function claimUpdate(env, eventId, pKey, staleAfterSeconds = 300) {
  requireDatabase(env);
  const inboxKey = await keyedIdentity(env.TELEGRAM_DATA_HMAC_KEY, "telegram:update", eventId);
  const claimToken = crypto.randomUUID();
  const now = Math.floor(Date.now() / 1000);
  const expires = now + 7 * 24 * 60 * 60;
  await env.DB.batch([
    env.DB.prepare("DELETE FROM telegram_inbox WHERE expires_at <= ?").bind(now),
    env.DB.prepare("DELETE FROM telegram_sessions WHERE expires_at <= ?").bind(now),
    env.DB.prepare("DELETE FROM telegram_audit WHERE expires_at <= ?").bind(now),
    env.DB.prepare("DELETE FROM telegram_workflows WHERE expires_at <= ?").bind(now),
  ]);
  const inserted = await env.DB
    .prepare(
      "INSERT OR IGNORE INTO telegram_inbox(event_id,principal_key,status,claim_token,workflow_id,created_at,updated_at,expires_at) VALUES(?,?, 'processing',?,NULL,?,?,?,?)",
    )
    .bind(inboxKey, pKey, claimToken, now, now, expires)
    .run();
  if (Number(inserted?.meta?.changes || 0) > 0) {
    return {claimed: true, workflowId: null, claimToken};
  }
  const row = await env.DB
    .prepare("SELECT status,claim_token,workflow_id,updated_at FROM telegram_inbox WHERE event_id=?")
    .bind(inboxKey)
    .first();
  if (!row) return {claimed: false, duplicate: true, workflowId: null};
  const status = String(row.status || "");
  if (status === "completed") {
    return {claimed: false, duplicate: true, workflowId: row.workflow_id ? String(row.workflow_id) : null};
  }
  if (status === "processing" && Number(row.updated_at || 0) > now - staleAfterSeconds) {
    return {claimed: false, duplicate: true, workflowId: row.workflow_id ? String(row.workflow_id) : null};
  }
  const reclaimed = await env.DB
    .prepare(
      "UPDATE telegram_inbox SET status='processing',claim_token=?,workflow_id=NULL,updated_at=?,expires_at=? WHERE event_id=? AND (status <> 'processing' OR updated_at <= ?)",
    )
    .bind(claimToken, now, expires, inboxKey, now - staleAfterSeconds)
    .run();
  if (Number(reclaimed?.meta?.changes || 0) !== 1) {
    return {claimed: false, duplicate: true, workflowId: row.workflow_id ? String(row.workflow_id) : null, claimToken: null};
  }
  return {claimed: true, workflowId: null, claimToken};
}

export async function completeUpdate(env, eventId, workflowId, claimToken) {
  requireDatabase(env);
  if (!claimToken) return;
  const inboxKey = await keyedIdentity(env.TELEGRAM_DATA_HMAC_KEY, "telegram:update", eventId);
  const now = Math.floor(Date.now() / 1000);
  await env.DB
    .prepare("UPDATE telegram_inbox SET status='completed',workflow_id=?,updated_at=? WHERE event_id=? AND claim_token=? AND status='processing'")
    .bind(workflowId || null, now, inboxKey, claimToken)
    .run();
}

export async function failUpdate(env, eventId, claimToken) {
  requireDatabase(env);
  if (!claimToken) return;
  const inboxKey = await keyedIdentity(env.TELEGRAM_DATA_HMAC_KEY, "telegram:update", eventId);
  const now = Math.floor(Date.now() / 1000);
  await env.DB
    .prepare("UPDATE telegram_inbox SET status='failed',updated_at=? WHERE event_id=? AND claim_token=? AND status='processing'")
    .bind(now, inboxKey, claimToken)
    .run();
}

export async function bindWorkflow(env, workflowId, pKey, ttlSeconds = 90 * 24 * 60 * 60) {
  requireDatabase(env);
  const now = Math.floor(Date.now() / 1000);
  await env.DB.prepare(
    "INSERT OR IGNORE INTO telegram_workflows(workflow_id,principal_key,created_at,last_seen_at,expires_at) VALUES(?,?,?,?,?)"
  ).bind(workflowId, pKey, now, now, now + ttlSeconds).run();
  const row = await env.DB.prepare(
    "SELECT principal_key FROM telegram_workflows WHERE workflow_id=?"
  ).bind(workflowId).first();
  if (!row || String(row.principal_key) !== pKey) throw new Error("workflow_owner_conflict");
}

export async function touchWorkflow(env, workflowId, pKey) {
  requireDatabase(env);
  const now = Math.floor(Date.now() / 1000);
  await env.DB.prepare(
    "UPDATE telegram_workflows SET last_seen_at=?,expires_at=? WHERE workflow_id=? AND principal_key=?"
  ).bind(now, now + 90 * 24 * 60 * 60, workflowId, pKey).run();
}

export async function ownsWorkflow(env, workflowId, pKey) {
  requireDatabase(env);
  const row = await env.DB.prepare(
    "SELECT workflow_id FROM telegram_workflows WHERE workflow_id=? AND principal_key=? AND expires_at>?"
  ).bind(workflowId, pKey, Math.floor(Date.now() / 1000)).first();
  return Boolean(row);
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
  const auditKey = await keyedIdentity(env.TELEGRAM_DATA_HMAC_KEY, "telegram:audit", eventId);
  await env.DB
    .prepare(
      "INSERT OR IGNORE INTO telegram_audit(event_id,principal_key,chat_key,action,workflow_id,intent_digest,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?)",
    )
    .bind(auditKey, pKey, cKey, action, workflowId, intentDigest, now, now + 30 * 24 * 60 * 60)
    .run();
}

export async function deleteUserData(env, pKey, cKey) {
  requireDatabase(env);
  await env.DB.batch([
    env.DB.prepare("DELETE FROM telegram_inbox WHERE principal_key=?").bind(pKey),
    env.DB.prepare("DELETE FROM telegram_audit WHERE principal_key=?").bind(pKey),
    env.DB.prepare("DELETE FROM telegram_consents WHERE principal_key=?").bind(pKey),
    env.DB.prepare("DELETE FROM telegram_workflows WHERE principal_key=?").bind(pKey),
    env.DB.prepare("DELETE FROM telegram_sessions WHERE chat_key=?").bind(cKey),
  ]);
}
