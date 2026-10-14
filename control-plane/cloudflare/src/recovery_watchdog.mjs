const REPOSITORY = "orionrayy/ai-determinism-engine";
const WORKFLOW_PATH = "orchestrator.yml";
const DEFAULT_STALE_SECONDS = 1800;
const DEFAULT_DISPATCH_COOLDOWN_SECONDS = 1800;
const CLAIM_LEASE_SECONDS = 90;
const FAILED_RETRY_SECONDS = 300;
const MAX_API_BODY_BYTES = 256 * 1024;

function structuredLog(event, details = {}) {
  console.log(JSON.stringify({component: "github-schedule-watchdog", event, ...details}));
}

function jsonResponse(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {"content-type": "application/json; charset=utf-8", "cache-control": "no-store"},
  });
}

function clampInteger(value, fallback, minimum, maximum) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) return fallback;
  return Math.max(minimum, Math.min(parsed, maximum));
}

async function githubRequest(token, path, init = {}) {
  const headers = {
    accept: "application/vnd.github+json",
    authorization: "Bearer " + token,
    "x-github-api-version": "2022-11-28",
    ...(init.headers || {}),
  };
  if (init.body !== undefined) headers["content-type"] = "application/json";
  const response = await fetch("https://api.github.com" + path, {...init, headers});
  const text = await response.text();
  if (new TextEncoder().encode(text).byteLength > MAX_API_BODY_BYTES) {
    throw new Error("github_api_response_too_large");
  }
  if (!response.ok) {
    throw new Error("github_api_http_" + response.status + ":" + text.replace(/\s+/g, " ").slice(0, 180));
  }
  if (response.status === 204 || !text) return null;
  try {
    return JSON.parse(text);
  } catch (_) {
    throw new Error("github_api_invalid_json");
  }
}

function timestampSeconds(run) {
  const parsed = Date.parse(run?.run_started_at || run?.created_at || "");
  return Number.isFinite(parsed) ? Math.floor(parsed / 1000) : null;
}

async function watchdogRequest(env, action, payload) {
  if (!env.RECOVERY_WATCHDOG) throw new Error("recovery_watchdog_binding_missing");
  const id = env.RECOVERY_WATCHDOG.idFromName(REPOSITORY + ":main");
  const stub = env.RECOVERY_WATCHDOG.get(id);
  const response = await stub.fetch("https://recovery-watchdog.internal/" + action, {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(payload),
  });
  const body = await response.text();
  if (!response.ok) throw new Error("recovery_watchdog_http_" + response.status + ":" + body.slice(0, 160));
  try {
    return JSON.parse(body);
  } catch (_) {
    throw new Error("recovery_watchdog_invalid_json");
  }
}

export class RecoveryWatchdog {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
  }

  async fetch(request) {
    if (request.method !== "POST") return jsonResponse({error: "method_not_allowed"}, 405);
    const url = new URL(request.url);
    if (!["/claim", "/finish"].includes(url.pathname)) return jsonResponse({error: "not_found"}, 404);
    const raw = await request.text();
    if (new TextEncoder().encode(raw).byteLength > 4096) return jsonResponse({error: "body_too_large"}, 413);
    let body;
    try {
      body = JSON.parse(raw);
    } catch (_) {
      return jsonResponse({error: "invalid_json"}, 400);
    }
    if (!body || typeof body !== "object" || Array.isArray(body)) return jsonResponse({error: "body_must_be_object"}, 400);
    const key = String(body.key || "").trim();
    const now = Number(body.now);
    if (!key || key.length > 180 || !Number.isInteger(now) || now < 0) {
      return jsonResponse({error: "watchdog_identity_invalid"}, 400);
    }

    if (url.pathname === "/claim") {
      const result = await this.ctx.storage.transaction(async (txn) => {
        const current = await txn.get("dispatch_state");
        if (current?.status === "inflight" && Number(current.lease_until || 0) > now) {
          return {allowed: false, reason: "claim_in_flight"};
        }
        if (current?.key === key && current.status === "sent" && Number(current.retry_after || 0) > now) {
          return {allowed: false, reason: "dispatch_cooldown"};
        }
        if (current?.key === key && current.status === "retryable" && Number(current.retry_after || 0) > now) {
          return {allowed: false, reason: "retry_cooldown"};
        }
        const claimId = crypto.randomUUID();
        await txn.put("dispatch_state", {
          key,
          status: "inflight",
          claim_id: claimId,
          lease_until: now + CLAIM_LEASE_SECONDS,
          retry_after: now + FAILED_RETRY_SECONDS,
        });
        return {allowed: true, claim_id: claimId};
      });
      return jsonResponse(result);
    }

    const claimId = String(body.claim_id || "");
    if (!claimId || typeof body.ok !== "boolean") return jsonResponse({error: "watchdog_finish_invalid"}, 400);
    const cooldownSeconds = clampInteger(body.cooldown_seconds, DEFAULT_DISPATCH_COOLDOWN_SECONDS, 300, 7200);
    const result = await this.ctx.storage.transaction(async (txn) => {
      const current = await txn.get("dispatch_state");
      if (!current || current.key !== key || current.status !== "inflight" || current.claim_id !== claimId) {
        return {updated: false, reason: "claim_mismatch"};
      }
      await txn.put("dispatch_state", body.ok ? {
        key,
        status: "sent",
        sent_at: now,
        retry_after: now + cooldownSeconds,
      } : {
        key,
        status: "retryable",
        failed_at: now,
        retry_after: now + FAILED_RETRY_SECONDS,
      });
      return {updated: true, status: body.ok ? "sent" : "retryable"};
    });
    return jsonResponse(result, result.updated ? 200 : 409);
  }
}

export async function runScheduledRecoveryWatchdog(controller, env) {
  const token = String(env?.GITHUB_RECOVERY_TOKEN || "").trim();
  if (!token) {
    structuredLog("disabled", {reason: "missing_github_recovery_token"});
    return {status: "disabled", reason: "missing_github_recovery_token"};
  }

  const scheduledMs = Number(controller?.scheduledTime);
  const nowMs = Number.isFinite(scheduledMs) && scheduledMs > 0 ? scheduledMs : Date.now();
  const now = Math.floor(nowMs / 1000);
  const staleSeconds = clampInteger(env.ORCHESTRATOR_WATCHDOG_STALE_SECONDS, DEFAULT_STALE_SECONDS, 900, 7200);
  const cooldownSeconds = clampInteger(env.ORCHESTRATOR_WATCHDOG_DISPATCH_COOLDOWN_SECONDS, DEFAULT_DISPATCH_COOLDOWN_SECONDS, 900, 7200);

  const path = "/repos/" + REPOSITORY + "/actions/workflows/" + WORKFLOW_PATH +
    "/runs?branch=main&event=schedule&per_page=10";
  const payload = await githubRequest(token, path, {method: "GET"});
  if (!payload || !Array.isArray(payload.workflow_runs)) {
    throw new Error("github_schedule_runs_missing");
  }
  const runs = payload.workflow_runs
    .filter((run) => run && run.event === "schedule" && run.head_branch === "main" && run.id !== undefined)
    .sort((a, b) => (timestampSeconds(b) ?? -1) - (timestampSeconds(a) ?? -1));
  if (!runs.length) {
    structuredLog("no_schedule_baseline", {repository: REPOSITORY, branch: "main"});
    return {status: "no_schedule_baseline"};
  }

  const latest = runs[0];
  const latestAt = timestampSeconds(latest);
  if (latestAt === null || latestAt > now) {
    structuredLog("latest_run_timestamp_unusable", {run_id: latest.id});
    return {status: "unknown_run_timestamp"};
  }
  const ageSeconds = Math.max(0, now - latestAt);
  const activeStatuses = new Set(["queued", "in_progress", "requested", "waiting", "pending"]);
  if (activeStatuses.has(String(latest.status || "").toLowerCase()) && ageSeconds < staleSeconds) {
    structuredLog("active_run", {run_id: latest.id, status: latest.status, age_seconds: ageSeconds});
    return {status: "active_run", run_id: latest.id};
  }
  if (ageSeconds < staleSeconds) {
    structuredLog("fresh_schedule", {run_id: latest.id, age_seconds: ageSeconds});
    return {status: "fresh_schedule", run_id: latest.id, age_seconds: ageSeconds};
  }

  const key = "schedule-gap:" + String(latest.id);
  const claim = await watchdogRequest(env, "claim", {key, now});
  if (!claim.allowed) {
    structuredLog("dispatch_suppressed", {run_id: latest.id, reason: claim.reason || "claim_denied"});
    return {status: "dispatch_suppressed", reason: claim.reason || "claim_denied", run_id: latest.id};
  }

  try {
    await githubRequest(token, "/repos/" + REPOSITORY + "/dispatches", {
      method: "POST",
      body: JSON.stringify({
        event_type: "orchestrator.recovery",
        client_payload: {
          source: "cloudflare-watchdog",
          event_id: "cloudflare-watchdog:" + String(latest.id),
          latest_schedule_run_id: latest.id,
          observed_at: new Date(nowMs).toISOString(),
          schedule_age_seconds: ageSeconds,
        },
      }),
    });
    const finished = await watchdogRequest(env, "finish", {
      key, claim_id: claim.claim_id, ok: true, now, cooldown_seconds: cooldownSeconds,
    });
    if (!finished.updated) throw new Error("watchdog_claim_finish_conflict");
    structuredLog("recovery_dispatch_requested", {
      run_id: latest.id,
      event_id: "cloudflare-watchdog:" + String(latest.id),
      age_seconds: ageSeconds,
    });
    return {status: "dispatch_requested", run_id: latest.id, age_seconds: ageSeconds};
  } catch (error) {
    try {
      await watchdogRequest(env, "finish", {key, claim_id: claim.claim_id, ok: false, now});
    } catch (_) {
      // Keep the original dispatch failure as the diagnostic signal.
    }
    const message = String(error?.message || "dispatch_failed").replace(/\s+/g, " ").slice(0, 240);
    structuredLog("recovery_dispatch_failed", {run_id: latest.id, error: message});
    return {status: "dispatch_failed", run_id: latest.id, error: message};
  }
}
