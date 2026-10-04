const E = new TextEncoder();
const SKEW = 300;
const MAX_BODY = 768 * 1024;

const json = (value, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: {"content-type":"application/json; charset=utf-8","cache-control":"no-store"}
  });

const now = () => Math.floor(Date.now() / 1000);

const hexBytes = (hex) => {
  if (!/^[0-9a-f]{64}$/i.test(hex)) throw new Error("invalid_signature");
  const out = new Uint8Array(32);
  for (let i = 0; i < 32; i += 1) out[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return out;
};

async function authenticated(request, env, body, path) {
  const ts = request.headers.get("X-Control-Plane-Timestamp") || "";
  const sig = request.headers.get("X-Control-Plane-Signature") || "";
  const asserted = Number(ts);
  if (!Number.isInteger(asserted) || Math.abs(now() - asserted) > SKEW) return false;
  const prefix = E.encode([ts, request.method.toUpperCase(), path].join("\n") + "\n");
  const signed = new Uint8Array(prefix.length + body.length);
  signed.set(prefix);
  signed.set(body, prefix.length);
  const key = await crypto.subtle.importKey("raw", E.encode(env.CONTROL_PLANE_SECRET), {name:"HMAC",hash:"SHA-256"}, false, ["verify"]);
  return crypto.subtle.verify("HMAC", key, hexBytes(sig), signed);
}

function conflict(message) {
  return new Response(JSON.stringify({error: message}), {
    status: 409, headers: {"content-type":"application/json"}
  });
}

function ttl(value) {
  const n = Number(value);
  return Number.isInteger(n) ? Math.max(60, Math.min(n, 3600)) : 1200;
}

export class WorkflowControlPlane {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
    ctx.blockConcurrencyWhile(async () => {
      ctx.storage.sql.exec("CREATE TABLE IF NOT EXISTS lease (singleton INTEGER PRIMARY KEY CHECK(singleton=1), owner TEXT NOT NULL, fence_epoch INTEGER NOT NULL, expires_at INTEGER NOT NULL)");
      try { ctx.storage.sql.exec("ALTER TABLE lease ADD COLUMN workflow_id TEXT NOT NULL DEFAULT ''"); } catch (_) {}
      ctx.storage.sql.exec("CREATE TABLE IF NOT EXISTS effect (effect_id TEXT PRIMARY KEY, semantic_digest TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('inflight','completed')), owner TEXT NOT NULL, fence_epoch INTEGER NOT NULL, created_at INTEGER NOT NULL, completed_at INTEGER, output_sha256 TEXT)");
      ctx.storage.sql.exec("CREATE TABLE IF NOT EXISTS workflow_state (singleton INTEGER PRIMARY KEY CHECK(singleton=1), state_version INTEGER NOT NULL, state_json TEXT NOT NULL, updated_at INTEGER NOT NULL)");
      ctx.storage.sql.exec("CREATE TABLE IF NOT EXISTS outbox (sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, payload_json TEXT NOT NULL, created_at INTEGER NOT NULL)");
      ctx.storage.sql.exec("CREATE INDEX IF NOT EXISTS outbox_created_idx ON outbox(created_at)");
      ctx.storage.sql.exec(
        "CREATE TABLE IF NOT EXISTS provider_rate (singleton INTEGER PRIMARY KEY CHECK(singleton=1), provider TEXT NOT NULL, tokens REAL NOT NULL, updated_at INTEGER NOT NULL)"
      );
      ctx.storage.sql.exec("CREATE TABLE IF NOT EXISTS recovery (singleton INTEGER PRIMARY KEY CHECK(singleton=1), due_at INTEGER NOT NULL, due INTEGER NOT NULL DEFAULT 0, event_id TEXT NOT NULL, fired_at INTEGER, claimed INTEGER NOT NULL DEFAULT 0, claim_owner TEXT, claim_expires_at INTEGER)");
      try { ctx.storage.sql.exec("ALTER TABLE recovery ADD COLUMN claimed INTEGER NOT NULL DEFAULT 0"); } catch (_) {}
      try { ctx.storage.sql.exec("ALTER TABLE recovery ADD COLUMN claim_owner TEXT"); } catch (_) {}
      try { ctx.storage.sql.exec("ALTER TABLE recovery ADD COLUMN claim_expires_at INTEGER"); } catch (_) {}
    });
  }

  lease() {
    const rows = this.ctx.storage.sql.exec("SELECT owner, workflow_id, fence_epoch, expires_at FROM lease WHERE singleton=1").toArray();
    return rows.length ? rows[0] : null;
  }

  requireLease(owner, fenceEpoch, at) {
    const row = this.lease();
    if (!row || String(row.owner) !== owner || Number(row.fence_epoch) !== Number(fenceEpoch) || Number(row.expires_at) <= at) {
      throw conflict("stale_or_missing_lease");
    }
  }

  acquire(body) {
    const owner = String(body.owner || "").trim();
    const workflowId = String(body.workflow_id || "").trim();
    if (!owner || owner.length > 128 || !workflowId) throw new Error("workflow_lease_identity_invalid");
    const at = now();
    const result = this.ctx.storage.transactionSync(() => {
      const current = this.lease();
      if (current && Number(current.expires_at) > at && String(current.owner) !== owner) {
        return {conflict:true, expires_at:Number(current.expires_at)};
      }
      const epoch = current && Number(current.expires_at) > at ? Number(current.fence_epoch) : (current ? Number(current.fence_epoch) + 1 : 1);
      const expires = at + ttl(body.ttl_seconds);
      this.ctx.storage.sql.exec(
        "INSERT INTO lease(singleton,owner,fence_epoch,expires_at,workflow_id) VALUES(1,?,?,?,?) ON CONFLICT(singleton) DO UPDATE SET owner=excluded.owner,fence_epoch=excluded.fence_epoch,expires_at=excluded.expires_at,workflow_id=excluded.workflow_id",
        owner, epoch, expires, workflowId
      );
      return {
        status: current && Number(current.expires_at) > at ? "renewed" : "acquired",
        owner,
        workflow_id: workflowId,
        fence_epoch: epoch,
        expires_at: expires
      };
    });
    return result.conflict ? json({error:"lease_held",expires_at:result.expires_at},409) : json(result);
  }

  renew(body) {
    const owner = String(body.owner || "").trim();
    const workflowId = String(body.workflow_id || "").trim();
    const epoch = Number(body.fence_epoch);
    const at = now();
    const result = this.ctx.storage.transactionSync(() => {
      const current = this.lease();
      if (!current || String(current.workflow_id || "") !== workflowId) {
        throw conflict("workflow_lease_identity_mismatch");
      }
      this.requireLease(owner, epoch, at);
      const expires = at + ttl(body.ttl_seconds);
      this.ctx.storage.sql.exec("UPDATE lease SET expires_at=? WHERE singleton=1 AND owner=? AND fence_epoch=?", expires, owner, epoch);
      return {status:"renewed", owner, workflow_id:workflowId, fence_epoch:epoch, expires_at:expires};
    });
    return json(result);
  }

  release(body) {
    const owner = String(body.owner || "").trim();
    const workflowId = String(body.workflow_id || "").trim();
    const epoch = Number(body.fence_epoch);
    return json({status:this.ctx.storage.transactionSync(() => {
      const current = this.lease();
      if (!current) return "already_released";
      if (
        String(current.owner) !== owner ||
        String(current.workflow_id || "") !== workflowId ||
        Number(current.fence_epoch) !== epoch
      ) throw conflict("stale_or_missing_lease");
      this.ctx.storage.sql.exec("DELETE FROM lease WHERE singleton=1");
      return "released";
    })});
  }

  claim(body) {
    const owner = String(body.owner || "").trim();
    const epoch = Number(body.fence_epoch);
    const effectId = String(body.effect_id || "").trim();
    const digest = String(body.semantic_digest || "").trim();
    if (!/^[0-9a-f]{64}$/.test(effectId) || !/^[0-9a-f]{64}$/.test(digest)) throw new Error("effect_identity_invalid");
    const at = now();
    const result = this.ctx.storage.transactionSync(() => {
      this.requireLease(owner, epoch, at);
      const rows = this.ctx.storage.sql.exec("SELECT effect_id,semantic_digest,status FROM effect WHERE effect_id=?", effectId).toArray();
      if (rows.length) {
        if (String(rows[0].semantic_digest) !== digest) throw conflict("effect_semantic_conflict");
        return {status:String(rows[0].status), effect_id:effectId, semantic_digest:digest};
      }
      this.ctx.storage.sql.exec("INSERT INTO effect(effect_id,semantic_digest,status,owner,fence_epoch,created_at,completed_at,output_sha256) VALUES(?,?, 'inflight',?,?,?,NULL,NULL)", effectId,digest,owner,epoch,at);
      return {status:"claimed",effect_id:effectId,semantic_digest:digest};
    });
    return json(result);
  }

  complete(body) {
    const owner = String(body.owner || "").trim();
    const epoch = Number(body.fence_epoch);
    const effectId = String(body.effect_id || "").trim();
    const digest = String(body.semantic_digest || "").trim();
    const output = String(body.output_sha256 || "").trim();
    const at = now();
    const result = this.ctx.storage.transactionSync(() => {
      this.requireLease(owner, epoch, at);
      const rows = this.ctx.storage.sql.exec("SELECT semantic_digest,status,owner,fence_epoch FROM effect WHERE effect_id=?", effectId).toArray();
      if (!rows.length) throw conflict("effect_missing");
      const row = rows[0];
      if (String(row.semantic_digest) !== digest) throw conflict("effect_semantic_conflict");
      if (String(row.status) === "completed") return {status:"already_completed"};
      if (String(row.owner) !== owner || Number(row.fence_epoch) !== epoch) throw conflict("effect_fence_conflict");
      this.ctx.storage.sql.exec("UPDATE effect SET status='completed',completed_at=?,output_sha256=? WHERE effect_id=?", at, output, effectId);
      return {status:"completed"};
    });
    return json(result);
  }

  providerRateAcquire(body) {
    const requested = String(body.provider || "").trim().toLowerCase();
    const LIMITS = {
      // Crossref list/search calls: polite pool is 3 req/s; we enforce a
      // deliberately conservative 2 req/s global gate to protect free access.
      crossref: {rate: 2, burst: 2},
      semantic_scholar: {rate: 5, burst: 5},
      europe_pmc: {rate: 5, burst: 5},
      openalex: {rate: 10, burst: 10},
    };
    let config = LIMITS[requested];
    if (!config) throw new Error("provider_rate_unsupported");
    const pool = String(body.pool || "public").trim().toLowerCase();
    if (requested === "crossref" && pool === "polite") {
      config = {rate: 3, burst: 3};
    } else if (requested === "crossref" && pool !== "public") {
      throw new Error("provider_rate_pool_invalid");
    }
    const at = Date.now() / 1000;
    const result = this.ctx.storage.transactionSync(() => {
      const rows = this.ctx.storage.sql.exec(
        "SELECT provider,tokens,updated_at FROM provider_rate WHERE singleton=1"
      ).toArray();
      let tokens = config.burst;
      let last = at;
      let provider = requested;
      if (rows.length) {
        provider = String(rows[0].provider || requested);
        if (provider !== requested) {
          throw conflict("provider_rate_identity_conflict");
        }
        tokens = Number(rows[0].tokens);
        last = Number(rows[0].updated_at);
      }
      tokens = Math.min(
        config.burst,
        Math.max(0, tokens + Math.max(0, at - last) * config.rate)
      );
      if (tokens < 1) {
        const retryAfter = (1 - tokens) / config.rate;
        this.ctx.storage.sql.exec(
          "INSERT INTO provider_rate(singleton,provider,tokens,updated_at) VALUES(1,?,?,?) " +
          "ON CONFLICT(singleton) DO UPDATE SET tokens=excluded.tokens,updated_at=excluded.updated_at",
          requested, tokens, at
        );
        return {granted:false,retry_after:Math.max(0.05, retryAfter)};
      }
      tokens -= 1;
      this.ctx.storage.sql.exec(
        "INSERT INTO provider_rate(singleton,provider,tokens,updated_at) VALUES(1,?,?,?) " +
        "ON CONFLICT(singleton) DO UPDATE SET tokens=excluded.tokens,updated_at=excluded.updated_at",
        requested, tokens, at
      );
      return {granted:true,retry_after:0};
    });
    return json({
      status: result.granted ? "granted" : "throttled",
      provider: requested,
      retry_after: Number(result.retry_after || 0),
    });
  }

  resourceAcquire(body) {
    const owner = String(body.owner || "").trim();
    const workflowId = String(body.workflow_id || "").trim();
    if (!owner || !workflowId) throw new Error("resource_lock_identity_invalid");
    const at = now();
    const result = this.ctx.storage.transactionSync(() => {
      const current = this.lease();
      if (
        current &&
        Number(current.expires_at) > at &&
        (
          String(current.owner) !== owner ||
          String(current.workflow_id || "") !== workflowId
        )
      ) {
        return {conflict: true, expires_at: Number(current.expires_at)};
      }
      const currentEpoch = current ? Number(current.fence_epoch) : 0;
      const renewing =
        Boolean(current) &&
        Number(current.expires_at) > at &&
        String(current.owner) === owner &&
        String(current.workflow_id || "") === workflowId;
      const epoch = renewing ? currentEpoch : currentEpoch + 1;
      const expires = at + ttl(body.ttl_seconds);
      this.ctx.storage.sql.exec(
        "INSERT INTO lease(singleton,owner,fence_epoch,expires_at,workflow_id) " +
        "VALUES(1,?,?,?,?) ON CONFLICT(singleton) DO UPDATE SET " +
        "owner=excluded.owner,fence_epoch=excluded.fence_epoch," +
        "expires_at=excluded.expires_at,workflow_id=excluded.workflow_id",
        owner, epoch, expires, workflowId
      );
      return {
        status:"acquired",
        owner,
        workflow_id:workflowId,
        fence_epoch:epoch,
        expires_at:expires
      };
    });
    return result.conflict
      ? json({error:"resource_lock_held",expires_at:result.expires_at},409)
      : json(result);
  }

  resourceRenew(body) {
    const owner = String(body.owner || "").trim();
    const workflowId = String(body.workflow_id || "").trim();
    const epoch = Number(body.fence_epoch);
    const at = now();
    const result = this.ctx.storage.transactionSync(() => {
      const row = this.lease();
      if (
        !row ||
        String(row.owner) !== owner ||
        String(row.workflow_id || "") !== workflowId ||
        Number(row.fence_epoch) !== epoch ||
        Number(row.expires_at) <= at
      ) {
        throw conflict("stale_or_missing_resource_lock");
      }
      const expires = at + ttl(body.ttl_seconds);
      this.ctx.storage.sql.exec(
        "UPDATE lease SET expires_at=? WHERE singleton=1 AND owner=? AND fence_epoch=?",
        expires, owner, epoch
      );
      return {
        status:"renewed",
        owner,
        workflow_id:workflowId,
        fence_epoch:epoch,
        expires_at:expires
      };
    });
    return json(result);
  }

  resourceRelease(body) {
    const owner = String(body.owner || "").trim();
    const workflowId = String(body.workflow_id || "").trim();
    const epoch = Number(body.fence_epoch);
    const result = this.ctx.storage.transactionSync(() => {
      const row = this.lease();
      if (!row) return "already_released";
      if (
        String(row.owner) !== owner ||
        String(row.workflow_id || "") !== workflowId ||
        Number(row.fence_epoch) !== epoch
      ) {
        throw conflict("stale_or_missing_resource_lock");
      }
      this.ctx.storage.sql.exec("DELETE FROM lease WHERE singleton=1");
      return "released";
    });
    return json({status:result});
  }

  async armRecovery(body) {
    const owner = String(body.owner || "").trim();
    const workflowId = String(body.workflow_id || "").trim();
    const epoch = Number(body.fence_epoch);
    const dueAt = Number(body.due_at);
    const eventId = String(body.event_id || "").trim();
    if (
      !owner ||
      !workflowId ||
      !Number.isInteger(epoch) ||
      !Number.isInteger(dueAt) ||
      !eventId ||
      eventId.length > 256
    ) {
      throw new Error("recovery_schedule_invalid");
    }
    const at = now();
    const scheduledAt = Math.max(at, dueAt);
    let result;
    this.ctx.storage.transactionSync(() => {
      this.requireLease(owner, epoch, at);
      const rows = this.ctx.storage.sql.exec(
        "SELECT due_at,due,event_id FROM recovery WHERE singleton=1"
      ).toArray();
      if (
        rows.length &&
        String(rows[0].event_id) === eventId &&
        Number(rows[0].due) === 0 &&
        Number(rows[0].due_at) >= scheduledAt
      ) {
        result = {
          status:"already_armed",
          due_at:Number(rows[0].due_at),
          event_id:eventId
        };
        return;
      }
      this.ctx.storage.sql.exec(
        "INSERT INTO recovery(singleton,due_at,due,event_id,fired_at,claimed,claim_owner,claim_expires_at) VALUES(1,?,0,?,NULL,0,NULL,NULL) " +
        "ON CONFLICT(singleton) DO UPDATE SET due_at=excluded.due_at,due=0,event_id=excluded.event_id,fired_at=NULL,claimed=0,claim_owner=NULL,claim_expires_at=NULL",
        scheduledAt,
        eventId
      );
      result = {
        status:"armed",
        due_at:scheduledAt,
        event_id:eventId
      };
    });
    if (result.status === "armed") {
      await this.ctx.storage.setAlarm(Number(result.due_at) * 1000);
    }
    return json({
      status:result.status,
      workflow_id:workflowId,
      due_at:Number(result.due_at),
      event_id:eventId
    });
  }

  claimRecovery(body) {
    const workflowId = String(body.workflow_id || "").trim();
    const eventId = String(body.event_id || "").trim();
    const owner = String(body.owner || "").trim();
    if (!workflowId || !eventId || !owner) throw new Error("recovery_claim_invalid");
    const at = now();
    const expiresAt = at + Math.max(30, Math.min(Number(body.ttl_seconds) || 300, 600));
    const result = this.ctx.storage.transactionSync(() => {
      const rows = this.ctx.storage.sql.exec(
        "SELECT due,event_id,claimed,claim_owner,claim_expires_at FROM recovery WHERE singleton=1"
      ).toArray();
      if (!rows.length) return "absent";
      const row = rows[0];
      if (String(row.event_id) !== eventId || Number(row.due) !== 1) return "not_due";
      const existingExpiry = Number(row.claim_expires_at || 0);
      if (Number(row.claimed) === 1 && existingExpiry > at && String(row.claim_owner || "") !== owner) {
        return "already_claimed";
      }
      this.ctx.storage.sql.exec(
        "UPDATE recovery SET claimed=1,claim_owner=?,claim_expires_at=? WHERE singleton=1",
        owner,
        expiresAt
      );
      return "claimed";
    });
    return result === "already_claimed"
      ? json({error:"recovery_claimed"},409)
      : json({status:result,workflow_id:workflowId,event_id:eventId,claim_owner:owner,claim_expires_at:expiresAt});
  }

  async clearRecovery(body) {
    const owner = String(body.owner || "").trim();
    const workflowId = String(body.workflow_id || "").trim();
    const epoch = Number(body.fence_epoch);
    if (!owner || !workflowId || !Number.isInteger(epoch)) {
      throw new Error("recovery_clear_invalid");
    }
    const at = now();
    this.ctx.storage.transactionSync(() => {
      this.requireLease(owner, epoch, at);
      this.ctx.storage.sql.exec("DELETE FROM recovery WHERE singleton=1");
    });
    await this.ctx.storage.deleteAlarm();
    return json({status:"cleared",workflow_id:workflowId});
  }

  ackRecovery(body) {
    const workflowId = String(body.workflow_id || "").trim();
    const eventId = String(body.event_id || "").trim();
    if (!workflowId || !eventId) throw new Error("recovery_ack_invalid");
    const result = this.ctx.storage.transactionSync(() => {
      const rows = this.ctx.storage.sql.exec(
        "SELECT due,event_id FROM recovery WHERE singleton=1"
      ).toArray();
      if (!rows.length) return "already_clear";
      const row = rows[0];
      if (String(row.event_id) !== eventId) return "stale_event";
      this.ctx.storage.sql.exec("DELETE FROM recovery WHERE singleton=1");
      return "acknowledged";
    });
    return json({status:result,workflow_id:workflowId,event_id:eventId});
  }

  readRecovery() {
    const rows = this.ctx.storage.sql.exec(
      "SELECT due_at,due,event_id,fired_at,claimed,claim_owner,claim_expires_at FROM recovery WHERE singleton=1"
    ).toArray();
    return json(rows.length ? {
      status:"stored",
      due_at:Number(rows[0].due_at),
      due:Boolean(rows[0].due),
      event_id:String(rows[0].event_id),
      fired_at:rows[0].fired_at == null ? null : Number(rows[0].fired_at),
      claimed:Boolean(rows[0].claimed),
      claim_owner:rows[0].claim_owner == null ? "" : String(rows[0].claim_owner),
      claim_expires_at:rows[0].claim_expires_at == null ? null : Number(rows[0].claim_expires_at)
    } : {status:"absent"});
  }

  async alarm(alarmInfo) {
    const at = now();
    const result = this.ctx.storage.transactionSync(() => {
      const rows = this.ctx.storage.sql.exec(
        "SELECT due_at,due,event_id FROM recovery WHERE singleton=1"
      ).toArray();
      if (!rows.length) return {status:"noop"};
      const row = rows[0];
      if (Number(row.due) === 1) {
        return {status:"already_due"};
      }
      if (Number(row.due_at) > at) {
        return {status:"not_due",due_at:Number(row.due_at)};
      }
      this.ctx.storage.sql.exec(
        "UPDATE recovery SET due=1,fired_at=?,claimed=0,claim_owner=NULL,claim_expires_at=NULL WHERE singleton=1",
        at
      );
      this.ctx.storage.sql.exec(
        "INSERT INTO outbox(event_type,payload_json,created_at) VALUES(?,?,?)",
        "workflow.recovery_due",
        JSON.stringify({
          event_id:String(row.event_id),
          fired_at:at,
          retry_count:Number(alarmInfo?.retryCount || 0)
        }),
        at
      );
      return {status:"due"};
    });
    if (result.status === "not_due") {
      await this.ctx.storage.setAlarm(Number(result.due_at) * 1000);
    }
  }

  readWorkflowState() {
    const rows = this.ctx.storage.sql.exec(
      "SELECT state_version,state_json,updated_at FROM workflow_state WHERE singleton=1"
    ).toArray();
    if (!rows.length) return json({status:"absent"});
    let state;
    try {
      state = JSON.parse(String(rows[0].state_json));
    } catch (_) {
      throw new Error("workflow_state_corrupt");
    }
    if (!state || typeof state !== "object" || Array.isArray(state)) {
      throw new Error("workflow_state_corrupt");
    }
    const recoveryRows = this.ctx.storage.sql.exec(
      "SELECT due_at,due,event_id,fired_at FROM recovery WHERE singleton=1"
    ).toArray();
    const recovery = recoveryRows.length ? {
      due_at:Number(recoveryRows[0].due_at),
      due:Boolean(recoveryRows[0].due),
      event_id:String(recoveryRows[0].event_id),
      fired_at:recoveryRows[0].fired_at == null ? null : Number(recoveryRows[0].fired_at)
    } : {
      due_at:null,
      due:false,
      event_id:"",
      fired_at:null
    };
    return json({
      status:"stored",
      state_version:Number(rows[0].state_version),
      updated_at:Number(rows[0].updated_at),
      state,
      recovery
    });
  }

  async writeWorkflowState(body) {
    const owner = String(body.owner || "").trim();
    const epoch = Number(body.fence_epoch);
    const expected = Number(body.expected_state_version);
    const state = body.state;
    const recovery = body.recovery;
    if (!owner || !Number.isInteger(epoch) || !Number.isInteger(expected)) {
      throw new Error("workflow_state_identity_invalid");
    }
    if (!state || typeof state !== "object" || Array.isArray(state)) {
      throw new Error("workflow_state_must_be_object");
    }
    if (recovery !== undefined && (
      !recovery ||
      typeof recovery !== "object" ||
      Array.isArray(recovery)
    )) {
      throw new Error("workflow_recovery_mutation_invalid");
    }
    const recoveryAction = recovery
      ? String(recovery.action || "none").trim().toLowerCase()
      : "none";
    if (!["none","arm","clear"].includes(recoveryAction)) {
      throw new Error("workflow_recovery_action_invalid");
    }
    let recoveryDueAt = null;
    let recoveryEventId = "";
    if (recoveryAction === "arm") {
      recoveryDueAt = Number(recovery.due_at);
      recoveryEventId = String(recovery.event_id || "").trim();
      if (!Number.isInteger(recoveryDueAt) || !recoveryEventId || recoveryEventId.length > 256) {
        throw new Error("workflow_recovery_schedule_invalid");
      }
    }

    const stateJson = JSON.stringify(state);
    if (stateJson.length > 600 * 1024) {
      throw new Error("workflow_state_too_large");
    }
    const at = now();
    const result = this.ctx.storage.transactionSync(() => {
      this.requireLease(owner, epoch, at);
      const rows = this.ctx.storage.sql.exec(
        "SELECT state_version FROM workflow_state WHERE singleton=1"
      ).toArray();
      const actual = rows.length ? Number(rows[0].state_version) : 0;
      if (actual !== expected) {
        throw new Response(
          JSON.stringify({
            error:"workflow_state_conflict",
            state_version:actual
          }),
          {status:409,headers:{"content-type":"application/json"}}
        );
      }
      const next = actual + 1;
      this.ctx.storage.sql.exec(
        "INSERT INTO workflow_state(singleton,state_version,state_json,updated_at) " +
        "VALUES(1,?,?,?) ON CONFLICT(singleton) DO UPDATE SET " +
        "state_version=excluded.state_version,state_json=excluded.state_json,updated_at=excluded.updated_at",
        next, stateJson, at
      );

      let recoveryResult = {action:"none",status:"unchanged"};
      if (recoveryAction === "clear") {
        this.ctx.storage.sql.exec("DELETE FROM recovery WHERE singleton=1");
        recoveryResult = {action:"clear",status:"cleared"};
      } else if (recoveryAction === "arm") {
        const scheduledAt = Math.max(at, recoveryDueAt);
        const currentRecovery = this.ctx.storage.sql.exec(
          "SELECT due_at,due,event_id FROM recovery WHERE singleton=1"
        ).toArray();
        if (
          currentRecovery.length &&
          String(currentRecovery[0].event_id) === recoveryEventId &&
          Number(currentRecovery[0].due) === 0 &&
          Number(currentRecovery[0].due_at) >= scheduledAt
        ) {
          recoveryResult = {
            action:"arm",
            status:"already_armed",
            due_at:Number(currentRecovery[0].due_at),
            event_id:recoveryEventId,
          };
        } else {
          this.ctx.storage.sql.exec(
            "INSERT INTO recovery(singleton,due_at,due,event_id,fired_at,claimed,claim_owner,claim_expires_at) VALUES(1,?,0,?,NULL,0,NULL,NULL) " +
            "ON CONFLICT(singleton) DO UPDATE SET due_at=excluded.due_at,due=0,event_id=excluded.event_id,fired_at=NULL,claimed=0,claim_owner=NULL,claim_expires_at=NULL",
            scheduledAt,
            recoveryEventId
          );
          recoveryResult = {
            action:"arm",
            status:"armed",
            due_at:scheduledAt,
            event_id:recoveryEventId,
          };
        }
      }
      return {
        status:"stored",
        state_version:next,
        updated_at:at,
        recovery:recoveryResult,
      };
    });

    if (
      result.recovery &&
      result.recovery.action === "arm" &&
      result.recovery.status === "armed"
    ) {
      await this.ctx.storage.setAlarm(Number(result.recovery.due_at) * 1000);
    } else if (
      result.recovery &&
      result.recovery.action === "clear" &&
      result.recovery.status === "cleared"
    ) {
      await this.ctx.storage.deleteAlarm();
    }
    return json(result);
  }

  appendOutbox(body) {
    const owner = String(body.owner || "").trim();
    const epoch = Number(body.fence_epoch);
    const eventType = String(body.event_type || "").trim();
    const payload = body.payload;
    if (!owner || !eventType || !payload || typeof payload !== "object" || Array.isArray(payload)) {
      throw new Error("outbox_event_invalid");
    }
    const raw = JSON.stringify(payload);
    if (raw.length > 16 * 1024) throw new Error("outbox_payload_too_large");
    const at = now();
    const sequence = this.ctx.storage.transactionSync(() => {
      this.requireLease(owner, epoch, at);
      const result = this.ctx.storage.sql.exec(
        "INSERT INTO outbox(event_type,payload_json,created_at) VALUES(?,?,?)",
        eventType, raw, at
      );
      this.ctx.storage.sql.exec(
        "DELETE FROM outbox WHERE sequence <= COALESCE((SELECT MAX(sequence) FROM outbox) - 256, -1)"
      );
      return Number(result.meta.last_row_id);
    });
    return json({status:"appended",sequence});
  }

  inspect(url) {
    const effectId = url.searchParams.get("effect_id") || "";
    if (!/^[0-9a-f]{64}$/.test(effectId)) throw new Error("effect_id_invalid");
    const rows = this.ctx.storage.sql.exec("SELECT effect_id,semantic_digest,status,fence_epoch,created_at,completed_at,output_sha256 FROM effect WHERE effect_id=?", effectId).toArray();
    return json(rows.length ? {
      status:String(rows[0].status), effect_id:String(rows[0].effect_id), semantic_digest:String(rows[0].semantic_digest),
      fence_epoch:Number(rows[0].fence_epoch), created_at:Number(rows[0].created_at),
      completed_at:rows[0].completed_at == null ? null : Number(rows[0].completed_at),
      output_sha256:String(rows[0].output_sha256 || "")
    } : {status:"absent",effect_id:effectId});
  }

  resolve(body) {
    const owner = String(body.owner || "").trim();
    const epoch = Number(body.fence_epoch);
    const effectId = String(body.effect_id || "").trim();
    const digest = String(body.semantic_digest || "").trim();
    const outcome = String(body.outcome || "").trim().toLowerCase();
    const output = String(body.output_sha256 || "").trim();
    const at = now();
    if (!["completed","not_applied","unknown"].includes(outcome)) throw new Error("resolution_outcome_invalid");
    const result = this.ctx.storage.transactionSync(() => {
      this.requireLease(owner, epoch, at);
      const rows = this.ctx.storage.sql.exec("SELECT semantic_digest,status FROM effect WHERE effect_id=?", effectId).toArray();
      if (!rows.length && outcome === "not_applied") return "not_applied";
      if (!rows.length && outcome === "completed") {
        this.ctx.storage.sql.exec(
          "INSERT INTO effect(effect_id,semantic_digest,status,owner,fence_epoch,created_at,completed_at,output_sha256) VALUES(?,?,'completed',?,?,?,?,?)",
          effectId,digest,owner,epoch,at,at,output
        );
        return "completed";
      }
      if (!rows.length) throw conflict("effect_missing");
      if (String(rows[0].semantic_digest) !== digest) throw conflict("effect_semantic_conflict");
      if (outcome === "completed") {
        this.ctx.storage.sql.exec("UPDATE effect SET status='completed',owner=?,fence_epoch=?,completed_at=?,output_sha256=? WHERE effect_id=?", owner,epoch,at,output,effectId);
        return "completed";
      }
      if (outcome === "not_applied") {
        this.ctx.storage.sql.exec("DELETE FROM effect WHERE effect_id=? AND status='inflight'", effectId);
        return "not_applied";
      }
      return "unknown";
    });
    return json({status:result});
  }

  async fetch(request) {
    const url = new URL(request.url);
    const raw = new Uint8Array(await request.arrayBuffer());
    if (raw.byteLength > MAX_BODY) return json({error:"body_too_large"},413);
    let ok = false;
    try { ok = await authenticated(request,this.env,raw,url.pathname); } catch (_) { ok = false; }
    if (!ok) return json({error:"unauthorized"},401);
    let body = {};
    try {
      if (raw.length) body = JSON.parse(new TextDecoder().decode(raw));
      if (!body || typeof body !== "object" || Array.isArray(body)) throw new Error("body_must_be_object");
    } catch (e) { return json({error:String(e.message || "invalid_json")},400); }
    try {
      if (url.pathname.startsWith("/v1/resources/")) {
        if (request.method === "POST" && url.pathname.endsWith("/rate/acquire")) return this.providerRateAcquire(body);
        if (request.method === "POST" && url.pathname.endsWith("/lease/acquire")) return this.resourceAcquire(body);
        if (request.method === "POST" && url.pathname.endsWith("/lease/renew")) return this.resourceRenew(body);
        if (request.method === "POST" && url.pathname.endsWith("/lease/release")) return this.resourceRelease(body);
        return json({error:"not_found"},404);
      }
      if (request.method === "GET" && url.pathname.endsWith("/state")) return this.readWorkflowState();
      if (request.method === "GET" && url.pathname.endsWith("/recovery")) return this.readRecovery();
      if (request.method === "POST" && url.pathname.endsWith("/recovery/arm")) return this.armRecovery(body);
      if (request.method === "POST" && url.pathname.endsWith("/recovery/claim")) return this.claimRecovery(body);
      if (request.method === "POST" && url.pathname.endsWith("/recovery/clear")) return this.clearRecovery(body);
      if (request.method === "POST" && url.pathname.endsWith("/recovery/ack")) return this.ackRecovery(body);
      if (request.method === "PUT" && url.pathname.endsWith("/state")) return this.writeWorkflowState(body);
      if (request.method === "POST" && url.pathname.endsWith("/outbox")) return this.appendOutbox(body);
      if (request.method === "POST" && url.pathname.endsWith("/lease/acquire")) return this.acquire(body);
      if (request.method === "POST" && url.pathname.endsWith("/lease/renew")) return this.renew(body);
      if (request.method === "POST" && url.pathname.endsWith("/lease/release")) return this.release(body);
      if (request.method === "POST" && url.pathname.endsWith("/effects/claim")) return this.claim(body);
      if (request.method === "POST" && url.pathname.endsWith("/effects/complete")) return this.complete(body);
      if (request.method === "GET" && url.pathname.endsWith("/effects/inspect")) return this.inspect(url);
      if (request.method === "POST" && url.pathname.endsWith("/effects/resolve")) return this.resolve(body);
      return json({error:"not_found"},404);
    } catch (e) {
      if (e instanceof Response) return e;
      return json({error:String(e.message || "request_failed")},400);
    }
  }
}

export default {
  async fetch(request,env) {
    const url = new URL(request.url);
    const workflowMatch = url.pathname.match(/^\/v1\/workflows\/([^/]+)(?:\/.*)?$/);
    const resourceMatch = url.pathname.match(/^\/v1\/resources\/([^/]+)(?:\/.*)?$/);
    if (!workflowMatch && !resourceMatch) return json({error:"not_found"},404);

    let logicalId;
    let objectNamePrefix;
    try {
      logicalId = decodeURIComponent((workflowMatch || resourceMatch)[1]);
      objectNamePrefix = workflowMatch ? "workflow:" : "resource:";
    } catch (_) {
      return json({error:"resource_id_invalid"},400);
    }
    if (!logicalId || logicalId.length > 128) return json({error:"resource_id_invalid"},400);

    const objectName = objectNamePrefix + logicalId;
    const stub = env.WORKFLOW_CONTROL.get(env.WORKFLOW_CONTROL.idFromName(objectName));
    return stub.fetch(request);
  }
};
