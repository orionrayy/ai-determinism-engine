const E = new TextEncoder();
const SKEW = 300;
const MAX_BODY = 65536;

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
      ctx.storage.sql.exec("CREATE TABLE IF NOT EXISTS effect (effect_id TEXT PRIMARY KEY, semantic_digest TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('inflight','completed')), owner TEXT NOT NULL, fence_epoch INTEGER NOT NULL, created_at INTEGER NOT NULL, completed_at INTEGER, output_sha256 TEXT)");
    });
  }

  lease() {
    const rows = this.ctx.storage.sql.exec("SELECT owner, fence_epoch, expires_at FROM lease WHERE singleton=1").toArray();
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
    if (!owner || owner.length > 128) throw new Error("owner_invalid");
    const at = now();
    const result = this.ctx.storage.transactionSync(() => {
      const current = this.lease();
      if (current && Number(current.expires_at) > at && String(current.owner) !== owner) {
        return {conflict:true, expires_at:Number(current.expires_at)};
      }
      const epoch = current && Number(current.expires_at) > at ? Number(current.fence_epoch) : (current ? Number(current.fence_epoch) + 1 : 1);
      const expires = at + ttl(body.ttl_seconds);
      this.ctx.storage.sql.exec(
        "INSERT INTO lease(singleton,owner,fence_epoch,expires_at) VALUES(1,?,?,?) ON CONFLICT(singleton) DO UPDATE SET owner=excluded.owner,fence_epoch=excluded.fence_epoch,expires_at=excluded.expires_at",
        owner, epoch, expires
      );
      return {status: current && Number(current.expires_at) > at ? "renewed" : "acquired", owner, fence_epoch:epoch, expires_at:expires};
    });
    return result.conflict ? json({error:"lease_held",expires_at:result.expires_at},409) : json(result);
  }

  renew(body) {
    const owner = String(body.owner || "").trim();
    const epoch = Number(body.fence_epoch);
    const at = now();
    const result = this.ctx.storage.transactionSync(() => {
      this.requireLease(owner, epoch, at);
      const expires = at + ttl(body.ttl_seconds);
      this.ctx.storage.sql.exec("UPDATE lease SET expires_at=? WHERE singleton=1 AND owner=? AND fence_epoch=?", expires, owner, epoch);
      return {status:"renewed", owner, fence_epoch:epoch, expires_at:expires};
    });
    return json(result);
  }

  release(body) {
    const owner = String(body.owner || "").trim();
    const epoch = Number(body.fence_epoch);
    return json({status:this.ctx.storage.transactionSync(() => {
      const current = this.lease();
      if (!current) return "already_released";
      if (String(current.owner) !== owner || Number(current.fence_epoch) !== epoch) throw conflict("stale_or_missing_lease");
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
    const match = url.pathname.match(/^\/v1\/workflows\/([^/]+)(?:\/.*)?$/);
    if (!match) return json({error:"not_found"},404);
    let workflowId;
    try { workflowId = decodeURIComponent(match[1]); } catch (_) { return json({error:"workflow_id_invalid"},400); }
    if (!workflowId || workflowId.length > 128) return json({error:"workflow_id_invalid"},400);
    const stub = env.WORKFLOW_CONTROL.get(env.WORKFLOW_CONTROL.idFromName(workflowId));
    return stub.fetch(request);
  }
};
