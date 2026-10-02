import { PrivateInput } from "./private_input_object.js";
import { ExecutionLease } from "./execution_lease_object.js";

const PROTOCOL = "ai-orchestrator.private-input/v2";
const MAX_BODY_BYTES = 128 * 1024;
const MAX_LEASE_TTL_SECONDS = 900;
const MIN_LEASE_TTL_SECONDS = 60;
const MAX_ATTEMPT_RESERVATION = 16;

const MAX_TTL_SECONDS = 7 * 24 * 60 * 60;
const MIN_TTL_SECONDS = 300;
const CLOCK_SKEW_SECONDS = 300;
const HEX64 = /^[0-9a-f]{64}$/;

function json(value) {
  return JSON.stringify(value);
}

function response(body, status = 200) {
  return new Response(json(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

function hex(bytes) {
  return [...new Uint8Array(bytes)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function fromHex(value) {
  if (!HEX64.test(value)) throw new Error("hex64_required");
  const out = new Uint8Array(value.length / 2);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = Number.parseInt(value.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

function constantTimeEqual(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) diff |= a[i] ^ b[i];
  return diff === 0;
}

async function hmac(secret, message) {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  return hex(await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(message),
  ));
}

async function verifyHmac(secret, method, path, timestamp, body, supplied) {
  if (!supplied.startsWith("sha256=")) return false;
  const signature = supplied.slice("sha256=".length);
  if (!HEX64.test(signature)) return false;
  const message = [
    method.toUpperCase(),
    path || "/",
    PROTOCOL,
    String(timestamp),
    body,
  ].join("\n");
  const expected = await hmac(secret, message);
  return constantTimeEqual(fromHex(expected), fromHex(signature));
}

async function auth(request, env) {
  const secret = String(env.ORCHESTRATOR_PRIVATE_INPUT_SECRET || "");
  if (!secret) return false;
  if ((request.headers.get("X-Orchestrator-Protocol") || "") !== PROTOCOL) return false;
  const timestampRaw = request.headers.get("X-Orchestrator-Timestamp") || "";
  if (!/^\d+$/.test(timestampRaw)) return false;
  const timestamp = Number(timestampRaw);
  if (!Number.isSafeInteger(timestamp)) return false;
  if (Math.abs(Math.floor(Date.now() / 1000) - timestamp) > CLOCK_SKEW_SECONDS) return false;
  const body = request.method === "GET" || request.method === "DELETE"
    ? ""
    : await request.clone().text();
  return verifyHmac(
    secret,
    request.method,
    new URL(request.url).pathname,
    timestamp,
    body,
    request.headers.get("X-Orchestrator-Signature") || "",
  );
}

function validateEnvelope(envelope, now, requireTtlWindow) {
  if (!envelope || typeof envelope !== "object" || Array.isArray(envelope)) {
    throw new Error("envelope_must_be_object");
  }
  const required = [
    "schema_version", "protocol", "input_ref", "execution_id",
    "intent_fingerprint", "input_digest", "expires_at", "payload",
  ];
  const allowed = new Set(required);
  for (const key of Object.keys(envelope)) {
    if (!allowed.has(key)) throw new Error("unknown_envelope_field");
  }
  for (const key of required) {
    if (!(key in envelope)) throw new Error("missing_" + key);
  }
  if (envelope.schema_version !== 1) throw new Error("schema_version_invalid");
  if (envelope.protocol !== PROTOCOL) throw new Error("protocol_invalid");
  for (const key of ["input_ref", "execution_id", "intent_fingerprint", "input_digest"]) {
    if (typeof envelope[key] !== "string" || !HEX64.test(envelope[key])) {
      throw new Error(key + "_invalid");
    }
  }
  if (!Number.isInteger(envelope.expires_at)) throw new Error("expires_at_invalid");
  if (!envelope.payload || typeof envelope.payload !== "object" || Array.isArray(envelope.payload)) {
    throw new Error("payload_invalid");
  }
  if (envelope.expires_at <= now) throw new Error("input_expired");
  if (requireTtlWindow) {
    const ttl = envelope.expires_at - now;
    if (ttl < MIN_TTL_SECONDS || ttl > MAX_TTL_SECONDS) throw new Error("ttl_out_of_bounds");
  }
}

async function deriveRef(executionId, digest, secret) {
  return hmac(secret, PROTOCOL + "\n" + executionId + "\n" + digest);
}

function privateInputStub(env, ref) {
  return env.PRIVATE_INPUTS.getByName(ref);
}

function validateLeaseRequest(value, kind) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("lease_request_must_be_object");
  }
  const allowed = new Set(
    kind === "acquire"
      ? ["subject", "owner_id", "ttl_seconds", "max_attempts", "initial_attempts"]
      : kind === "reserve"
        ? ["subject", "owner_id", "count", "max_attempts", "ttl_seconds"]
        : ["subject", "owner_id"],
  );
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) throw new Error("unknown_lease_field");
  }
  if (typeof value.subject !== "string" || value.subject.length < 1 || value.subject.length > 128) {
    throw new Error("lease_subject_invalid");
  }
  if (typeof value.owner_id !== "string" || value.owner_id.length < 1 || value.owner_id.length > 128) {
    throw new Error("lease_owner_invalid");
  }
  if (kind === "acquire") {
    if (!Number.isInteger(value.ttl_seconds) ||
        value.ttl_seconds < MIN_LEASE_TTL_SECONDS ||
        value.ttl_seconds > MAX_LEASE_TTL_SECONDS) {
      throw new Error("lease_ttl_invalid");
    }
    if (!Number.isInteger(value.max_attempts) ||
        value.max_attempts < 1 ||
        value.max_attempts > 256 ||
        !Number.isInteger(value.initial_attempts) ||
        value.initial_attempts < 0 ||
        value.initial_attempts > value.max_attempts) {
      throw new Error("lease_attempt_budget_invalid");
    }
  }
  if (kind === "reserve") {
    if (!Number.isInteger(value.count) || value.count < 1 || value.count > MAX_ATTEMPT_RESERVATION) {
      throw new Error("lease_count_invalid");
    }
    if (!Number.isInteger(value.max_attempts) || value.max_attempts < 1 || value.max_attempts > 256) {
      throw new Error("lease_max_attempts_invalid");
    }
    if (!Number.isInteger(value.ttl_seconds) ||
        value.ttl_seconds < MIN_LEASE_TTL_SECONDS ||
        value.ttl_seconds > MAX_LEASE_TTL_SECONDS) {
      throw new Error("lease_ttl_invalid");
    }
  }
}

async function leaseRequest(request, env, kind) {
  const body = await request.text();
  if (new TextEncoder().encode(body).byteLength > 16 * 1024) {
    return response({ ok: false, error: "lease_request_too_large" }, 413);
  }
  let value;
  try {
    value = JSON.parse(body);
    validateLeaseRequest(value, kind);
  } catch (error) {
    return response({
      ok: false,
      error: error instanceof Error ? error.message : "invalid_lease_request",
    }, 400);
  }

  const stub = env.EXECUTION_LEASES.getByName(value.subject);
  const now = Math.floor(Date.now() / 1000);
  if (kind === "acquire") {
    const result = await stub.acquire(
      value.subject,
      value.owner_id,
      now,
      value.ttl_seconds,
      value.max_attempts,
      value.initial_attempts,
    );
    return response(result, result.conflict || result.budget_exhausted ? 409 : 200);
  }
  if (kind === "reserve") {
    const result = await stub.reserve(
      value.subject,
      value.owner_id,
      now,
      value.count,
      value.max_attempts,
      value.ttl_seconds,
    );
    return response(result, result.ok ? 200 : 409);
  }
  const result = await stub.release(value.subject, value.owner_id);
  return response(result);
}

async function handlePost(request, env) {
  const body = await request.text();
  if (new TextEncoder().encode(body).byteLength > MAX_BODY_BYTES) {
    return response({ ok: false, error: "request_too_large" }, 413);
  }
  let envelope;
  try {
    envelope = JSON.parse(body);
    validateEnvelope(envelope, Math.floor(Date.now() / 1000), true);
  } catch (error) {
    return response({ ok: false, error: error instanceof Error ? error.message : "invalid_envelope" }, 400);
  }
  const secret = String(env.ORCHESTRATOR_PRIVATE_INPUT_SECRET || "");
  const expectedRef = await deriveRef(envelope.execution_id, envelope.input_digest, secret);
  if (!constantTimeEqual(fromHex(expectedRef), fromHex(envelope.input_ref))) {
    return response({ ok: false, error: "input_ref_mismatch" }, 400);
  }
  const result = await privateInputStub(env, envelope.input_ref).putInput(envelope);
  if (result.conflict) {
    return response({ ok: false, error: "input_ref_conflict" }, 409);
  }
  return response(result, result.created ? 201 : 200);
}

async function handleGet(env, ref) {
  if (!HEX64.test(ref)) return response({ ok: false, error: "input_ref_invalid" }, 400);
  const result = await privateInputStub(env, ref).getInput(ref);
  if (result === null) return response({ ok: false, error: "input_not_found" }, 404);
  return response(result);
}

async function handleDelete(env, ref) {
  if (!HEX64.test(ref)) return response({ ok: false, error: "input_ref_invalid" }, 400);
  return response(await privateInputStub(env, ref).deleteInput(ref));
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/health" && request.method === "GET") {
      return response({
        ok: true,
        service: "ai-orchestrator-private-input",
        protocol: PROTOCOL,
        storage: "durable-object-sqlite",
      });
    }
    if (!["POST", "GET", "DELETE"].includes(request.method)) {
      return response({ ok: false, error: "method_not_allowed" }, 405);
    }
    if (!(await auth(request, env))) {
      return response({ ok: false, error: "unauthorized" }, 401);
    }
    try {
      if (url.pathname === "/v1/inputs" && request.method === "POST") {
        return await handlePost(request, env);
      }
      if (url.pathname === "/v1/leases/acquire" && request.method === "POST") {
        return await leaseRequest(request, env, "acquire");
      }
      if (url.pathname === "/v1/leases/reserve-attempt" && request.method === "POST") {
        return await leaseRequest(request, env, "reserve");
      }
      if (url.pathname === "/v1/leases/release" && request.method === "POST") {
        return await leaseRequest(request, env, "release");
      }
      const match = url.pathname.match(/^\/v1\/inputs\/([0-9a-f]{64})$/);
      if (match && request.method === "GET") return await handleGet(env, match[1]);
      if (match && request.method === "DELETE") return await handleDelete(env, match[1]);
      return response({ ok: false, error: "not_found" }, 404);
    } catch {
      return response({ ok: false, error: "private_input_internal_error" }, 500);
    }
  },
};


export { PrivateInput, ExecutionLease };
