import { PrivateInput } from "./private_input_object.js";

const PROTOCOL = "ai-orchestrator.private-input/v2";
const MAX_BODY_BYTES = 128 * 1024;
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

function validateEnvelope(envelope, now) {
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
  const ttl = envelope.expires_at - now;
  if (ttl < MIN_TTL_SECONDS || ttl > MAX_TTL_SECONDS) throw new Error("ttl_out_of_bounds");
  if (!envelope.payload || typeof envelope.payload !== "object" || Array.isArray(envelope.payload)) {
    throw new Error("payload_invalid");
  }
  if (envelope.expires_at <= now) throw new Error("input_expired");
}

async function deriveRef(executionId, digest, secret) {
  return hmac(secret, PROTOCOL + "\n" + executionId + "\n" + digest);
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
        const body = await request.text();
        if (new TextEncoder().encode(body).byteLength > MAX_BODY_BYTES) {
          return response({ ok: false, error: "request_too_large" }, 413);
        }
        const envelope = JSON.parse(body);
        validateEnvelope(envelope, Math.floor(Date.now() / 1000));
        const secret = String(env.ORCHESTRATOR_PRIVATE_INPUT_SECRET || "");
        const expectedRef = await deriveRef(envelope.execution_id, envelope.input_digest, secret);
        if (!constantTimeEqual(fromHex(expectedRef), fromHex(envelope.input_ref))) {
          return response({ ok: false, error: "input_ref_mismatch" }, 400);
        }
        const result = await env.PRIVATE_INPUTS.getByName(envelope.input_ref).putInput(envelope);
        if (result.conflict) return response({ ok: false, error: "input_ref_conflict" }, 409);
        return response(result, result.created ? 201 : 200);
      }
      const match = url.pathname.match(/^\/v1\/inputs\/([0-9a-f]{64})$/);
      if (match && request.method === "GET") {
        const result = await env.PRIVATE_INPUTS.getByName(match[1]).getInput(match[1]);
        if (result === null) return response({ ok: false, error: "input_not_found" }, 404);
        return response(result);
      }
      if (match && request.method === "DELETE") {
        const result = await env.PRIVATE_INPUTS.getByName(match[1]).deleteInput(match[1]);
        return response(result);
      }
      return response({ ok: false, error: "not_found" }, 404);
    } catch (error) {
      const message = error instanceof Error ? error.message : "private_input_internal_error";
      if (/input_expired|ttl_out_of_bounds|invalid|unknown_|missing_|payload_|protocol_/.test(message)) {
        return response({ ok: false, error: message }, 400);
      }
      return response({ ok: false, error: "private_input_internal_error" }, 500);
    }
  },
};

export { PrivateInput };
