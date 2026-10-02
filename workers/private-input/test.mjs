import assert from "node:assert/strict";
import worker from "./src/index.js";

const SECRET = "test-private-secret";
const encoder = new TextEncoder();
const store = new Map();

async function hmac(message) {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const digest = await crypto.subtle.sign("HMAC", key, encoder.encode(message));
  return [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function sign(method, path, timestamp, body) {
  const msg = [method, path, "ai-orchestrator.private-input/v2", String(timestamp), body].join("\n");
  return "sha256=" + await hmac(msg);
}

function env() {
  return {
    ORCHESTRATOR_PRIVATE_INPUT_SECRET: SECRET,
    PRIVATE_INPUTS: {
      async get(key) {
        return store.get(key) ?? null;
      },
      async put(key, value, options) {
        store.set(key, value);
        assert.equal(typeof options.expiration, "number");
      },
      async delete(key) {
        store.delete(key);
      },
    },
  };
}

async function call(method, path, body = "") {
  const timestamp = Math.floor(Date.now() / 1000);
  const headers = {
    "X-Orchestrator-Protocol": "ai-orchestrator.private-input/v2",
    "X-Orchestrator-Timestamp": String(timestamp),
    "X-Orchestrator-Signature": await sign(method, path, timestamp, body),
  };
  if (method === "POST") headers["Content-Type"] = "application/json";
  return worker.fetch(
    new Request("https://example.test" + path, {
      method,
      body: method === "GET" || method === "DELETE" ? undefined : body,
      headers,
    }),
    env(),
  );
}

const executionId = "e".repeat(64);
const digest = "d".repeat(64);
const intent = "f".repeat(64);
const expiresAt = Math.floor(Date.now() / 1000) + 3600;
const ref = await hmac(
  "ai-orchestrator.private-input/v2\n" + executionId + "\n" + digest,
);
const envelope = {
  schema_version: 1,
  protocol: "ai-orchestrator.private-input/v2",
  input_ref: ref,
  execution_id: executionId,
  intent_fingerprint: intent,
  input_digest: digest,
  expires_at: expiresAt,
  payload: { title: "private" },
};

let // Protocol and schema gates must fail closed.
response = await worker.fetch(
  new Request("https://example.test/v1/inputs", {
    method: "POST",
    body: JSON.stringify(envelope),
    headers: {
      "X-Orchestrator-Protocol": "ai-orchestrator.private-input/v2",
      "X-Orchestrator-Timestamp": "123abc",
      "X-Orchestrator-Signature": await sign("POST", "/v1/inputs", 123, JSON.stringify(envelope)),
      "Content-Type": "application/json",
    },
  }),
  env(),
);
assert.equal(response.status, 401);

const unknownEnvelope = { ...envelope, extra: true };
response = await call("POST", "/v1/inputs", JSON.stringify(unknownEnvelope));
assert.equal(response.status, 400);

response = await call("POST", "/v1/inputs", JSON.stringify(envelope));
assert.equal(response.status, 201);
let data = await response.json();
assert.equal(data.ok, true);
assert.equal(data.input_ref, ref);

response = await call("POST", "/v1/inputs", JSON.stringify(envelope));
assert.equal(response.status, 200);

response = await call("GET", "/v1/inputs/" + ref);
assert.equal(response.status, 200);
data = await response.json();
assert.deepEqual(data.payload, { title: "private" });
assert.equal(data.execution_id, executionId);

response = await worker.fetch(
  new Request("https://example.test/v1/inputs", {
    method: "POST",
    body: JSON.stringify(envelope),
    headers: {
      "X-Orchestrator-Protocol": "ai-orchestrator.private-input/v2",
      "X-Orchestrator-Timestamp": String(Math.floor(Date.now() / 1000)),
      "X-Orchestrator-Signature": "sha256=" + "0".repeat(64),
      "Content-Type": "application/json",
    },
  }),
  env(),
);
assert.equal(response.status, 401);

response = await call("DELETE", "/v1/inputs/" + ref);
assert.equal(response.status, 200);

response = await call("GET", "/v1/inputs/" + ref);
assert.equal(response.status, 404);

console.log("private-input worker tests: OK");
