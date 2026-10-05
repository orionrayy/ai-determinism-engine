import {canonicalJson, PROTOCOL} from "./protocol.js";
import {hmacHex} from "./crypto.js";

const encoder = new TextEncoder();
const MAX_RESPONSE_BYTES = 64 * 1024;

function requireGateway(env) {
  if (!env.GATEWAY_URL || !env.GATEWAY_SHARED_SECRET) {
    throw new Error("gateway_not_configured");
  }
}

export async function dispatchToGateway(env, payload, eventId) {
  requireGateway(env);
  const body = canonicalJson(payload);
  const path = "/event";
  const timestamp = String(Math.floor(Date.now() / 1000));
  const signatureInput = timestamp + "\nPOST\n" + path + "\n" + eventId + "\n" + body;
  const signature = "sha256=" + (await hmacHex(env.GATEWAY_SHARED_SECRET, signatureInput));

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(new URL(path, env.GATEWAY_URL), {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "accept": "application/json",
        "user-agent": "Vorenyx-Telegram-Control/1.0",
        "X-Orchestrator-Timestamp": timestamp,
        "X-Orchestrator-Signature": signature,
        "Idempotency-Key": eventId,
        "X-Orchestrator-Protocol": PROTOCOL,
      },
      body,
      signal: controller.signal,
    });
    const raw = new Uint8Array(await response.arrayBuffer());
    if (raw.byteLength > MAX_RESPONSE_BYTES) throw new Error("gateway_response_too_large");
    let result = {};
    if (raw.byteLength) result = JSON.parse(new TextDecoder().decode(raw));
    if (!response.ok) throw new Error(String(result?.error || "gateway_dispatch_failed"));
    return result;
  } finally {
    clearTimeout(timeout);
  }
}

export function buildRunPayload(goal, mode, eventId) {
  return {
    domain: "telegram",
    operation: mode === "live" ? "prompt_live" : "prompt",
    input: {goal},
    requested_mode: mode,
    event_id: eventId,
    idempotency_key: eventId,
    source: "telegram",
  };
}

export function buildWorkflowPayload(operation, workflowId, eventId, approveHighRisk = false) {
  const payload = {
    domain: "orchestration",
    operation,
    input: {workflow_id: workflowId},
    workflow_id: workflowId,
    requested_mode: "dry-run",
    event_id: eventId,
    idempotency_key: eventId,
    source: "telegram",
  };
  if (approveHighRisk) payload.approve_high_risk = true;
  return payload;
}
