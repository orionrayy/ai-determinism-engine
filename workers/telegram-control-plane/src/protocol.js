import {sha256Hex} from "./crypto.js";

export const MAX_UPDATE_BYTES = 256 * 1024;
export const MAX_GOAL_CHARS = 3500;
export const MAX_CALLBACK_BYTES = 256;
export const PROTOCOL = "ai-orchestrator.telegram-control/v1";

function canonicalize(value) {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return "[" + value.map(canonicalize).join(",") + "]";
  }
  return (
    "{" +
    Object.keys(value)
      .sort()
      .map((key) => JSON.stringify(key) + ":" + canonicalize(value[key]))
      .join(",") +
    "}"
  );
}

export function canonicalJson(value) {
  return canonicalize(value);
}

export function parseCommand(text) {
  const raw = String(text || "").trim();
  if (!raw.startsWith("/")) return {kind: "text", text: raw, args: ""};
  const match = raw.match(/^\/([^\s]+)(?:\s+([\s\S]*))?$/);
  if (!match) return {kind: "text", text: raw, args: ""};
  const [commandPart, args = ""] = match.slice(1);
  const command = commandPart.split("@", 1)[0].toLowerCase();
  return {kind: "command", command, args: args.trim()};
}

export function splitFirstArg(args) {
  const raw = String(args || "").trim();
  if (!raw) return ["", ""];
  const index = raw.indexOf(" ");
  if (index < 0) return [raw, ""];
  return [raw.slice(0, index), raw.slice(index + 1).trim()];
}

export function normalizeGoal(value) {
  const goal = String(value || "").trim();
  if (!goal) throw new Error("goal_required");
  if (goal.length > MAX_GOAL_CHARS) throw new Error("goal_too_long");
  return goal;
}

export function normalizeWorkflowId(value) {
  const id = String(value || "").trim();
  if (!/^[A-Za-z0-9_.:-]{1,128}$/.test(id)) {
    throw new Error("workflow_id_invalid");
  }
  return id;
}

export function telegramEventId(updateId) {
  const id = Number(updateId);
  if (!Number.isSafeInteger(id) || id < 0) throw new Error("update_id_invalid");
  return "tg:" + String(id);
}

export async function intentDigest(command, argumentsText) {
  return sha256Hex(
    canonicalJson({
      protocol: PROTOCOL,
      command: String(command || "").toLowerCase(),
      arguments: String(argumentsText || "").trim(),
    }),
  );
}

export function callbackAction(data) {
  const raw = String(data || "");
  if (new TextEncoder().encode(raw).byteLength > MAX_CALLBACK_BYTES) {
    throw new Error("callback_too_large");
  }
  const [namespace, action] = raw.split(":", 2);
  if (namespace !== "consent") throw new Error("callback_invalid");
  if (!["accept", "revoke"].includes(action)) throw new Error("callback_invalid");
  return action;
}
