import {hmacHex, sha256Hex} from "./crypto.js";
import {canonicalJson} from "./protocol.js";
import {readGitWorkflowSummary} from "./gateway.js";
import {normalizeWorkflowId} from "./protocol.js";

const MAX_RESPONSE_BYTES = 700 * 1024;

export async function readWorkflowState(env, workflowId) {
  if (!env.CONTROL_PLANE_URL || !env.CONTROL_PLANE_SECRET) {
    throw new Error("control_plane_not_configured");
  }
  const id = normalizeWorkflowId(workflowId);
  const path = "/v1/workflows/" + encodeURIComponent(id) + "/state";
  const timestamp = String(Math.floor(Date.now() / 1000));
  const signature = "sha256=" + (
    await hmacHex(
      env.CONTROL_PLANE_SECRET,
      timestamp + "\nGET\n" + path + "\n\n" + "",
    )
  );
  const response = await fetch(new URL(path, env.CONTROL_PLANE_URL), {
    method: "GET",
    headers: {
      "accept": "application/json",
      "X-Control-Plane-Timestamp": timestamp,
      "X-Control-Plane-Signature": signature,
      "X-Control-Plane-Request-ID": "",
    },
  });
  const raw = new Uint8Array(await response.arrayBuffer());
  if (raw.byteLength > MAX_RESPONSE_BYTES) throw new Error("control_plane_response_too_large");
  let result = {};
  if (raw.byteLength) result = JSON.parse(new TextDecoder().decode(raw));
  if (!response.ok) throw new Error(String(result?.error || "control_plane_request_failed"));
  return result;
}

export function summarizeState(payload) {
  const state = payload?.state;
  if (!state || typeof state !== "object" || Array.isArray(state)) {
    throw new Error("workflow_state_invalid");
  }
  const nodes = Array.isArray(state.nodes) ? state.nodes : [];
  const counts = {};
  for (const node of nodes) {
    const status = String(node?.status || "unknown");
    counts[status] = (counts[status] || 0) + 1;
  }
  const pending_approvals = nodes
    .filter((node) => String(node?.status || "") === "waiting_approval")
    .slice(0, 16)
    .map((node) => ({
      node_id: String(node?.id || ""),
      risk: String(node?.risk || "unknown"),
      tool: String(node?.tool || ""),
      approval_issue: node?.input?.approval_issue ? Number(node.input.approval_issue) : null,
    }));
  return {
    workflow_id: String(state.id || ""),
    status: String(state.status || "unknown"),
    mode: String(state.execution_mode || (state.live ? "live" : "dry-run")),
    updated_at: String(state.updated_at || payload.updated_at || ""),
    node_counts: counts,
    pending_approvals,
    failed_node: state.failed_node ? String(state.failed_node) : null,
    replan_count: Number(state.replan_count || 0),
    attempts_used: Number(state.attempts_used || 0),
  };
}


export async function readWorkflowSummary(env, workflowId) {
  let controlPlaneError = null;
  if (env.CONTROL_PLANE_URL && env.CONTROL_PLANE_SECRET) {
    try {
      const payload = await readWorkflowState(env, workflowId);
      return summarizeState(payload);
    } catch (error) {
      controlPlaneError = error;
      if (Number(error?.status || 0) === 404) {
        controlPlaneError = null;
      }
    }
  }
  try {
    return await readGitWorkflowSummary(env, workflowId);
  } catch (error) {
    const code = String(error?.message || "workflow_status_unavailable");
    if (code === "distributed_control_plane_required" && controlPlaneError) {
      throw controlPlaneError;
    }
    if (code === "distributed_control_plane_required") {
      throw new Error("control_plane_required");
    }
    if (controlPlaneError) throw controlPlaneError;
    throw error;
  }
}

export async function statusDigest(summary) {
  return sha256Hex(canonicalJson(summary));
}
