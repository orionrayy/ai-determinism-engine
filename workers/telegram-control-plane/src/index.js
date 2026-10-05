import {constantTimeEqual, isAllowedUser, isApprover, isPrivateMessage} from "./policy.js";
import {callbackAction, normalizeGoal, normalizeWorkflowId, parseCommand, splitFirstArg, telegramEventId} from "./protocol.js";
import {audit, bindWorkflow, chatKey, claimUpdate, completeUpdate, deleteUserData, failUpdate, getConsent, getSession, hasDatabase, ownsWorkflow, principalKey, revokeConsent, saveSession, setConsent, touchWorkflow} from "./db.js";
import {buildRunPayload, buildWorkflowPayload, dispatchToGateway} from "./gateway.js";
import {readWorkflowState, summarizeState} from "./control_plane.js";

const POLICY_VERSION = "2026-10-05";
const JSON_HEADERS = {"content-type": "application/json; charset=utf-8", "cache-control": "no-store"};
const PRIVACY_TEXT = `VORENYX Orchestrator Core — Privacy Notice

This bot is a third-party application and is not operated by Telegram.

Purpose: receive commands and user-submitted prompts that you explicitly send to the bot, route them to the VORENYX orchestration system, and return bounded workflow status/results.

Data minimization:
- Telegram user/chat identifiers are used only for authorization and operational correlation.
- Persistent identifiers are HMAC-derived keys; raw Telegram IDs are not stored in the database.
- Optional session data is encrypted at rest with an application-managed AES-256-GCM key.
- Audit records contain action names, workflow IDs and cryptographic digests; prompt/message bodies are not stored in the Telegram database.
- Group/channel messages are not accepted by this bot.

AI/data use:
- Data received from Telegram is not collected for model training, fine-tuning, benchmarking datasets, or unrelated analytics.
- A prompt you explicitly submit may be forwarded to the orchestration engine and, according to its own execution policy, to an external provider required to perform the requested operation.

Your controls:
- Use /privacy for this notice.
- Use /delete_me to remove the bot's stored database records associated with your account.
- Use /prompt off to disable direct text prompting.

By selecting Authorize, you give active, revocable consent for the processing described above.`;

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {status, headers: JSON_HEADERS});
}

function typeName(error) {
  return String(error?.name || "Error");
}

async function telegramApi(env, method, payload) {
  if (!env.TELEGRAM_BOT_TOKEN) throw new Error("telegram_bot_token_missing");
  const response = await fetch(
    "https://api.telegram.org/bot" + env.TELEGRAM_BOT_TOKEN + "/" + method,
    {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify(payload),
    },
  );
  const raw = new Uint8Array(await response.arrayBuffer());
  if (raw.byteLength > 128 * 1024) throw new Error("telegram_api_response_too_large");
  const data = raw.byteLength ? JSON.parse(new TextDecoder().decode(raw)) : {};
  if (!response.ok || data.ok !== true) throw new Error(String(data?.description || "telegram_api_failed"));
  return data.result;
}

async function sendText(env, chatId, text, replyMarkup) {
  const payload = {
    chat_id: chatId,
    text: String(text || "").slice(0, 3900),
    disable_web_page_preview: true,
  };
  if (replyMarkup) payload.reply_markup = replyMarkup;
  return telegramApi(env, "sendMessage", payload);
}

async function answerCallback(env, callbackId) {
  try {
    await telegramApi(env, "answerCallbackQuery", {callback_query_id: callbackId});
  } catch (_) {}
}

function commandMenu() {
  return {
    inline_keyboard: [
      [{text: "Authorize", callback_data: "consent:accept"}, {text: "Privacy", callback_data: "consent:privacy"}],
      [{text: "Revoke", callback_data: "consent:revoke"}],
    ],
  };
}


async function claimMutableUpdate(env, userId, eventId) {
  const pKey = await principalKey(env, userId);
  const result = await claimUpdate(env, eventId, pKey);
  return {pKey, claimed: Boolean(result.claimed), workflowId: result.workflowId || null};
}

async function completeMutableUpdate(env, eventId, workflowId, claimToken) {
  try {
    await completeUpdate(env, eventId, workflowId, claimToken);
  } catch (_) {
    // Gateway/orchestrator idempotency remains the execution backstop.
  }
}

async function failMutableUpdate(env, eventId, claimToken) {
  try {
    await failUpdate(env, eventId, claimToken);
  } catch (_) {}
}

function helpText() {
  return [
    "VORENYX Orchestrator Core",
    "",
    "/run <goal> — create a dry-run workflow",
    "/runlive <goal> — request live execution",
    "/status <workflow_id> — bounded status view",
    "/resume <workflow_id> — continue an existing workflow",
    "/approve <workflow_id> — approve high-risk continuation (authorized approvers only)",
    "/prompt on|off — direct text prompting mode",
    "/last — status of your last workflow",
    "/id — show your Telegram user ID",
    "/privacy — privacy notice",
    "/delete_me — delete stored bot records",
    "/help — this help",
    "",
    "Live side effects remain subject to the orchestrator's policy and approval gates.",
  ].join("\n");
}

async function requireConsent(env, userId) {
  if (!hasDatabase(env)) throw new Error("database_not_configured");
  const pKey = await principalKey(env, userId);
  const consent = await getConsent(env, pKey);
  if (!consent) throw new Error("consent_required");
  return pKey;
}

async function recordCommand(env, userId, chatId, eventId, action, workflowId, digest) {
  if (!hasDatabase(env)) return;
  try {
    const pKey = await principalKey(env, userId);
    const cKey = await chatKey(env, chatId);
    await audit(env, {
      eventId,
      pKey,
      cKey,
      action,
      workflowId,
      intentDigest: digest,
    });
  } catch (error) {
    console.error("telegram audit persistence failed", typeName(error));
  }
}

async function requireWorkflowAccess(env, userId, chatId, workflowId) {
  const pKey = await principalKey(env, userId);
  if (await ownsWorkflow(env, workflowId, pKey)) {
    await touchWorkflow(env, workflowId, pKey);
    return;
  }
  if (isApprover(env, userId)) return;
  const cKey = await chatKey(env, chatId);
  const session = await getSession(env, cKey);
  if (session?.last_workflow_id === workflowId) return;
  throw new Error("workflow_access_denied");
}

async function dispatchAndReply(env, message, eventId, payload, action) {
  const result = await dispatchToGateway(env, payload, eventId);
  const workflowId = String(result?.workflow_id || payload?.workflow_id || "");
  if (workflowId) {
    const pKey = await principalKey(env, message.from.id);
    try {
      await bindWorkflow(env, workflowId, pKey);
    } catch (error) {
      console.error("telegram workflow ownership persistence failed", typeName(error));
    }
    try {
      const cKey = await chatKey(env, message.chat.id);
      const session = (await getSession(env, cKey)) || {prompt_mode: false, last_workflow_id: null};
      session.last_workflow_id = workflowId;
      await saveSession(env, cKey, session);
    } catch (error) {
      console.error("telegram session persistence failed", typeName(error));
    }
  }
  try {
    await recordCommand(env, message.from.id, message.chat.id, eventId, action, workflowId || null, null);
  } catch (error) {
    console.error("telegram audit persistence failed", typeName(error));
  }
  return workflowId || "accepted";
}

async function handleCallback(env, update) {
  const callback = update.callback_query;
  const message = callback?.message;
  const user = callback?.from;
  if (!callback || !message || !user || message.chat.type !== "private") {
    await answerCallback(env, callback?.id);
    return;
  }
  await answerCallback(env, callback.id);
  if (!isAllowedUser(env, user.id)) {
    await sendText(env, message.chat.id, "Access denied.");
    return;
  }
  const callbackEventId = "callback:" + telegramEventId(update.update_id);
  let mutationEventId = callbackEventId;
  let mutationClaimToken = null;
  try {
    const action = callbackAction(callback.data);
    if (action === "privacy") {
      await sendText(env, message.chat.id, PRIVACY_TEXT);
      return;
    }
    const pKey = await principalKey(env, user.id);
    mutationEventId = callbackEventId + ":" + action;
    const claimed = await claimUpdate(env, mutationEventId, pKey);
    if (!claimed.claimed) return;
    mutationClaimToken = claimed.claimToken;
    if (action === "accept") {
      await setConsent(env, pKey, POLICY_VERSION);
      await completeMutableUpdate(env, mutationEventId, null, mutationClaimToken);
      await sendText(env, message.chat.id, "Authorization recorded.\n\n" + helpText());
      return;
    }
    await revokeConsent(env, pKey);
    await completeMutableUpdate(env, mutationEventId, null, mutationClaimToken);
    await sendText(env, message.chat.id, "Consent revoked. New execution commands are disabled until you authorize again.");
  } catch (error) {
    await failMutableUpdate(env, mutationEventId, mutationClaimToken);
    const code = String(error?.message || "request_failed");
    await sendText(env, message.chat.id, code === "database_not_configured"
      ? "Database persistence is not configured; authorization cannot be recorded yet."
      : "Request could not be completed.");
  }
}

async function handleMessage(env, update) {
  const message = update.message;
  if (!message || !isPrivateMessage(message)) return;
  const userId = message.from.id;
  const chatId = message.chat.id;
  const eventId = telegramEventId(update.update_id);
  let mutationClaimToken = null;
  const parsed = parseCommand(message.text);

  if (parsed.kind === "command" && parsed.command === "id") {
    await sendText(env, chatId, "Your Telegram user ID: " + String(userId));
    return;
  }

  if (!isAllowedUser(env, userId)) {
    await sendText(env, chatId, "Access denied.");
    return;
  }

  try {
    if (parsed.kind === "command") {
      switch (parsed.command) {
        case "start":
          await sendText(
            env,
            chatId,
            PRIVACY_TEXT + "\n\nAuthorization is required before any orchestration command.",
            commandMenu(),
          );
          return;
        case "help":
          await sendText(env, chatId, helpText());
          return;
        case "privacy":
          await sendText(env, chatId, PRIVACY_TEXT);
          return;
        case "revoke": {
          const pKey = await principalKey(env, userId);
          const claim = await claimMutableUpdate(env, userId, eventId);
          if (!claim.claimed) return;
          await revokeConsent(env, pKey);
          await completeMutableUpdate(env, eventId, null, mutationClaimToken);
          await sendText(env, chatId, "Consent revoked. New execution commands are disabled until you authorize again.");
          return;
        }
        case "delete_me": {
          const pKey = await principalKey(env, userId);
          const cKey = await chatKey(env, chatId);
          const claim = await claimMutableUpdate(env, userId, eventId);
          if (!claim.claimed) return;
          mutationClaimToken = claim.claimToken;
          await deleteUserData(env, pKey, cKey);
          await completeMutableUpdate(env, eventId, null, mutationClaimToken);
          await sendText(env, chatId, "Stored bot records associated with this account were deleted.");
          return;
        }
        case "prompt": {
          await requireConsent(env, userId);
          const [mode] = splitFirstArg(parsed.args);
          if (!["on", "off"].includes(mode)) throw new Error("usage_prompt_on_off");
          const cKey = await chatKey(env, chatId);
          const session = (await getSession(env, cKey)) || {prompt_mode: false, last_workflow_id: null};
          const claim = await claimMutableUpdate(env, userId, eventId);
          if (!claim.claimed) return;
          mutationClaimToken = claim.claimToken;
          session.prompt_mode = mode === "on";
          await saveSession(env, cKey, session);
          await completeMutableUpdate(env, eventId, null);
          await sendText(env, chatId, "Direct text prompting: " + mode.toUpperCase());
          return;
        }
        case "run":
        case "runlive": {
          await requireConsent(env, userId);
          const claim = await claimMutableUpdate(env, userId, eventId);
          if (!claim.claimed) return;
          mutationClaimToken = claim.claimToken;
          const goal = normalizeGoal(parsed.args);
          const mode = parsed.command === "runlive" ? "live" : "dry-run";
          const workflowId = await dispatchAndReply(
            env,
            message,
            eventId,
            buildRunPayload(goal, mode, eventId),
            mode === "live" ? "runlive" : "run",
          );
          await completeMutableUpdate(env, eventId, workflowId, mutationClaimToken);
          await sendText(env, chatId, mode === "live"
            ? "Live execution requested. Workflow: " + workflowId + "\nHigh-risk effects still require approval."
            : "Dry-run accepted. Workflow: " + workflowId);
          return;
        }
        case "status": {
          await requireConsent(env, userId);
          const [workflowIdArg] = splitFirstArg(parsed.args);
          const workflowId = normalizeWorkflowId(workflowIdArg);
          await requireWorkflowAccess(env, userId, chatId, workflowId);
          const state = await readWorkflowState(env, workflowId);
          const summary = summarizeState(state);
          await recordCommand(
            env,
            userId,
            chatId,
            eventId,
            "status",
            workflowId,
            null,
          );
          await sendText(env, chatId, [
            "Workflow status",
            "ID: " + summary.workflow_id,
            "State: " + summary.status,
            "Mode: " + summary.mode,
            "Updated: " + summary.updated_at,
            "Nodes: " + JSON.stringify(summary.node_counts),
            "Failed node: " + String(summary.failed_node || "none"),
            "Replans: " + String(summary.replan_count),
            "Attempts: " + String(summary.attempts_used),
          ].join("\n"));
          return;
        }
        case "resume": {
          await requireConsent(env, userId);
          const claim = await claimMutableUpdate(env, userId, eventId);
          if (!claim.claimed) return;
          mutationClaimToken = claim.claimToken;
          const [workflowIdArg] = splitFirstArg(parsed.args);
          const workflowId = normalizeWorkflowId(workflowIdArg);
          await requireWorkflowAccess(env, userId, workflowId);
          const result = await dispatchToGateway(
            env,
            buildWorkflowPayload("resume", workflowId, eventId),
            eventId,
          );
          await completeMutableUpdate(env, eventId, workflowId, mutationClaimToken);
          await recordCommand(env, userId, chatId, eventId, "resume", workflowId, null);
          await sendText(env, chatId, "Resume requested for workflow " + workflowId + ".\nGateway: " + String(result?.queued ? "accepted" : "submitted"));
          return;
        }
        case "approve": {
          if (!isApprover(env, userId)) {
            await sendText(env, chatId, "Approval access denied.");
            return;
          }
          await requireConsent(env, userId);
          const claim = await claimMutableUpdate(env, userId, eventId);
          if (!claim.claimed) return;
          mutationClaimToken = claim.claimToken;
          const [workflowIdArg] = splitFirstArg(parsed.args);
          const workflowId = normalizeWorkflowId(workflowIdArg);
          await dispatchToGateway(
            env,
            buildWorkflowPayload("approve", workflowId, eventId, true),
            eventId,
          );
          await completeMutableUpdate(env, eventId, workflowId, mutationClaimToken);
          await recordCommand(env, userId, chatId, eventId, "approve", workflowId, null);
          await sendText(env, chatId, "Approval signal submitted for workflow " + workflowId + ". The orchestrator will still enforce its own validation and policy.");
          return;
        }
        case "last": {
          await requireConsent(env, userId);
          const cKey = await chatKey(env, chatId);
          const session = await getSession(env, cKey);
          const workflowId = normalizeWorkflowId(session?.last_workflow_id || "");
          const state = await readWorkflowState(env, workflowId);
          const summary = summarizeState(state);
          await sendText(env, chatId, [
            "Last workflow",
            "ID: " + summary.workflow_id,
            "State: " + summary.status,
            "Mode: " + summary.mode,
            "Nodes: " + JSON.stringify(summary.node_counts),
          ].join("\n"));
          return;
        }
        default:
          await sendText(env, chatId, helpText());
          return;
      }
    }

    await requireConsent(env, userId);
    const cKey = await chatKey(env, chatId);
    const session = await getSession(env, cKey);
    if (!session?.prompt_mode) {
      await sendText(env, chatId, "Direct text prompting is off. Use /prompt on, then send a prompt.");
      return;
    }
    const goal = normalizeGoal(message.text || "");
    const claim = await claimMutableUpdate(env, userId, eventId);
    if (!claim.claimed) return;
    mutationClaimToken = claim.claimToken;
    const workflowId = await dispatchAndReply(
      env,
      message,
      eventId,
      buildRunPayload(goal, "dry-run", eventId),
      "prompt_text",
    );
    await completeMutableUpdate(env, eventId, workflowId, mutationClaimToken);
    await sendText(env, chatId, "Prompt accepted as dry-run. Workflow: " + workflowId);

  } catch (error) {
    const code = String(error?.message || "request_failed");
    const userMessage = {
      database_not_configured: "Database persistence is not configured.",
      data_encryption_key_missing: "Bot storage encryption is not configured.",
      data_hmac_key_missing: "Bot data integrity key is not configured.",
      consent_required: "Authorize first with /start.",
      goal_required: "Provide a goal after the command.",
      goal_too_long: "Goal is too long.",
      workflow_id_invalid: "Workflow ID is invalid or missing.",
      control_plane_not_configured: "Control-plane status access is not configured.",
      gateway_not_configured: "Gateway dispatch is not configured.",
      workflow_access_denied: "You are not authorized to manage this workflow.",
    }[code] || "Request could not be completed.";
    await sendText(env, chatId, userMessage);
  }
}

function privacyResponse() {
  return new Response(
    "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>VORENYX Orchestrator Privacy</title></head><body><main style=\"max-width:760px;margin:40px auto;padding:0 18px;font-family:system-ui,sans-serif;white-space:pre-wrap\">" +
      PRIVACY_TEXT.replace(/&/g, "&amp;").replace(/</g, "&lt;") +
      "</main></body></html>",
    {status: 200, headers: {"content-type": "text/html; charset=utf-8", "cache-control": "public, max-age=300"}},
  );
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/privacy") return privacyResponse();
    if (request.method === "GET" && url.pathname === "/health") return jsonResponse({ok: true, service: "telegram-control-plane"});
    if (request.method !== "POST" || url.pathname !== "/telegram/webhook") {
      return jsonResponse({error: "not_found"}, 404);
    }
    const secret = env.TELEGRAM_WEBHOOK_SECRET;
    const supplied = request.headers.get("X-Telegram-Bot-Api-Secret-Token") || "";
    if (!secret || !constantTimeEqual(secret, supplied)) {
      return jsonResponse({error: "unauthorized"}, 401);
    }

    const raw = new Uint8Array(await request.arrayBuffer());
    if (raw.byteLength > 256 * 1024) return jsonResponse({error: "update_too_large"}, 413);

    let update;
    try {
      update = JSON.parse(new TextDecoder().decode(raw));
    } catch (_) {
      return jsonResponse({error: "invalid_json"}, 400);
    }
    try {
      if (update.callback_query) {
        await handleCallback(env, update);
      } else if (update.message) {
        await handleMessage(env, update);
      }
    } catch (_) {
      // Do not log or persist Telegram payloads, which may contain user prompts.
    }
    return jsonResponse({ok: true});
  },
};
