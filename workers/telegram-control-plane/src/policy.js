const CSV_RE = /[^,]+/;

export function parseCsv(value) {
  return new Set(
    String(value || "")
      .split(",")
      .map((item) => item.trim())
      .filter((item) => CSV_RE.test(item)),
  );
}

export function isAllowedUser(env, userId) {
  return parseCsv(env.TELEGRAM_ALLOWED_USER_IDS).has(String(userId));
}

export function isApprover(env, userId) {
  return parseCsv(env.TELEGRAM_APPROVER_USER_IDS).has(String(userId));
}

export function isPrivateMessage(message) {
  return message?.chat?.type === "private" && message?.from?.id != null;
}

export function isSafeWorkflowId(value) {
  return /^[A-Za-z0-9_.:-]{1,128}$/.test(String(value || ""));
}

export function constantTimeEqual(a, b) {
  const left = new TextEncoder().encode(String(a || ""));
  const right = new TextEncoder().encode(String(b || ""));
  let diff = left.length ^ right.length;
  const length = Math.max(left.length, right.length);
  for (let i = 0; i < length; i += 1) {
    diff |= (left[i] || 0) ^ (right[i] || 0);
  }
  return diff === 0;
}
