const encoder = new TextEncoder();

function bytesToBase64url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function base64urlToBytes(value) {
  const normalized = String(value || "").replace(/-/g, "+").replace(/_/g, "/");
  const padded = normalized + "=".repeat((4 - (normalized.length % 4)) % 4);
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function hmacBytes(secret, message) {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(String(secret || "")),
    {name: "HMAC", hash: "SHA-256"},
    false,
    ["sign"],
  );
  return new Uint8Array(
    await crypto.subtle.sign("HMAC", key, typeof message === "string" ? encoder.encode(message) : message),
  );
}

export async function hmacHex(secret, message) {
  const bytes = await hmacBytes(secret, message);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function sha256Hex(value) {
  const data = typeof value === "string" ? encoder.encode(value) : value;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", data));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function keyedIdentity(secret, namespace, value) {
  return hmacHex(secret, String(namespace) + ":" + String(value));
}

function encryptionKeyBytes(secret) {
  const raw = String(secret || "");
  if (!raw) throw new Error("data_encryption_key_missing");
  try {
    const decoded = base64urlToBytes(raw);
    if (decoded.length === 32) return decoded;
  } catch (_) {}
  const utf8 = encoder.encode(raw);
  if (utf8.length === 32) return utf8;
  throw new Error("data_encryption_key_must_be_32_bytes_or_base64url_32_bytes");
}

export async function encryptJson(secret, value) {
  const keyBytes = encryptionKeyBytes(secret);
  const key = await crypto.subtle.importKey(
    "raw",
    keyBytes,
    {name: "AES-GCM"},
    false,
    ["encrypt"],
  );
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const plaintext = encoder.encode(JSON.stringify(value));
  const ciphertext = new Uint8Array(
    await crypto.subtle.encrypt({name: "AES-GCM", iv}, key, plaintext),
  );
  return bytesToBase64url(iv) + "." + bytesToBase64url(ciphertext);
}

export async function decryptJson(secret, packed) {
  const [ivEncoded, ciphertextEncoded] = String(packed || "").split(".", 2);
  if (!ivEncoded || !ciphertextEncoded) throw new Error("encrypted_record_invalid");
  const keyBytes = encryptionKeyBytes(secret);
  const key = await crypto.subtle.importKey(
    "raw",
    keyBytes,
    {name: "AES-GCM"},
    false,
    ["decrypt"],
  );
  const plaintext = await crypto.subtle.decrypt(
    {name: "AES-GCM", iv: base64urlToBytes(ivEncoded)},
    key,
    base64urlToBytes(ciphertextEncoded),
  );
  return JSON.parse(new TextDecoder().decode(plaintext));
}
