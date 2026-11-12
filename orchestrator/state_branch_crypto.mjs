import crypto from 'node:crypto';
import { OBJECT_ROOT, VERSION, MAX_STATE_FILE_BYTES, StateBranchError, normalizeStatePath, pathToken, aad } from './state_branch_core.mjs';

export function encryptStateObject(relativePath, value, keys, context) {
  const normalized = normalizeStatePath(relativePath);
  const data = Buffer.isBuffer(value) ? value : Buffer.from(value);
  if (data.length > MAX_STATE_FILE_BYTES) {
    throw new StateBranchError(`State object exceeds ${MAX_STATE_FILE_BYTES} bytes.`, 'STATE_FILE_TOO_LARGE');
  }
  const token = pathToken(normalized, keys.pathKey);
  const nonce = crypto.randomBytes(12);
  const payload = Buffer.from(JSON.stringify({
    version: VERSION,
    path: normalized,
    sha256: crypto.createHash('sha256').update(data).digest('hex'),
    data: data.toString('base64'),
  }), 'utf8');
  const cipher = crypto.createCipheriv('aes-256-gcm', keys.encryptionKey, nonce);
  cipher.setAAD(aad(context, token));
  const ciphertext = Buffer.concat([cipher.update(payload), cipher.final()]);
  const envelope = {
    version: VERSION,
    algorithm: 'aes-256-gcm',
    nonce: nonce.toString('base64'),
    tag: cipher.getAuthTag().toString('base64'),
    ciphertext: ciphertext.toString('base64'),
  };
  return { token, objectPath: `${OBJECT_ROOT}/${token}.enc`, bytes: Buffer.from(`${JSON.stringify(envelope)}\n`, 'utf8') };
}

export function decryptStateObject(envelopeBytes, expectedToken, keys, context) {
  let envelope;
  try {
    envelope = JSON.parse(Buffer.from(envelopeBytes).toString('utf8'));
  } catch {
    throw new StateBranchError('Encrypted state envelope is invalid JSON.', 'STATE_ENVELOPE_INVALID');
  }
  if (envelope?.version !== VERSION || envelope?.algorithm !== 'aes-256-gcm') {
    throw new StateBranchError('Encrypted state envelope version or algorithm is unsupported.', 'STATE_ENVELOPE_UNSUPPORTED');
  }
  const decodeCanonical = (value, name) => {
    if (typeof value !== 'string') throw new Error(name);
    const decoded = Buffer.from(value, 'base64');
    if (decoded.toString('base64') !== value) throw new Error(name);
    return decoded;
  };
  let nonce; let tag; let ciphertext;
  try {
    nonce = decodeCanonical(envelope.nonce, 'nonce');
    tag = decodeCanonical(envelope.tag, 'tag');
    ciphertext = decodeCanonical(envelope.ciphertext, 'ciphertext');
  } catch {
    throw new StateBranchError('Encrypted state envelope encoding is invalid.', 'STATE_ENVELOPE_INVALID');
  }
  if (nonce.length !== 12 || tag.length !== 16 || ciphertext.length === 0) {
    throw new StateBranchError('Encrypted state envelope dimensions are invalid.', 'STATE_ENVELOPE_INVALID');
  }
  let plaintext;
  try {
    const decipher = crypto.createDecipheriv('aes-256-gcm', keys.encryptionKey, nonce);
    decipher.setAAD(aad(context, expectedToken));
    decipher.setAuthTag(tag);
    plaintext = Buffer.concat([decipher.update(ciphertext), decipher.final()]);
  } catch {
    throw new StateBranchError('State authentication failed; refusing recovery or fallback.', 'STATE_AUTH_FAILED');
  }
  let record;
  try {
    record = JSON.parse(plaintext.toString('utf8'));
  } catch {
    throw new StateBranchError('Decrypted state record is invalid JSON.', 'STATE_RECORD_INVALID');
  }
  const relativePath = normalizeStatePath(record?.path);
  if (record?.version !== VERSION || pathToken(relativePath, keys.pathKey) !== expectedToken || typeof record.data !== 'string') {
    throw new StateBranchError('Decrypted state identity is invalid.', 'STATE_RECORD_INVALID');
  }
  let data;
  try {
    data = decodeCanonical(record.data, 'data');
  } catch {
    throw new StateBranchError('Decrypted state data encoding is invalid.', 'STATE_RECORD_INVALID');
  }
  const digest = crypto.createHash('sha256').update(data).digest('hex');
  if (data.length > MAX_STATE_FILE_BYTES || digest !== record.sha256) {
    throw new StateBranchError('Decrypted state digest or size is invalid.', 'STATE_DIGEST_INVALID');
  }
  return { path: relativePath, data, sha256: digest };
}
