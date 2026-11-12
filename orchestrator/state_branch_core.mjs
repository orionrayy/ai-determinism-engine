import crypto from 'node:crypto';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

export const DEFAULT_STATE_BRANCH = 'orchestrator-state';
export const OBJECT_ROOT = '.orchestrator-state/v1';
export const MAX_STATE_FILES = 10_000;
export const MAX_STATE_FILE_BYTES = 16 * 1024 * 1024;
export const MAX_STATE_TOTAL_BYTES = 64 * 1024 * 1024;
const BASE_FILE = 'orchestrator-state-base.json';
export const VERSION = 1;
export const RESEARCH_CACHE = '.orchestrator/research-cache';

export class StateBranchError extends Error {
  constructor(message, code = 'STATE_BRANCH_ERROR') {
    super(message);
    this.name = 'StateBranchError';
    this.code = code;
  }
}

export function runGit(root, args, options = {}) {
  const env = { ...process.env, ...(options.env ?? {}) };
  // The master key is consumed only in-process and is never forwarded to Git.
  delete env.ORCHESTRATOR_STATE_ENCRYPTION_KEY;
  const result = spawnSync('git', args, {
    cwd: root,
    input: options.input,
    encoding: options.binary ? null : 'utf8',
    env,
    maxBuffer: options.maxBuffer ?? 96 * 1024 * 1024,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  if (result.error) {
    throw new StateBranchError(`git ${args[0]} could not start: ${result.error.message}`, 'GIT_EXECUTION_ERROR');
  }
  if (result.status !== 0 && !options.allowFailure) {
    const stderr = Buffer.isBuffer(result.stderr) ? result.stderr.toString('utf8') : String(result.stderr ?? '');
    throw new StateBranchError(`git ${args[0]} failed (${result.status}): ${stderr.replace(/\s+/g, ' ').slice(0, 500)}`, 'GIT_COMMAND_FAILED');
  }
  return options.binary ? result.stdout : String(result.stdout ?? '');
}

export function resolveContext(env = process.env) {
  const repository = String(env.GITHUB_REPOSITORY ?? '').trim();
  const branch = String(env.ORCHESTRATOR_STATE_BRANCH ?? DEFAULT_STATE_BRANCH).trim();
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository)) {
    throw new StateBranchError('GITHUB_REPOSITORY is missing or invalid.', 'REPOSITORY_INVALID');
  }
  if (!/^[A-Za-z0-9._/-]+$/.test(branch) || branch.startsWith('-') || branch.includes('..') || branch.endsWith('/') || branch.includes('//')) {
    throw new StateBranchError('ORCHESTRATOR_STATE_BRANCH is invalid.', 'BRANCH_INVALID');
  }
  return { repository, branch };
}

export function resolveKeys(env, repository) {
  const raw = String(env.ORCHESTRATOR_STATE_ENCRYPTION_KEY ?? '').trim();
  if (!/^[A-Za-z0-9+/]{43}=$/.test(raw)) {
    throw new StateBranchError(
      'ORCHESTRATOR_STATE_ENCRYPTION_KEY must be canonical base64 for exactly 32 random bytes. Refusing plaintext fallback.',
      'STATE_KEY_MISSING_OR_INVALID',
    );
  }
  const master = Buffer.from(raw, 'base64');
  if (master.length !== 32 || master.toString('base64') !== raw) {
    throw new StateBranchError('State encryption key must decode to exactly 32 bytes.', 'STATE_KEY_MISSING_OR_INVALID');
  }
  const salt = Buffer.from(`ade-orchestrator-state-v1:${repository}`, 'utf8');
  const encryptionKey = Buffer.from(crypto.hkdfSync('sha256', master, salt, Buffer.from('aes-256-gcm'), 32));
  const pathKey = Buffer.from(crypto.hkdfSync('sha256', master, salt, Buffer.from('hmac-path-token'), 32));
  master.fill(0);
  return { encryptionKey, pathKey };
}

export function normalizeStatePath(value) {
  const raw = String(value ?? '').replaceAll('\\', '/');
  if (!raw.startsWith('.orchestrator/')) {
    throw new StateBranchError('State path is outside .orchestrator.', 'STATE_PATH_INVALID');
  }
  const normalized = path.posix.normalize(raw);
  const parts = normalized.split('/');
  if (normalized !== raw || parts.some((part) => !part || part === '.' || part === '..') || normalized.startsWith('../') || normalized.includes('//')) {
    throw new StateBranchError('State path is not canonical.', 'STATE_PATH_INVALID');
  }
  if (parts[1] === 'research-cache') {
    throw new StateBranchError('Research cache is ephemeral and excluded from state persistence.', 'STATE_PATH_EXCLUDED');
  }
  return normalized;
}

export function pathToken(relativePath, pathKey) {
  const normalized = normalizeStatePath(relativePath);
  return crypto.createHmac('sha256', pathKey)
    .update('ade-orchestrator-state-path-v1\0', 'utf8')
    .update(normalized, 'utf8')
    .digest('hex');
}

export function aad(context, token) {
  return Buffer.from(`ade-orchestrator-state-v1\n${context.repository}\n${context.branch}\n${token}`, 'utf8');
}
