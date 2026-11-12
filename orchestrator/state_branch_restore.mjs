import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { VERSION, OBJECT_ROOT, MAX_STATE_FILES, MAX_STATE_FILE_BYTES, MAX_STATE_TOTAL_BYTES, StateBranchError, resolveContext, resolveKeys, runGit, normalizeStatePath } from './state_branch_core.mjs';
import { decryptStateObject } from './state_branch_crypto.mjs';
import { snapshotDirectory } from './state_branch_snapshot.mjs';

const BASE_FILE = 'orchestrator-state-base.json';

export function baseCheckpointPath(root, env) {
  const runIdentity = String(env.GITHUB_RUN_ID ?? env.CI_JOB_ID ?? crypto.createHash('sha256').update(path.resolve(root)).digest('hex').slice(0, 16));
  const safe = runIdentity.replace(/[^A-Za-z0-9_.-]/g, '_').slice(0, 80);
  return path.join(os.tmpdir(), `${BASE_FILE.replace('.json', '')}-${safe}.json`);
}

export function writeBaseCheckpoint(filePath, commit, snapshot) {
  const files = Object.fromEntries([...snapshot].sort(([a], [b]) => a.localeCompare(b)).map(([name, bytes]) => [name, bytes.toString('base64')]));
  const temporary = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify({ version: VERSION, commit, files })}\n`, { mode: 0o600 });
  fs.renameSync(temporary, filePath);
}

export function readBaseCheckpoint(filePath) {
  let record;
  try { record = JSON.parse(fs.readFileSync(filePath, 'utf8')); }
  catch { throw new StateBranchError('State base checkpoint is missing or corrupt; restore must succeed before persist.', 'STATE_BASE_MISSING'); }
  if (record?.version !== VERSION || !record.files || typeof record.files !== 'object' || Array.isArray(record.files)) {
    throw new StateBranchError('State base checkpoint format is invalid.', 'STATE_BASE_INVALID');
  }
  const snapshot = new Map();
  for (const [name, encoded] of Object.entries(record.files)) {
    const normalized = normalizeStatePath(name);
    if (typeof encoded !== 'string') throw new StateBranchError('State base checkpoint entry is invalid.', 'STATE_BASE_INVALID');
    const bytes = Buffer.from(encoded, 'base64');
    if (bytes.toString('base64') !== encoded || bytes.length > MAX_STATE_FILE_BYTES) throw new StateBranchError('State base checkpoint entry is invalid.', 'STATE_BASE_INVALID');
    snapshot.set(normalized, bytes);
  }
  return { commit: record.commit ?? null, snapshot };
}

export function remoteBranchCommit(root, branch) {
  const ref = `refs/heads/${branch}`;
  const remote = runGit(root, ['ls-remote', '--heads', 'origin', ref], { allowFailure: true });
  if (!remote.trim()) return null;
  runGit(root, ['fetch', '--no-tags', 'origin', `+${ref}:refs/remotes/origin/${branch}`]);
  const commit = runGit(root, ['rev-parse', `refs/remotes/origin/${branch}`]).trim();
  if (!/^[a-f0-9]{40,64}$/.test(commit)) throw new StateBranchError('Remote state branch reference is malformed.', 'STATE_BRANCH_REF_INVALID');
  return commit;
}

export function readRemoteSnapshot(root, commit, keys, ctx) {
  const result = new Map();
  const objects = new Map();
  if (!commit) return { snapshot: result, objects };
  const names = runGit(root, ['ls-tree', '-r', '--name-only', commit]).split('\n').filter(Boolean);
  if (names.length > MAX_STATE_FILES) throw new StateBranchError('Remote state branch exceeds object-count limit.', 'STATE_FILE_COUNT_LIMIT');
  let totalBytes = 0;
  for (const objectPath of names) {
    const match = new RegExp(`^${OBJECT_ROOT.replaceAll('.', '\\.').replaceAll('/', '\\/')}\\/([a-f0-9]{64})\\.enc$`).exec(objectPath);
    if (!match) throw new StateBranchError('State branch contains an unexpected plaintext or non-object path.', 'STATE_BRANCH_CONTENT_INVALID');
    const token = match[1];
    const envelope = runGit(root, ['show', `${commit}:${objectPath}`], { binary: true, maxBuffer: MAX_STATE_FILE_BYTES * 2 });
    const record = decryptStateObject(envelope, token, keys, ctx);
    if (result.has(record.path)) throw new StateBranchError('State branch contains duplicate logical objects.', 'STATE_DUPLICATE_OBJECT');
    totalBytes += record.data.length;
    if (totalBytes > MAX_STATE_TOTAL_BYTES) throw new StateBranchError('Remote state exceeds total-size limit.', 'STATE_TOTAL_SIZE_LIMIT');
    result.set(record.path, record.data);
    objects.set(record.path, { objectPath, token });
  }
  return { snapshot: result, objects };
}

function clearRestorableState(stateDir) {
  if (!fs.existsSync(stateDir)) { fs.mkdirSync(stateDir, { recursive: true, mode: 0o700 }); return; }
  const stat = fs.lstatSync(stateDir);
  if (!stat.isDirectory() || stat.isSymbolicLink()) throw new StateBranchError('.orchestrator is not a safe directory.', 'STATE_ROOT_INVALID');
  for (const name of fs.readdirSync(stateDir)) {
    const full = path.join(stateDir, name);
    const item = fs.lstatSync(full);
    if (item.isSymbolicLink()) throw new StateBranchError('Symlink found in local state; refusing automatic cleanup.', 'STATE_SYMLINK_BLOCKED');
    if (name === 'research-cache') continue;
    fs.rmSync(full, { recursive: true, force: false });
  }
}

function restoreFiles(stateDir, snapshot) {
  clearRestorableState(stateDir);
  for (const [relativePath, bytes] of snapshot) {
    const normalized = normalizeStatePath(relativePath);
    const destination = path.resolve(stateDir, normalized.slice('.orchestrator/'.length));
    if (!destination.startsWith(`${path.resolve(stateDir)}${path.sep}`)) throw new StateBranchError('Restored path escaped state directory.', 'STATE_PATH_INVALID');
    fs.mkdirSync(path.dirname(destination), { recursive: true, mode: 0o700 });
    fs.writeFileSync(destination, bytes, { mode: 0o600 });
  }
}

export function restoreState({ root = process.cwd(), env = process.env } = {}) {
  const ctx = resolveContext(env);
  const keys = resolveKeys(env, ctx.repository);
  const commit = remoteBranchCommit(root, ctx.branch);
  let snapshot;
  if (commit) {
    snapshot = readRemoteSnapshot(root, commit, keys, ctx).snapshot;
    restoreFiles(path.join(root, '.orchestrator'), snapshot);
    console.log(`Restored ${snapshot.size} encrypted orchestrator state object(s) from ${commit.slice(0, 12)}.`);
  } else {
    // One-time migration path: keep the checked-out legacy state locally; the next
    // successful persist encrypts it before creating the dedicated state branch.
    snapshot = new Map();
    console.log('Encrypted state branch is absent; retaining local state for one-time encrypted bootstrap.');
  }
  writeBaseCheckpoint(baseCheckpointPath(root, env), commit, snapshot);
  return { restored: Boolean(commit), bootstrap: !commit, files: snapshot.size, commit };
}
