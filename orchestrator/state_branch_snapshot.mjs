import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { MAX_STATE_FILES, MAX_STATE_FILE_BYTES, MAX_STATE_TOTAL_BYTES, RESEARCH_CACHE, StateBranchError, normalizeStatePath } from './state_branch_core.mjs';

export function snapshotDirectory(stateDir) {
  const snapshot = new Map();
  const root = path.resolve(stateDir);
  if (!fs.existsSync(root)) return snapshot;
  const rootStat = fs.lstatSync(root);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) {
    throw new StateBranchError('.orchestrator must be a real directory, not a symlink.', 'STATE_ROOT_INVALID');
  }
  let totalBytes = 0;
  function walk(current) {
    const entries = fs.readdirSync(current, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      const stat = fs.lstatSync(fullPath);
      if (stat.isSymbolicLink()) throw new StateBranchError('Symlinks are forbidden in persisted orchestrator state.', 'STATE_SYMLINK_BLOCKED');
      const relative = `.orchestrator/${path.relative(root, fullPath).split(path.sep).join('/')}`;
      if (relative === RESEARCH_CACHE || relative.startsWith(`${RESEARCH_CACHE}/`)) continue;
      if (stat.isDirectory()) walk(fullPath);
      else if (stat.isFile()) {
        if (snapshot.size >= MAX_STATE_FILES) throw new StateBranchError('Orchestrator state exceeds file-count limit.', 'STATE_FILE_COUNT_LIMIT');
        const bytes = fs.readFileSync(fullPath);
        if (bytes.length > MAX_STATE_FILE_BYTES) throw new StateBranchError('Orchestrator state file exceeds size limit.', 'STATE_FILE_TOO_LARGE');
        totalBytes += bytes.length;
        if (totalBytes > MAX_STATE_TOTAL_BYTES) throw new StateBranchError('Orchestrator state exceeds total-size limit.', 'STATE_TOTAL_SIZE_LIMIT');
        snapshot.set(normalizeStatePath(relative), bytes);
      } else throw new StateBranchError('Unsupported filesystem object in orchestrator state.', 'STATE_FILE_TYPE_BLOCKED');
    }
  }
  walk(root);
  return snapshot;
}

const digest = (value) => value === undefined ? null : crypto.createHash('sha256').update(value).digest('hex');

export function mergeSnapshots(base, local, remote) {
  const merged = new Map(remote);
  const conflicts = [];
  const paths = new Set([...base.keys(), ...local.keys(), ...remote.keys()]);
  for (const relativePath of paths) {
    const baseHash = digest(base.get(relativePath));
    const localHash = digest(local.get(relativePath));
    const remoteHash = digest(remote.get(relativePath));
    const localChanged = localHash !== baseHash;
    const remoteChanged = remoteHash !== baseHash;
    if (localChanged && remoteChanged && localHash !== remoteHash) {
      conflicts.push(relativePath);
      continue;
    }
    if (localChanged) {
      if (local.has(relativePath)) merged.set(relativePath, local.get(relativePath));
      else merged.delete(relativePath);
    }
  }
  if (conflicts.length) {
    throw new StateBranchError(`Concurrent state update conflict on ${conflicts.length} object(s); refusing overwrite.`, 'STATE_WRITE_CONFLICT');
  }
  return merged;
}
