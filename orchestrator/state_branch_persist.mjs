import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { StateBranchError, resolveContext, resolveKeys, runGit } from './state_branch_core.mjs';
import { encryptStateObject } from './state_branch_crypto.mjs';
import { snapshotDirectory, mergeSnapshots } from './state_branch_snapshot.mjs';
import { baseCheckpointPath, readBaseCheckpoint, remoteBranchCommit, readRemoteSnapshot, writeBaseCheckpoint } from './state_branch_restore.mjs';

function commitSnapshot(root, parentCommit, remoteRecords, merged, keys, ctx) {
  const indexPath = path.join(os.tmpdir(), `ade-state-index-${process.pid}-${crypto.randomBytes(6).toString('hex')}`);
  const env = { GIT_INDEX_FILE: indexPath };
  try {
    if (parentCommit) runGit(root, ['read-tree', parentCommit], { env });
    else runGit(root, ['read-tree', '--empty'], { env });
    for (const [relativePath, object] of remoteRecords) {
      const next = merged.get(relativePath);
      const currentBytes = remoteRecords.get(relativePath)?.bytes;
      if (!merged.has(relativePath) || !currentBytes || !Buffer.from(currentBytes).equals(Buffer.from(next))) {
        runGit(root, ['update-index', '--force-remove', '--', object.objectPath], { env, allowFailure: true });
      }
    }
    for (const [relativePath, bytes] of merged) {
      const existing = remoteRecords.get(relativePath);
      if (existing?.bytes && Buffer.from(existing.bytes).equals(Buffer.from(bytes))) continue;
      const encrypted = encryptStateObject(relativePath, bytes, keys, ctx);
      const blob = runGit(root, ['hash-object', '-w', '--stdin'], { input: encrypted.bytes }).trim();
      runGit(root, ['update-index', '--add', '--cacheinfo', `100644,${blob},${encrypted.objectPath}`], { env });
    }
    const tree = runGit(root, ['write-tree'], { env }).trim();
    if (parentCommit) {
      const parentTree = runGit(root, ['rev-parse', `${parentCommit}^{tree}`]).trim();
      if (tree === parentTree) return null;
    }
    const args = ['commit-tree', tree];
    if (parentCommit) args.push('-p', parentCommit);
    args.push('-m', 'chore(orchestrator): persist encrypted workflow state');
    const commit = runGit(root, args, {
      env: {
        GIT_AUTHOR_NAME: 'github-actions[bot]',
        GIT_AUTHOR_EMAIL: '41898282+github-actions[bot]@users.noreply.github.com',
        GIT_COMMITTER_NAME: 'github-actions[bot]',
        GIT_COMMITTER_EMAIL: '41898282+github-actions[bot]@users.noreply.github.com',
      },
    }).trim();
    return commit;
  } finally {
    try { fs.rmSync(indexPath, { force: true }); } catch { /* best effort */ }
  }
}

function snapshotWithRemoteMetadata(root, commit, keys, ctx) {
  return readRemoteSnapshot(root, commit, keys, ctx);
}

export function persistState({ root = process.cwd(), env = process.env, retries = 5 } = {}) {
  const ctx = resolveContext(env);
  const keys = resolveKeys(env, ctx.repository);
  const baseFile = baseCheckpointPath(root, env);
  const base = readBaseCheckpoint(baseFile);
  const local = snapshotDirectory(path.join(root, '.orchestrator'));
  let lastError;
  for (let attempt = 1; attempt <= Math.max(1, retries); attempt += 1) {
    const latestCommit = remoteBranchCommit(root, ctx.branch);
    if (base.commit && !latestCommit) throw new StateBranchError('Encrypted state branch disappeared; refusing legacy fallback.', 'STATE_BRANCH_DISAPPEARED');
    const remote = snapshotWithRemoteMetadata(root, latestCommit, keys, ctx);
    // The exact restore-time base snapshot is kept in runner-local temporary
    // storage. Reusing it avoids depending on shallow-clone history depth.
    const merged = mergeSnapshots(base.snapshot, local, remote.snapshot);
    const remoteRecords = new Map();
    for (const [relativePath, object] of remote.objects) {
      remoteRecords.set(relativePath, {
        ...object,
        bytes: remote.snapshot.get(relativePath),
      });
    }
    const nextCommit = commitSnapshot(root, latestCommit, remoteRecords, merged, keys, ctx);
    if (!nextCommit) {
      writeBaseCheckpoint(baseFile, latestCommit, remote.snapshot);
      console.log(`Encrypted orchestrator state unchanged; ${remote.snapshot.size} object(s) already current.`);
      return { persisted: false, files: remote.snapshot.size, commit: latestCommit };
    }
    try {
      runGit(root, ['push', 'origin', `${nextCommit}:refs/heads/${ctx.branch}`]);
      const verified = remoteBranchCommit(root, ctx.branch);
      if (verified !== nextCommit) throw new StateBranchError('State branch push could not be verified.', 'STATE_PUSH_UNVERIFIED');
      writeBaseCheckpoint(baseFile, verified, merged);
      console.log(`Persisted ${merged.size} encrypted orchestrator state object(s) to ${verified.slice(0, 12)}.`);
      return { persisted: true, files: merged.size, commit: verified };
    } catch (error) {
      lastError = error;
      if (error?.code === 'STATE_WRITE_CONFLICT' || attempt === retries) break;
      // Refetch and repeat the three-way merge. No force push is permitted.
    }
  }
  throw new StateBranchError(
    `Encrypted state persistence failed after ${retries} bounded attempt(s): ${String(lastError?.message ?? lastError).slice(0, 400)}`,
    lastError?.code === 'STATE_WRITE_CONFLICT' ? 'STATE_WRITE_CONFLICT' : 'STATE_PUSH_FAILED',
  );
}
