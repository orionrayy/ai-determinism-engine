#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { StateBranchError } from './state_branch_core.mjs';
import { restoreState } from './state_branch_restore.mjs';
import { persistState } from './state_branch_persist.mjs';

export { DEFAULT_STATE_BRANCH, OBJECT_ROOT, MAX_STATE_FILES, MAX_STATE_FILE_BYTES, MAX_STATE_TOTAL_BYTES, VERSION, StateBranchError, runGit, resolveContext, resolveKeys, normalizeStatePath, pathToken, aad } from './state_branch_core.mjs';
export { encryptStateObject, decryptStateObject } from './state_branch_crypto.mjs';
export { snapshotDirectory, mergeSnapshots } from './state_branch_snapshot.mjs';
export { restoreState, baseCheckpointPath, writeBaseCheckpoint, readBaseCheckpoint, remoteBranchCommit, readRemoteSnapshot } from './state_branch_restore.mjs';
export { persistState } from './state_branch_persist.mjs';

function main() {
  const mode = String(process.argv[2] ?? '').trim();
  if (mode === 'restore') restoreState();
  else if (mode === 'persist') persistState();
  else throw new StateBranchError('Usage: node orchestrator/state_branch.mjs <restore|persist>.', 'CLI_USAGE');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try { main(); }
  catch (error) {
    console.error(`orchestrator state ${error?.code ?? 'ERROR'}: ${String(error?.message ?? error).replace(/\s+/g, ' ').slice(0, 700)}`);
    process.exitCode = 1;
  }
}
