import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { StateBranchError, decryptStateObject, encryptStateObject, mergeSnapshots, normalizeStatePath, persistState, remoteBranchCommit, restoreState, snapshotDirectory } from './state_branch.mjs';

function makeKeys() {
  const master = crypto.randomBytes(32);
  return {
    encryptionKey: Buffer.from(crypto.hkdfSync('sha256', master, Buffer.from('ade-test'), Buffer.from('aes-256-gcm'), 32)),
    pathKey: Buffer.from(crypto.hkdfSync('sha256', master, Buffer.from('ade-test'), Buffer.from('hmac-path-token'), 32)),
  };
}
function git(root, ...args) {
  return execFileSync('git', args, { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
}
function setupRemote(root) {
  fs.mkdirSync(root, { recursive: true });
  git(root, 'init', '-b', 'main');
  git(root, 'config', 'user.name', 'Test Operator');
  git(root, 'config', 'user.email', 'operator@example.invalid');
  fs.mkdirSync(path.join(root, '.orchestrator'), { recursive: true });
  fs.writeFileSync(path.join(root, '.orchestrator', 'state.json'), JSON.stringify({ version: 1, workflows: {} }) + '\n');
  git(root, 'add', '.orchestrator/state.json');
  git(root, 'commit', '-m', 'initial source tree');
  const bare = root + '-origin.git';
  execFileSync('git', ['init', '--bare', bare], { encoding: 'utf8' });
  git(root, 'remote', 'add', 'origin', bare);
  git(root, 'push', '-u', 'origin', 'main');
  return bare;
}
function runtimeEnv(root) {
  return {
    ...process.env,
    GITHUB_REPOSITORY: 'test-owner/test-repo',
    ORCHESTRATOR_STATE_BRANCH: 'orchestrator-state',
    ORCHESTRATOR_STATE_ENCRYPTION_KEY: crypto.randomBytes(32).toString('base64'),
    GITHUB_RUN_ID: 'test-' + path.basename(root),
  };
}

test('AES-256-GCM authenticates repository, branch and opaque object token', () => {
  const context = { repository: 'test-owner/test-repo', branch: 'orchestrator-state' };
  const keys = makeKeys();
  const encrypted = encryptStateObject('.orchestrator/workflows/a.json', Buffer.from('private-state'), keys, context);
  const decoded = decryptStateObject(encrypted.bytes, encrypted.token, keys, context);
  assert.equal(decoded.path, '.orchestrator/workflows/a.json');
  assert.equal(decoded.data.toString(), 'private-state');
  assert.throws(() => decryptStateObject(encrypted.bytes, encrypted.token, keys, { ...context, branch: 'wrong' }), error => error.code === 'STATE_AUTH_FAILED');
});

test('path traversal and ephemeral research cache are rejected', () => {
  assert.throws(() => normalizeStatePath('.orchestrator/../secret'), StateBranchError);
  assert.throws(() => normalizeStatePath('.orchestrator/research-cache/cache.json'), error => error.code === 'STATE_PATH_EXCLUDED');
});

test('three-way merge combines disjoint writes and rejects same-path conflicts', () => {
  const base = new Map([['.orchestrator/state.json', Buffer.from('base')]]);
  const local = new Map(base);
  local.set('.orchestrator/state.json', Buffer.from('local'));
  local.set('.orchestrator/workflows/a.json', Buffer.from('a'));
  const remote = new Map(base);
  remote.set('.orchestrator/workflows/b.json', Buffer.from('b'));
  const merged = mergeSnapshots(base, local, remote);
  assert.equal(merged.get('.orchestrator/workflows/b.json').toString(), 'b');
  assert.equal(merged.get('.orchestrator/state.json').toString(), 'local');
  const conflict = new Map(remote);
  conflict.set('.orchestrator/state.json', Buffer.from('remote'));
  assert.throws(() => mergeSnapshots(base, local, conflict), error => error.code === 'STATE_WRITE_CONFLICT');
});

test('git integration persists ciphertext only and restores state', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ade-state-git-'));
  const env = runtimeEnv(root);
  let bare;
  const inspect = root + '-inspect';
  try {
    bare = setupRemote(root);
    assert.equal(restoreState({ root, env }).bootstrap, true);
    fs.mkdirSync(path.join(root, '.orchestrator', 'workflows'), { recursive: true });
    fs.writeFileSync(path.join(root, '.orchestrator', 'workflows', 'wf.json'), JSON.stringify({ goal: 'PRIVATE_MARKER' }) + '\n');
    assert.equal(persistState({ root, env }).persisted, true);
    execFileSync('git', ['clone', '--branch', 'orchestrator-state', bare, inspect], { encoding: 'utf8' });
    const names = git(inspect, 'ls-tree', '-r', '--name-only', 'HEAD').split('\n').filter(Boolean);
    assert.equal(names.length, 2);
    assert.ok(names.every(name => /^\.orchestrator-state\/v1\/[a-f0-9]{64}\.enc$/.test(name)));
    assert.equal(names.map(name => fs.readFileSync(path.join(inspect, name), 'utf8')).join('').includes('PRIVATE_MARKER'), false);
    fs.rmSync(path.join(root, '.orchestrator'), { recursive: true, force: true });
    assert.equal(restoreState({ root, env }).restored, true);
    assert.match(fs.readFileSync(path.join(root, '.orchestrator', 'workflows', 'wf.json'), 'utf8'), /PRIVATE_MARKER/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
    if (bare) fs.rmSync(bare, { recursive: true, force: true });
    fs.rmSync(inspect, { recursive: true, force: true });
  }
});

test('remote lookup errors cannot masquerade as a missing state branch', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ade-state-remote-error-'));
  try {
    git(root, 'init');
    git(root, 'remote', 'add', 'origin', path.join(root, 'missing-origin.git'));
    assert.throws(
      () => remoteBranchCommit(root, 'orchestrator-state'),
      error => error.code === 'GIT_COMMAND_FAILED',
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('missing state key fails closed', () => {
  assert.throws(() => restoreState({ root: process.cwd(), env: { GITHUB_REPOSITORY: 'owner/repo' } }), error => error.code === 'STATE_KEY_MISSING_OR_INVALID');
});
