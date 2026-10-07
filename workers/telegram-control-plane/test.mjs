import test from 'node:test';
import assert from 'node:assert/strict';

import {callbackAction, canonicalJson, normalizeGoal, normalizeWorkflowId, parseCommand, splitFirstArg, telegramEventId} from './src/protocol.js';
import {constantTimeEqual, isAllowedUser, isApprover, isPrivateMessage, isSafeWorkflowId} from './src/policy.js';
import {claimUpdate, completeUpdate} from './src/db.js';

test('canonical JSON sorts object keys recursively', () => {
  assert.equal(canonicalJson({b: 2, a: {d: 4, c: 3}}), '{"a":{"c":3,"d":4},"b":2}');
});

test('command parser handles whitespace and command suffixes', () => {
  assert.deepEqual(parseCommand('/run@VORENYX_ORCHESTRATOR_CORE_BOT  inspect repo'), {kind:'command', command:'run', args:'inspect repo'});
  assert.deepEqual(parseCommand('plain text'), {kind:'text', text:'plain text', args:''});
});

test('goal and workflow bounds are enforced', () => {
  assert.equal(normalizeGoal('  hello  '), 'hello');
  assert.throws(() => normalizeGoal(''), /goal_required/);
  assert.throws(() => normalizeGoal('x'.repeat(3501)), /goal_too_long/);
  assert.equal(normalizeWorkflowId('wf:1'), 'wf:1');
  assert.throws(() => normalizeWorkflowId('bad id'), /workflow_id_invalid/);
  assert.equal(isSafeWorkflowId('wf:1'), true);
  assert.equal(isSafeWorkflowId('bad/id'), false);
});

test('event identity is deterministic and bounded', () => {
  assert.equal(telegramEventId(77), 'tg:77');
  assert.throws(() => telegramEventId(-1), /update_id_invalid/);
  assert.equal(splitFirstArg('wf:1 do something')[0], 'wf:1');
});

test('callback policy only accepts consent actions', () => {
  assert.equal(callbackAction('consent:accept'), 'accept');
  assert.equal(callbackAction('consent:privacy'), 'privacy');
  assert.equal(callbackAction('consent:revoke'), 'revoke');
  assert.throws(() => callbackAction('run:live'), /callback_invalid/);
});

test('authorization is explicit and private-chat scoped', () => {
  const env = {TELEGRAM_ALLOWED_USER_IDS:'10,20', TELEGRAM_APPROVER_USER_IDS:'20'};
  assert.equal(isAllowedUser(env, 10), true);
  assert.equal(isAllowedUser(env, 30), false);
  assert.equal(isApprover(env, 20), true);
  assert.equal(isApprover(env, 10), false);
  assert.equal(isPrivateMessage({chat:{type:'private'}, from:{id:10}}), true);
  assert.equal(isPrivateMessage({chat:{type:'group'}, from:{id:10}}), false);
});

test('constant-time equality helper distinguishes mismatches', () => {
  assert.equal(constantTimeEqual('abc', 'abc'), true);
  assert.equal(constantTimeEqual('abc', 'abd'), false);
  assert.equal(constantTimeEqual('abc', 'abcd'), false);
});

test('telegram worker source contains no bot-token-shaped secret', async () => {
  const fs = await import('node:fs/promises');
  const {globSync} = await import('node:fs');
  const paths = globSync('workers/telegram-control-plane/**/*.{js,json,jsonc,mjs}');
  const tokenPattern = /\b\d{6,12}:[A-Za-z0-9_-]{20,}\b/;
  for (const path of paths) {
    const content = await fs.readFile(path, 'utf8');
    assert.equal(tokenPattern.test(content), false, path);
  }
});

class FakeD1 {
  constructor() {
    this.row = null;
    this.executed = [];
  }

  batch(statements) {
    return Promise.all(statements.map((statement) => statement.run()));
  }

  prepare(sql) {
    this.executed.push(sql);
    return {
      bind: (...args) => {
        const placeholderCount = (sql.match(/\?/g) || []).length;
        assert.equal(
          args.length,
          placeholderCount,
          'D1 placeholder/bind arity mismatch: ' + sql,
        );
        return {
          run: async () => this.run(sql, args),
          first: async () => this.first(sql, args),
        };
      },
    };
  }

  async run(sql, args) {
    if (sql.startsWith('DELETE FROM')) return {meta: {changes: 0}};
    if (sql.startsWith('INSERT OR IGNORE INTO telegram_inbox')) {
      if (this.row) return {meta: {changes: 0}};
      this.row = {
        event_id: args[0],
        principal_key: args[1],
        status: 'processing',
        claim_token: args[2],
        workflow_id: null,
        created_at: args[3],
        updated_at: args[4],
        expires_at: args[5],
      };
      return {meta: {changes: 1}};
    }
    if (sql.startsWith("UPDATE telegram_inbox SET status='processing'")) {
      if (!this.row || this.row.event_id !== args[3]) return {meta: {changes: 0}};
      if (!(this.row.status !== 'processing' || this.row.updated_at <= args[4])) {
        return {meta: {changes: 0}};
      }
      this.row.status = 'processing';
      this.row.claim_token = args[0];
      this.row.workflow_id = null;
      this.row.updated_at = args[1];
      this.row.expires_at = args[2];
      return {meta: {changes: 1}};
    }
    if (sql.startsWith("UPDATE telegram_inbox SET status='completed'")) {
      if (!this.row || this.row.event_id !== args[2] || this.row.claim_token !== args[3] || this.row.status !== 'processing') {
        return {meta: {changes: 0}};
      }
      this.row.status = 'completed';
      this.row.workflow_id = args[0];
      this.row.updated_at = args[1];
      return {meta: {changes: 1}};
    }
    throw new Error('Unhandled fake D1 SQL: ' + sql);
  }

  async first(sql, args) {
    if (sql.startsWith('SELECT status,claim_token,workflow_id,updated_at FROM telegram_inbox')) {
      if (!this.row || this.row.event_id !== args[0]) return null;
      return {
        status: this.row.status,
        claim_token: this.row.claim_token,
        workflow_id: this.row.workflow_id,
        updated_at: this.row.updated_at,
      };
    }
    throw new Error('Unhandled fake D1 SELECT: ' + sql);
  }
}

test('Telegram inbox claim is single-flight, completable, and reclaimable after staleness', async () => {
  const db = new FakeD1();
  const env = {
    DB: db,
    TELEGRAM_DATA_HMAC_KEY: 'test-hmac-key',
    TELEGRAM_DATA_ENCRYPTION_KEY: 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
  };

  const first = await claimUpdate(env, 'tg:100', 'principal');
  assert.equal(first.claimed, true);
  assert.ok(first.claimToken);

  await completeUpdate(env, 'tg:100', 'wf:100', first.claimToken);

  const duplicate = await claimUpdate(env, 'tg:100', 'principal');
  assert.equal(duplicate.claimed, false);
  assert.equal(duplicate.duplicate, true);
  assert.equal(duplicate.workflowId, 'wf:100');

  db.row.status = 'processing';
  db.row.updated_at = Math.floor(Date.now() / 1000) - 301;
  const reclaimed = await claimUpdate(env, 'tg:100', 'principal', 300);
  assert.equal(reclaimed.claimed, true);
  assert.ok(reclaimed.claimToken);
});
