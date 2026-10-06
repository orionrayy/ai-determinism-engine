import test from 'node:test';
import assert from 'node:assert/strict';

import {callbackAction, canonicalJson, normalizeGoal, normalizeWorkflowId, parseCommand, splitFirstArg, telegramEventId} from './src/protocol.js';
import {constantTimeEqual, isAllowedUser, isApprover, isPrivateMessage, isSafeWorkflowId} from './src/policy.js';

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