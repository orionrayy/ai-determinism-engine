# Native Worker Bridge v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a durable native-worker handoff path that pauses exact DAG nodes and resumes them through a validated one-time callback.

**Architecture:** Extend the orchestrator state machine with `waiting_native_worker`, add a protocol/credential helper, expose a callback route on the existing gateway, and route `orchestrator.native_result` events to the exact workflow/node. Keep the worker vendor-neutral and public-safe.

**Tech Stack:** Python 3.12 stdlib, GitHub Actions, GitHub repository_dispatch, existing gateway.

**Spec:** docs/superpowers/specs/native-worker-bridge-v1.md

## Global Constraints

- Protocol: `ai-orchestrator.native-worker/v1`.
- Execution ID: SHA-256(`workflow_id + ":" + node_id`).
- Native task payloads require `public_safe=true`.
- Callback tokens are one-time, expiry-bound, SHA-256 hashed in persisted state.
- Invalid, expired, mismatched, or replayed callbacks must not mutate workflow state.
- High-risk/irreversible connector actions remain approval-gated.
- Do not add external dependencies.

## Review Focus

- Callback with valid token but wrong node/execution_id must not complete another node.
- Callback replay after completion must not mutate output.
- Expired token must be rejected without changing status.
- Native task generation without `public_safe=true` must fail closed.
- Worker error callback must produce a retryable failed node rather than a completed node.

### Task 1: Native protocol helper

**Files:**
- Create: `orchestrator/native_worker.py`
- Test: `orchestrator/test_native_worker.py`

**Interfaces:**
- `execution_id(workflow_id: str, node_id: str) -> str`
- `create_task(workflow_id, node_id, connector, action, goal, node_input, callback_url, expires_at) -> tuple[dict, str]`
- `hash_token(token: str) -> str`
- `verify_token(token: str, token_hash: str) -> bool`

- [ ] Step 1: Write failing tests for deterministic IDs, public-safe enforcement, and token hashing.
- [ ] Step 2: Run `python -m unittest orchestrator.test_native_worker -v`; expected FAIL because helpers do not exist.
- [ ] Step 3: Implement the minimal helper.
- [ ] Step 4: Run focused tests; expected PASS.
- [ ] Step 5: Commit.

### Task 2: State-machine handoff and callback application

**Files:**
- Modify: `orchestrator/orchestrator.py`
- Test: `orchestrator/test_orchestrator.py`

**Interfaces:**
- Add `waiting_native_worker` transitions.
- `queue_native_worker_task(workflow, node, registry) -> dict`
- `apply_native_result(workflow, payload) -> str`

- [ ] Step 1: Write failing tests for pause/resume, exact binding, expiry rejection, replay rejection, and worker error.
- [ ] Step 2: Run focused tests; expected FAIL on missing state/handler behavior.
- [ ] Step 3: Implement queue/resume using existing GitHub issue adapter and persisted state.
- [ ] Step 4: Run focused tests; expected PASS.
- [ ] Step 5: Commit.

### Task 3: Gateway callback transport

**Files:**
- Modify: `gateway.py`
- Test: `test_gateway.py`

**Interfaces:**
- Add `POST /native-result`.
- Dispatch `orchestrator.native_result` with exact payload.
- Enforce JSON, size and protocol validation.
- Preserve existing `/event` authentication.

- [ ] Step 1: Write failing callback route tests.
- [ ] Step 2: Run gateway tests; expected FAIL because route does not exist.
- [ ] Step 3: Implement route.
- [ ] Step 4: Run focused tests; expected PASS.
- [ ] Step 5: Commit.

### Task 4: GitHub Actions routing and worker contract

**Files:**
- Modify: `.github/workflows/orchestrator.yml`
- Create: `NATIVE_WORKER_PROTOCOL.md`
- Create: `native-worker-prompt.md`
- Modify: `ORCHESTRATION_MEMORY.md`
- Modify: `.github/workflows/orchestrator-tests.yml`
- Test: `test_actions_config.py`

**Interfaces:**
- Trigger `repository_dispatch` type `orchestrator.native_result`.
- Pass exact workflow/node identifiers into the worker invocation.
- Document native-worker task claiming and callback format.
- Document public-safe and approval constraints.

- [ ] Step 1: Add failing workflow/config tests for the new event and exact workflow binding.
- [ ] Step 2: Run workflow/config tests; expected FAIL until routing exists.
- [ ] Step 3: Implement workflow routing and documentation.
- [ ] Step 4: Run full test suite; expected PASS.
- [ ] Step 5: Commit.

## Final Verification

Run the full Python unit-test suite and compile all changed Python files.
