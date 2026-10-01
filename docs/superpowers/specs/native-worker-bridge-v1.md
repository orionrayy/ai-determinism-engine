# Native Worker Bridge v1 — Architecture Specification

## Goal
Allow the GitHub-based orchestrator to pause a DAG node for execution by the ChatGPT-native worker, then resume the exact workflow when the worker returns a validated result.

## Scope
The bridge is a transport/protocol boundary. It does not impersonate or directly invoke ChatGPT connector sessions. The native worker invokes an actually connected ChatGPT connector.

## State model
Add `waiting_native_worker`.
Node lifecycle: `ready -> running -> waiting_native_worker -> ready/completed/failed/cancelled`.

## Envelope
Protocol: `ai-orchestrator.native-worker/v1`.
Required: protocol, workflow_id, node_id, execution_id, connector, action, goal, input, callback, expires_at.
`execution_id = SHA-256(workflow_id + ":" + node_id)`.

## Callback
Gateway route: `POST /native-result`.
Body: protocol, workflow_id, node_id, execution_id, token, result, optional error.
The gateway validates JSON/size/protocol/identifiers and dispatches `orchestrator.native_result`.
The orchestrator validates the one-time token hash, expiry, exact execution binding, and replay state before mutating workflow state.

## Security
Native tasks require `public_safe=true` because the current repository-backed transport is public. Never place secrets in task payloads/issues. Callback tokens are high-entropy, one-time and expiry-bound. Invalid/expired/replayed callbacks cause no workflow mutation. High-risk connector actions remain behind the existing approval gate.

## Transport
1. Orchestrator creates and persists a native task.
2. Worker claims it and uses its connected connector.
3. Worker posts the callback.
4. Gateway dispatches the exact workflow/node result event.
5. GitHub Actions resumes that workflow.
6. Orchestrator validates and applies the result.
7. Normal DAG continuation resumes.

## Failure handling
Invalid/expired/replayed callback: no mutation. Worker error: failed node, then normal retry/replan policy. Missing callback: node remains waiting and can be reconciled by scheduled resumes.

## Non-goals
Direct GitHub Actions access to ChatGPT connector sessions; vendor-specific OAuth; arbitrary connector execution without explicit metadata.
