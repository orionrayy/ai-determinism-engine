# Native Worker Protocol v1

Protocol: `ai-orchestrator.native-worker/v1`

A native task is created when the orchestrator cannot execute a connector action directly and needs a ChatGPT-native worker.

## Task source

Tasks are created as GitHub Issues with title:

`[ORCHESTRATOR NATIVE] <workflow_id> / <node_id>`

The issue body contains `TASK_JSON`, a callback URL, and a single-use callback token.

Only tasks with `input.public_safe=true` are eligible in v1 because the repository-backed transport is public.

## Worker procedure

1. Read the exact task issue.
2. Validate protocol, workflow_id, node_id, execution_id, connector, action, expiry, and public_safe.
3. Use the explicitly named connected ChatGPT connector. Do not substitute another connector without an orchestrator replan.
4. Perform only the declared action and preserve the task intent.
5. Return exactly one of `result` or `error`.
6. POST the callback body to the supplied callback URL using an HTTP-capable tool available to the worker session.

## Callback body

```json
{
  "protocol": "ai-orchestrator.native-worker/v1",
  "workflow_id": "...",
  "node_id": "...",
  "execution_id": "...",
  "token": "...",
  "result": {}
}
```

Error form:

```json
{
  "protocol": "ai-orchestrator.native-worker/v1",
  "workflow_id": "...",
  "node_id": "...",
  "execution_id": "...",
  "token": "...",
  "error": {
    "type": "worker_error",
    "message": "..."
  }
}
```

The worker must never expose unrelated secrets, credentials, private user data, or task tokens outside the callback operation.

A successful callback is accepted only when the orchestrator verifies the exact workflow/node binding, unexpired token hash, and waiting state. Replays are rejected.
