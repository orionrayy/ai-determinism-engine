# Free private-input store

This Worker is the zero-dollar private-input backend for the orchestration engine.

It uses Cloudflare Workers Free + SQLite-backed Durable Objects. Private inputs and execution leases are isolated by opaque subject/name. SQLite storage is transactional and strongly consistent; private-input rows use expiry alarms, while execution-lease rows retain the attempt ledger for seven days after ownership release/expiry. This avoids the cross-region read-after-write delay of Workers KV.

Current Workers Free Durable Object allowances include 100,000 requests/day, 13,000 GB-s/day, 5 million SQLite rows read/day, 100,000 rows written/day, and 5 GB total SQLite storage. These are finite free quotas, not unlimited capacity.

## Deploy

Run from the repository root through the manual GitHub Actions workflow:

`Actions → Deploy private input worker → Run workflow`

The workflow requires these repository secrets:
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`
- `ORCHESTRATOR_PRIVATE_INPUT_SECRET`

The deployment uses Wrangler `4.146.0` and Node `24`. The Worker URL is health-checked after deployment.

After deployment, set the resulting Worker URL as `ORCHESTRATOR_PRIVATE_INPUT_URL` in:
- the gateway service environment;
- the AI Orchestrator GitHub Actions secret store.

Keep `ORCHESTRATOR_PRIVATE_INPUT_SECRET` identical on both sides.

Endpoints:
- GET /health
- authenticated POST /v1/inputs
- authenticated GET /v1/inputs/{input_ref}
- authenticated DELETE /v1/inputs/{input_ref}

The protocol signs HTTP method, request path, protocol version, timestamp, and exact body. POST is idempotent by `input_ref`; conflicting envelopes are rejected. The Durable Object rejects expired values and removes them through its alarm.

The repository intentionally does not claim that the Worker is deployed until a real account deployment and authenticated health check have succeeded.
