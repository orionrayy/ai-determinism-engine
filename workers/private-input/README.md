# Free private-input store

This Worker is the zero-dollar private-input backend for the orchestration engine.

It uses Cloudflare Workers Free + Workers KV. The Worker stores the private input envelope under its opaque `input_ref`; KV expiration enforces the requested TTL. Current free limits are finite: Workers allows 100,000 requests/day, while KV allows 100,000 reads/day and 1,000 writes/deletes per day with 1 GB stored data. That is suitable for a small control plane but not unlimited.

## Deploy

Run from this directory:

`npx wrangler@4.146.0 deploy`

Wrangler can automatically provision a KV resource when the binding has no ID. Keep the shared secret as a Worker secret; never put it in `wrangler.jsonc` or source.

For GitHub Actions deployment, configure `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` as repository secrets, then run the dedicated deployment workflow.

After deployment, set the Worker URL as `ORCHESTRATOR_PRIVATE_INPUT_URL` in both the gateway and the orchestration workflow, and set the same `ORCHESTRATOR_PRIVATE_INPUT_SECRET` on both sides.

Endpoints:
- GET /health
- authenticated POST /v1/inputs
- authenticated GET /v1/inputs/{input_ref}
- authenticated DELETE /v1/inputs/{input_ref}

The protocol signs HTTP method, request path, protocol version, timestamp, and exact body. Replayed POSTs are idempotent by `input_ref`; conflicting envelopes are rejected.
