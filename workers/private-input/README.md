# Free private-input store

This Worker is the optional zero-dollar private-input backend for the orchestration engine.

It uses Cloudflare Workers Free with SQLite-backed Durable Objects. Private inputs are keyed by an opaque HMAC-derived reference and expire automatically through Durable Object alarms.

The free tier has finite quotas; it is not unlimited. The current design intentionally requires no paid database, queue, broker, or orchestration SaaS.

## Deployment

Run `Deploy private input worker` manually from GitHub Actions.

Required repository secrets:
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`
- `ORCHESTRATOR_PRIVATE_INPUT_SECRET`

After successful deployment, set the Worker `workers.dev` URL as `ORCHESTRATOR_PRIVATE_INPUT_URL` and use the same secret in the orchestrator and gateway environments.

The control plane remains usable without this Worker in dry-run mode. Live structured connector execution is fail-closed until the private-input channel is configured and verified.

## Security boundary

The gateway sends only an opaque reference plus execution identity, intent fingerprint, digest, and idempotency metadata through GitHub Actions. Raw structured input is fetched just before connector execution and removed from the node input immediately afterward.

The Worker signs requests with HMAC-SHA256 over HTTP method, path, protocol, timestamp, and exact body. Input references are deterministically derived from protocol, execution identity, and input digest.

Deployment is not considered verified until the workflow reports a successful `/health` response.