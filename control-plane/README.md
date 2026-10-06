# Distributed control plane

This is an optional, free-first coordination plane for live orchestration.

Runtime activation requires both:
ORCHESTRATOR_CONTROL_PLANE_URL
ORCHESTRATOR_CONTROL_PLANE_SECRET

The runner uses one Durable Object instance per workflow ID. The object stores
a workflow lease with a monotonically increasing fence epoch and a durable
effect ledger. An effect is only executable after its stable effect ID is
claimed. Existing inflight or completed claims are never auto-replayed.

This is intentionally effectively-once/fail-closed, not a mathematical
exactly-once guarantee for arbitrary external providers. A provider can still
return an ambiguous outcome; that outcome is reconciled before the workflow
can continue.

Deploy from control-plane/cloudflare with Wrangler, set CONTROL_PLANE_SECRET
as a Worker secret, then put the deployed HTTPS URL and the same secret in the
GitHub Actions environment.

Never commit the secret or store it in .orchestrator state.

Git remains source/audit persistence. Durable Objects handle coordination.
Do not add Cloudflare Queues solely for this control path; the free operation
budget is too small for a high-volume event bus.
