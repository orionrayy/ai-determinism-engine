# Sugarnymous Runtime Reliability Case Study — 2026-10-17

Scope: read-only inspection of the connected Cloudflare account and published Floot project metadata. This records verified control-plane evidence and prior route behavior; it does not claim an older incident is still active on the latest deployed version. No webhook, secret, route, Worker deployment, production setting, or project code was changed.

## Findings

### Confirmed historical 503 on native Worker health route

Cloudflare Workers Observability returned one active issue group for \`sugarnymous-native-edge\`:

- Issue \`c1a6937b-c2d3-4e69-bc15-167c91520cd4\`: \`HttpServerError · HTTP 503\`, two occurrences.
- Both occurrences were \`GET /_cloudflare-stage-health\`; their traces included a \`hyperdrive_connect\` span.
- Both occurrences referenced script version \`f8b22501-8877-4772-9ecc-9d00a692a2f5\` (version 29), at approximately 2026-10-17 06:07 UTC.
- Version 33 was deployed around 2026-10-17 06:12 UTC. The issue group remains active, but its recorded occurrences refer to the older version. A fresh post-v33 occurrence is required before claiming the latest deployment still fails.
- The inspected native Worker health handler performs a direct Hyperdrive \`select 1\` probe and then a Kysely query against \`users\`. A 503 is consistent with a DB/Hyperdrive readiness failure; the observed span supports investigation of that path, but the exact internal health reason is not included in the issue occurrence. Root cause remains unconfirmed.

### Route contract discrepancy

- Cloudflare \`sugarnymous-staging-edge\` points at \`https://sugarnymous-prerelease.floot.app\`.
- Its special health route is \`/_cloudflare-stage-health\`. It queries origin \`/_api/health\` and requires JSON \`status="ok"\` and \`database="ok"\`.
- Other paths, including \`/health\`, are passed through unchanged to the Floot origin. The previously observed staging \`/health\` 404 is consistent with this route implementation: it is not a special route on the staging Worker.
- Current native Worker source has explicit \`/_cloudflare-stage-health\`, \`/_api/health\` and \`/health\` branches. Host + route must be treated together; these paths are not interchangeable across edge and origin hosts.
- The landing page and \`/privacy\` had previously worked, but that does not prove Telegram webhook, database, or origin health.

### Observability and live-verification limitation

- Cloudflare resource metadata reports Observability disabled for both \`sugarnymous-staging-edge\` and \`sugarnymous-native-edge\`, despite a version annotation that mentions enabling sanitized observability. Control-plane settings, not version-message text, are authoritative.
- Direct synthetic HTTP requests launched from the connected tool runner were blocked by its outbound request policy and did not reach the public hosts. Floot's metered project tools/log access were also paused by the free-plan daily build-action cap during this audit. These restrictions are tooling limitations, not proof that the public app is down.
- Recheck preview/browser logs when Floot tool access resets and an editor browser session is open. Do not enable more verbose telemetry or change Worker settings without reviewing data minimization and approval.

## Reliability practices transferred to Aurum

1. Test endpoint + host + method + status + JSON body contract; do not presume \`/health\`, \`/_api/health\` and staging-only health routes are aliases.
2. Separate landing-page availability, backend readiness, database readiness and webhook readiness into independent probes.
3. Attribute error occurrences to exact deployed version IDs. Keep a historical issue separate from a current incident until newer telemetry confirms it.
4. Verify actual observability configuration, not only a deployment annotation. Redact tokens/payloads and avoid sensitive logs.
5. For dependencies, emit safe stage-specific health reasons and per-hop timings. Do not hide every DB failure behind an indistinguishable 503 when a sanitized error code is possible.
6. While debugging, avoid webhook/production cutovers and retry loops; use read-only evidence first, then preview/staging contracts.
