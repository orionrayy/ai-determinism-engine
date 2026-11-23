# Robin OSINT Integration Architecture

**Status:** selected implementation design  
**Upstream:** `apurvsinghgautam/robin`  
**Pinned upstream commit:** `b5d4f5f509fac39d65d3a28fd1d78b56efbeeaf4`  
**Engine:** `orionrayy/ai-determinism-engine`

## Goal

Expose Robin as an optional, policy-controlled OSINT capability inside the AI orchestration engine while preserving the engine as the sole workflow authority.

## Architecture decision

Robin is retained as a Git submodule at `vendor/robin`, pinned to the audited upstream commit. The engine builds Robin's own Dockerfile from that pinned source when the optional runtime is enabled. This retains upstream attribution and license terms without forking or copying Robin implementation code into the engine.

A small standard-library Python adapter speaks MCP JSON-RPC over Docker stdio. It does not add an MCP SDK dependency to the engine, expose an HTTP port, run Robin's Streamlit UI, or create Cloudflare resources. Robin's existing image entrypoint starts Tor in the container and launches the MCP server on stdio.

The engine exposes one stateful Robin action: `robin_investigate`. The other Robin tools are session-scoped; splitting search, filtering, scraping, and synthesis across separately spawned MCP servers would lose Robin's session state. The one-call action already runs those stages and persists the completed report.

## Runtime and cost policy

- Runtime is opt-in. The GitHub Actions runner only initializes the Robin submodule and builds the image when the repository variable `ORCHESTRATOR_ROBIN_ENABLED=true` is set.
- The image tag is derived from the exact upstream commit; floating `latest` tags are not used.
- Robin receives only the existing engine `GEMINI_API_KEY`, mapped to Robin's `GOOGLE_API_KEY`. No additional provider keys are forwarded.
- The selected model must pass the engine's existing time-bounded Gemini free-model allowlist when `ORCHESTRATOR_FREE_ONLY=true`.
- The integration adds no Python runtime dependency, paid API, Cloudflare Worker, queue, or database.
- Docker build and Tor network access consume runner time/network and are not represented as zero-cost infrastructure guarantees.

## Authorization and safety boundary

- The tool is available only when `ROBIN_ENABLE_TOKEN=true`; the workflow maps this from the repository variable above. Default is disabled.
- Live mode and the engine's existing high-risk approval gate are required.
- The caller must supply an explicit supported preset, set `user_chose=true`, and provide an authorization basis that is present verbatim in the original goal. The planner must not invent user consent or authorization.
- Supported presets are limited to `threat_intel` and `ransomware_malware`. The `personal_identity` and `corporate_espionage` presets are excluded from the engine adapter.
- Query, instruction, scrape depth, thread count, and response size are bounded. The adapter adds instructions to exclude credentials and personal identifiers from the report.
- Scraped pages and Robin's generated report remain untrusted input. Their existing delimiters are retained; downstream agents are told never to follow instructions embedded in the report.
- Robin's text-only search/scrape behavior is retained. This integration does not add file, media, malware, exploit, credential-access, purchase, or account-access capabilities.

## Effects, retries, and persistence

Robin's investigation makes external network requests through Tor and saves the result inside Robin's investigations directory. It is therefore high-risk and side-effecting. Its effect contract uses engine identity and fencing, with retry blocked and no automatic reconciliation. An uncertain result must not be run again automatically.

The investigation volume is scoped to the workflow/node identity. On a confirmed successful response, the engine's normal durable workflow output/checkpoint becomes the durable result and the temporary Docker volume is removed. On an uncertain failure, automatic retry and replan remain blocked. Runner-local Robin files are not promised to survive destruction of a GitHub-hosted runner.

## Cloudflare boundary

The engine remains on its existing GitHub Actions execution fabric. Robin is not suitable for a Cloudflare Worker because it needs a Linux container, Tor daemon, and long-running MCP stdio process. No Cloudflare resource is provisioned or changed.

A read-only Cloudflare inventory refresh was attempted during this integration but the connected Cloudflare API returned 401/403 authentication errors for account resource-list operations. Current Cloudflare inventory therefore remains unverified; no Cloudflare changes were made.

## Acceptance criteria

1. The upstream submodule points to the exact audited commit and preserves the MIT license in the upstream tree.
2. MCP stdio initialization, tool discovery, bounded request/response handling, and tool-call error handling work without an engine-side MCP dependency.
3. The engine routes an explicit `capability:osint` to Robin only when enabled.
4. The live action requires the existing high-risk approval path and the existing Gemini free-model policy.
5. Missing consent, unsupported presets, missing authorization basis, disabled runtime, missing Docker/image, and missing credentials fail closed.
6. The effect contract prevents blind retry/replan after an uncertain invocation.
7. The standard engine test workflow validates imports, unit tests, registry JSON, and workflow syntax.
8. No Cloudflare resource, secret, or production deployment is changed.
