# Browser Automation Foundation

This directory owns the browser automation boundary. The first runtime is the official Microsoft Playwright MCP server, not a custom browser implementation.

## Runtime

- Pinned upstream package: @playwright/mcp 0.0.86 (Apache-2.0).
- Node.js 20+ and an MCP client are required.
- Chromium is downloaded automatically on first browser use.
- Stdio is the default transport; no public HTTP listener is started.
- A persistent per-workspace browser profile and an isolated ephemeral profile are provided.
- Browser state, downloads, screenshots, traces, and storage-state files are kept under ignored local paths.

## Start

Windows PowerShell:
    .\browser\playwright\start.ps1

Linux / macOS:
    bash browser/playwright/start.sh

Alternatively, run the package script from its directory:
    cd browser/playwright
    npm run browser:mcp

There is no project-level npm install requirement. The launcher uses npx --yes with an exact package version. The first invocation fetches the pinned MCP package; the browser binary is downloaded on first browser use.

For an MCP client, copy browser/playwright/mcp-client.example.json and replace its absolute config path with the actual repository path. The client must support local stdio MCP servers.

## Session modes

- mcp.config.json: persistent per-workspace profile stored under browser/playwright/.state/profile. Sign in manually in the browser. Do not use this profile for untrusted tasks or share it across unrelated users.
- mcp.config.ephemeral.json: in-memory isolated context for public, credential-free automation and CI smoke checks.
- mcp.config.full.example.json: opt-in extended capabilities (network, storage, devtools, PDF, vision). Review permissions before enabling them.

Never commit .state/, browser profiles, cookies, access tokens, storage-state JSON, HAR files, downloaded private documents, screenshots containing private data, or trace archives.

## Python policy

orchestrator/browser_policy.py is a pure-stdlib policy preflight. It does not launch a browser and does not enforce approvals by itself. A future orchestration adapter must call it before dispatching MCP actions, obtain approval from the trusted supervisor for high-risk effects, and validate returned evidence before completing a workflow node.

Run its regression suite from the repository root:
    python -m unittest orchestrator.test_browser_policy -v

Run local configuration checks:
    node browser/playwright/doctor.mjs

## Scope boundary

This foundation does not yet wire a local MCP process into the hosted ChatGPT connector runtime or GitHub Actions worker. The MCP server runs on the machine where it is configured. The Python orchestration bridge, durable session broker, remote worker pool, and cross-runtime evidence reconciliation are later phases.
