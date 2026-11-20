# Browser Automation Architecture

Status: foundation implementation, not yet an orchestrator execution adapter  
Owner: orionrayy/ai-determinism-engine  
Runtime: Microsoft Playwright MCP  
Policy module: orchestrator/browser_policy.py

## 1. Design objectives

1. Keep browser control based on open standards and an upstream-supported runtime.
2. Avoid paid browser-agent dependencies, usage-metered hosted execution, and model-provider lock-in.
3. Make local execution the default. Remote browser execution must be a separately declared runtime with its own cost, trust, network, and approval contract.
4. Separate reasoning/planning from browser execution. Playwright MCP performs browser operations; the orchestration supervisor remains authoritative for intent, risk, retries, approvals, and state.
5. Preserve deterministic evidence and recoverability without persisting credentials or full page payloads into workflow state.
6. Allow Chromium-first development while keeping future Firefox, WebKit, CDP, and remote endpoints as replaceable execution options.

## 2. Runtime stack

| Layer | Responsibility | Current status |
|---|---|---|
| AI / workflow planner | Define goal, task sequence, dependencies | Existing orchestrator |
| Browser policy | Validate task shape, URL, effect class, approval digest | Implemented as pure Python preflight |
| MCP transport | Expose typed browser tools over stdio | Official Playwright MCP configuration |
| Browser engine | Chromium navigation and interaction | Upstream Playwright |
| Evidence boundary | Produce bounded evidence metadata and artifact hashes | Python contract implemented; adapter integration pending |
| Durable workflow state | Store identity, checkpoints, outcome, evidence references | Existing orchestrator; browser adapter integration pending |
| Human approval | Approve exact high-risk intent before dispatch | Existing orchestrator approval model; browser-specific binding pending |

The repository does not fork Microsoft Playwright MCP. Upstream provides navigation, accessibility snapshots, tabs, form interaction, screenshots, waiting, and optional network/storage/testing capabilities. Use structured accessibility snapshots for normal interaction; use screenshots when visual state is relevant.

## 3. Runtime and version policy

- Runtime package: @playwright/mcp 0.0.86, pinned rather than latest.
- Package license: Apache-2.0.
- Node.js: 20 or newer.
- Transport: stdio by default. No listener on 0.0.0.0 is enabled by default.
- Browser: Chromium baseline. Cross-browser coverage is a later compatibility milestone.
- Dependency installation: npx --yes downloads the pinned package into the npm cache; this does not require npm install or a project-local dependency tree.
- Version updates must be deliberate: update the exact pin, inspect upstream release notes, rerun config checks and browser smoke tests, then update this document.

## 4. Session and credential handling

Two modes are supplied:

- Persistent per-workspace profile for manually authenticated workflows. It remains local to the machine and must not be committed, uploaded as an artifact, or shared across unrelated users.
- Ephemeral isolated context for public or credential-free automation.

Do not export or log cookies, local storage, authentication headers, passwords, one-time codes, or browser profile contents. Do not put credentials into task goals, query strings, source control, screenshots, traces, or workflow state. Use manual sign-in in a trusted local browser. A future secret integration must use an approved OS/provider secret store and narrowly scoped credentials.

The default profile is not a security sandbox. It is local persistent state. High-risk tasks should use a dedicated profile with the minimum required account access.

## 5. Task contract and risk policy

Every browser task must declare:

- Stable workflow and node identifiers.
- A supported Playwright MCP action.
- An effect class: read_only, reversible, external_write, financial, destructive, account_admin, or unknown.
- An optional HTTPS target URL. HTTP is permitted only for loopback development targets.
- A bounded JSON input object.

High-risk classes (external_write, financial, destructive, account_admin, and unknown) require trusted human approval. Approval is bound to a semantic SHA-256 digest of workflow, node, action, target, effect, and input; approval for one task must not authorize a modified task.

The effect classification must come from the trusted planner/supervisor, not from page content or the browser. Page text is untrusted data and must never be interpreted as policy instructions. When the downstream effect of an action is ambiguous, classify it as unknown and fail closed pending approval.

The preflight module is not an enforcement boundary until a runtime adapter calls it before every matching MCP action. Do not describe this first tranche as end-to-end approval enforcement.

## 6. URL and network safety

The Python preflight rejects unsupported schemes, embedded URL credentials, malformed URLs, and non-loopback plaintext HTTP. It stores only the target origin in evidence, not the URL path, query, or fragment.

This is not complete SSRF protection. Browser redirects, DNS rebinding, subresources, downloads, service workers, WebSockets, and page-initiated requests need network-level controls. Playwright MCP allowed-origin settings are useful guardrails but are not a security boundary and do not constrain redirects. When the runtime is moved to a server/CI worker, enforce outbound egress policy at the container/host boundary and separately test redirect and private-network behavior.

Do not expose an HTTP MCP endpoint to the public internet. A future remote transport must use authenticated transport, strict host binding, private networking, tenant isolation, and bounded session lifetime.

## 7. Effects, retries, and idempotency

Navigation, snapshots, waiting, and screenshots are observational operations. Clicks, typing, form filling, uploads, and dialog handling are interactions whose downstream effects depend on the page.

For external writes, financial actions, destructive operations, account administration, or unknown effects:

1. Require approval bound to the semantic digest.
2. Execute only after trusted approval.
3. Capture bounded evidence and operation identifiers.
4. If the browser disconnects or a response is ambiguous after submission, mark the outcome uncertain.
5. Reconcile the target website state before any retry.
6. Never blindly replay a submit, purchase, delete, publish, transfer, or account-change action after an uncertain result.

Browser-level click success is not proof that a business transaction completed. Validate the resulting application state independently.

## 8. Evidence contract

Evidence records contain only workflow/node identity, action, target origin, timestamp, outcome, and optional SHA-256 hashes for output/screenshot artifacts. They deliberately exclude page text, full URLs, cookies, headers, storage values, and raw browser traces.

Artifacts live outside Git in browser/playwright/.state/. Any future artifact uploader must have explicit retention, redaction, access-control, and size policies. Evidence hashes prove byte identity, not source authenticity or business correctness.

## 9. Implementation roadmap

### Phase 1 — local runtime foundation (this change)

- Pin official Playwright MCP.
- Supply persistent, ephemeral, and opt-in extended configurations.
- Add Windows and POSIX startup scripts without a project-level install step.
- Add Python task policy, approval-digest binding, evidence contract, JSON schemas, and regression tests.
- Add CI syntax/config validation.

### Phase 2 — orchestrator execution adapter

- Add a typed MCP client adapter with bounded request/response sizes and explicit tool allowlists.
- Invoke policy preflight before each browser action.
- Bind trusted approval receipts to task digests.
- Add effect classification and uncertainty semantics to workflow nodes.
- Return browser artifacts by content-addressed reference, not raw page payload.
- Do not add browser capabilities to orchestrator/tools.json until actual MCP transport and adapter are executable in the orchestrator environment.

### Phase 3 — durable sessions and recovery

- Add session ownership/lease, workspace isolation, TTL, and lifecycle events.
- Store session metadata, not authentication material, in workflow state.
- Reconcile pending side effects before retries and after worker restarts.
- Add trace, screenshot, console, network, and download evidence with strict redaction and resource caps.

### Phase 4 — optional remote runtimes

- Add a local/remote runtime interface.
- Evaluate self-hosted container workers first.
- Consider Cloudflare Browser Run only for workloads that fit its quota and execution constraints; do not assume it is unlimited or required.
- Require explicit free-tier and data-residency certification before activating a hosted provider.
- Preserve Playwright API compatibility so runtime replacement does not change workflow semantics.

## 10. Acceptance criteria

The foundation is complete only when:

- Node and config checks pass on CI.
- Python policy regression tests pass.
- MCP process launches on a supported desktop environment.
- A public test site can be navigated, inspected, interacted with, and captured with an evidence record.
- Ephemeral and persistent profile behavior is verified separately.
- High-risk approval digest mismatch is rejected.
- An interrupted external-write test proves no blind replay occurs.

This commit can validate source/config and policy tests. A real browser launch and login/profile verification still require a Node.js 20+ local machine with an installed browser binary and compatible local MCP client.
