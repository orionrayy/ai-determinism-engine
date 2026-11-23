# Robin OSINT MCP Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Integrate pinned Robin dark-web OSINT as an optional, approval-gated capability in the AI orchestration engine.

**Architecture:** Keep Robin as a Git submodule pinned to `b5d4f5f509fac39d65d3a28fd1d78b56efbeeaf4`; use Robin's Dockerfile and a standard-library MCP stdio adapter. Register a separate `osint` capability, with fail-closed user authorization, Gemini free-model policy, and non-retryable effect semantics.

**Tech Stack:** Python 3.12 engine, Python stdlib, MCP JSON-RPC stdio, Docker, Git submodules, GitHub Actions, unittest.

**Spec:** `docs/architecture/robin-osint-integration.md`

## Global Constraints

- Robin source is pinned to commit `b5d4f5f509fac39d65d3a28fd1d78b56efbeeaf4`; never use a floating upstream branch or image tag.
- Runtime remains disabled unless `ORCHESTRATOR_ROBIN_ENABLED=true`.
- `ROBIN_ENABLE_TOKEN=true` is the exact runtime enablement token.
- Only `threat_intel` and `ransomware_malware` presets are accepted.
- `user_chose=true` and an explicit authorization basis present in the original goal are mandatory.
- Reuse `GEMINI_API_KEY` as Robin's `GOOGLE_API_KEY`; pass no other provider secrets.
- In hard free mode, the selected Robin model must pass `gemini_model_allowed` and its time-bounded allowlist.
- Robin output remains untrusted and delimited; no instruction inside scraped text is followed.
- The Robin effect contract is `identity=engine`, `retry=blocked`, `reconciliation=none`, `fencing=engine`.
- No Cloudflare deployment, new hosted service, paid dependency, production secret mutation, or destructive operation.

## Validation boundary

The latest GitHub Actions run validates the adapter and engine integration, but it does not build or launch the Robin container because `ORCHESTRATOR_ROBIN_ENABLED` remains unset by default. No live Tor investigation or provider request was performed. The high-level `robin_investigate` pipeline is integrated; the Streamlit UI and separate low-level MCP session tools are not individually surfaced as engine nodes. A runtime smoke test requires an explicit enablement and approved live workflow.

## Review Focus

- Explicit consent spoofed by planner output rather than present in the original goal — test exact goal binding.
- Personal-identity or corporate-espionage presets accidentally enabled — test strict preset allowlist.
- MCP notifications, malformed frames, oversized responses, and mismatched request IDs — test bounded JSON-RPC handling.
- Tor bootstrap/runtime failures after effect start — ensure uncertainty is marked and never retried.
- Missing/expired free-model approval, Docker, or API key — test fail-closed preconditions.

---

### Task 1: Robin MCP stdio adapter

**Files:**
- Create: `orchestrator/robin_mcp.py`
- Test: `orchestrator/test_robin_mcp.py`

**Interfaces:**
- Produces `validate_investigation_input(payload: Mapping[str, Any], goal: str) -> dict[str, Any]`.
- Produces `MCPStdioClient(command: Sequence[str], env: Mapping[str, str], timeout: float)` with `initialize()`, `list_tools()`, `call_tool(name, arguments)`, and `close()`.
- Produces `execute_robin_mcp(node: Any, goal: str, model: str) -> dict[str, Any]`.
- The adapter accepts no free-form URL, shell command, Docker image override, or provider-key override from workflow/node input.

- [x] **Step 1: Write failing validation tests** for missing preset, false `user_chose`, authorization basis absent from the original goal, unsupported presets, allowed defensive presets, and bounded query/instructions.
- [x] **Step 2: Run** `python -m unittest discover -s orchestrator -p "test_robin_mcp.py" -v`; expected: failures because the adapter module/functions do not exist.
- [x] **Step 3: Write failing MCP protocol tests** using a real local Python child process that implements initialize, tools/list, and tools/call; cover notifications, malformed responses, and oversized messages.
- [x] **Step 4: Implement the minimal stdlib JSON-RPC stdio client** with bounded messages, request deadlines, process cleanup, and no sampling capability.
- [x] **Step 5: Implement the Robin Docker invocation** with the pinned image name, workflow/node-derived volume, only `GOOGLE_API_KEY` and `ROBIN_MODEL` forwarded, Tor health polling, UTF-8 bounded report output, untrusted-result metadata, and best-effort report redaction.
- [x] **Step 6: Preserve the named volume after the non-idempotent tool starts; remove it only after `persist_workflow()` succeeds. Keep it on uncertain failures or persistence errors; do not delete results before the durable checkpoint exists.
- [x] **Step 7: Verify** the adapter tests pass and inspect the subprocess argv to prove no secret value or untrusted argument is interpolated into a shell command.

### Task 2: Engine routing, policy, and effect admission

**Files:**
- Modify: `orchestrator/tools.json`
- Modify: `orchestrator/orchestrator.py`
- Modify: `orchestrator/llm_planner.py`
- Test: `orchestrator/test_robin_mcp.py`

**Interfaces:**
- Registry declares `capability:osint` with `robin_mcp` as its only/default adapter.
- `robin_mcp.required_env` is `GEMINI_API_KEY`; `enabled_env` is `ROBIN_ENABLE_TOKEN`; `allowed_actions` contains only `investigate`.
- `execute_node` delegates to `execute_robin_mcp(node, goal, model)` after the existing live effect-contract and free-only gates.
- Planner omits `robin_mcp` unless `ROBIN_ENABLE_TOKEN == "true"`.

- [x] **Step 1: Write failing registry/policy tests** for disabled/enabled routing, hard-free model enforcement, high-risk classification, and blocked retry contract.
- [x] **Step 2: Run** the focused test module and verify the new policy assertions fail before production changes.
- [x] **Step 3: Register** `capability:osint` and the high-risk `robin_mcp.investigate` effect contract in `tools.json`.
- [x] **Step 4: Add planner gating and planner instructions** requiring user-supplied preset, `user_chose=true`, and authorization basis in the original goal.
- [x] **Step 5: Route the engine executor** to the adapter and reuse `configured_gemini_model` plus `gemini_model_allowed`.
- [x] **Step 6: Verify** focused tests plus import/registry checks.

### Task 3: Pinned upstream source and opt-in runner integration

**Files:**
- Create: `.gitmodules`
- Create gitlink: `vendor/robin` at upstream commit `b5d4f5f509fac39d65d3a28fd1d78b56efbeeaf4`
- Modify: `.github/workflows/orchestrator.yml`
- Test: `orchestrator/test_robin_mcp.py`

**Interfaces:**
- The workflow maps repository variable `ORCHESTRATOR_ROBIN_ENABLED` to `ROBIN_ENABLE_TOKEN`.
- When enabled, the workflow initializes the pinned submodule and builds image `ai-orchestrator-robin:b5d4f5f509fa` before orchestration.
- When disabled (default), it neither fetches the submodule nor builds the Docker image.

- [x] **Step 1: Write failing workflow/submodule tests** for exact commit pin, default-disabled gating, and immutable image tag.
- [x] **Step 2: Run** the focused tests and verify they fail before the workflow/submodule changes.
- [x] **Step 3: Add** the MIT upstream submodule pointer and conditional Docker build step.
- [x] **Step 4: Add** only the existing Gemini secret/model environment mapping required by Robin; never add a new provider key.
- [x] **Step 5: Verify** JSON, workflow syntax, gitlink SHA, and actionlint through the engine CI workflow.

### Task 4: Canonical documentation and memory

**Files:**
- Create: `docs/architecture/robin-osint-integration.md`
- Create: `docs/superpowers/plans/2026-11-23-robin-osint-mcp-integration.md`
- Modify: `README.md`
- Modify: `ORCHESTRATION_MEMORY.md`

- [x] **Step 1: Add** the architecture and implementation plan documents.
- [x] **Step 2: Document** the Robin capability, runtime enablement, security boundary, free-model coupling, and limits in the README.
- [x] **Step 3: Add** a dated canonical-memory entry with upstream SHA, integration state, tests/CI evidence, and the Cloudflare inventory authentication limitation.
- [x] **Step 4: Review** all changes for stale claims, license omission, unbounded costs, external side effects, and false completion claims.
