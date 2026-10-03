# v67 Epistemic Boundary Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Harden the v66 control-plane boundaries and add a free-first evidence/epistemic layer without introducing paid runtime dependencies or a second durable runtime.

**Architecture:** Keep GitHub Actions + stdlib Python as the durable control plane. Fix connector idempotency/CI coverage first, then normalize evidence into canonical source records with independent-source semantics. Add optional public/free provider adapters and claim-level validation primitives without making multi-agent debate mandatory.

**Tech Stack:** Python 3.12 stdlib, GitHub Actions, existing connector bridge, existing agent protocol.

**Spec:** User request in this conversation; v66 audit findings and repository main at fff0864ac2e05a120754e291d07ed0751e5dc103.

## Global Constraints
- Free-first: no paid runtime service, database, broker, queue, or new runtime dependency.
- External side effects remain behind existing approval/idempotency/reconciliation controls.
- Preserve deterministic routing, bounded inputs/outputs, and existing protocol identities.
- Every behavioral change gets a regression test before implementation.

## Review Focus
- Connector cache expiry must never crash idempotency handling.
- Bridge runtime and orchestrator result-contract schemas must remain symmetric.
- CI must execute root bridge runtime tests, not merely compile them.
- Research source counts must represent canonical independent evidence rather than provider count.
- Provider/model availability policy must be centralized rather than duplicated.

### Task 1: Connector idempotency regression
Files: Modify test_bridge_runtime.py, bridge_runtime.py.
- [ ] Add a test proving expired 3-tuple cache entries are cleaned without exception.
- [ ] Run the focused test and observe the tuple-unpack failure.
- [ ] Fix cleanup to unpack the 3-tuple.
- [ ] Run focused and bridge runtime tests.
- [ ] Commit.

### Task 2: CI runtime coverage
Files: Modify .github/workflows/orchestrator-tests.yml.
- [ ] Add test_bridge_runtime.py to the actual unit-test command.
- [ ] Add bridge_runtime.py and bridge_server.py to PR path triggers if absent.
- [ ] Validate YAML/actionlint through CI.
- [ ] Commit.

### Task 3: Connector protocol symmetry
Files: Modify bridge_runtime.py, test_bridge_runtime.py.
- [ ] Add result-required/result-types support to bridge normalization.
- [ ] Add regression showing discovery preserves result contract.
- [ ] Run bridge runtime tests.
- [ ] Commit.

### Task 4: Evidence model
Files: Create orchestrator/evidence_records.py, tests.
- [ ] Add canonical source identity and deterministic deduplication.
- [ ] Add source/provider/primaryity/accessibility metadata.
- [ ] Add independent-source counting based on canonical identity.
- [ ] Add tests for cross-provider duplicates and independent evidence.
- [ ] Commit.

### Task 5: Research provider fabric
Files: Create orchestrator/research_providers.py, tests, registry updates.
- [ ] Add bounded stdlib adapters for OpenAlex, Semantic Scholar, Europe PMC, and CORE where no credential is needed.
- [ ] Respect free-only budgets and provider-specific rate limits via deterministic sequential fallback.
- [ ] Normalize into evidence records.
- [ ] Commit.

### Task 6: Unified execution policy
Files: Modify orchestrator/capability_graph.py, orchestrator/orchestrator.py, tests.
- [ ] Add one policy predicate for free/model/env/health eligibility.
- [ ] Reuse it in router and executor.
- [ ] Add regression preventing selection of a paid/unavailable Gemini model.
- [ ] Commit.

### Task 7: Epistemic output validation
Files: Create focused module and tests.
- [ ] Add claim/evidence reference validation and coverage computation.
- [ ] Preserve unresolved/contested claims instead of forcing consensus.
- [ ] Commit.

### Task 8: Final verification
- [ ] Run complete Python test suite.
- [ ] Run Actionlint/CI verification through GitHub Actions.
- [ ] Inspect changed files and branch compare.
- [ ] Create/update PR only; do not merge without a separate approval.