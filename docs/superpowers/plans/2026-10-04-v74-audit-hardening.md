# V74 Audit Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the v67-v74 orchestration stack by closing scheduled-recovery, distributed-authority, execution-duplication, and epistemic-admission boundaries while preserving a strict $0/free-first runtime.

**Architecture:** GitHub Actions remains the compute substrate; Cloudflare Durable Objects remain the distributed authority only for workflows explicitly assigned to that authority mode; Git remains an audit/snapshot substrate. Recovery becomes a small deterministic policy layer, and execution/epistemic checks are centralized so the same safety semantics cannot diverge between one-step and batched execution.

**Tech Stack:** Python 3.12, GitHub Actions, Cloudflare Durable Objects/SQLite, stdlib-only runtime primitives, existing public/free research APIs.

**Spec:** Prior approved audit/design in conversation; this plan records the executable consolidation slice.

## Global Constraints

- Preserve `ORCHESTRATOR_FREE_ONLY=true` and do not add paid runtime dependencies.
- Do not silently downgrade a distributed-authority workflow to Git-only live execution.
- Waiting for approval is a normal durable state, not a polling-recovery condition.
- Recovery failures must be isolated per workflow candidate.
- No LLM self-reported evidence-independence field may be authoritative.
- Selective evidence abstention must be an actual admission gate, not telemetry-only.
- Keep external side effects fail-closed when outcome or fencing is uncertain.

## Review Focus

- Empty repository state must make scheduled recovery succeed with no candidate workflows.
- An active GitHub run must never be rearmed merely because local state is older than five minutes.
- A distributed-authority workflow must fail closed when its control plane is unavailable.
- A high-confidence epistemic verdict with unsupported coverage must not pass the admission gate.
- Debate routing must compute evidence independence from structured evidence rather than trust a self-reported count.

---

### Task 1: Scheduled recovery policy extraction

**Files:**
- Create: `orchestrator/recovery_policy.py`
- Test: `orchestrator/test_recovery_policy.py`
- Modify: `.github/workflows/orchestrator.yml`

**Interfaces:**
- `run_is_active(status: str | None) -> bool`
- `recovery_reasons(workflow: Mapping[str, Any], *, now: datetime, active_run_status: str | None = None, stale_after_seconds: int = 300) -> list[str]`
- `recovery_event_id(workflow: Mapping[str, Any], reason: str, generation: str | None = None) -> str`

- [ ] **Step 1:** Add failing tests for empty state, active-run suppression, approval suppression, uncertain-effect recovery, and stable generation-bound recovery identities.
- [ ] **Step 2:** Run the focused tests and confirm import/behavior failures.
- [ ] **Step 3:** Implement the pure recovery policy.
- [ ] **Step 4:** Replace inline scheduler decision logic with the policy and isolate per-candidate dispatch failures.
- [ ] **Step 5:** Use `gh api --method POST`, safe `git add -A .orchestrator`, and current-run status checks.
- [ ] **Step 6:** Run the focused tests and full orchestrator suite.

### Task 2: Immutable execution authority

**Files:**
- Create: `orchestrator/authority.py`
- Test: `orchestrator/test_authority.py`
- Modify: `orchestrator/state_schema.py`
- Modify: `orchestrator/orchestrator.py`

**Interfaces:**
- `authority_mode_for_workflow(live: bool, control_plane_available: bool) -> str`
- `infer_legacy_authority(workflow: Mapping[str, Any]) -> str`
- `require_runtime_authority(workflow: Mapping[str, Any], control_plane_configured: bool, control_plane_active: bool) -> None`

- [ ] **Step 1:** Add failing tests for immutable authority selection, legacy inference, and no distributed-to-Git downgrade.
- [ ] **Step 2:** Run focused tests and verify failure.
- [ ] **Step 3:** Implement authority helpers and state migration defaults.
- [ ] **Step 4:** Make workflow creation persist `authority_mode`; make load/persist prefer the correct authority.
- [ ] **Step 5:** Run focused tests, compile checks, and the full suite.

### Task 3: Execution-path consolidation

**Files:**
- Create: `orchestrator/execution_runtime.py`
- Test: `orchestrator/test_execution_runtime.py`
- Modify: `orchestrator/orchestrator.py`

**Interfaces:**
- Centralize the side-effect lifecycle in one runtime helper used by single-node and batched execution paths.
- Reuse the existing checkpoint, retry, lease, effect-claim, and validation callbacks rather than duplicating their semantics.

- [ ] **Step 1:** Add regression tests for identical admission/fencing semantics across single-node and batch paths.
- [ ] **Step 2:** Run focused tests and confirm failure.
- [ ] **Step 3:** Extract the smallest shared execution lifecycle without changing public node/state contracts.
- [ ] **Step 4:** Delete only the duplicated path logic made unreachable by the extraction.
- [ ] **Step 5:** Run full unit/eval checks.

### Task 4: Epistemic admission repair

**Files:**
- Modify: `orchestrator/epistemic_validation.py`
- Modify: `orchestrator/selective_evidence_gate.py`
- Modify: `orchestrator/epistemic_deliberation.py`
- Test: `orchestrator/test_epistemic_validation.py`
- Test: `orchestrator/test_epistemic_deliberation.py`

**Interfaces:**
- Selective gate consumes canonical `coverage` keys, not an invented `supported_coverage` alias.
- A selective abstention blocks validation admission.
- Independence is recomputed from evidence records.
- Low confidence with insufficient/contested evidence routes to abstention or deliberation rather than silent acceptance.

- [ ] **Step 1:** Add failing tests for the `supported_coverage` mismatch, real blocking abstention, and self-reported independence poisoning.
- [ ] **Step 2:** Run focused tests and confirm failure.
- [ ] **Step 3:** Implement the minimal repairs.
- [ ] **Step 4:** Run the complete epistemic/evaluation suite.

### Task 5: Final verification and PR review

**Files:**
- Modify: `ORCHESTRATION_MEMORY.md` only after the integration branch has a verified final state.

- [ ] **Step 1:** Run Python compileall, YAML/actionlint checks, full unittest discovery, and evaluation harness.
- [ ] **Step 2:** Inspect Git diff and workflow runs.
- [ ] **Step 3:** Perform an independent code review of the whole branch.
- [ ] **Step 4:** Update canonical memory to the verified integration baseline and create a ready-for-review PR; do not merge into `main` automatically.
