# v75 Recovery, Authority, and Epistemic Boundary Hardening Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Harden the v74 orchestration line so unattended recovery, workflow authority, ingress idempotency, and epistemic admission remain deterministic and fail-closed at $0 runtime cost.

**Architecture:** Add a small pure scheduled-recovery policy module, make workflow state authority explicit and immutable, reuse the existing control-plane session for one-step execution, and make epistemic gates/debate consume validated evidence structures. Keep GitHub Actions as compute/watchdog, Git as audit/snapshot, and the existing optional Cloudflare Durable Object as live authority; do not add another workflow engine.

**Tech Stack:** Python 3.12 stdlib, GitHub Actions YAML, existing Cloudflare Worker, unittest.

**Spec:** docs/superpowers/specs/2026-10-04-v75-recovery-authority-epistemic-design.md

## Global Constraints

- Production orchestration remains GitHub Actions + stdlib Python.
- ORCHESTRATOR_FREE_ONLY=true remains authoritative for live production workflow execution.
- No paid database, broker, queue, workflow engine, or runtime dependency is added.
- Git remains an audit/eventual-snapshot layer for distributed workflows.
- Recovery remains bounded, deterministic, idempotent, and fail-closed for ambiguous effects.
- Existing state remains migratable and readable.

## Review Focus

1. Empty sharded state and missing shard directory must not turn recovery into a pathspec failure — Task 1 tests the no-work/empty-state path.
2. A stale-looking workflow with a still-running GitHub run must never be rearmed — Task 1 tests active run suppression.
3. A distributed workflow must not execute from Git when control-plane configuration disappears — Task 2 tests fail-closed authority.
4. An idempotency-only ingress request must map to the same workflow without a false collision — Task 2 tests stored idempotency identity.
5. A high-confidence epistemic result must not pass when selective evidence checks abstain, and a fake self-reported source count must not satisfy deliberation — Task 3 tests both boundaries.

---

### Task 1: Deterministic scheduled recovery

**Files:**
- Create: orchestrator/scheduled_recovery.py
- Create: orchestrator/test_scheduled_recovery.py
- Modify: .github/workflows/orchestrator.yml
- Modify: test_actions_config.py

**Interfaces:**
- Consumes workflow dictionaries plus a UTC datetime and an optional GitHub run status.
- Produces pure recovery decisions and a stable event ID.
- Scheduled workflow uses the helpers but remains responsible for GitHub API I/O and aggregate failure reporting.

- [ ] Step 1: Write the failing unit tests

Add tests for waiting_approval exclusion; stale-running threshold behavior; active GitHub run suppression for queued/in_progress/waiting/requested/pending; missing or unknown run status eligibility; stable recovery event IDs when only updated_at changes; changed IDs when recovery-relevant state changes; and empty state producing zero candidates.

Run: python -m unittest orchestrator.test_scheduled_recovery -v
Expected: FAIL because the new helpers do not exist.

Add an action-config assertion that scheduled recovery dispatch contains --method POST and waiting_approval is absent from the scheduler candidate expression.
Run: python -m unittest test_actions_config -v
Expected: FAIL for the same reason.

- [ ] Step 2: Implement minimal deterministic recovery policy

Implement:
- DEFAULT_STALE_SECONDS = 1500
- ACTIVE_RUN_STATUSES = {queued, in_progress, waiting, requested, pending}
- is_stale_running(workflow, now, stale_seconds=DEFAULT_STALE_SECONDS) -> bool
- is_recovery_candidate(workflow, now, *, active_run_status=None, stale_seconds=DEFAULT_STALE_SECONDS) -> bool
- recovery_generation(workflow) -> str
- recovery_event_id(workflow) -> str

recovery_generation hashes only deterministic recovery-relevant fields: workflow ID, status, failed-node identity, node status/error execution-uncertain markers, barrier-failed execution markers, federation ID/status, and control-plane state version. Do not include mutable wall-clock fields or GitHub run IDs.

- [ ] Step 3: Patch scheduled workflow

Use is_recovery_candidate for candidate selection.

For a stored GitHub run ID on a running candidate, query GET /repos/{repo}/actions/runs/{run_id} via gh api. Treat active statuses as non-recoverable for this scheduler pass; treat 404, missing, or unknown as eligible.

Dispatch repository events with gh api --method POST repos/$REPOSITORY/dispatches --input -.

Continue processing all candidates after an individual dispatch failure. Collect a bounded error summary and exit nonzero only after all candidates have been attempted.

Remove waiting_approval from scheduler recovery.

Use recovery_event_id for stable idempotency.

Guard compaction staging so a missing .orchestrator/workflows directory is harmless.

- [ ] Step 4: Run focused tests

Run: python -m unittest orchestrator.test_scheduled_recovery test_actions_config -v
Expected: PASS.

- [ ] Step 5: Commit

git add orchestrator/scheduled_recovery.py orchestrator/test_scheduled_recovery.py .github/workflows/orchestrator.yml test_actions_config.py
git commit -m "fix: harden scheduled recovery semantics"

---

### Task 2: Immutable workflow authority and ingress idempotency

**Files:**
- Modify: orchestrator/state_schema.py
- Modify: orchestrator/orchestrator.py
- Modify: orchestrator/test_orchestrator.py

**Interfaces:**
- Add authority constants AUTHORITY_GIT_DURABLE = git_durable and AUTHORITY_DISTRIBUTED_CONTROL_PLANE = distributed_control_plane.
- Add migration/default logic that preserves explicit distributed control-plane workflows and otherwise defaults to Git authority.
- Add a single authority gate used by control-plane session, workflow load, and persistence.
- Persist idempotency_key from create_workflow().

- [ ] Step 1: Write the failing tests

Add tests for live workflow creation with control-plane configuration selecting distributed authority; workflow creation without control-plane configuration selecting Git authority; legacy explicit control-plane state migrating to distributed authority; distributed workflow with missing control-plane configuration failing before Git-only live execution; Git-authority workflow continuing to use Git even when control-plane configuration later appears; run_one_step and run_workflow sharing the same control-plane session path; and idempotency-only ingress storing idempotency_key and resolving repeated requests to the same workflow.

Run: python -m unittest orchestrator.test_orchestrator -v
Expected: FAIL on the new assertions.

- [ ] Step 2: Implement schema/migration authority

Add the authority constants and migrate every workflow so explicit distributed state becomes distributed authority and all other workflows become Git authority. New workflow creation chooses authority from live mode plus current control-plane configuration. Do not persist secrets. Do not change authority later because environment configuration changes.

- [ ] Step 3: Enforce authority at load/persist/session boundaries

Make load_workflow inspect locally persisted workflow authority before deciding whether the remote control plane is authoritative.

For distributed authority: configured control plane means remote state is authoritative; unavailable control plane means fail closed with a dependency error rather than executing from Git.

For Git authority: always use Git state even if control-plane configuration appears later.

Make persist_workflow reject distributed live persistence without an active control-plane lease instead of falling through to _write_workflow_shard().

Replace the manual control-plane acquisition block in run_one_step with the existing control_plane_session context manager.

- [ ] Step 4: Persist ingress identity

Add idempotency_key to the workflow object returned by create_workflow when supplied. Preserve null/absent behavior when no key is supplied.

- [ ] Step 5: Run focused tests

Run: python -m unittest orchestrator.test_orchestrator -v
Expected: PASS.

- [ ] Step 6: Run full Python suite

Run: python -m unittest discover -s orchestrator -p "test_*.py" -v && python -m unittest test_gateway test_bridge_runtime test_actions_config -v
Expected: PASS.

- [ ] Step 7: Commit

git add orchestrator/state_schema.py orchestrator/orchestrator.py orchestrator/test_orchestrator.py
git commit -m "fix: make workflow authority immutable"

---

### Task 3: Evidence gate and blind deliberation correctness

**Files:**
- Modify: orchestrator/epistemic_validation.py
- Modify: orchestrator/epistemic_deliberation.py
- Modify: orchestrator/test_epistemic_validation.py
- Modify: orchestrator/test_epistemic_deliberation.py

**Interfaces:**
- claim_coverage exposes supported_coverage.
- validate_epistemic_output treats selective abstention as a failed admission result.
- debate_decision derives evidence weakness from validated evidence_records via distinct evidence-work identity.
- blind_challenge_view returns opaque candidate IDs without agent identity/role metadata.

- [ ] Step 1: Write the failing tests

Validation tests: high-confidence output with evidence coverage 1.0 and supported coverage 1.0 does not abstain; high-confidence output failing supported coverage abstains and returns passed == False; selective abstention cannot coexist with passed == True; claim coverage exposes supported_coverage.

Deliberation tests: independent_source_count=999 with no valid evidence records still triggers insufficient evidence; valid evidence records meeting the work-count threshold do not trigger evidence weakness; blind view contains candidate_id but no agent_id or agent_role; candidate ordering is stable for identical redacted content.

Run: python -m unittest orchestrator.test_epistemic_validation orchestrator.test_epistemic_deliberation -v
Expected: FAIL before implementation.

- [ ] Step 2: Fix coverage and admission semantics

Set supported_coverage equal to the existing supported claim coverage metric. Compute final passed as base validation success AND not selective abstention.

- [ ] Step 3: Remove self-reported evidence authority

Use count_distinct_evidence_works from the existing evidence independence module on structured evidence records. Never use the proposal independent_source_count field as an authority input.

- [ ] Step 4: Make blind challenge views truly opaque

Redact identity, role, voting, and consensus metadata before ordering. Hash each redacted candidate object deterministically, sort by digest, and assign sequential opaque IDs candidate_1, candidate_2, etc. Preserve claims/evidence needed for critique.

- [ ] Step 5: Run focused tests

Run: python -m unittest orchestrator.test_epistemic_validation orchestrator.test_epistemic_deliberation -v
Expected: PASS.

- [ ] Step 6: Run full Python suite and offline evaluation

Run: python -m unittest discover -s orchestrator -p "test_*.py" -v && python -m unittest test_gateway test_bridge_runtime test_actions_config -v
Then: ORCHESTRATOR_FREE_ONLY=true python -m orchestrator.eval_harness --json
Expected: PASS and no live provider calls.

- [ ] Step 7: Commit

git add orchestrator/epistemic_validation.py orchestrator/epistemic_deliberation.py orchestrator/test_epistemic_validation.py orchestrator/test_epistemic_deliberation.py
git commit -m "fix: harden epistemic admission and blind debate"

---

### Task 4: Branch-level verification and review gate

**Files:**
- No production changes unless verification exposes a defect.
- Optional: ORCHESTRATION_MEMORY.md update if implementation diverges from the v75 spec.

**Interfaces:**
- Deliver a single feature branch based on feat/v74-selective-evidence-gate.
- Main remains untouched until review/merge.

- [ ] Step 1: Compile Python and Worker

Run: python -m compileall -q orchestrator
Run: node --check control-plane/cloudflare/src/worker.js
Expected: PASS.

- [ ] Step 2: Run actionlint

Use the repository's pinned actionlint workflow command or CI workflow itself.
Expected: PASS with no workflow syntax errors.

- [ ] Step 3: Run full verification

Use the repository's orchestrator-tests workflow on the feature branch/PR. Check unit tests, actionlint, Node syntax, and offline evaluation.

- [ ] Step 4: Self-review the diff

Check no paid runtime dependency; no silent authority downgrade; no repeated waiting_approval recovery; no unbounded scheduler loop; no new self-reported evidence authority; no unrelated refactor.

- [ ] Step 5: Request code review

Open or update the pull request from feat/v75-recovery-authority-epistemic. Do not merge until branch verification and review pass.

- [ ] Step 6: Final integration decision

Only after green verification, review the 261-commit v74 lineage for a single integration point toward main. Do not merge historical v67-v74 PRs one-by-one.