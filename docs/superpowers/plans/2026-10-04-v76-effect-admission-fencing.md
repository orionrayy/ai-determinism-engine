# v76 Effect Admission and Lease Fencing Implementation Plan

> Execute task-by-task with test-first development and verify every acceptance criterion.

## Task 1 — RED tests

Files:
- Create orchestrator/test_effect_admission.py
- Modify orchestrator/test_orchestrator.py

Tests:
- a lease with less than expected+safety seconds remaining is rejected;
- sufficient horizon is admitted;
- the same lease epoch is allowed after renewal when only expiry changes;
- a side-effect execution path renews rather than reacquires the workflow lease;
- retry renewal updates the lease object used by completion;
- blocked admission produces no tool execution.

Run focused tests and confirm RED.

## Task 2 — GREEN implementation

Files:
- Modify orchestrator/orchestrator.py

Implement a pure ensure_effect_lease_horizon helper plus bounded environment parsing. Refactor the side-effect handling so the current session lease is renewed before claim; horizon is checked before claim; no second acquire occurs; before_attempt renews and stores the renewed lease; horizon is checked before every actual side-effect attempt; completion uses the current lease.

## Task 3 — Verification

Run:
- python -m compileall -q orchestrator
- node --check control-plane/cloudflare/src/worker.js
- full orchestrator unittest suite
- root gateway/bridge/actions config suite
- existing control-plane evaluation harness under ORCHESTRATOR_FREE_ONLY=true
- inspect changed-file diff for dependency and authority regressions.

## Acceptance

1. No side effect starts unless the lease horizon is sufficient.
2. No redundant workflow lease acquisition occurs inside an active session.
3. Latest renewed lease is used for effect completion.
4. Lease-failure paths remain fail-closed and recoverable.
5. No paid dependency is introduced.