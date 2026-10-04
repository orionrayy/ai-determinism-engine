# v76 Effect Admission and Lease Fencing

## Goal

Prevent unsafe external side-effect execution when the distributed control-plane lease is close to expiry, and eliminate lease ownership drift caused by repeated acquisition inside a workflow turn.

## Constraints

- $0/free-first architecture remains unchanged.
- No new external service or dependency.
- Git-only workflows keep their existing durability barrier path.
- Distributed workflows continue to use the existing Durable Object control plane.
- No provider is assumed to support fencing unless its adapter contract says so.
- Unknown external outcomes remain fail-closed.

## Changes

1. Remove the second workflow-lease acquisition immediately before effect claim. The control_plane_session lease is authoritative for the entire supervisor turn.
2. Introduce a deterministic effect-attempt lease horizon policy: default expected single-attempt duration 120 seconds; default safety margin 30 seconds. Execution is admitted only when the current lease has at least expected + safety seconds remaining. Both values are configurable through bounded environment configuration.
3. Before effect claim, renew the current lease and validate its remaining horizon.
4. Before every retry attempt, renew the lease, replace the local lease object with the renewed lease, then validate the horizon. Completion must use the latest lease object.
5. For lease horizon failure, do not invoke the external tool. Record a dependency/control-plane failure and leave the workflow recoverable.
6. Preserve current GitHub/connector reconciliation semantics; no broadening of automatic side-effect replay.
7. Add regression tests for no second lease acquire; insufficient lease horizon blocks tool execution; sufficient horizon admits execution; retry renewal replaces the lease expiration seen by completion; lease renewal preserves fence epoch while extending expiry; control-plane completion receives the latest lease object; and Git-only side effects remain governed by the durability barrier.

## Non-goals

- No full ExecutionRuntime consolidation.
- No Durable Object Alarm subsystem.
- No new evidence provider.
- No migration away from GitHub Actions.