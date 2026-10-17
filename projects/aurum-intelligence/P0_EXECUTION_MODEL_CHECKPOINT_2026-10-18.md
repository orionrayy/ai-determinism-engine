# P0 execution-accounting checkpoint — 2026-10-18

Status: candidate increment on the existing research PR branch; hosted CI pending.
Target: Aurum Intelligence research pipeline in orionrayy/ai-determinism-engine.
Decision gate: NO_TRADE; live execution disabled.

## Delivered

- Dependency-free deterministic closed-trade accounting using entry/exit Bid/Ask snapshots.
- Correct long Ask/Bid and short Bid/Ask quote-side accounting; spread is not charged twice.
- Separate adverse slippage, per-side commission, signed swap PnL and optional net R.
- UTC-aware timestamp, positive non-crossed quote, chronology, nonnegative cost/contract-size, lot, rollover and currency checks.
- Unknown material costs suppress net PnL; declared assumptions remain explicit estimates.
- Evidence references are syntax-checked only. Manual-review eligibility at most; live execution authorization is invariantly false.
- Standard-library regression suite for accounting and fail-closed behavior.

## Local validation

- unittest: 16 tests passed on staged candidate files.
- py_compile: passed.
- Hosted PR CI is still required and may reveal integration failures.

## Known limitations

This is accounting logic, not a fill simulator. It does not handle partial fills, rejection, queue position, limit-order fill probability, market impact, latency distributions, gaps/stop execution, margin/liquidation, instrument-specific triple-swap calendars or currency conversion. No real data or observed broker costs were introduced by this change.

## Next gates

1. Hosted CI passes the existing online-benchmark suite plus these tests.
2. Add broker/instrument-specific profiles from authoritative specifications and authorized historical execution evidence.
3. Implement chronological quote replay with actual Bid/Ask, conservative path ambiguity and fill rules.
4. Report gross, spread, slippage, commission, swap and net PnL independently.
5. Compare OOS net expectancy and probability scores to preregistered causal baselines with dependence-aware intervals. Unit tests do not promote a model.
