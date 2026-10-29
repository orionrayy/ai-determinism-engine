# SOL Sentinel — Orchestration Memory Addendum

Date: 2026-10-29, Asia/Jakarta.

This file supplements, but does not replace, the canonical `ORCHESTRATION_MEMORY.md`. It is deliberately stored as a separate, small project-specific addendum so SOL Sentinel's operational context can be versioned without rewriting the entire large control-plane memory file.

## Project identity

- Repository: https://github.com/orionrayy/ai-determinism-engine
- Development branch: `feat/sol-sentinel-paper-trader-v1`
- Draft PR: https://github.com/orionrayy/ai-determinism-engine/pull/178
- Project directory: `projects/sol-sentinel/`
- Runtime mode: read-only research/scanner; paper trading is a later phase.
- Live transaction execution: disabled and unimplemented in Phase 0.

## User goal and capital context

The user reports IDR 300,000 deposited in their KYC-verified Reku account and expresses an aspirational portfolio target of IDR 3,500,000. The target implies approximately 11.67x equity and +1,066.7% net return. Do not promise, imply, or optimize as if this outcome were guaranteed or owed by a deadline.

A Reku screenshot showed a point-in-time limit-buy form at IDR 1,820,000 with IDR 150,000 selected, an estimated 0.08241759 SOL, IDR 300,000 available, and zero open orders. It does not prove any order was submitted or filled.

## Implementation state

The branch contains:
- `README.md`
- `SPEC.md`
- `RISK_POLICY.md`
- `SOURCES.md`
- `SESSION_LOG.md`
- `CONVERSATION_CONTEXT.md`
- `RUNBOOK_TERMUX.md`
- `sol_sentinel.py`
- `test_sol_sentinel.py`
- `.github/workflows/sol-sentinel-tests.yml`
- this addendum

CI run `40899451163` succeeded for branch head `39ef55b1f8c7da160f901894524318184cffa13d`. Subsequent documentation-only commits should be checked against the latest CI run before claiming current head is green.

## Execution and safety policy

1. The scanner uses Binance public SOLUSDT 1m/5m/15m candles only as a reference feed; it does not represent Reku SOL/IDR executable prices.
2. Scanner uses Python standard library and has no order-placement, withdrawal, wallet creation, key handling, or transaction-signing capability.
3. Actual Reku fee schedules, minimum order size, withdrawal costs and limits, supported API operations, and terms for automation must be verified from current official sources before live use.
4. Never use undocumented/private endpoints to bypass platform controls.
5. Do not place secrets, wallet private keys, seed phrases, API tokens, personal identifiers, raw account screenshots, or private transaction details in this public repository.
6. A live execution phase requires explicit user approval tied to exact action/parameters, a dedicated wallet or credential boundary, a durable idempotency identity, safe uncertain-outcome reconciliation, order limits, cost/slippage gates, emergency stop, and independent validation.
7. No leverage, martingale, loss-chasing, or automatic risk escalation to force the target.
8. If data is stale/inconsistent, costs exceed modeled edge, or transaction state is unknown, fail closed and do not trade.

## Next work

1. Have the user run the scanner on Termux and report only non-sensitive output.
2. Add historical data capture/replay and a durable, cost-aware paper ledger.
3. Obtain realistic venue-specific fee/spread/slippage assumptions.
4. Run chronological out-of-sample evaluation and report expectancy, profit factor, maximum drawdown, sample size, and cost sensitivity.
5. Review the official Reku API/terms and feasible execution path.
6. Only then consider a live adapter after an explicit approval gate.

## Conversation continuity

The user requested persistence of context, source links, tools, decisions, and session progress. The structured user-facing summary is `CONVERSATION_CONTEXT.md`; detailed technical choices are `SPEC.md`; current verified milestones are `SESSION_LOG.md`; source and tool URLs are in `SOURCES.md`.

A direct update of the full root `ORCHESTRATION_MEMORY.md` was attempted but blocked by a tool safety check; no specific cause was returned. This separate addendum is the safe, small-file workaround. Do not state that the canonical memory itself was modified.

## Update — 2026-10-29, paper research pipeline

The feature branch now includes bounded public-candle capture, a cost-aware offline backtest/SQLite ledger, and fixed-parameter chronological 60/20/20 evaluation with a distinct out-of-sample segment. Associated offline tests are included in the project's CI.

Latest verified CI: run `40906164183`, commit `41f60c53ff7801d3b3729c27f5bd338854538386`, success. No user-device testing, live market order, account connection, or financial side effect has occurred. Keep all downloaded candles and reports with account-specific values on-device; never commit private data into this public repository.


## Cost model and workflow update — 2026-10-29

The scanner and offline paper engine separate fee, spread, and slippage assumptions. The new `.github/workflows/sol-sentinel-research.yml` is a free-first research workflow for bounded public candle capture, offline backtesting, chronological evaluation, and short-retention artifacts. It is not a live trading workflow. Its first manual dispatch is still required to validate actual GitHub Actions public-data reachability and report generation.


## Continuity archive update — 2026-10-29

The detailed conversational decision timeline, APIs/repositories, GitHub tool inventory, known rejections, CI debugging evidence and continuation plan are recorded in `EXECUTION_LOG.md` and indexed from `README.md`. Before any further writes, reread the current PR head and latest CI. The last fully inspected green run before the journal commits was `40910240121` (commit `e20c3ce0b49969cbaa08706a44fc765e47e01db9`, 32 tests passed). This needs to be refreshed after the current commit sequence.


## Latest checkpoint — 2026-10-29

- Full user-facing decisions, source/API inventory, GitHub connector tools, debugging/rejection history, CI runs and continuation steps: `projects/sol-sentinel/EXECUTION_LOG.md`.
- Project README indexes the journal; the context and session log have current status entries.
- Last verified green head before these final handoff notes: `8d8c91738b5f237e715206f662db417bbb509b63`; CI run `40911083700` passed compilation and 32 offline tests.
- PR #178 is still a draft and unmerged. The PR's recorded base SHA differs from the currently observed `main` SHA; inspect and reconcile before merge.
- No live trading or wallet operation was performed. End-to-end network research, Termux runtime, and historical profitability remain unverified.


## Historical data and cost-model checkpoint — 2026-10-29

Successful 30-day research run `40914239047` captured 43,200 public Binance SOLUSDT 1m candles with a dated USD/IDR reference. Baseline/60-20-20 segments had no trades. Initial sensitivity run `40915191550` found no trades with the Reku Pro fee reference even at the Pro fee-only floor. A theoretical Lightning fee-only floor (zero spread/slippage) made 10 simulated trades but lost about IDR 3,893 (-1.30%). This theoretical floor omits quote friction and is not a live scenario.

The later expanded-statistics run `40915499872` failed closed because Frankfurter timed out. The FX reference fetcher now has bounded retries and passing offline tests; retry-enabled full pipeline run `40916209196` was in progress at the last read. Recheck the final result before updating status. No live trade, wallet operation or actual target attainment has occurred.


## Latest checkpoint — 2026-10-30 WIB

SOL Sentinel's public-only research pipeline now has verified current Reku market-data handling:
- Latest CI [40921862301](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921862301) passed 63 offline tests.
- Research run [40921592237](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921592237) completed successfully with a 30-day, 43,200-candle Binance SOLUSDT pipeline and artifact [12918995790](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921592237/artifacts/12918995790).
- Reku order book now uses the verified public query `symbol=SOL_IDR`; the snapshot probe returned 40 levels per side and 27.210884 bps spread. Public chart uses 5m first because f=1 returned HTTP 500.
- The legacy Reku `/activitytrans` route returned HTTP 404. Collector fails fast and marks it unavailable; do not use incomplete chart data as a substitute for 30 days of native history.
- No strategy edge has been demonstrated: baseline and chronological segments still yield zero trades. Live mode remains disabled. PR #178 is draft/unmerged and requires base-SHA reconciliation.

## 2026-10-30 implementation checkpoint — spread and diagnostics

- Verified patch head: `7d74015ecb99c0f1768baddd40b1643399c5bea2`, based on previous branch head `29488b5ecc55fe080980a38bd900753c7d70a135`.
- CI runs [40928032367](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032367) and PR run [40928041385](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928041385) passed compile + 70 offline tests.
- Spread formula remains midpoint convention. Historical `198.0198` vs `200` bps discrepancy for bid 99/ask 101 comes from ask-price denominator versus midpoint, respectively. Current code already used midpoint; the patch adds robust type/numeric validation and regression tests rather than changing the expected value.
- New diagnostics report multi-label rejection reasons, exact rejection combinations, order attempts/fills, minimum-order rejections, gross/average win/loss and net expectancy. Zero-trade win rate and expectancy are null to represent undefined statistics.
- Simulated minimum order is explicitly configured: Reku Pro Rp30,000 and theoretical Lightning floor Rp5,000, from Reku's public help page. The current research run [40928032346](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032346) was in progress when recorded; append its final artifact/metrics only after inspection.
- No live trade, cancellation, money transfer, withdrawal, or credential use. Termux remains untested. PR #178 remains draft and unmerged; do not merge until current main/PR state is reconciled.
