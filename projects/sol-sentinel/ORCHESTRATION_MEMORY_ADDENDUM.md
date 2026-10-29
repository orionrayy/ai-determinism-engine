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