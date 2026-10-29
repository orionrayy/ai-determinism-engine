# SOL Sentinel

A free-first, read-only SOL scalping research and paper-trading project intended to integrate with `orionrayy/ai-determinism-engine`.

**Current phase: Phase 0 — public-market scanner and research.** This branch does not place orders, connect to Reku accounts, create wallets, sign transactions, or move funds. It cannot guarantee a financial target.

## Objective

Investigate whether a rule-based SOL strategy can produce positive net expectancy after realistic trading costs. The user's aspirational account-value target is IDR 3,500,000 from an initial IDR 300,000. This is a goal for measurement, not a forecast or promise. Achieving it would require approximately 11.67x total equity (about +1,066.7%), which is not a credible guaranteed single-session outcome.

## Status

- [x] Inspected repository head and created an isolated implementation branch.
- [x] Defined scope, risk constraints, sources, event log, and Termux runbook.
- [x] Added a dependency-free Python read-only scanner that reads public Binance SOL/USDT candles.
- [x] Added offline unit tests and a CI workflow.
- [ ] Run the scanner on the user's device and validate network/API access in their environment.
- [ ] Confirm Reku live fee schedule, minimum order, withdrawal availability/cost, and official API terms before any live path.
- [ ] Collect historical data and perform out-of-sample testing including costs.
- [ ] Implement a paper-position ledger and report realized/unrealized simulated P&L.
- [ ] Evaluate a documented, policy-compliant live execution route only after the above gates pass.

## Quick start

Requires Python 3.10+ and network access; no third-party package is required.

```sh
python projects/sol-sentinel/sol_sentinel.py
python projects/sol-sentinel/sol_sentinel.py --json
python projects/sol-sentinel/sol_sentinel.py --watch-seconds 30
python projects/sol-sentinel/capture_history.py --symbol SOLUSDT --interval 1m --days 3 --output ~/.sol-sentinel/SOLUSDT_1m.csv
python projects/sol-sentinel/paper_engine.py --csv ~/.sol-sentinel/SOLUSDT_1m.csv --report-path ~/.sol-sentinel/backtest.json
python projects/sol-sentinel/evaluate_splits.py --csv ~/.sol-sentinel/SOLUSDT_1m.csv --output ~/.sol-sentinel/evaluation.json
python projects/sol-sentinel/sol_sentinel.py --fee-bps-round-trip 42.11 --spread-bps-round-trip 10 --slippage-bps-round-trip 20
python -m unittest discover -s projects/sol-sentinel -p 'test_*.py' -v
```

The fee/slippage parameters are assumptions and must be set to the actual selected venue and execution path. The default data venue is Binance public SOL/USDT, not Reku SOL/IDR. Historical capture fails closed on gaps; the current backtest requires contiguous 1-minute candles. Never treat a backtest as a forecast.

## Files

- `SPEC.md`: system contract, DAG, interfaces, phases, and acceptance criteria.
- `RISK_POLICY.md`: hard safety limits and live-trading gates.
- `SOURCES.md`: source inventory, official documentation, and verification caveats.
- `CONVERSATION_CONTEXT.md`: user-facing conversation digest and continuation handoff.
- `EXECUTION_LOG.md`: chronological user-facing request/decision timeline, API/tool inventory, CI evidence, rejection history and continuation checklist.
- `ORCHESTRATION_MEMORY_ADDENDUM.md`: project-specific continuity note supplementing the root canonical memory.
- `SESSION_LOG.md`: user-confirmed project context, decisions, and progress checkpoints.
- `RUNBOOK_TERMUX.md`: install, run, validate, and troubleshoot on Android.
- `sol_sentinel.py`: standard-library scanner, indicators, signal gating, and JSONL audit output.
- `capture_history.py`: bounded public Binance 1m history capture with CSV hash and metadata sidecar.
- `paper_engine.py`: cost-aware offline backtest with SQLite paper ledger; never sends orders.
- `evaluate_splits.py`: fixed-parameter chronological 60/20/20 evaluation, including an untouched out-of-sample segment.
- `test_paper_engine.py`: replay, data-integrity, and ledger tests.
- `test_sol_sentinel.py`: offline tests.
- `../../.github/workflows/sol-sentinel-tests.yml`: continuous integration for this module.

## Integration principle

The existing orchestrator is the supervisory control plane. The scanner is a small deterministic worker. LLMs can critique strategy reports and propose offline experiments, but may not override hard risk rules, handle private keys, or sign/send transactions. Live effects require explicit approval, provider-policy verification, idempotency, durable intent, reconciliation, and a kill switch.

## Privacy

The repository is public. Do not commit wallet seed phrases, private keys, API secrets, account identifiers, screenshots with personal information, or private transaction records. The session log captures only information necessary to track the project and user-requested target.


## Automated historical research

The GitHub Actions workflow `.github/workflows/sol-sentinel-research.yml` can capture a bounded public candle window, run the offline backtest, and evaluate fixed parameters on chronological 60/20/20 splits. It is research automation only, not a live trading bot. Because GitHub requires `workflow_dispatch` and scheduled workflows to exist on the default branch, the workflow must be merged to the default branch before it can be run manually from Actions or execute on its daily schedule. Artifacts are retained for seven days.


## Currency consistency

The scanner's market feed is SOLUSDT but the paper engine's reference capital is IDR. The history capture records a dated daily USD/IDR reference, and the simulator converts OHLC prices to IDR before calculating quantity, stop/target levels and nominal P&L. This is a unit-conversion proxy, not a live Reku price; USDT/USD basis and intra-window FX variation are not modeled. The engine fails closed if FX metadata is absent or stale unless a verified manual rate is supplied.


## Current checkpoint (2026-10-29)

- Latest previously verified CI run: [40910240121](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40910240121), passed compile and 32 offline tests for commit `e20c3ce0b49969cbaa08706a44fc765e47e01db9`.
- Current development remains on `feat/sol-sentinel-paper-trader-v1`; draft PR #178 is open and not merged.
- Full project continuity journal: [EXECUTION_LOG.md](EXECUTION_LOG.md). Re-check the newest head and its CI after any further commits; this status is a point-in-time checkpoint.
