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
- [ ] Verify Reku public endpoint availability and live spread/order-book snapshot through the current GitHub Actions run; then validate access on the user's Termux device.
- [x] Verify Reku's published Pro Mode fee components and minimum order from official pages; re-check the final order preview before any trade.
- [ ] Confirm actual SOL withdrawal availability/cost and supported authenticated order API terms before any live path.
- [x] Capture 30 days of historical reference candles and run fixed chronological 60/20/20 backtest + cost sensitivity; current rules did not pass the baseline trade gate.
- [x] Implement offline paper ledger, cost model, drawdown reporting, and predeclared sensitivity scenarios.
- [x] Add a read-only adapter for documented Reku public SOL/IDR price, bid/ask, and order-book endpoints.
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
python projects/sol-sentinel/cost_sensitivity.py --csv ~/.sol-sentinel/SOLUSDT_1m.csv --output ~/.sol-sentinel/cost_sensitivity.json
python projects/sol-sentinel/reku_market.py --symbol SOL --binance-csv ~/.sol-sentinel/SOLUSDT_1m.csv --output ~/.sol-sentinel/reku_sol_idr_market.json
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
- `CHAT_TRANSCRIPT.md`: verbatim user prompts for this project sequence plus execution milestone digest.
- `ORCHESTRATION_MEMORY_ADDENDUM.md`: project-specific continuity note supplementing the root canonical memory.
- `SESSION_LOG.md`: user-confirmed project context, decisions, and progress checkpoints.
- `RUNBOOK_TERMUX.md`: install, run, validate, and troubleshoot on Android.
- `sol_sentinel.py`: standard-library scanner, indicators, signal gating, and JSONL audit output.
- `capture_history.py`: bounded public Binance 1m history capture with CSV hash and metadata sidecar.
- `paper_engine.py`: cost-aware offline backtest with SQLite paper ledger; never sends orders.
- `evaluate_splits.py`: fixed-parameter chronological 60/20/20 evaluation, including an untouched out-of-sample segment.
- `cost_sensitivity.py`: predeclared scenarios compare published Reku Pro fee components with clearly labeled lower-friction and fee-floor assumptions; no automatic selection by P&L.
- `reku_market.py`: read-only public Reku SOL/IDR price, bid/ask, order book, spread, top-five depth and optional comparison with Binance reference prices.
- `test_reku_market.py`: offline tests for quote parsing, spread math, and cross-venue reference calculations.
- `test_cost_sensitivity.py`: scenario and invariant tests.
- `test_paper_engine.py`: replay, data-integrity, and ledger tests.
- `test_sol_sentinel.py`: offline tests.
- `../../.github/workflows/sol-sentinel-tests.yml`: continuous integration for this module.

## Integration principle

The existing orchestrator is the supervisory control plane. The scanner is a small deterministic worker. LLMs can critique strategy reports and propose offline experiments, but may not override hard risk rules, handle private keys, or sign/send transactions. Live effects require explicit approval, provider-policy verification, idempotency, durable intent, reconciliation, and a kill switch.

## Privacy

The repository is public. Do not commit wallet seed phrases, private keys, API secrets, account identifiers, screenshots with personal information, or private transaction records. The session log captures only information necessary to track the project and user-requested target.


## Automated historical research

The GitHub Actions workflow `.github/workflows/sol-sentinel-research.yml` can capture a bounded public candle window, run the offline backtest, evaluate fixed parameters on chronological 60/20/20 splits, and compare predeclared cost scenarios. It is research automation only, not a live trading bot. Because GitHub requires `workflow_dispatch` and scheduled workflows to exist on the default branch, the workflow must be merged to the default branch before it can be run manually from Actions or execute on its daily schedule. Artifacts are retained for seven days.


## Currency consistency

The scanner's market feed is SOLUSDT but the paper engine's reference capital is IDR. The history capture records a dated daily USD/IDR reference, and the simulator converts OHLC prices to IDR before calculating quantity, stop/target levels and nominal P&L. This is a unit-conversion proxy, not a live Reku price; USDT/USD basis and intra-window FX variation are not modeled. The engine fails closed if FX metadata is absent or stale unless a verified manual rate is supplied.


## Current checkpoint (2026-10-29)

- Latest previously verified CI run: [40910240121](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40910240121), passed compile and 32 offline tests for commit `e20c3ce0b49969cbaa08706a44fc765e47e01db9`.
- Current development remains on `feat/sol-sentinel-paper-trader-v1`; draft PR #178 is open and not merged.
- Full project continuity journal: [EXECUTION_LOG.md](EXECUTION_LOG.md). Re-check the newest head and its CI after any further commits; this status is a point-in-time checkpoint.


## Research checkpoint — 2026-10-29

- Public historical capture now uses Binance's official market-data-only host after GitHub Actions returned HTTP 451 from the general API host.
- Successful 30-day pipeline: [research run 40914239047](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40914239047), 43,200 candles, baseline strategy 0 trades.
- Fixed-parameter cost sensitivity found no Pro-mode trades even at the fee-only floor (spread/slippage set to zero). A Lightning fee-only mathematical floor produced 10 trades and approximately IDR -3,893 net over the dataset, before any actual quote spread/slippage; this is not a live execution estimate.
- Details, run IDs, artifacts, tool/source inventory, and exact user request quotations are in [EXECUTION_LOG.md](EXECUTION_LOG.md) and [CHAT_TRANSCRIPT.md](CHAT_TRANSCRIPT.md).


## Reku public market adapter

The adapter reads the documented public Reku API only: coin catalogue, price, bid/ask and order book. It does not authenticate, place orders or infer order-execution permission. The public API reference is legacy documentation; the current GitHub Actions run is testing live endpoint availability. A report marked `PARTIAL` or `ERROR` must not be used to approve a trade. The 30-day baseline has not demonstrated a tradable edge; zero trades and the negative fee-only Lightning floor are explicit warnings, not reasons to force entries.
