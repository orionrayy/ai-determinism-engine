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
- `reku_market.py`: read-only public Reku SOL/IDR price, bid/ask, order-book, and optional cross-venue comparison.
- `test_reku_market.py`: tests for quote parsing, midpoint spread math, order-book fallback, and cross-venue reference calculations.
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


## Latest verified checkpoint — 2026-10-30 WIB

- Current branch head before these documentation commits: `ca3e13391634609377617ef26ca6ef255059de3d`; latest CI [40921862301](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921862301) passed **63 offline tests**.
- Full public research run [40921592237](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921592237) succeeded and published [artifact 12918995790](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921592237/artifacts/12918995790).
- Current Reku order-book query `symbol=SOL_IDR` now returns a valid book; the timestamped probe had 40 levels per side and spread 27.21 bps. The chart probe uses 5m first; native 1m chart request returned HTTP 500.
- The documented legacy `/activitytrans` request returns HTTP 404. Native Reku history is explicitly marked unavailable; historical backtest uses Binance public reference candles.
- The baseline and chronological segments still yield zero trades. No live trading code is enabled. PR #178 remains draft/unmerged and needs base-SHA reconciliation.
- Detailed evidence and continuation steps: [EXECUTION_LOG.md](EXECUTION_LOG.md).

## Implementation checkpoint — 2026-10-30 WIB

- Latest code commit: `7d74015ecb99c0f1768baddd40b1643399c5bea2`. Spread parsing now rejects invalid quote types/numbers safely and has regression tests for midpoint bps, crossed/zero/negative, missing, non-finite, numeric strings and extreme magnitudes.
- The backtest reports filter rejection combinations, signal rejection counts, order attempts/fills, minimum-order rejections and trade expectancy/win rate with undefined metrics represented as `null` when there are no trades. Simulated entry checks include configured venue minimum notional (Reku Pro Rp30,000; Lightning Rp5,000 reference).
- CI [40928032367](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032367) and PR CI [40928041385](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928041385) both passed compilation and **70 offline unit tests**. Research workflow [40928032346](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032346) was still running when this checkpoint was first recorded.
- PR #178 remains draft/open/unmerged. No live order functionality was used; no Termux device validation was performed. Research status remains nonvalidated until current research artifact and independent metrics are inspected.


## Latest native-venue checkpoint — 2026-10-30 WIB

- Read-only INDODAX public market-data and historical-candle adapter is implemented. Public smoke-test evidence: 43,200 contiguous SOL/IDR 1m candles, 30 days, SHA-256 `9dd63e357e3a8f4ba168bccd65edf3fe621d3758884a97549e8425c812ab2316`; workflow [#40948205968](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40948205968). Data was captured by GitHub Actions from the official public endpoint; no account credentials, withdrawals, wallet signatures or orders were used.
- Native IDR backtest now uses identity FX conversion and requires explicit INDODAX-specific costs/minimums; Reku sensitivity presets refuse native IDR input. Source metadata SHA-256/data-status checks fail closed. Paper/live safety gate remains closed.
- Latest verified tests at the code checkpoint immediately before this note: [CI #40948901551](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40948901551), 135 tests, 0 failures. Latest complete 30-day Binance research [run #40948530609](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40948530609) still had zero candidates/trades in the baseline and all fixed time splits; the theoretical fee-only floor remains net negative. A full research run on any newer code head must be treated separately.
- Source identity is now explicit: Binance SOLUSDT remains research reference only; Reku is an acquisition/on-ramp candidate only; INDODAX SOL/IDR is the initial CEX adapter candidate, but its exact current fee mode and a user's destination network acceptance are not certified by public market-data tests.
- See `PLATFORM_ADAPTER_DECISION_2026-10-30.md`, `RUNBOOK_TERMUX.md`, and `EXECUTION_LOG.md`. PR #178 remains draft/open/unmerged and its branch history is diverged from `main`; do not merge or reconcile destructively.


## Latest verified checkpoint — public depth quote and native IDR history

- Successful read-only research [#40950218320](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950218320) captured 43,200 native INDODAX SOL/IDR 1m candles and a current top-of-book quote. Dataset SHA-256 `c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30`; data_status OK; no FX conversion.
- Live public depth snapshot: `market_quote_status=OK`; bid Rp1,777,113; ask Rp1,779,923; spread ~15.7997 bps at 2026-10-29T20:37:47Z. Earlier snapshot was ~25.6669 bps; these are point-in-time quotes, not a spread history.
- Latest exact feature-head CI [#40950649418](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950649418) passed 137 offline tests. A successful CI run does not establish profitability.
- Current 30-day strategy research remains zero candidates/trades across fixed 60/20/20 splits. Native INDODAX data is collected but still must not be backtested using Reku fees; selected fee mode and current all-in fee components must be verified and explicitly supplied.
- Reku remains acquisition/on-ramp candidate only, INDODAX is a candidate execution venue, and Jupiter remains quote-only. No live executor, signer, transfer sender or order route exists. Strategy readiness remains `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`.


## Local diagnostics and venue-cost audit

- termux_doctor.py and test_termux_doctor.py: dependency-free Termux/runtime, SQLite recovery/idempotency, public INDODAX metadata, and streaming dataset diagnostics. This script is safe and read-only; GitHub CI does not validate a user's handset.
- INDODAX_FEE_MODEL_2026-10-30.md: dated official Pro all-in fee model, minimum-order constraints, explicit backtest cost decomposition, and unresolved Lite/withdrawal/spread evidence.
- Native INDODAX evaluation: [research run #40993527618](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40993527618), artifact #12949927917. Current-run workflow [#40994319663](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40994319663) is being validated separately.

## Latest repeated native INDODAX rolling-window result — 2026-10-30

- Most recent successful current-run native venue replay: [workflow #40994685989](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40994685989), artifact #12951295388.
- Dataset SHA-256: d76434d4c9019249e924e72c5b05d0b200c0dfe3c4709c02a6d821fa0b22327c; 43,200 contiguous native INDODAX SOL/IDR one-minute candles; no FX conversion.
- All fixed Pro scenarios (taker-current, taker-stress and maker-current sensitivity) yielded zero candidates, fills and trades in development, validation and OOS. This newer rolling window did not reproduce the single maker-only development trade from #40994319663.
- Current state remains RESEARCH_ONLY / INSUFFICIENT_EVIDENCE. See EXECUTION_LOG.md and the official-fee audit for the full split metrics and limitations.

## INDODAX depth collection

Added `indodax_microstructure.py` for finite, unauthenticated public SOL/IDR depth sampling and SQLite archiving. It validates sorted multi-level books and calculates observed spread/depth distributions from captured snapshots only. See `RUNBOOK_TERMUX.md`. The collector has offline fixture tests; a live endpoint capture and actual Android Termux validation remain outstanding.


## Depth-aware paper execution (engineering component)

The new `depth_execution.py` module implements deterministic multi-level book walking, Decimal VWAP, per-snapshot shared liquidity, residual order lifecycle (GTC/IOC, cancel and expiry), and a persistent SQLite event/fill/portfolio journal with reconciliation and failure-injection tests. It is intentionally separate from the legacy top-of-book adapter. See `RUNBOOK_TERMUX.md` for the offline test command and constraints. It does not connect itself to a forward market-data loop and does not submit or sign orders.
