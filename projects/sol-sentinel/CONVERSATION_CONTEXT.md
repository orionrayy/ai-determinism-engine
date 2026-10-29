# Conversation Context and Handoff

This document preserves the user-facing context and decisions needed to continue SOL Sentinel across sessions. It is a structured digest, not a verbatim dump of every assistant response or internal reasoning. Do not store private keys, API tokens, passwords, seed phrases, or raw personal screenshots in this public repository.

## Dialogue timeline (2026-10-29, Asia/Jakarta)

1. User requested an aggressive automated scalping project with an aspirational IDR 500,000 to IDR 3,000,000 outcome in one night and asked which asset could make sense.
2. User narrowed initial scanning to SOL.
3. User said no exchange was set up, capital was cash, and asked for a highly automated, free-first system that the assistant could inspect and operate, while asking about a no-KYC route.
4. Decision: avoid relying on an unverified no-KYC IDR on-ramp. Prioritize a self-custody wallet and a DEX only for funds already on-chain; treat the fiat-to-crypto funding path separately and legally.
5. User requested free API, repo, and project discovery. Research identified Binance public market data, DEX Screener, Jupiter docs, Helius free tier, Solana SDKs (anza-xyz/kit, solders), and LiteSVM as possible components. Their free-tier terms and limits must be rechecked before production.
6. User said they have an already-KYC-verified Reku account and asked whether they should buy SOL there.
7. User confirmed IDR 300,000 had been deposited and was available; target changed to IDR 3,500,000.
8. User screenshot of Reku SOL Pro showed the Buy tab, Limit Order, limit price IDR 1,820,000, nominal order amount IDR 150,000, estimated quantity 0.08241759 SOL, available balance IDR 300,000, and Open Orders (0). This is only a point-in-time UI snapshot; it does not show a submitted or filled order.
9. User asked to build and prepare the system and preserve project context, source links, tools, workflow, history and conversation in this GitHub repository.
10. Engineering decision: a financial outcome cannot be guaranteed. First create a read-only scanner and CI, then collect data, build a cost-aware paper engine, evaluate out-of-sample performance, and only later consider live execution behind explicit safety gates.

## Key quantitative context

- Initial capital reported: IDR 300,000.
- Aspirational end equity: IDR 3,500,000.
- Required profit: IDR 3,200,000.
- Equity multiple: about 11.67x.
- Required net return: about +1,066.7%.
- This target is not a forecast, deadline quota, or guaranteed result. The system must not escalate leverage or risk to force the target.
- Initial paper-only parameters: risk budget 0.5% equity/trade (IDR 1,500 at initial equity), daily loss lock 3% (IDR 9,000), no leverage, no martingale, no averaging down.
- A candidate IDR 150,000 first exposure and IDR 150,000 reserve were discussed as a conservative planning assumption, not an instruction to buy.

## Current implementation

Repository: https://github.com/orionrayy/ai-determinism-engine
Feature branch: feat/sol-sentinel-paper-trader-v1
Project home: https://github.com/orionrayy/ai-determinism-engine/tree/feat/sol-sentinel-paper-trader-v1/projects/sol-sentinel

Implemented on the branch:
- README.md: project status and quick start.
- SPEC.md: components, DAG, scope, acceptance criteria, and rollout gates.
- RISK_POLICY.md: risk controls and fail-closed rules.
- SOURCES.md: APIs, repositories, official docs, quotas, and caveats.
- SESSION_LOG.md: durable decisions and progress.
- RUNBOOK_TERMUX.md: how to run on Android.
- sol_sentinel.py: standard-library, read-only scanner using completed Binance SOLUSDT 1m/5m/15m candles.
- test_sol_sentinel.py: offline unit tests.
- .github/workflows/sol-sentinel-tests.yml: free CI compile + unit tests.

CI run:
- Run URL: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40898402582
- Commit checked: e34a1cd8d74d76967df580cd563bfbbb20ac6d7f
- Conclusion: success; compile and offline unit tests passed.
- Later session-log-only code/doc updates may trigger another workflow; inspect the latest run before claiming the latest commit is green.

## Critical limitations / next steps

1. Scanner uses Binance public SOLUSDT data as a reference feed, not Reku SOL/IDR execution prices. API reachability on Termux has not yet been confirmed.
2. Fees and slippage passed to the scanner are assumptions. Verify the current fee schedule and exact order preview on Reku before using the model.
3. Current scanner does not open positions, place/cancel orders, create a wallet, sign a transaction, or withdraw assets.
4. Reku's currently supported official authenticated order API and terms for automation have not been verified. Do not automate through undocumented private endpoints.
5. Add durable historical capture/replay and a cost-aware paper position/fill ledger.
6. Validate on chronologically separated train/validation/out-of-sample data; report net expectancy, profit factor, max drawdown, sample size, and cost sensitivity.
7. Only after strategy and execution validation may a documented live path be evaluated, with explicit user approval, isolated signer, idempotency, reconciliation, strict exposure/loss/slippage caps, and user-accessible kill switch.

## Sources and tools

Canonical links and caveats are maintained in SOURCES.md. Relevant official references include:
- Binance spot market-data docs: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
- DEX Screener API: https://docs.dexscreener.com/api/reference
- Reku status: https://reku.id/status
- Reku historical API specification, unverified for current automation support: https://api.reku.id/v2/openapi.json
- Jupiter docs: https://developers.jup.ag/docs
- Helius pricing: https://www.helius.dev/pricing
- Solana TypeScript SDK: https://github.com/anza-xyz/kit
- Solders: https://github.com/kevinheavey/solders
- LiteSVM: https://github.com/LiteSVM/litesvm
- OJK PADK No. 3/2026: https://www.ojk.go.id/id/regulasi/Pages/PADK-Nomor-3-Tahun-2026-Penyelenggaraan-Perdagangan-Aset-Keuangan-Digital-termasuk-Aset-Kripto.aspx

## Privacy warning

This repository is public. Anything committed here is publicly readable. Only the user-provided project budget/target and the minimum necessary screenshot observations have been retained to preserve task continuity. Never add account identifiers, personally identifying data, private transaction identifiers, credentials, wallet secrets, or unredacted screenshots.


## Continued implementation checkpoint — 2026-10-29

Additional files now present:
- `capture_history.py`: bounded public-only Binance candle capture, CSV hash and metadata sidecar; no order API.
- `paper_engine.py`: offline cost-aware historical simulation and SQLite paper ledger.
- `test_paper_engine.py`: offline backtest/ledger tests.
- `evaluate_splits.py`: fixed-parameter chronological 60/20/20 evaluation with an untouched out-of-sample segment.
- `test_evaluate_splits.py`: tests for exact split boundaries, insufficient history, invalid ratios, and timestamp gaps.

Latest verified CI run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40906164183 — success for commit `41f60c53ff7801d3b3729c27f5bd338854538386`. The compile and offline unit-test steps both passed.

The backtest has not yet been run against a downloaded historical CSV on the user's device. No actual profitability, exchange connectivity, Reku execution support, or goal attainment has been established. Next actions: run capture and tests in Termux, run the basic backtest and chronological evaluator with verified fee parameters, inspect the trade count and per-segment results, and keep all private data local.


## Cost-model / research-workflow checkpoint — 2026-10-29

The scanner and paper engine now model venue fee, quoted spread, and slippage as distinct parameters. The paper engine applies adverse spread and slippage adjustments to modeled fills, retains exchange fees separately, and reports execution-friction cost. Negative spread values are rejected. A free-first, manual/scheduled research workflow now captures public Binance candle history, runs the offline backtest and fixed-parameter 60/20/20 evaluation, and uploads artifacts for seven days. It does not send orders.

The workflow has been committed but not manually dispatched during this session; API reachability and actual report artifacts still need validation. Never interpret the simulation as current account equity or as proof of profitability.


## Conversation, APIs, tools and progress archive

The detailed user-facing project timeline, API/repository inventory, GitHub connector operation inventory, tool rejection history, CI run IDs, current status and next-step checklist are now preserved in [EXECUTION_LOG.md](EXECUTION_LOG.md). This is a structured digest rather than a verbatim export of the entire displayed chat; private reasoning and secrets are intentionally not stored.


## Latest read-back verification — 2026-10-29

- The detailed journal is [EXECUTION_LOG.md](EXECUTION_LOG.md).
- Latest verified head at this checkpoint: `8d8c91738b5f237e715206f662db417bbb509b63`; CI run [40911083700](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40911083700) passed compile and 32 offline tests.
- PR #178 remains draft/open/unmerged. Observed main SHA `d5672c84a0ae184c3a95d3f925925887ece71aa4` differs from the PR-reported base SHA `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`; synchronize/review before merge.
- Research workflow full network run, Termux testing and historical profitability remain unverified.


## Historical validation checkpoint — 2026-10-29

- The first end-to-end capture failed due HTTP 451 from the general Binance REST host. Switching to the official market-data-only host `https://data-api.binance.vision/api/v3/klines` resolved that failure.
- Successful 7-day run: [40914013908](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40914013908), 10,080 1m candles; zero trades under the baseline strategy.
- Successful 30-day run: [40914239047](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40914239047), 43,200 1m candles; baseline and fixed chronological 60/20/20 splits yielded zero trades.
- Initial cost-sensitivity report: [40915191550](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40915191550). Reku Pro baseline (published fee components plus assumed spread/slippage) and Pro fee-only theoretical floor produced zero trades. Lightning fee-only theoretical floor with zero spread/slippage generated 10 trades but lost approximately IDR 3,893 (-1.30%); this is not a realistic executable quote.
- A later rerun (40915499872) failed closed because the daily FX reference request timed out. The FX fetcher now has bounded retries, with unit tests, and a retry-enabled full pipeline run `40916209196` is in progress at the last checkpoint. Do not claim that last run succeeded until its final status and artifacts are inspected.
- The exact user prompt sequence and major execution milestones are preserved in `CHAT_TRANSCRIPT.md`; the detailed tool/API inventory and run IDs are in `EXECUTION_LOG.md`.


## Latest continuation checkpoint — 2026-10-30 WIB

Current verified code head is `ca3e13391634609377617ef26ca6ef255059de3d` on `feat/sol-sentinel-paper-trader-v1`. CI run [40921862301](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921862301) passed 63 offline tests. Full research run [40921592237](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921592237) completed successfully with artifact [12918995790](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921592237/artifacts/12918995790).

The live Reku read-only probe now successfully calls the order book using `symbol=SOL_IDR`; a sample snapshot returned bid IDR 1,835,000, ask IDR 1,840,000 and spread 27.21 bps. The chart endpoint is usable at 5-minute interval; the 1-minute request returned HTTP 500 and is no longer probed first. The legacy `/activitytrans` route returns HTTP 404, so native Reku transaction history remains unavailable and the collector records the blocker rather than repeatedly retrying.

The 30-day Binance reference backtest still produces zero trades under the current signal/cost gate. This means no profitability evidence has been established. Live orders remain disabled. PR #178 remains draft/unmerged; reconcile its recorded base SHA with the current `main` SHA before merge. Read `EXECUTION_LOG.md` for the detailed event and test history.

## Latest implementation checkpoint — 2026-10-30 WIB

Current verified source branch head is `7d74015ecb99c0f1768baddd40b1643399c5bea2`. CI [40928032367](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032367) and [40928041385](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928041385) succeeded; each recorded 70 tests and no failures. The patch hardens spread inputs, guards midpoint spread definition, adds modeled Reku minimum notional (Pro Rp30,000 / Lightning Rp5,000), adds signal-rejection combinations and simulated-order-size counters, and reports expectancy/trade stats with undefined zero-trade metrics as null.

The historic 43,200-candle baseline remains zero trades. Observed filter counts in run [40921592237](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921592237) are multi-label; no single filter has been proven to be the sole cause. Latest research workflow [40928032346](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032346) was in progress when this checkpoint was written, and must be rechecked before claiming a post-patch research result.

The PR is still open/draft/unmerged. No real orders or financial side effects occurred. Android/Termux was not accessed and remains unvalidated. Root `ORCHESTRATION_MEMORY.md` remains unmodified by this branch commit; a separate additive checkpoint branch/PR is being prepared to avoid colliding with the evolved `main`.
