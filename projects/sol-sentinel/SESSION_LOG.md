# SOL Sentinel — Session Log / Durable Context

Timezone: Asia/Jakarta. This log contains user-visible project decisions and verified execution checkpoints, not private chain-of-thought.

## 2026-10-29 — Project initiation

### User goal
Build a highly automated, efficient, carefully calibrated SOL scalping project targeting equity of IDR 3,500,000 from reported starting capital IDR 300,000. User requests free-first APIs/open-source projects and that sources, tools, decisions, and project history be stored in GitHub so the assistant can inspect the current implementation in future sessions.

### Decisions
- Keep SOL as the initial candidate because it fits the Solana/Jupiter execution design and has large-market liquidity; do not claim it will achieve the target.
- Target is an aspiration, not a guaranteed outcome. IDR 300,000 to IDR 3,500,000 implies 11.6667x equity and +1,066.7% net return.
- Primary fiat on-ramp is the user's existing verified Reku account; no KYC-bypass route is assumed.
- Do not assume Reku supports official authenticated automated order placement until current official API docs and terms verify it.
- First deliver a read-only market scanner using public market data, followed by strategy research and a cost-aware paper simulator.
- No live trade, withdrawal, wallet creation, or real order has been submitted by the assistant.
- Do not store credentials, private keys, seed phrases, account IDs, or personal screenshots in the public repository.

### Screenshot facts supplied by user
A Reku SOL Pro screen showed: buy side selected; order type Limit Order; limit price IDR 1,820,000; buy amount IDR 150,000; estimated size 0.08241759 SOL; available balance IDR 300,000; order count shown as zero. This is a point-in-time UI snapshot only. It does not prove a buy order was submitted or filled.

### Financial risk design (initial paper assumptions)
- Up to IDR 150,000 may be considered for a future manual initial exposure only after the user checks the live quote/fees; keep the other IDR 150,000 in IDR during early validation.
- Initial simulated risk budget: 0.5% equity per trade, IDR 1,500 when equity is IDR 300,000.
- Initial simulated daily stop: 3% equity, IDR 9,000 at IDR 300,000.
- No leverage, martingale, loss-chasing, automatic position escalation, or averaging down.
- Actual loss may exceed planned risk due to gaps, spread, slippage, platform/network issues, or failed/partial execution.

### Candidate free-first sources
- Binance public klines: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
- DEX Screener API: https://docs.dexscreener.com/api/reference
- Reku status: https://reku.id/status
- Reku historical API spec (must revalidate): https://api.reku.id/v2/openapi.json
- Jupiter docs: https://developers.jup.ag/docs
- Helius free tier: https://www.helius.dev/pricing
- Solana TypeScript SDK: https://github.com/anza-xyz/kit
- Solana Python toolkit: https://github.com/kevinheavey/solders
- LiteSVM: https://github.com/LiteSVM/litesvm
- Cloudflare Workers free limits: https://developers.cloudflare.com/workers/platform/limits/
- OJK PADK No. 3/2026: https://www.ojk.go.id/id/regulasi/Pages/PADK-Nomor-3-Tahun-2026-Penyelenggaraan-Perdagangan-Aset-Keuangan-Digital-termasuk-Aset-Kripto.aspx

### Repository checkpoint
- Repository: https://github.com/orionrayy/ai-determinism-engine
- Main tree SHA observed: eaa864a554c2c995ce73bd9f6b4d07206ecc56c5
- Branch created: feat/sol-sentinel-paper-trader-v1
- No changes to main branch in this checkpoint.
- Planned artifacts: project spec, risk policy, sources, this session log, Termux runbook, scanner, offline tests, CI test workflow.

## Current implementation state

- Branch: `feat/sol-sentinel-paper-trader-v1`.
- Scanner code, specification, risk policy, sources, conversation context, session log, Termux runbook, unit tests, and CI workflow are committed on the feature branch.
- GitHub Actions run `40898402582` for commit `e34a1cd8d74d76967df580cd563bfbbb20ac6d7f`: workflow conclusion `success`; compile and offline-unit-test steps passed.
- The scanner is read-only and explicitly sets `order_placement_enabled=false`; it has not been run from the user's Termux device yet.
- No wallet was created, no SOL was purchased, and no live order, transfer, or withdrawal was submitted by the assistant.
- This is an engineering checkpoint, not a performance claim. Strategy profitability and the IDR 3,500,000 target remain unverified.

### Next work items
1. Run the scanner from the user's Termux device and verify live public market-data reachability.
2. Add durable candle capture/replay and a cost-aware paper-position ledger.
3. Validate Reku's official fee schedule, minimum order, withdrawal fee/minimum, and whether any official authenticated order API is supported and permitted.
4. Research and test historical data chronologically (training, validation, untouched out-of-sample) with costs, spread, slippage, no-fill and latency assumptions.
5. Record expectancy, profit factor, maximum drawdown, sample size and sensitivity analysis; do not tune only to a target outcome.
6. Require explicit user approval and a separate security review before any live execution integration.


## 2026-10-29 — Paper engine and chronological evaluation checkpoint

### Implemented
- `paper_engine.py`: offline-only cost-aware backtest for contiguous 1-minute OHLCV CSV, with completed 5m/15m aggregation, deterministic signal hypothesis, modeled entry/exit fees and per-side slippage, position sizing, stop/target/max-hold exits, conservative stop-first same-bar handling, drawdown metrics, and SQLite ledger persistence with deterministic run identity.
- `capture_history.py`: bounded public Binance kline history capture for up to 30 days, capped pages, bounded retries, gap validation, CSV, SHA-256, and metadata sidecar. Read-only only.
- `evaluate_splits.py`: fixed-parameter chronological 60/20/20 development/validation/out-of-sample evaluation. No automatic parameter optimization. Each segment uses independent initial capital and results are not compounded across segments.
- `test_paper_engine.py` and `test_evaluate_splits.py`: offline validation for history parsing, aggregation, configuration guards, gap rejection, ledger idempotency, and split boundaries.
- CI compile list now includes scanner, capture, paper engine, evaluator, and tests. Unit test discovery covers all test modules.

### Debug and validation history
- Initial CI uncovered fixture issues: an epoch timestamp that did not align to complete five-minute buckets and a gap fixture that produced only 99 rows. Fixtures were corrected without relaxing production data-integrity checks.
- Latest verified workflow run: `40906164183`, commit `41f60c53ff7801d3b3729c27f5bd338854538386`, conclusion `success`. Compile and offline unit-test steps passed.
- Do not carry forward stale failed runs as current status; the latest run above is green for its exact commit.
- Termux runtime/network access has not yet been executed or verified from the user's device.
- No real order, wallet creation, signing, transfer, withdrawal, or trade was executed.

### Next steps
1. Test public history capture and scanner on Termux; retain capture CSV and metadata locally, not in the public repo.
2. Run the backtest and chronological evaluation on real historical data with actual current venue fee assumptions and conservative slippage.
3. Inspect sample count, per-segment net P&L, profit factor, drawdown, cost sensitivity, and trade logs. Do not claim strategy edge if trade count is small or segments disagree.
4. Improve the fill/cost model and include trade-level spread and latency assumptions before paper deployment.
5. Verify current official Reku market-data/API docs, terms, minimum order/withdrawal and fees.
6. Keep live execution blocked until an official permitted route, isolated signer, idempotency/reconciliation, kill switch, risk caps, out-of-sample evidence, and explicit user approval are all present.


## 2026-10-29 — Explicit execution friction and research workflow

- The cost model now separates round-trip fee, quoted spread, and slippage assumptions in the read-only scanner.
- The offline backtest models round-trip spread through an adverse half-spread adjustment at entry and exit, applies slippage as a separate per-side price adjustment, keeps exchange fees separate in the ledger, and reports modeled execution-friction costs.
- The chronological evaluator accepts a separate spread assumption.
- Added validation that rejects negative spread values, plus a bounded public-only `.github/workflows/sol-sentinel-research.yml` that can capture historical candles, run offline backtest/evaluation, and upload artifacts for seven days. This workflow has not yet been manually dispatched in this session; its market-data step and artifacts need a real run to validate network/runtime behavior.
- The live trading route remains unimplemented; no orders, transfers or wallet operations have occurred.


## 2026-10-29 — Durable conversation/tool journal checkpoint

- Added `EXECUTION_LOG.md` to preserve the user-facing request/decision timeline, the source/API inventory, GitHub connector operations used, known tool rejections, CI debugging history, and exact continuation checklist. It deliberately excludes private chain-of-thought, secrets, and raw screenshots.
- Added a link from the project README to the execution journal.
- Latest fully inspected green CI baseline before these new documentation commits: run `40910240121`, commit `e20c3ce0b49969cbaa08706a44fc765e47e01db9`; Python compile and offline tests succeeded, log reported 32 tests in 0.082 seconds.
- Current branch is `feat/sol-sentinel-paper-trader-v1`; draft PR #178 remains open/unmerged. The canonical root `ORCHESTRATION_MEMORY.md` is unchanged; project addendum is used instead because the large-file update was rejected.
- The current project includes scanner, bounded history capture, daily USD/IDR reference metadata, IDR unit conversion, cost-aware offline backtest/SQLite ledger, fixed-parameter chronological 60/20/20 evaluator, unit tests, CI, and a bounded research workflow. No historical dataset has yet been used to claim profitability.
- Next verification: inspect latest CI after this journal update, then validate one end-to-end history-capture/backtest/evaluator run. Research workflow is not yet dispatched; on-demand/scheduled invocation normally requires the workflow on the default branch.
