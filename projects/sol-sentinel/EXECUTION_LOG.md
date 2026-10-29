# SOL Sentinel — Conversation and Execution Journal

Last updated: 2026-10-29 (Asia/Jakarta).
Repository: https://github.com/orionrayy/ai-determinism-engine
Branch: `feat/sol-sentinel-paper-trader-v1`
Draft PR: https://github.com/orionrayy/ai-determinism-engine/pull/178

This journal preserves the user-facing project history, decisions, source/API inventory, tools used, CI evidence, errors and continuation work. It is a structured operational record—not a verbatim dump of every rendered chat message or private internal reasoning. Never add secrets, private keys, wallet seed phrases, API tokens, account IDs, unredacted screenshots or private transaction records.

## 1. User request and decision timeline

### A. Initial trading target
- User first asked for an aggressive automated scalping project for the night, with an initial aspiration of IDR 500,000 to IDR 3,000,000, and asked which assets could make that possible.
- User narrowed the first scanner to SOL.
- We recorded that short-term asset selection cannot make a six-fold one-night outcome reliable. The system should not treat a financial target as a promise or escalate risk merely to reach it.

### B. Exchange and KYC constraint
- User said there was no exchange account yet, the funds were still cash, and requested an efficient, intelligent, free-first automation system that the assistant could inspect and help operate.
- User asked for no-KYC possibilities.
- We decided not to rely on a purported no-KYC IDR fiat on-ramp unless it could be verified as safe, legal, and operational. A DEX does not accept paper currency directly.
- User then clarified that their Reku account is already KYC-verified and asked whether they could buy the asset there.
- Decision: use the existing Reku account as the potential fiat on-ramp; do not assume Reku offers an official supported automated-order API until its current docs and terms confirm it.

### C. Actual capital and target change
- User reported that IDR 300,000 had been deposited to Reku.
- The aspirational equity target changed to IDR 3,500,000.
- Required gain: IDR 3,200,000; 11.6667x total equity; approximately +1,066.7% net return.
- The target is not a forecast or guaranteed outcome.

### D. Reku screenshot facts
The user shared a screenshot of Reku SOL Pro showing:
- Buy tab; Limit Order;
- limit price IDR 1,820,000;
- nominal order amount IDR 150,000;
- estimated amount 0.08241759 SOL;
- available balance IDR 300,000;
- Open Orders (0).
This is a point-in-time UI snapshot only. It does not show a submitted or filled order. The screenshot should not be committed to the public repository.

### E. Build and preserve everything
- User asked to build and prepare the automation, with all conversation context, sources, tools, sequence and progress saved in GitHub.
- Initial root-memory update to the large `ORCHESTRATION_MEMORY.md` was rejected by a GitHub tool safety check; no detailed reason was provided, so do not state a definitive cause.
- Fallback: create `projects/sol-sentinel/ORCHESTRATION_MEMORY_ADDENDUM.md`, `CONVERSATION_CONTEXT.md`, `SESSION_LOG.md`, and this journal. These supplement the root canonical file; they do not replace or modify the root file.

## 2. Current implementation inventory

Project home: https://github.com/orionrayy/ai-determinism-engine/tree/feat/sol-sentinel-paper-trader-v1/projects/sol-sentinel

- `README.md` — project scope, run commands, automation and safety caveats.
- `SPEC.md` — architecture, DAG, strategy hypothesis and acceptance gates.
- `RISK_POLICY.md` — risk boundaries, fail-closed behavior, cost assumptions and secret/approval controls.
- `SOURCES.md` — APIs, repositories, official documentation, limits and caveats.
- `SESSION_LOG.md` — progress checkpoints and verified CI evidence.
- `CONVERSATION_CONTEXT.md` — compact continuity handoff and user-facing timeline.
- `ORCHESTRATION_MEMORY_ADDENDUM.md` — project-specific orchestrator memory; supplements the canonical root file.
- `EXECUTION_LOG.md` — this operational journal.
- `RUNBOOK_TERMUX.md` — Android/Termux runbook.
- `sol_sentinel.py` — Python standard-library, read-only scanner for public Binance SOLUSDT candles at 1m/5m/15m; EMA/RSI/ATR/relative volume, signal reasons, cost-gate assumptions and JSONL audit.
- `capture_history.py` — bounded public candle capture (1–30 days, page cap and retries), CSV plus hash/metadata sidecar, and dated USD/IDR reference lookup.
- `paper_engine.py` — offline cost-aware backtest, candle aggregation, fee/spread/slippage modeling, risk sizing, stop/target/time exits, drawdown metrics and idempotent SQLite paper ledger.
- `evaluate_splits.py` — fixed-parameter chronological 60/20/20 development/validation/out-of-sample evaluation.
- `test_sol_sentinel.py`, `test_capture_history.py`, `test_paper_engine.py`, `test_evaluate_splits.py` — offline tests.
- `.github/workflows/sol-sentinel-tests.yml` — compile and offline unit-test CI.
- `.github/workflows/sol-sentinel-research.yml` — bounded public history capture, offline backtest, chronological evaluation, and seven-day artifact retention. Because the workflow is only on the feature branch and is not on the default branch, GitHub's normal manual/scheduled execution path has not yet been used to validate its full network/report path.

## 3. Architecture and hard boundaries

Current DAG:
1. Public-data source preflight.
2. Fetch completed 1m, 5m and 15m candles.
3. Validate timestamps, history, candle bounds and freshness; fail closed.
4. Calculate indicators and a multi-timeframe candidate signal.
5. Model round-trip fees, spread and slippage separately.
6. Apply deterministic risk/cost gates.
7. Record either NO_TRADE or a paper-only candidate in JSONL.
8. Capture public history and run offline backtest / chronological holdout reports.
9. Save SQLite simulation ledger locally.
10. Expose report/log summaries for later assistant review.

Safety status:
- No live order placement, order cancellation, account login, wallet generation, private-key handling, transaction signing, transfer or withdrawal is implemented.
- No secret or user account credential has been committed.
- No real trade or wallet operation has been performed by the assistant.
- Live execution must remain disabled until an official supported route and current terms are verified, realistic venue costs are measured, out-of-sample results are reviewed, a signer is isolated, idempotency/reconciliation and kill switch are tested, strict caps are enforced, and the user approves the exact deployment.

## 4. Market data, conversion and cost assumptions

### Primary public candle source
- Binance Spot API docs: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
- Public endpoint used by the project: `GET https://api.binance.com/api/v3/klines`.
- Data symbol: `SOLUSDT`; completed candles only.
- Binance data are a reference feed, not executable Reku SOL/IDR prices.

### Daily quote-currency conversion reference
- Frankfurter docs: https://frankfurter.dev/
- Current endpoint recorded in code: https://api.frankfurter.dev/v2/rate/USD/IDR
- The collector stores the daily USD/IDR reference and its date in the CSV metadata sidecar.
- The simulator converts SOLUSDT OHLC values into IDR units before nominal sizing and P&L calculations. This keeps capital and price units consistent but is not a trading quote.
- USDT is treated as a USD proxy; the model does not capture USDT premium/discount, intraperiod FX variation, or Reku-specific SOL/IDR prices. A constant daily conversion is not proof of actual realized IDR P&L.
- The engine should reject absent/stale FX metadata unless the caller supplies a verified manual rate and descriptive source label.

### Cost model
- Scanner models round-trip fee, round-trip spread and round-trip slippage as separate parameters.
- Backtest/evaluator model entry fee, exit fee, round-trip spread and per-side slippage; spread and slippage are represented as adverse fill adjustments and exchange fees remain separate.
- Current defaults are placeholders for research/sensitivity only, not verified current Reku costs.
- Historical reports must not be called profitable evidence until measured venue fees/spreads, sufficient trade samples and representative regimes are considered.

## 5. Free-first API, repository and runtime inventory

- Binance public market data: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
- DEX Screener API (secondary pool/liquidity information): https://docs.dexscreener.com/api/reference
- Reku coin/status page: https://reku.id/status
- Reku historical API schema (current automated-order support and terms unverified; do not use private/undocumented endpoints): https://api.reku.id/v2/openapi.json
- Jupiter Developer docs (future Solana self-custody swap option only after funds are on-chain): https://developers.jup.ag/docs
- Helius docs/pricing (optional chain monitoring; free tier terms can change): https://www.helius.dev/docs and https://www.helius.dev/pricing
- Solana TypeScript SDK: https://github.com/anza-xyz/kit
- Solders Python/Rust toolkit: https://github.com/kevinheavey/solders
- LiteSVM local VM/testing: https://github.com/LiteSVM/litesvm
- Cloudflare Workers limits: https://developers.cloudflare.com/workers/platform/limits/
- GitHub Actions: https://docs.github.com/actions
- OJK PADK No. 3/2026: https://www.ojk.go.id/id/regulasi/Pages/PADK-Nomor-3-Tahun-2026-Penyelenggaraan-Perdagangan-Aset-Keuangan-Digital-termasuk-Aset-Kripto.aspx

Use is purpose-specific:
- Binance: reference candles only.
- Frankfurter: dated USD/IDR conversion reference only.
- DEX Screener/GeckoTerminal: secondary pool validation; not execution prices.
- Jupiter/Helius/Solana SDK/LiteSVM: future research options; they are not currently used to execute any transaction.
- Reku: user's existing fiat on-ramp candidate; automated order API permission/support remains unverified.
- GitHub Actions: offline tests and bounded research jobs, not a latency-critical scalping runtime.
- Termux: intended first user runtime; has not yet been tested from the user's device.

## 6. Tools / connector operations used

GitHub connector actions used during this project:
- `mcp__GitHub__search_installed_repositories_v2` — locate/check the installed repository.
- `mcp__GitHub__fetch` — inspect public GitHub API resources, PRs, trees, branches and Actions run metadata.
- `mcp__GitHub__fetch_file` — read repository files and exact commit/blob metadata.
- `mcp__GitHub__create_branch` — create the isolated development branch.
- `mcp__GitHub__create_file` — add project source, tests, documentation and workflows.
- `mcp__GitHub__update_file` — apply scoped file changes with expected blob SHA.
- `mcp__GitHub__create_pull_request` — open draft PR #178.
- `mcp__GitHub__update_pull_request` — refresh PR title/body/status summary.
- `mcp__GitHub__fetch_workflow_run_jobs` — inspect workflow/check step status.
- `mcp__GitHub__fetch_workflow_job_logs` — read test failure/success output.

No automated trading connector, Reku authenticated API or wallet signer was used. All GitHub writes were restricted to the feature branch and draft PR. The canonical root memory was not modified.

## 7. CI evidence and debugging history

### Confirmed green runs
- Run 40905304275: scanner/early paper-engine tests passed after fixing fixture assumptions.
- Run 40906164183: compile and offline tests passed for early capture/backtest/evaluation pipeline.
- Run 40907162332: latest explicitly inspected run at that checkpoint passed; log showed 25 tests in 0.077 seconds.
- Run 40908767558: CI passed after stop-gap / exit-friction test fixture fixes.
- Run 40910240121: latest run confirmed in this journal's initial inspection; commit `e20c3ce0b49969cbaa08706a44fc765e47e01db9`; compile and offline tests passed. Job log reported 32 tests in 0.082 seconds. See https://github.com/orionrayy/ai-determinism-engine/actions/runs/40910240121.

### Failures discovered and resolved
- Five-minute aggregation correctly discarded partial buckets; one test fixture incorrectly expected a partial bucket to be retained. The timestamp fixture was aligned to a complete bucket.
- CSV gap test used only 99 valid rows after skipping one candle and failed the minimum-history check before the intended gap assertion. The fixture was expanded to 101 iterations so 100 rows remained.
- New stop-gap tests initially called a test helper with unsupported `high`/`low` parameters; the tests were changed to explicit OHLC Bar instances.
- These issues were fixed without loosening the production data-integrity checks.

### Tool rejections
- Updating the large root `ORCHESTRATION_MEMORY.md` was blocked by a tool safety check without a reason code. Used the project-specific addendum instead.
- A later attempted direct patch of a code file was also blocked once; the same kind of scoped update later succeeded through the normal GitHub update-file operation. Do not claim the tool's hidden reason is known.
- No write was forced through a lower-level undocumented API.

## 8. Current repository checkpoint

- Branch: `feat/sol-sentinel-paper-trader-v1`
- PR: #178, draft/open, not merged: https://github.com/orionrayy/ai-determinism-engine/pull/178
- Latest head at time of journal creation: `e20c3ce0b49969cbaa08706a44fc765e47e01db9`
- Current project files and state must always be re-read from GitHub before future edits; do not assume this recorded SHA is still the latest after subsequent commits.
- No changes were made directly to `main` in this project work.
- The main canonical `ORCHESTRATION_MEMORY.md` remains unmodified; the addendum supplements it.

## 9. Outstanding work / exact next steps

1. Re-read current PR head and CI before the next edit; current verified baseline is run 40910240121.
2. Consider reviewing/merging the draft PR only after file review. The research workflow uses `workflow_dispatch` and `schedule`; GitHub requires the workflow to be present on the default branch for normal manual/scheduled invocation. It has not yet been end-to-end dispatched in this session.
3. Run `capture_history.py` on GitHub Actions or Termux to test real public API and Frankfurter network reachability; keep CSV/report artifacts private/local when they contain any user-specific context (historical public candles themselves are public).
4. Run `paper_engine.py` and `evaluate_splits.py` with captured data; verify the metadata sidecar's FX date and explicit source label.
5. Inspect per-segment sample sizes, trade count, net expectancy, profit factor, max drawdown, loss streaks, total fee/friction costs, and sensitivity to worse spread/slippage. No parameter tuning against the out-of-sample segment.
6. Add measured quote spread or a documented source-specific spread model; improve latency, no-fill, partial-fill and venue-price-difference simulation.
7. Verify current Reku official fees, minimum order, withdrawals and supported authenticated order API terms. Do not use undocumented endpoints.
8. Validate from Android/Termux. Report only non-sensitive logs.
9. Keep live trading disabled until official route, risk caps, wallet/signer isolation, durable idempotency, uncertain-outcome reconciliation, kill switch, and explicit user approval are in place.
10. Keep financial target as an aspiration. Never state that the IDR 3,500,000 target has been achieved unless independently verified through actual account/equity records.

## 10. Privacy, legal and operational constraints

This repository is public. Preserve user-requested project context without including private identifiers or credentials. KYC avoidance is not a design objective for bypassing legal requirements; self-custody/DEX activity does not imply exemption from applicable rules. IDR 300,000 is the reported starting capital; IDR 3,500,000 is a target only. The project must never trade on an LLM's confidence alone.


## 11. Latest checkpoint after continuity journal was added

Read-back verification at 2026-10-29:
- PR #178 remains open and draft, not merged.
- Latest branch head observed at this checkpoint: `8d8c91738b5f237e715206f662db417bbb509b63`.
- Latest verified CI run for that exact head: [40911083700](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40911083700), conclusion `success`. Compile and offline tests passed; log reports 32 tests in 0.099 seconds.
- Current `main` SHA observed: `d5672c84a0ae184c3a95d3f925925887ece71aa4`; the PR object still reported base SHA `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`. Reconcile this difference before any merge; no rebase/merge was performed.
- Repository tree includes `EXECUTION_LOG.md` plus the scanner, capture, paper engine, split evaluator, tests and project docs listed in section 2.
- This validates source compilation and offline unit tests only. The historical research workflow has not been run end-to-end on actual network data; no profitability or live trading claim is established.
- Subsequent edits will create a new head and may trigger new CI; re-read GitHub before relying on this snapshot.


## 12. Follow-up status refresh — 2026-10-29

A fresh read-back immediately before this note showed:
- PR #178: open, draft, not merged.
- Branch head before this documentation-only refresh: `6c6729338423b19ff16fadc09686f972d4a42665`.
- CI run [40911562946](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40911562946) passed for that exact head. Compile passed; 32 offline tests passed in 0.073 seconds.
- Current `main` SHA observed remains `d5672c84a0ae184c3a95d3f925925887ece71aa4`; PR base SHA remains `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`. Do not merge without reconciling.
- No end-to-end public history capture/backtest report has been run yet, no Termux runtime check has been done, and no live trading action has occurred.

This entry is documentation-only. It changes no scanner or trading logic. Since every new commit can trigger CI, the exact final head and its CI should be obtained from the PR/Actions links above rather than inferred from a fixed SHA stored in a log.


## 13. End-to-end historical research and cost sensitivity — 2026-10-29

### Endpoint failure and recovery
- First pipeline run `40913304159` failed during the public candle capture with HTTP 451 from the general Binance REST host. No backtest was run on that failed attempt.
- Official Binance docs identify `https://data-api.binance.vision` as the market-data-only host for public endpoints, including `GET /api/v3/klines`: https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md
- Collector and scanner now use `https://data-api.binance.vision/api/v3/klines`. Subsequent runs succeeded.

### Successful seven-day baseline
- Run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40914013908
- Artifact: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40914013908/artifacts/12915391700
- Data: 10,080 completed 1m candles, 2026-10-22 through 2026-10-29.
- FX reference: USD/IDR 18,158, dated 2026-10-29 (Frankfurter; USD used as a proxy for USDT, not a trading quote).
- Result: 0 candidate entries / 0 closed trades; initial/final simulation equity both IDR 300,000. This is no-trade/no-evidence, not breakeven performance.

### Successful thirty-day baseline
- Run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40914239047
- Artifact: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40914239047/artifacts/12914478212
- Data: 43,200 completed 1m candles, 2026-09-29 through 2026-10-29.
- FX reference: USD/IDR 18,158, dated 2026-10-29; used as a single reference rate across the dataset. This does not reproduce historic FX or USDT basis.
- Baseline strategy: 0 candidates, 0 trades, IDR 0 simulated P&L because no position was opened.
- Development segment: 25,920 bars; 119 otherwise technically eligible setups rejected by the modeled net-edge/cost gate.
- Validation segment: 8,640 bars; 43 such setups rejected by the cost gate.
- Out-of-sample segment: 8,640 bars; 34 such setups rejected by the cost gate.
- Across the 30-day split evaluation, the cost gate rejected 196 technically qualifying setups in total (119 + 43 + 34). The main Pro baseline modeled 72.11 bps round-trip friction: 42.11 bps fee reference + 10 bps assumed spread + 20 bps assumed round-trip slippage. With the minimum net-edge buffer of 25 bps, the required gross movement budget in the signal gate is 97.11 bps. These spread/slippage values are assumptions, not measurements.
- The reason counters overlap for bars that fail multiple technical checks; the specific `estimated_net_edge_below_threshold` count is only recorded when other technical criteria passed first.

### Current published Reku costs checked
- Official fee page: https://reku.id/fees/crypto (checked 2026-10-29).
- Pro buy: 0.1% trading + 0% PPN + 0.00555% CFX = 0.10555% (10.555 bps).
- Pro sell: 0.1% trading + 0.21% PPh + 0.00555% CFX = 0.31555% (31.555 bps).
- Approximate sum at equal nominal value: 42.11 bps before spread/slippage. Refresh the current fee page and actual order preview before any real use.
- CFX adjustment announcement, effective 2026-10-01: https://reku.id/information/informasi-penyesuaian-biaya-cfx

### Fixed cost-sensitivity comparison
- Code: `cost_sensitivity.py`; tests: `test_cost_sensitivity.py`.
- Research run with the sensitivity module: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40915191550
- Artifact: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40915191550/artifacts/12914798752
- Scenarios are predeclared; technical rules do not change, and no scenario is auto-selected by historical performance.
- Reku Pro baseline: 72.11 bps total modeled cost; zero candidates/trades.
- Pro with lower hypothetical execution friction: 57.11 bps; zero candidates/trades.
- Pro fee-only theoretical floor (zero spread/slippage): 42.11 bps; zero candidates/trades.
- Lightning fee-only theoretical floor (zero quote spread/slippage): 22.11 bps; 10 closed trades, net simulated P&L approximately IDR -3,893, net return -1.30%, max drawdown about 1.62%. The scenario omits real Lightning quote spread/slippage and must not be treated as executable or as a suggestion to trade.
- This result reinforces that loosening cost assumptions does not demonstrate an edge; the only scenario that generated trades in this dataset was an unrealistic fee-floor sensitivity and it lost money.

### Test and artifact status
- CI run after the sensitivity module: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40915091684; success.
- Run log reported 35 unit tests passing at the preceding sensitivity code checkpoint.
- A second, final research run with expanded per-scenario statistics was started at run `40915499872`; read its final status before describing it as complete. Its run history/artifact should supersede older sensitivity results once successful.
- Artifacts expire seven days after creation. Never infer a permanent report path from a transient artifact URL.


## 14. FX timeout failure and bounded-retry recovery — 2026-10-29

- Run `40915499872` failed after history capture: the Frankfurter request timed out while attaching the daily USD/IDR reference. The CSV metadata recorded the error; the backtest correctly failed closed and did not use a fabricated or stale FX rate. Because the backtest step failed, the expanded sensitivity summary from that run was not produced.
- Fixed `capture_history.py` so `fetch_quote_to_idr_reference` performs up to three bounded retries on transient timeout/network/rate-limit/server failures, with exponential short pauses. Non-retryable HTTP statuses and invalid/stale rates still fail closed.
- Added offline tests for transient timeout recovery and exhausting the retry bound. CI for the capture module/tests passed on run `40916160706`; the job reported the full project suite passing. Current retry-enabled full research workflow run: `40916209196`, launched on head `6ea5fd72945045c03ae2112fe3d9e870f2586c0a`; its final status must be read before claiming end-to-end success.
- The seven-day and earlier 30-day successful research runs remain valid evidence for their exact artifacts. The earlier 30-day sensitivity result from run `40915191550` remains usable as a preliminary report, but that workflow version did not publish all the extra per-scenario metrics added later.
- Current conclusion does not change: the Pro baseline has zero trades over the 30-day sample; the only trade-producing scenario observed was a deliberately unrealistic Lightning fee-only floor with zero spread and slippage, and that scenario lost roughly IDR 3,893 over 10 trades.


## 15. Live API investigation and adapter repair — 2026-10-30 WIB

### Confirmed fixes
- The latest unit-test CI is [run 40921862301](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921862301), passing **63 tests** on head `ca3e13391634609377617ef26ca6ef255059de3d`. It covers interval-validation order, chart freshness/fallback, symbol-based order-book queries, HTTP error-body diagnostics, and fail-fast handling of a missing history endpoint.
- A live read-only research pipeline completed successfully at [run 40921592237](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921592237), with artifact [12918995790](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921592237/artifacts/12918995790). This run's code head was `b3b8efe3b633118f52e59517aa9d26aebf79d839`; the current head adds a further test only.
- The current Reku order-book API rejected the legacy `?id=51` query with HTTP 400 and body `{"error":"invalid 'symbol'"}`. A bounded symbol query using `symbol=SOL_IDR` now succeeds. The validated snapshot returned 40 levels per side, best bid IDR 1,835,000, best ask IDR 1,840,000, and a spread of IDR 5,000 / **27.210884 bps**. Values are a timestamped public snapshot, not a current-after-checkpoint quote.
- The API returned HTTP 500 for the unsupported 1-minute chart probe. The adapter now tries documented 5-minute data first, then a 15-minute fallback. The successful snapshot returned 84 chart points at 5-minute interval, latest bar timestamp 2026-10-29 17:25 UTC; this limited chart response is not treated as 30-day history.
- The legacy POST `/v2/activitytrans` route returned HTTP 404. The history collector now fails fast on that definitive 404 instead of repeating the unsupported request for each month, and records `error_category=endpoint_not_found`. Native Reku historical activity remains **unavailable/unverified**; no fabricated replacement candles are created.

### Research result and interpretation
- The full research workflow succeeded with 43,200 public Binance SOLUSDT one-minute candles covering 2026-09-29 through 2026-10-29, and a USD/IDR reference of 18,158 dated 2026-10-29.
- The baseline and chronological 60/20/20 segments still produced **zero trades**. Zero return here means no position opened; it is not evidence of profitability or safety.
- The live spread is used only as a fixed sensitivity reference across the historical window, not presented as a historical spread series. With the Reku Pro fee components (42.11 bps), observed spread snapshot (27.21 bps), and assumed 20 bps round-trip slippage, the modeled total is about 89.32 bps, or about 114.32 bps including the configured 25 bps minimum net-edge buffer. The current-spread scenario also did not produce a trade.
- No live order, cancellation, transfer, withdrawal, wallet operation, or credential use occurred. Live execution remains disabled.

### Current GitHub state and next gates
- Current branch head: `ca3e13391634609377617ef26ca6ef255059de3d`.
- [PR #178](https://github.com/orionrayy/ai-determinism-engine/pull/178) is still draft/open and unmerged.
- The PR records base SHA `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`, while the observed current `main` SHA is `d5672c84a0ae184c3a95d3f925925887ece71aa4`. Reconcile this difference before any merge.
- Remaining blockers: verify a current documented source for native Reku history; validate the scanner/backtest on the user's Termux device; improve strategy research because the tested rules still generate no trade; and keep live execution blocked until provider authorization, measured costs, risk controls, reconciliation, kill switch, out-of-sample evidence, and explicit approval are satisfied.
