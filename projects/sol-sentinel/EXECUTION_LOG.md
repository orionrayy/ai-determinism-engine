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

## 16. Spread, minimum-order and backtest telemetry patch — 2026-10-30 WIB

### Baseline and execution identity
- Commit immediately before patch: `29488b5ecc55fe080980a38bd900753c7d70a135`.
- Patch commit: `7d74015ecb99c0f1768baddd40b1643399c5bea2`, branch `feat/sol-sentinel-paper-trader-v1`.
- Automated code test workflow: [40928032367](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032367), conclusion `success`; PR-triggered duplicate [40928041385](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928041385), conclusion `success`. Both logs report compilation completed and `Ran 70 tests`, `OK`, zero test failures.
- Research workflow [40928032346](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032346) was running when this log entry was initially authored. Record its final result only after checking the completed run and artifact.

### Spread incident: cause and invariant
- Historical report `198.0198 bps` for bid 99 / ask 101 corresponds to `(ask - bid) / ask * 10,000`. It does not use the price midpoint.
- The project invariant is conventional midpoint spread: `mid = (bid + ask)/2`; `spread_bps = (ask - bid)/mid * 10,000`. With bid 99 and ask 101, spread 2 / midpoint 100 gives **200 bps**. The currently inspected pre-patch implementation already used the correct midpoint denominator; therefore the historical discrepancy is attributed to an older denominator/fixture mismatch, not an active defect in the current formula. The patch hardens the input contract and adds regression coverage instead of changing the expected 200 bps.
- The test now rejects missing/nonnumeric, boolean, zero/negative, non-finite and crossed quotes; accepts numeric strings and a zero-spread quote; and exercises extreme finite magnitudes. A prior captured sample 1,835,000 / 1,840,000 calculates 27.210884 bps using midpoint 1,837,500. This is not a current market quote.

### Code changes
- `reku_market.py`: safely converts public quote values, disallows booleans, handles conversion errors explicitly, and computes midpoint using a half-sum to reduce overflow risk.
- `test_reku_market.py`: added five focused regression tests for bps denominator, invalid price classes, zero spread / numeric string compatibility, and extreme floats.
- `paper_engine.py`: added a configurable minimum simulated order notional, a fail-closed notional predicate, explicit signal rejection-combination telemetry, entry attempts/fills/size rejections, daily-loss-lock blocked-bar counts, gross P&L, average win/loss, expectancy, average hold, and `null` win-rate/expectancy when no trades exist.
- `evaluate_splits.py` and `cost_sensitivity.py`: propagate diagnostic fields so JSON research artefacts and each split retain counts and metrics.
- `test_paper_engine.py` and `test_cost_sensitivity.py`: regression coverage for minimum-notional assumptions and the semantic treatment of zero trades.
- Simulated minimum notional defaults to Reku Pro Rp30,000. The predeclared Lightning fee-floor scenario uses Rp5,000. These are modeled venue floors from the sources below, not proof of order precision/minimum rules for another venue.

### Official source records checked on 2026-10-30
- Reku fee table: https://reku.id/fees/crypto — Pro buy components 0.1% + 0.00555% CFX; Pro sell components 0.1% + 0.21% PPh + 0.00555% CFX.
- Reku minimum order: https://reku.id/help/batas-minimal-rupiah-dalam-order-jual-beli — page states Rp30,000 for Pro and Rp5,000 for Lightning (page dated 2025-11-19).
- Binance Spot REST API: https://developers.binance.com/en/docs/products/spot/rest-api — timestamps are milliseconds by default and market-data-only endpoints are documented separately.
- Binance symbol filters: https://github.com/binance/binance-spot-api-docs/blob/master/filters.md — filters include price tick size, lot step and minimum notional; this is venue-specific and must not be transferred to Reku.
- Reku legacy public API docs: https://api.reku.id/ — treated as legacy documentation; runtime behavior and native history availability must be reverified.
- These are source records, not authenticated order-path approval. Binance candles remain reference history, not executable Reku price history.

### Proven telemetry interpretation and limits
- Previous 30-day historical run [40921592237](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40921592237) captured 43,200 Binance SOLUSDT 1m candles (2026-09-29 to 2026-10-29) and had zero trades in the baseline and each 60/20/20 segment. Its baseline report logged 25,919 signal evaluations and 391 warm-up bars; rejection counts included no 1m breakout (23,913), missing 1m relative-volume confirmation (20,740), unconfirmed 5m momentum (16,998), and unconfirmed 15m trend (15,313). Counts are multi-label rejection events and can exceed the number of evaluations; they are not mutually exclusive.
- These data support that the stacked filters frequently rejected bars, but do not establish that any single filter alone caused zero trades. The current research run must be read to establish whether post-patch counters remain consistent.
- No live order, cancellation, transfer, withdrawal, wallet operation or credential use occurred. No Termux device or Android process was accessed. PR #178 remains draft/unmerged.


## Final verified research result — 2026-10-30 (SOL_SENTINEL_SESSION_2026-10-30_VERIFIED_FINAL)

The preceding section supersedes earlier notes that the latest research workflow was still running. Research run #25 completed successfully at https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032346 and published artifact #12921441578. Current finding: 43,200 one-minute Binance SOLUSDT candles; hash \`ccc89cec753644f11248bc75ac8b7b13fd9042ef4eb1d5a72074a75cdb9e8b26\`; conversion reference USD/IDR 18,158 dated 2026-10-29. The 60/20/20 segments contain 25,920 / 8,640 / 8,640 bars and each shows zero candidate signals, entry attempts/fills and closed trades. Do not report zero return as profit or safety; expectancy, win rate and PF are undefined with zero trades. Full-window multi-label counters and combo details appear in the same-named checkpoint in \`ORCHESTRATION_MEMORY_ADDENDUM.md\` and the reviewable audit \`projects/sol-sentinel/RESEARCH_AUDIT_2026-10-30.md\` in PR #180.

Run #268 at https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928041385 passed 70 tests. Midpoint spread behavior is consistent with its fixture: bid 100 / ask 102 is about 198.0198 bps; bid 99 / ask 101 is 200 bps. The code now rejects invalid/crossed quotes and tests extreme values. Reku Pro modeled costs are 72.11 bps with assumed spread/slippage; a one-snapshot sensitivity is 89.469781 bps; the theoretical Lightning fee-only floor produces 10 trades but loses IDR 3,893.219 net. Native Reku activity history remains blocked by HTTP 404. Keep mode \`RESEARCH_ONLY\` / \`INSUFFICIENT_EVIDENCE\`; no live orders occurred, and Termux remains untested.


## Current checkpoint after ledger / candle-time fixes — 2026-10-30 (SOL_SENTINEL_SESSION_2026-10-30_POST_AUDIT_FOLLOWUP)

This is a documentation follow-up to the earlier “research still running” entry. Research run #27 completed successfully on engine commit \`833e57a13022c3efe6ff5115df2b2e0697072817\`, dataset SHA-256 \`6e140e14be2e05d30819b96e90e37372ee13df55bb7d4f31fc9c9382bef86c14\`, 43,200 bars from 2026-09-29 18:23 UTC to 2026-10-29 18:22:59.999 UTC. Run #27 artifact #12922686248; output shows zero candidates/trades in the full backtest and every 60/20/20 split, with null expectancy/win-rate/PF. Reku Pro costs remain modeled at 72.11 bps including assumed spread/slippage (97.11 including the 25 bps edge buffer), and the single-spread snapshot scenario is 89.469781 bps. The zero-spread/slippage Lightning floor remains negative by about IDR 3,893.22.

After the code-only ledger fix, the candle-time/OHLCV guard, and three hand-calculated EMA/RSI/ATR fixtures, branch head \`8fb11f7081a355958add549052b1ecec90464423\` passed [Tests #278](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40930666859): 79 tests in 0.141s, 0 failed. The research workflow checked the engine commit immediately preceding the test-only commit; that lineage is documented in the audit. No live orders were used; Termux has not been tested; native Reku activity history remains unavailable from the legacy 404 endpoint.


## Final branch reconciliation and evidence pointer — 2026-10-30 (SOL_SENTINEL_SESSION_2026-10-30_FINAL_RECONCILIATION)

Final verification: branch head \`5f18753d4f5f059195b9b2e815627652e6564ca8\` passed 79 tests in [workflow #280](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40931067383); duplicate #279 also passed. Research #27 succeeded on code commit \`833e57a13022c3efe6ff5115df2b2e0697072817\`, dataset hash \`6e140e14be2e05d30819b96e90e37372ee13df55bb7d4f31fc9c9382bef86c14\`, artifact #12922686248, with zero trades in all splits.

Reconciliation is **not cleared**: compare current \`main\` \`d5672c84...\` against PR #178 head reports diverged history, merge base \`eaa864a...\`, 167 commits ahead and two behind, 27 file changes. GitHub's \`mergeable=true\` is not a merge authorization. PR #178 stays draft/open. The latest canonical memory and detailed audit are available through draft PR #180. No force-push, merge, or main branch modification occurred.


## Final compare refresh — 2026-10-30

The last fresh GitHub compare, performed after all current documentation commits were pushed, reports current \`main\` \`d5672c84a0ae184c3a95d3f925925887ece71aa4\` versus feature head \`3d2054da51ebb34f98380c50257e61735e650475\`: \`status=diverged\`, merge base \`eaa864a554c2c995ce73bd9f6b4d07206ecc56c5\`, \`ahead_by=168\`, \`behind_by=2\`, 27 files in PR #178. This supersedes the earlier point-in-time count of 167 commits ahead recorded in previous notes. \`mergeable=true\` is only API metadata and does not authorize merge; keep PR #178 draft and obtain review of the full history delta before integration. No rebase, force-push, merge, or change to main was performed.


## Continuation update — native INDODAX history and cross-venue safe backtest (2026-10-30 WIB)

### Verified implementation and CI

- Feature branch checkpoint before this log append: `e25b2990433673d3a79db8912c504e96b4bce204`.
- Latest verified code/unit-test check at that checkpoint: [SOL Sentinel Tests #40948901551](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40948901551), successful, **135 tests passed, 0 failed**. The test suite is offline except for separate research workflow steps; this CI success alone is not live-trading readiness.
- New files: `execution_adapters.py`, `test_execution_adapters.py`, `PLATFORM_ADAPTER_DECISION_2026-10-30.md`, `capture_indodax_history.py`, `test_capture_indodax_history.py`.
- The adapter exposes separate market data, capital-source metadata, custody, transfer-planning, order execution, signer, chain submitter and ledger contracts. Only public market data, mock fixtures, in-memory ledger and deterministic paper execution exist. There is no live order adapter or signer.
- Native IDR denomination integration is present in the paper engine and chronological evaluator: capture metadata identifies venue/pair/currency, SHA-256 and data status are checked when present, native IDR bars use FX factor 1.0, and reports state source denomination and fee model. Native INDODAX data require explicit `--fee-model indodax-custom`, entry/exit fees, round-trip spread, slippage, minimum order value and a source label. The Reku fee-sensitivity presets reject native IDR files instead of silently reusing the wrong venue's cost model.
- Guard fixes/regressions: conflicting duplicate market/history candles, invalid fraction/zero-value Reku trade rows, stale or invalid quote/route data, non-boolean verified-network flags, zero-value transfer remainder, missing/stale transfer fee evidence, tick/lot/minimum validation, same-key different-order replay, and live-mode gate remain fail-closed.

### Public-data smoke test and strategy evidence

- Read-only native INDODAX capture passed in [research workflow #40948205968](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40948205968) on workflow/code checkpoint `e7cc7ab2a810028af5a13d43c011674792a08a57`: 30 days / 43,200 contiguous 1m native SOL/IDR candles, SHA-256 `9dd63e357e3a8f4ba168bccd65edf3fe621d3758884a97549e8425c812ab2316`, range 2026-09-29 20:22:00 UTC to 2026-10-29 20:21:59.999 UTC. `data_status=OK`, source `https://indodax.com/tradingview/history_v2`, `fx_conversion_applied=false`. This is a successful public endpoint smoke test from GitHub's runner, not a user's device or account/deposit verification.
- On the same workflow family, native Reku activity history still returned HTTP 404 for legacy `/v2/activitytrans`. Reku's separate public quote snapshot returned `data_status=OK`; that does not restore the unavailable history endpoint.
- The full 30-day Binance SOLUSDT research pass in [run #40948530609](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40948530609) succeeded, dataset SHA-256 `08cb2e654c42e198d71817d2e92d7c354f4ef0206a7e9a352a2eff4b486395ae`, 43,200 bars. The baseline and all fixed 60/20/20 segments still showed zero candidates/trades; the theoretical Reku-Lightning fee-only floor was the exception with 10 simulated trades and approximately IDR -3,893.22, assuming zero spread/slippage. No strategy parameters were relaxed. This native INDODAX capture was not yet backtested with INDODAX fees.
- A new full research run on the latest code head was queued/running when this note was written. Do not claim a new full research result until its exact run and artifact have completed and been inspected.

### Current blockers and handoff

- PR #178 remains draft/open/unmerged. At the most recent compare, the branch was diverged from `main`, 235 commits ahead / 2 behind, merge base `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`, 34 changed files. This includes substantial unrelated/recent history; do not merge/rebase/reset/force-push blindly.
- PR #180 remains draft/open/unmerged and carries the canonical memory/audit branch. Append this checkpoint to `ORCHESTRATION_MEMORY.md` and `RESEARCH_AUDIT_2026-10-30.md` on the docs branch as the session's continuity record.
- Do not move funds. Before native INDODAX backtest, read the actual current profile fee components and select Lite/Pro minimum explicitly. Current official help lists minimum IDR 10,000 on Lite and IDR 25,000 for direct Pro (with Pro-interface orders IDR 10,000–24,999 routed through Lite); market metadata is not a substitute for mode constraints. Current network-specific SOL deposit/withdraw fees and the destination deposit screen must be checked before any separately approved transfer.
- Termux/Android compatibility remains untested. Current strategy remains `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`.


## Live INDODAX depth probe and latest 30-day capture — 2026-10-30 WIB

- Latest completed research workflow [#40950218320](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950218320) succeeded on workflow head `89a8d7181954a1524379d3b03fa77e703eb9331d`. It ran compile/offline tests (137 passed) and the public research capture. Latest exact feature-head test run after subsequent documentation changes: [#40950649418](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950649418), 137 passed, 0 failed.
- Read-only native INDODAX SOL/IDR capture: `data_status=OK`, 43,200 contiguous 1m bars, 30-day range 2026-09-29 20:36:00 UTC through 2026-10-29 20:35:59.999 UTC, dataset SHA-256 `c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30`, no FX conversion.
- Public INDODAX depth probe: `market_quote_status=OK`, `market_quote_error=null`, quote observed at 2026-10-29T20:37:47.062073Z, bid Rp1,777,113, ask Rp1,779,923, mid Rp1,778,518, spread ~15.7997 bps. Orders remain disabled. Earlier run #40950150440 showed bid Rp1,775,886 / ask Rp1,780,450 / spread ~25.6669 bps. These are two point-in-time observations, not an historical spread curve or guaranteed fills.
- The run also reports current public market filters: price tick Rp1, quantity step 1e-8 SOL, market minimum notional Rp10,000, minimum base quantity 0.00563136 SOL. The current account/mode minimum may be higher (official help lists Rp25,000 direct Pro / Rp10,000 Lite); current account all-in fees remain unverified here.
- Strategy output continues to show zero candidates/trades on the Binance SOLUSDT baseline and each fixed 60/20/20 split; expectancy, win rate and profit factor remain undefined. The fee-only hypothetical Reku Lightning floor simulates 10 trades at net about -Rp3,893.22 with zero spread/slippage, so it is not executable evidence. Reku history route `/v2/activitytrans` still returns HTTP 404; the failure is isolated as an optional non-fatal collector result, while separate public market data is OK.
- All probes use unauthenticated public market data. No account API, transfers, orders, wallet signing or secrets were used. Current release state remains `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`; Termux is still untested.

## Session continuation — 2026-10-30: native INDODAX replay, Termux diagnostics, and partial-fill simulation

### Verified native INDODAX evaluation

- Research workflow #40993486012 failed at the first newly-added replay command because shell line continuations were doubled; the command therefore received no --csv. Compile/tests and preceding capture stages ran. This defect was corrected and successful replay #40993527618 completed on commit b1ff4fe57bfb3118e31bc5314b21c5baa91732da; artifact #12949927917.
- The locked CSV from artifact #12931446637 was re-downloaded and checked. SHA-256 c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30; 43,200 contiguous one-minute candles; native INDODAX SOL/IDR, IDR denomination, no FX conversion. Timestamp continuity, close-time boundaries, finite OHLCV and OHLC ordering passed.
- The original artifact's backtest.json and evaluation.json were Binance SOLUSDT with Reku assumptions. They were not a native INDODAX result. The newly-added native replay is the first valid fee-separated INDODAX replay.
- Frozen replay #40993527618, direct Pro min order Rp25,000: taker snapshot cost = 20.555 + 41.555 + 15.799671 + 2×5 = 87.909671 bps; taker stress = 20.555 + 41.555 + 25.6669 + 2×10 = 107.7769 bps; maker sensitivity = 10.555 + 31.555 + 15.799671 + 2×5 = 67.909671 bps. All three had zero candidates, attempts, fills and trades on dev/validation/OOS. P&L zero reflects no positions; expectancy/win-rate/profit-factor undefined.
- Correct the earlier UI summary arithmetic: 82.11 and 72.11 bps were incorrect totals. The correct frozen-data totals are 87.909671 and 67.909671 bps respectively. Stress 107.7769 bps was correct.

### Same-run replay and new dataset

- Research workflow #40994319663 succeeded on commit 92810145ac60487a87730c977f3b26c6bb3a4ecb; artifact #12950538188. The workflow no longer downloads historical artifact #12931446637 by fixed ID, avoiding the seven-day artifact-retention dependency. It evaluates the native CSV captured within the same run.
- New dataset SHA-256 909b24977537110d2bd0ca114d5715d29646eb36b6ed9a268bf9470b64a05c59; 43,200 one-minute bars from 2026-09-30T02:54:00Z to 2026-10-30T02:53:59.999Z. Native INDODAX SOL/IDR; no FX conversion.
- Current snapshot at 2026-10-30T02:55:48.533456Z: bid Rp1,729,198, ask Rp1,729,703, spread 2.920003 bps. It is one point-in-time quote, not a historical spread distribution.
- Same-run current spread taker = 20.555 + 41.555 + 2.920003 + 2×5 = 75.030003 bps. Taker stress = 107.7769 bps. Maker sensitivity = 10.555 + 31.555 + 2.920003 + 2×5 = 55.030003 bps.
- Taker-current and taker-stress: zero candidates/trades in all three splits. Maker sensitivity: one development candidate/fill/closed trade only; validation and OOS zero. It ran 9 one-minute bars (entry 2026-10-15T14:57:00Z; exit 15:05:59.999Z), quantity 0.07239403390849532 SOL, net +Rp466.818, exchange fees Rp574.698, modeled execution friction Rp175.991. One development trade cannot substantiate an edge; its 100% win-rate and infinite profit factor are non-informative. Maker queue, post-only, and actual maker fills are not modeled; the one snapshot spread was held constant across historical rows.
- Current-run overlapping filter-rejection events (taker-current): breakout 40,510; relative volume 35,882; 5m momentum 28,299; 15m trend 25,989; 5m RSI 23,415; 1m exhaustion/missing 3,142; net-cost edge gate 143. Counts overlap and are not unique rejected bars. No strategy parameters were relaxed.

### Execution/tooling changes

- Added projects/sol-sentinel/INDODAX_FEE_MODEL_2026-10-30.md with official Pro all-in fee schedule, mode minimums, tick/quantity metadata, cost decomposition, timestamped quote observations, and explicit Lite/account/withdrawal uncertainty. Official Pro IDR rates used: maker buy 10.555 bps; taker buy 20.555; maker sell 31.555; taker sell 41.555; direct Pro minimum Rp25,000. All-in values include CFX/sell tax, so do not add them twice.
- Added standard-library-only projects/sol-sentinel/termux_doctor.py and tests. It checks Python/Termux detection, stdlib, memory/disk, core project compile, SQLite close/reopen/idempotent replay, optional public INDODAX pair catalogue GET, and optional streaming native dataset/hash/sidecar validation. Runbook has commands. No user's Android handset was accessed.
- Tests at commit 50c0c00dc4bc5a78ab212dc20bd4894fb99783e6 passed 146 tests in #40994692715. The later Termux-doctor smoke step without network (python projects/sol-sentinel/termux_doctor.py --skip-network --json) passed in #40994852707 on commit 8ceeb6a2feb96691d9d323ced053c3a294869174.
- Added top-of-book bid/ask quantities to MarketQuote, best-level quantity aggregation to the INDODAX depth adapter, PARTIALLY_FILLED_SIMULATED state, and paper fills capped to visible best-level quantity. Tests cover partial size, zero size and negative size. It does not sweep deeper levels, resume remaining quantity, or model maker queue/partial-fill timing.
- The current-run public Reku snapshot was OK with bid Rp1,725,000, ask Rp1,730,000, spread 28.9436 bps and 40 levels each side. Its legacy activity-history endpoint still returned HTTP 404. This does not prove a Reku account purchase API. Reku stays an on-ramp architecture candidate, not the required bot trading venue.
- Jupiter remains a public quote-only probe. It does not send taker, sign or submit; it still lacks comprehensive expiry, slippageBps, route/liquidity and priority-fee validation. A quote is not a fill.
- The scanner JSONL/watch loop is read-only observation, not a complete forward-paper portfolio monitor with persistent balances, venue-aware order lifecycle, partial remainder, risk-state recovery and reconciled P&L.

### Safety, repository and handoff

- No real orders, purchases, transfers, deposits, withdrawals, signatures, signers, private keys, seed phrases, passwords or API secrets were used or saved.
- INDODAX spread and slippage are not statistically validated. The spread references are a small number of single quotes; slippage 5/10 bps per side are declared assumptions. Lite-specific all-in fees, the user's account fee display, network-specific SOL withdrawal fees/minimums and both-side destination acceptance remain unverified.
- PR #178 remains draft/open/unmerged. At the following compare snapshot it was diverged from main; do not merge/rebase/reset blindly. PR #180 also remains draft/open/unmerged. The root canonical memory in main is not updated.
- Current status remains RESEARCH_ONLY / INSUFFICIENT_EVIDENCE. No OOS trade exists for any scenario.
- Next gates: build repeated read-only INDODAX depth sampling; model depth walk and remaining partial orders; verify account mode fee preview; complete Jupiter V2 field/expiry validation; validate supported network & deposit/withdrawal conditions without moving funds; test Termux on actual handset; run forward-paper ledger/monitor for a meaningful sample; then re-evaluate OOS evidence without retuning.

## Latest repeat-capture checkpoint — workflow #40994685989 (2026-10-30)

- Newest successful same-run native INDODAX research workflow: #40994685989, commit 50c0c00dc4bc5a78ab212dc20bd4894fb99783e6, artifact #12951295388. This run supersedes the one-trade maker result in #40994319663 for current status; the earlier single dev trade was not reproduced on the next rolling 30-day window.
- Dataset SHA-256 d76434d4c9019249e924e72c5b05d0b200c0dfe3c4709c02a6d821fa0b22327c; 43,200 contiguous one-minute native INDODAX SOL/IDR candles from 2026-09-30T03:01:00Z through 2026-10-30T03:00:59.999Z; fx_conversion_applied=false.
- Current-run INDODAX quote at 2026-10-30T03:02:06.143046Z: bid Rp1,737,289, ask Rp1,739,906, midpoint Rp1,738,597.5, spread 15.0523626084 bps. Market filters: tick Rp1, quantity step 1e-8 SOL, public API notional Rp10,000, minimum base quantity 0.00576894 SOL. These remain point-in-time public metadata/quote values.
- Current spread-snapshot Pro taker modeled total = 20.555 + 41.555 + 15.052363 + 2×5 = 87.162363 bps. Pro taker stress remains 107.7769 bps. Pro maker snapshot sensitivity = 10.555 + 31.555 + 15.052363 + 2×5 = 67.162363 bps.
- All three scenarios (Pro taker current snapshot, taker stress, maker current snapshot) produced zero candidates, zero entry attempts, zero simulated fills, zero opened/closed positions and zero trades in development, validation and out-of-sample. P&L zero is no-trade accounting, not profit. Expectancy, win rate and profit factor are undefined.
- Taker-current overlapping filter counts: 1m breakout 40,506; 1m relative volume 35,873; 5m momentum 28,306; 15m trend 25,981; 5m RSI 23,415; 1m exhaustion/missing 3,135; estimated net edge cost gate 143. Counts overlap.
- The previous run #40994319663 had one maker-only development trade (+Rp466.818 net) and no validation/OOS trades; the latest rolling window #40994685989 did not reproduce it. This is direct evidence that the lone trade is not stable/reproducible and cannot support an edge claim.
- Latest public Reku snapshot in this run: bid Rp1,725,000 / ask Rp1,730,000, spread 28.94356 bps; 40 depth levels each side. The 5m public chart returned 75 points with latest point age about 77 seconds; chart timestamp/interval semantics remain unverified. Reku activity-history legacy route remains HTTP 404. No account purchase API is established.
- CI offline suite #40994692715: 146 tests passed. Doctor smoke #40994852707: 146 tests passed and Termux doctor --skip-network --json succeeded. This validates GitHub runner only; user's Android/Termux device remains untested.
- Status remains RESEARCH_ONLY / INSUFFICIENT_EVIDENCE. No parameter relaxation, real order, purchase, transfer, signer, deposit or withdrawal.
## Chronology and look-ahead code review — 2026-10-30

- Inspected evaluate_splits.py: ratios are fixed 60/20/20; slices are disjoint and chronological; each segment checks one-minute continuity; each segment is evaluated independently with no parameter optimizer. The segment report warns that capital is reset independently per split.
- Inspected paper_engine.py signal path: 5m/15m bars are aggregated only from complete contiguous minute buckets; higher-timeframe lookups use bisect_right on close_time <= the 1m decision-bar close, so incomplete future 5m/15m bars are not consumed.
- Feature calculations inspected for this path use prior windows: relative-volume baseline uses the preceding 20 bars; breakout prior-high uses preceding 20 highs. Signals are decided at the 1m bar close and candidate entries are deferred to the next 1m candle open, not filled on the signal candle close.
- Intrabar exits use a documented conservative stop-first assumption if stop and target are both touched in one OHLC candle. End-of-data open positions are force-closed at the last close as an explicit simulation convention.
- No look-ahead defect was found in these inspected paths. Limitations remain: indicator warmup is rebuilt inside each isolated segment (discarding initial warmup bars rather than using prior-segment bars), OHLC cannot reveal exact intrabar path, fill/slippage/queue assumptions are not calibrated, and the review is code-path inspection rather than a formal proof.

## INDODAX multi-level depth collector — engineering checkpoint (2026-10-30)

- Added `projects/sol-sentinel/indodax_microstructure.py` and `test_indodax_microstructure.py`. The module is Python standard-library only and uses the unauthenticated public `GET https://indodax.com/api/depth/solidr` endpoint.
- It parses up to 500 ordered price/quantity levels per side; rejects empty, malformed, non-finite, non-positive, duplicate/unsorted and crossed books; accepts a locked zero-spread book; computes midpoint quoted spread in bps; and preserves an exchange timestamp only if the payload supplies an integer millisecond timestamp.
- SQLite stores observations and every captured depth level separately, with a timestamped content-derived observation key, plus request-attempt status/error and latency. Identical same-time book observations are idempotent; a changed book or later timestamp remains a new observation. WAL and transaction boundaries are enabled.
- Summary computes mean, median, nearest-rank P95 and population standard deviation for quoted spread and bid/ask IDR depth inside the configured midpoint band. SQL performs distribution and depth aggregation; the complete archive is not loaded into RAM for summaries.
- Sampling defaults to 30 snapshots at 10 seconds, requires >=1 second between requests, caps one run at 10,000 samples and 500 levels/side, caps HTTP response at 1 MB, and stops after three consecutive failures. Only public GET requests are used.
- Local validation on the authored collector: `py_compile` succeeded; all 22 new parser/client/SQLite/collection tests passed, with `ResourceWarning` treated as an error. These are fixture/offline tests. No live endpoint data was captured, so empirical spread/depth distribution fields remain empty until a real collection is run.
- Updated Termux doctor module compilation list and CI compile step. GitHub CI will be the authoritative full-suite validation for the new branch head. Android-device execution is still untested.
- Safety boundary unchanged: no order, account, wallet, signer, deposit, withdrawal, transfer or on-chain submission capability was added.

## Current acceptance impact

- Microstructure collector implementation and offline regression tests: implemented and locally validated.
- Empirical spread/depth distribution: pending actual public snapshots.
- Multi-level paper execution, residual lifecycle, persistent forward-paper portfolio/risk monitor, Jupiter full response validation and actual Termux-device validation: remain outstanding.


## Multi-level paper execution engine — checkpoint 2026-10-30 (Asia/Jakarta)

- Added `projects/sol-sentinel/depth_execution.py` and `test_depth_execution.py` as a separate, Python-standard-library paper-only subsystem. Existing `MarketQuote` and `PaperExecutionAdapter` remain unchanged.
- Source design: immutable content-addressed `DepthSnapshot`; Decimal book levels; market/limit requests; GTC/IOC; explicit order states; per-level `Fill` rows; cumulative VWAP; stable order/client IDs; timestamped fee and precision contract.
- Execution walks every observed opposing level up to the remaining order quantity and limit. The snapshot's liquidity is shared across all paper orders using reserved/consumed quantities; identical order/snapshot cycles are idempotent. No liquidity beyond the snapshot is invented.
- SQLite stores paper account, orders, immutable snapshots and levels, per-level consumption, fill records, append-only events, and unique execution cycles. Each matching cycle uses one `BEGIN IMMEDIATE` transaction containing fills, liquidity consumption, order state, events and portfolio changes. On pre-commit fault injection the complete write rolls back. After-commit lost-response retries return the stored cycle.
- GTC residuals remain open for later eligible snapshots. IOC residuals are cancelled after their single attempt. Explicit cancel/expiry preserve historical fills. Portfolio cash, fees, SOL inventory, average cost and realized P&L are reconstructed from immutable fills and checked against persisted balances. Recovery fails closed on detected tampering/inconsistency rather than silently overwriting values.
- Added 34 offline regression tests, including collector snapshot adapter compatibility and idempotent create-order retry when the same request terms are retried with a different attempt timestamp. Local validation: `py_compile` passed; `python -W error::ResourceWarning -m unittest` reported 34 tests, OK.
- GitHub CI has not yet run on the resulting commit at the time of this log append. Do not report GitHub validation until the exact head workflow is completed.
- Limitations: no maker queue/post-only simulation; no native exchange matching-engine parity; no automatic SQLite archive-to-execution streaming; no unattended forward-paper cycle; no daily-loss/cooldown/circuit-breaker full risk integration; no actual Android/Termux device run; no live order book sample was generated by these fixture tests.
- Safety: module has no exchange execution, balance access, wallet, signer, deposit, withdrawal, transfer or chain-submit interface. No live financial action occurred.


## Shared depth-liquidity pool hardening — 2026-10-30

- Follow-up source audit identified that timestamp-specific snapshot IDs alone could let identical side-book levels be counted as fresh liquidity on every polling sample. Added independent bid/ask pool fingerprints keyed by venue, symbol, side and the complete ordered price/quantity sequence.
- Snapshot IDs remain timestamped and immutable for audit; per-side pool consumption is shared across exact unchanged observations, even when observed_at changes. Bid-side changes do not reset ask-side liquidity; a changed side-book fingerprint creates a new observed pool.
- Added durable `liquidity_pools`, `liquidity_pool_levels`, and `snapshot_liquidity_pools` tables. Per-snapshot consumption remains auditable. Reconciliation now verifies both snapshot-level fill totals and aggregate side-pool consumption against all fill rows using that pool.
- Startup backfills side-pool mappings for existing snapshots. It aggregates prior snapshot consumption for identical side books and fails closed if total use exceeds captured size. A restart-migration regression test exercises this path.
- Added three pool tests: unchanged ask side at a later timestamp cannot be consumed twice; a bid-side-only change does not reset ask liquidity; and a changed ask-side depth creates a distinct pool. The migration test verifies previous per-snapshot fills are carried into the shared pool.
- Local suite now reports 38 tests, OK, with `ResourceWarning` promoted to error. GitHub CI on the follow-up commit is pending; the previous commit's CI passed 202 tests.
- This policy is intentionally conservative because the data feed does not contain trade-level queue depletion/replenishment evidence. An exact side-book fingerprint change is treated as a new observed liquidity state, not proof that the matching engine would refill the same orders.

 
## 2026-10-30 — v1.3 integrated, activated and merged

- Exact source branch `feat/sol-sentinel-v1.3-integrated` was built from current `main`, using the audited Sentinel implementation files and latest documentation/canonical memory. It had 1 commit ahead / 0 behind (48 files). The diverged historical PR #178 was not merged.
- Clean PR #181 merged on 2026-10-30 as `58ec3df095c60ccc6868bf0cbccb3c18384419eb`. Main push Orchestrator Tests #41007000945 succeeded.
- PR CI #41006890361: 216 tests passed, 0 failed; compile passed; Termux Doctor 6 PASS / 1 WARN / 0 FAIL on Ubuntu; immediate activation job succeeded. Orchestrator CI #41006890245 also passed.
- Activation artifact #12955034631; source SHA `5bf6c2cc9598d49c4ecec4458373538b3afa18f5`; cycle `7e08efadca684497f5341b3bdb760ad18cb62857`; 610 native INDODAX SOL/IDR 1m bars. Result was NO_TRADE, risk NORMAL, IDR 300,000 cash, 0 SOL, 0 order/fill/fee/P&L, consistent recovery/reconciliation.
- Actual depth collector summary at this cycle: 17 observations, 17 successes, 0 errors; mean spread 9.634764 bps / median 11.105374 / P95 16.071871. These are a newly accumulated small sample, not calibrated friction or queue data.
- Persistent state branch checkpoint: `sol-sentinel-paper-state`, commit `c3f259242d450d64b4baeb01ae083c178fd4d8b0`.
- Five-minute forward-paper scheduled workflow now exists on default branch; first post-merge cron invocation has not yet been separately observed. No 30-day pre-activation gate exists.
- Research remains insufficient; no profitability claim. Physical Android/Termux validation and account-specific fee-tier verification remain outstanding. Live trading remains disabled and unimplemented.
