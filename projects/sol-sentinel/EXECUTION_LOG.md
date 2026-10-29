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
