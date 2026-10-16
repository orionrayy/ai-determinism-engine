# Roadmap and acceptance criteria

## P0 — reconcile code and checkpoint
- Reconcile v1.0.0-reported timestamp validator changes with exact tracked source; mixed aware/naive timestamps must raise a controlled typed validation error, not TypeError.
- Test unsorted timestamps, missing/duplicate source indices, invalid OHLC, non-finite values and negative volume.
- Quality report should include time monotonicity, source-index continuity, duplicate count, interval median/p95, session gaps, irregular intervals, zero volume, flat candles, price jumps, null/invalid rows, provenance and data hash.
- Regenerate manifest/test/benchmark outputs from the exact release code and preserve failures as evidence.

Acceptance: fail-closed typed errors, deterministic benchmark artifacts, tests reproducible from clean extraction.

## P1 — widen data and regime coverage
- Acquire licensed multi-year XAUUSD data with documented origin, timezone/DST, broker/instrument, volume semantics, timeframe and Bid/Ask/tick availability.
- Sample multiple years, sessions, trending/ranging and volatility regimes plus point-in-time news windows.
- Keep source index and timestamps audited; no concatenation of disconnected remote offset ranges.
- Use chronological train/model-selection/calibration/final-holdout partitions and matured-label purging where labels overlap.

Acceptance: provenance manifest complete, multi-regime data passes structural and temporal gates, untouched final holdout maintained.

## P2 — execution-aware strategy evaluation
- Replay with broker-relevant Bid/Ask or ticks; include dynamic spread, commission, swap, slippage, gaps and timeout rules.
- Report net expectancy, R distribution, drawdown, exposure, tail loss, trade count, profit factor, Brier/log loss, calibration/reliability and confidence intervals.
- Compare against no-trade, naive/base-rate and fixed-rule baselines; control multiple testing and parameter search.

Acceptance: any proposed edge survives measured costs, regime slices, uncertainty and untouched holdout.

## P3 — model governance and risk gate
- Separate direction and TP-before-SL targets; calibration only on held-out forecasts.
- Require sufficient mature labels; otherwise return `insufficient_history` and no probability.
- Log data/features/model hashes and every decision.
- Risk gate should require validated model/data, acceptable spread/session conditions, risk limits, confidence threshold and stale-feed checks.

Acceptance: OOS calibrated forecasts beat base-rate comparators consistently; no in-sample score turns on live output.

## P4 — point-in-time event and cross-asset evidence
- Add macro/news release timestamps, revisions/vintages, source latency and event windows.
- Test DXY, yields/real rates, volatility and other features without look-ahead; report ablations against price-only baseline.
- Track stale/conflicting provider states and abstain on data-quality failure.

Acceptance: chronological event-study protocol and ablation tests pass without causal overclaim.

## P5 — integration, security and operations
- Test Telegram/OANDA sandbox integration with authorized credentials; account connectors remain separate, consent-gated modules.
- MT5/cTrader account connection is not implemented; OANDA adapter currently covers market data only.
- Implement least privilege, token rotation/revocation, idempotency, webhook-secret checks, audit logs, rate-limits, health probes and rollback.
- Paper-trade before any live order routing; keep order execution disabled pending dedicated safety review.

Acceptance: end-to-end sandbox/paper evidence, reversible deployment, no secrets in source/logs, signed-off security and risk plan.

## v1.1.0 follow-up — online dataset evidence gate

- [x] Reconcile and build the source from the v1.0.0 archive; regenerate release manifest and SHA-256; re-extract and rerun tests.
- [x] Add a bounded, fixed-host public dataset fetcher with 16 MiB limit, bounded retries, strict schema/finite/OHLC/time-order checks and source checksum.
- [x] Add the 50 timestamp-spread × 5 baseline protocol, 1,200 counterfactual path contracts over five anchors, and a 240-case software robustness suite.
- [x] Add the CI workflow, test module, and JSON/CSV/Markdown evidence artifacts without storing the raw online dataset in Git.
- [x] Verified first successful hosted workflow log and uploaded artifact: 35,954 upstream rows, hash and schema captured, 50 timestamp cases, 250 baseline rows, 1,200/1,200 market paths and 240/240 software cases. DS51 remains `reference_only` until source rights/timezone/broker provenance are reviewed.
- [ ] Acquire licensed broker Bid/Ask/tick data and document timezone/DST, symbol contract, volume semantics, spread/commission and fill information.
- [ ] Run chronological multi-fold/out-of-sample evaluation on broader independent time regimes; keep all trade outcomes matured and preserve an untouched final holdout.


## Final v1.1.0 rebuild — verified 2026-10-16

After adding the source-register entry and the portable online benchmark suite, the release was rebuilt from a clean verification path. Final artifact SHA-256 is 259103a2bf406498048e31c8ea97970c9f1ecd6bbf7636bd3bed07d7ca0ecb6e (245,370 bytes; 98 files in ZIP; manifest lists 51 sources). Final verification: pytest 169 passed; unittest discovery 102 passed; compileall PASS; source dataset audit has zero hard failures and one expected unknown-timezone warning; baseline and online-smoke benchmark entry points PASS; HTML IDs/inline JS checks PASS; extraction test reran pytest and passed. The source remains a research prototype, and remote acquisition of DS51 is pending the hosted GitHub Actions run.


## Final hosted verification — 2026-10-16, run 38851436990

The GitHub Actions run is **green**, including unit tests, live acquisition, dataset/count assertions, job summary and artifact upload. Run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990
Artifact: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990/artifacts/12025522071
Artifact SHA-256: `7c661df1afafc372fd2816f2074ceb05aa7af0f8c7a388e64f07aa6b4bc43ef9` (9,208 bytes; retention expires 2026-10-30 04:35:53Z).

Fresh online sample: 35,954 rows, 3,070,226 bytes, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`; timestamp range `2026-03-26T02:30:00+00:00`–`2026-09-25T20:40:00+00:00`; median interval 300 seconds; 136 intervals longer than 600 seconds. The remote report ran 50 timestamp cases / 250 rule comparisons, 1,200/1,200 market-path contracts and 240/240 software robustness contracts. Runtime 458.588 ms. All counts/assertions passed; decision output remains `NO_TRADE`.

The +00:00 offset is the representation seen in the file, not independent proof of the provider's intended timezone. Broker, quote-side, license scope and fill semantics remain unverified. DS51 remains `reference_only`. The positive PnL of the breakout baseline comes from only seven trades and cannot establish alpha. See `benchmarks/live_dataset_benchmark_2026-10-16.md` and its JSON companion.

## v1.2.0 checkpoint

- [x] Add the Sederhana/Tingkat lanjut mode switch to Mini App screens with accessible pressed state and local persistence.
- [x] Add Telegram mode selection and SQLite persistence with backward-compatible schema migration.
- [x] Regression-test invalid enum values, existing preference preservation, mode UI contract and NO_TRADE guardrail.
- [x] Rebuild v1.2.0 and verify tests, static browser-script syntax, HTML IDs and extracted release archive.
- [x] Inspect successful hosted online benchmark with 35,954 public dataset rows, 250 history-case comparisons, 1,200 market path tests and 240 software robustness scenarios.
- [ ] Review market-data licensing/rights scope and provenance before changing DS51 from reference_only.
- [ ] Bring in licensed broker-specific bid/ask/tick data and true execution cost/contract semantics.
- [ ] Expand chronological evaluation to multiple non-overlapping regimes and independent sources; produce enough out-of-sample TP-before-SL forecasts to evaluate calibration.
- [ ] Complete browser/mobile accessibility and authorized read-only provider/Telegram integration tests.
- [ ] Keep live order execution gated behind paper-trading evidence, least-privilege security review, independent risk approval and rollback plan.

## v1.3.0 full-history benchmark verified

- [x] Hosted full-history benchmark passed on the public 35,954-row XAUUSD M5 sample.
- [x] Replay every eligible next-bar opportunity after warm-up; 35,893 decisions per fixed rule.
- [x] Report five disjoint chronological folds, with fold 5 exactly the final 20% tail.
- [x] Add causal prequential probability baselines and proper scoring/reliability metrics plus approximate block-bootstrap uncertainty.
- [x] 1,200/1,200 market-path contracts and 240/240 software robustness cases pass.
- [x] Persist full metrics as Markdown and JSON under projects/aurum-intelligence/benchmarks/.
- [ ] Confirm intended-use rights, provenance and timezone semantics for the online source.
- [ ] Acquire licensed broker Bid/Ask/tick data and measured costs/fill semantics.
- [ ] Validate across independent chronological regimes and an untouched external holdout.
- [ ] Validate calibration for the exact TP-before-SL target and paper-trade before production.

## v1.3.1 hosted CI checkpoint

- [x] Temporal calibration unit tests and 240/240 synthetic contract scenarios passed in hosted CI run #38897894594.
- [x] Hosted dataset acquisition, full-history replay, five chronological folds, 1,200 market-path and 240 robustness assertions passed.
- [x] Evidence uploaded as artifact #12044756186; SHA-256 `a2561de0161e79dbaac794988b5545cd8f32fe2cf55d38c875e8aed91e99d9b3`.
- [ ] Data rights/provenance/timezone review, licensed Bid/Ask/tick data and measured costs.
- [ ] Independent market-regime holdout and exact-target calibration with sufficient real OOS observations.
- [ ] Paper trading, security/risk review and approval before any live execution.
