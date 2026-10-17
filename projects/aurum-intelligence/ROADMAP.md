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


## Multi-source real-dataset benchmark — PR #165 checkpoint (2026-10-17)

### Completed implementation / evidence
- [x] Add fixed-host, byte-bounded readers for community XAUUSD M5/M1 files from two public repositories; normalize explicit timestamps to UTC and report naive-timezone assumptions.
- [x] Validate each file independently, preserve each SHA-256, time range and gap statistics; never concatenate or merge source bars.
- [x] Run 50 timestamp-spread cases × 5 fixed rules per source, full-history replay, chronological tail diagnostics, and causal direction-probability baselines.
- [x] Run 720 synthetic market-path and 240 software robustness contracts per dataset; label synthetic cases separately from observed market records.
- [x] Add exact-timestamp M5 overlap analysis, relative price-difference compatibility thresholds (0.25% median / 0.50% P95), and a hard block on cross-source confirmation/production readiness when series are divergent.
- [x] CI [run #39014942332](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39014942332) passed on PR #165 head 707f563c7b4382e5a35a18197d1a0ec4100fc305; 8 temporal-calibration tests and 22 source/benchmark unit tests passed; live download/benchmark assertions and report upload passed.
- [x] Persist report JSON/Markdown and hashes as artifact #12094680550; raw input datasets are not uploaded to the benchmark artifact.

### Findings that prevent a source-level readiness claim
- Data rows sum to 140,164 across 3 files; this is not a deduplicated/independent sample size.
- The two M5 sources have 11,509 exact UTC timestamp matches, 97.933% close-to-close direction agreement, but median relative close-price delta 1.3580% and P95 delta 2.0576%, against 0.25% / 0.50% diagnostic limits. Cross-source status is DIVERGENT, independence UNVERIFIED.
- Every active fixed-rule baseline is net-negative over full history on each dataset under assumed costs. The simple causal direction-rate estimators show no meaningful Brier skill over the constant 0.5 comparator. The benchmark result does not prove alpha or calibrated TP-before-SL probability.
- Keep NO_TRADE; predictive_model_validated=false, deployment_calibrated=false, production_ready=false, cross_source_confirmation_allowed=false, and live order execution disabled.

### Next engineering gates (still open)
- [ ] Investigate M5 price-level divergence: compare timestamp semantics, bar label/open-close convention, upstream publisher/feed lineage and sample candles before deciding whether the source can be compared or should be excluded.
- [ ] Verify original source-data rights and commercial-use scope; repository license alone is insufficient.
- [ ] Add an independently verified broker/feed source or licensed Bid/Ask/tick data; record source identity and timestamp/quote-side semantics.
- [ ] Benchmark execution-aware strategies with measured spread, commission, slippage, swap and realistic fill/timeout behavior.
- [ ] Evaluate exact TP-before-SL labels and calibration on mature, chronological OOS outcomes with an untouched holdout; retain abstention when sample size or evidence is inadequate.
- [ ] Complete paper-trading, security/risk and live Telegram/provider acceptance before any production signal/order capability.


## Final hosted candidate verification — 2026-10-17

- [x] Run final candidate PR head e9f73719bc9acfbd0f9e85c777a97d856b76e1b5 through hosted CI: run #39015468873 PASS; 8 temporal-calibration tests and 22 Python unit/regression tests pass; live data acquisition, benchmark, final contracts and artifact upload pass.
- [x] Save the report artifact #12094870433 (33,975 bytes; SHA-256 8a23015c04bcd27821d41be4abebcf408e1f57df40a9fd399e406b98bad26bb3; expiration 2026-10-31 07:57:28 UTC). Raw input datasets are not in the artifact.
- [x] Complete the bounded timestamp-offset probe without applying any shifted timestamps.
- [x] Persist the final checkpoint in IMPLEMENTATION_STATUS.json and PROGRESS_LOG.md.
- [ ] Investigate M5 price-level divergence. Exact timestamp overlap is 11,509 bars; direction agreement is 97.933%, yet median absolute close delta is 63.515 price units and median relative delta is 1.3580%. Relative P95 is 2.0576% against 0.50% diagnostic limit; status remains DIVERGENT.
- [ ] Audit candle timestamp/label conventions and original feed lineage. The best observed offset (-7 hours) improves median relative delta by only 6.3458%; under the 20% material threshold there is no candidate shift. Do not shift or merge the source bars.
- [ ] Verify the legal rights and intended commercial use of source data, plus its broker/quote-side and timezone provenance.
- [ ] Acquire independently documented broker Bid/Ask/tick history with real spreads, commissions, slippage, swap, and fill semantics.
- [ ] Re-run execution-aware strategy replay, exact TP-before-SL calibration, mature-label leakage controls, independent chronological regimes and untouched final holdout.
- [ ] Keep NO_TRADE, cross-source confirmation disabled, production readiness false, and live execution disabled until evidence gates and paper-trading/security approval pass.


## Signed/monthly M5 discrepancy findings — final run #39015804227 (2026-10-17)

- [x] Add source-B-minus-A signed price bias, close price quantiles, signed OHLC deltas, percent above/below, UTC-month discrepancy buckets and five deterministic paired-candle snapshots.
- [x] Validate the additions with 22 Python unit/regression tests and hosted multi-source CI; preserve the summary and raw-source-free evidence artifact #12094359744.
- [x] Verify exact report JSON and aggregate/monthly coverage contracts: monthly timestamp buckets sum to the exact overlap count; five sample candles are returned; source bars remain separate.
- [ ] Investigate time-varying price discrepancy. Across 11,509 matched M5 timestamps, the median signed B−A delta is -63.515 price units; source B is below source A in 95.10% of matched timestamps. Median signed delta varies from -94.105 in March to -68.200 in April and -56.710 in May; one endpoint paired sample is almost equal (-0.235). Do not treat the difference as a stable additive offset or normalize it away.
- [ ] Verify upstream data lineage, quote-side, UTC bar-label semantics, volume semantics and usage rights. The GetData README describes bar-open UTC/tick volume; Sai describes mid-quote and stores timestamps without explicit timezone, interpreted as UTC from its README. These descriptions do not establish equivalent or independent pricing feeds.
- [ ] Compare samples and source hashes with a licensed broker Bid/Ask/tick feed, then rerun per-regime execution-aware benchmarks and exact TP-before-SL calibration on an untouched holdout.
- [ ] Retain NO_TRADE, no cross-source confirmation, no deployment calibration, and no live order execution until the previous data-validity phase is resolved and the later research/security gates pass.
