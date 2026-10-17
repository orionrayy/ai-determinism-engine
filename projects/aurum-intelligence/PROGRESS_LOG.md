# Progress log

## 2026-10-16 — repository checkpoint
- Confirmed the connected memory repository is `orionrayy/ai-determinism-engine`, branch `main`; created this project-specific directory without modifying orchestrator modules.
- Available complete source snapshot: v0.9.0, 85 ZIP entries. It includes application, tests, source register, canonical 60-candle sample, benchmark outputs, data downloader, Mini App, config templates and documentation.
- Validation evidence reported by the v0.9.0 report: 126 pytest passed; 92 unittest passed; Python compileall passed; Mini App JavaScript syntax passed.
- Benchmark evidence: 50 chronological decision timestamps × 5 fixed baselines = 250 rows; 18 TP/SL windows × 5 baselines = 90 rows; separate software robustness matrix 240/240; separate synthetic market-path matrix 240/240.
- Data breadth still inadequate: 60 candles from one short session, 50 source links registered but only one sampled. The two model candidates both return `insufficient_history` with zero OOS predictions.
- Remote data inspection found a large Hugging Face history but some distant source-row regions were not globally chronological. A separate 500-row contiguous region was inspected but not written to canonical data. Do not merge arbitrary offsets; enforce exact source-index continuity and timestamp monotonicity.
- v1.0.0 reported follow-on work: catch mixed timezone-aware/naive timestamp input as a typed validation failure instead of uncaught TypeError; add diagnostics for session gaps, irregular intervals, zero volume, flat bars, price jumps and source-index continuity. A complete v1.0.0 source ZIP was not present in the workspace, so these remain pending source reconciliation.

## Release history
- v0.1–v0.2: Telegram/Mini App starter; chart and technical-analysis UX.
- v0.5.1: expanded market data, persistence, cross-asset context, fixed-rule replay, provider and scenario helpers.
- v0.7.0: more structured case-based benchmarking, 50-reference source register, bounded data downloader and regression tests.
- v0.8.0: TP-before-SL replay and separate 240-case synthetic market-path tests.
- v0.9.0: experimental matured-label TP-before-SL logistic candidate; temporal-leakage tests; zero predictions on the small bundled dataset.
- v1.0.0 progress notes: dataset-validation edge-case hardening and expanded quality diagnostics reported; reconcile against actual code before tagging release.

## Restart protocol
1. Verify source archive hash and extract the exact source snapshot.
2. Rerun tests and benchmark commands; record timestamp, runtime, dependencies, data hash and code version.
3. Reconcile v1.0.0 timestamp validator changes and test mixed TZ, duplicates, shuffled/missing indices, impossible OHLC and malformed numerical values.
4. Acquire larger licensed and documented data. Validate all dates/indices. Split sessions and folds by chronology; never create continuity between separate offsets.
5. Run fixed baselines, walk-forward, calibration interval, and untouched final holdout after realistic costs. Preserve raw metrics and failed runs.
6. Keep live decision gate at `NO_TRADE` until out-of-sample calibration, net-of-cost edge, uncertainty bounds and paper-trading acceptance criteria pass.

## 2026-10-16 — v1.1.0 release reconstruction and online CI

- Retrieved the exact v1.0.0 source archive from the project artifact store, verified its SHA-256, extracted it and reran pytest/unittest/compileall and the release builder.
- New release builder produced `aurum-intelligence-v1.1.0.zip` (223,353 bytes; SHA-256 `4f3aafa682d1777a1dc6d220ad65b4b931b76760dafa8700818f8d1c0c18e623`). The builder verified its source, generated manifests/checks and fresh ZIP extraction.
- Base v1.1 package release checks: pytest 153 passed; unittest 92 passed; compileall passed; dataset integrity audit had no hard issues (only unknown timezone warning); bundled fixed baseline benchmark passed; external smoke benchmark passed (50 decision cases, 250 rule rows, 1,200 market stress paths, 240 software stress inputs); extracted-ZIP pytest rerun passed.
- Added DS51 to the register as reference-only: GetData Finance public XAUUSD M5 OHLCV CSV. Public-source README/schema discovery is not considered a successful data acquisition; retain checksum, timestamp-order, data quality, rights, and licensing checks before promotion.
- Added two stdlib-only benchmark runners and regression tests: the release-source runner (ISO-8601 and legacy timestamps, strict CSV/OHLC/chronology rules) and a portable project-folder runner intended for GitHub-hosted online acquisition.
- Portable runner offline smoke on bundled 60-bar sample: 30 timestamps × 5 baselines = 150 evaluations; 480 market paths across 2 anchors; 240 software robustness scenarios; all expected contracts passed. This is an offline smoke, not the live 35k+ upstream dataset.
- Added GitHub Actions workflow `.github/workflows/aurum-online-dataset-benchmark.yml` to run the parser tests, fetch current public data on a hosted runner (bounded to 16 MiB and three download attempts), require 50 cases/250 comparisons/200+ market and software cases, and upload audit artifacts without committing raw source data.
- Current hosted live data result remains **PENDING** until the workflow execution and uploaded report have been inspected. Do not infer success from YAML existence or local fixture success.


## Final v1.1.0 rebuild — verified 2026-10-16

After adding the source-register entry and the portable online benchmark suite, the release was rebuilt from a clean verification path. Final artifact SHA-256 is 259103a2bf406498048e31c8ea97970c9f1ecd6bbf7636bd3bed07d7ca0ecb6e (245,370 bytes; 98 files in ZIP; manifest lists 51 sources). Final verification: pytest 169 passed; unittest discovery 102 passed; compileall PASS; source dataset audit has zero hard failures and one expected unknown-timezone warning; baseline and online-smoke benchmark entry points PASS; HTML IDs/inline JS checks PASS; extraction test reran pytest and passed. The source remains a research prototype, and remote acquisition of DS51 is pending the hosted GitHub Actions run.


## 2026-10-16 — live hosted data acquired; artifact-upload retry

- Hosted GitHub Actions run `38851340128` successfully ran all six portable benchmark unit tests, fetched the live allow-listed DS51 CSV, completed all dataset/schema assertions, generated the benchmark and passed every count assertion.
- Observed upstream sample: **35,954 rows**, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`; benchmark runtime **458.635 ms** in the Ubuntu hosted runner.
- Historical replay: **50 chronological timestamp cases × 5 fixed baselines = 250 rows**. Synthetic market path matrix: **1,200/1,200 pass**. Software robustness matrix: **240/240 pass**. Decision gate stayed `NO_TRADE`.
- The run did not finish green because its final artifact upload step detected a literal backslash in the interpolated artifact name. The workflow YAML was corrected, but workflow file-only update does not by itself prove a new hosted run was queued; a follow-up PR synchronization commit and fresh run are required.
- DS51 remains `reference_only`: source schema/download is verified, but license for intended downstream use, timezone, broker and quote-side provenance are not independently established. The raw online source has not been committed to Git.


## Final hosted verification — 2026-10-16, run 38851436990

The GitHub Actions run is **green**, including unit tests, live acquisition, dataset/count assertions, job summary and artifact upload. Run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990
Artifact: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990/artifacts/12025522071
Artifact SHA-256: `7c661df1afafc372fd2816f2074ceb05aa7af0f8c7a388e64f07aa6b4bc43ef9` (9,208 bytes; retention expires 2026-10-30 04:35:53Z).

Fresh online sample: 35,954 rows, 3,070,226 bytes, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`; timestamp range `2026-03-26T02:30:00+00:00`–`2026-09-25T20:40:00+00:00`; median interval 300 seconds; 136 intervals longer than 600 seconds. The remote report ran 50 timestamp cases / 250 rule comparisons, 1,200/1,200 market-path contracts and 240/240 software robustness contracts. Runtime 458.588 ms. All counts/assertions passed; decision output remains `NO_TRADE`.

The +00:00 offset is the representation seen in the file, not independent proof of the provider's intended timezone. Broker, quote-side, license scope and fill semantics remain unverified. DS51 remains `reference_only`. The positive PnL of the breakout baseline comes from only seven trades and cannot establish alpha. See `benchmarks/live_dataset_benchmark_2026-10-16.md` and its JSON companion.

## 2026-10-16 — v1.2.0 UX increment and clean release verification

- Added explicit Sederhana/Tingkat lanjut progressive-disclosure controls in webapp/index.html. Advanced technical metrics, evidence notes, cross-asset context and detailed personalization are hidden by default in simple mode; core workflow and the NO TRADE state remain visible.
- Added Telegram settings callbacks for the two modes. SQLite user_preferences includes display_mode; initialization migrates existing databases in place, and the stored value is constrained to simple/advanced.
- Added tests for mode selection, accessibility state, local browser persistence, preservation of the decision gate and legacy database migration.
- Fixed stale hard-coded version labels in benchmark/download scripts to read VERSION.
- Release: v1.2.0, 260,781 bytes, SHA-256 19912b690e74af802a7170316703d40048a67731a1eb26d806c9929f048d2db3; 105 ZIP entries, manifest lists 104 non-manifest files.
- Clean verification: pytest 176 passed; unittest 109 passed; compileall, Mini App JavaScript syntax, 48 unique HTML IDs and clean ZIP extraction all pass.
- Bundled-data smoke: 60 rows, 50 observed decision points × 5 fixed rules = 250 evaluations, 1,200/1,200 counterfactual market paths and 240/240 software contracts passed. These are not independent samples or prediction-accuracy claims.
- Preserved hosted live benchmark run #38851629049: 35,954 rows, hash acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db, 250 rule comparisons, 1,200/1,200 market scenarios and 240/240 software cases. Data ends on 2026-09-25 and is not live-current at this checkpoint.
- Remaining work: review data-rights scope and provenance; acquire licensed broker Bid/Ask/tick data; run multiple-regime chronological validation and TP-before-SL calibration; perform browser/mobile and authorized read-only integration tests. Production and live trading remain disabled.

## 2026-10-16 — v1.3.0 full-history benchmark verified

- Clean source release: 278,402 bytes, SHA-256 53dabb5e86c8ce74691790e4c865d2624667cf531e572818bc571527bd4c2b67; 107 ZIP entries / 106 manifest entries.
- 181 pytest, 109 app unittest, 11 online benchmark tests; compileall, Mini App syntax, 48 unique HTML IDs, ZIP integrity and extracted pytest all passed.
- Hosted run #38888666520 on d790435b9f09d623a795aa6ab7ee4d1732515aa5 passed. Dataset: 35,954 rows, SHA-256 acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db, 2026-03-26 through 2026-09-25.
- Full replay: 35,893 eligible decisions per rule. Five disjoint folds: 7,175 / 7,176 / 7,176 / 7,176 / 7,190; the last fold equals the final tail.
- 1,200/1,200 market-path checks and 240/240 software robustness checks passed.
- All four active baselines lose in full and tail under assumed round-trip costs of 0.50 price units/trade. Selected-case breakout +2.37 across seven trades becomes -2,290.29 across 4,035 trades in the full replay.
- Prequential baselines: constant Brier .250000, expanding .250019 (improvement interval crosses zero), rolling-288 .250749 (worse). No calibrated TP-before-SL model exists.
- Maintain NO_TRADE. Rights/provenance/timezone, broker Bid/Ask/tick costs, independent regimes, external holdout and paper trading remain open.

## v1.3.1 hosted CI PASS — 2026-10-16

- Run #38897894594 passed against commit `6017eec3141fb803907583abeca006a230e90be4`: temporal calibration tests and 240 synthetic scenarios passed; live dataset fetch, full-history replay and count assertions passed; artifact upload succeeded.
- Artifact #12044756186, 14,966 bytes, SHA-256 `a2561de0161e79dbaac794988b5545cd8f32fe2cf55d38c875e8aed91e99d9b3`. Expiration 2026-10-30.
- Local v1.3.1 clean release stays 185 pytest / 109 unittest; archive entry root corrected to `aurum-intelligence-v1.3.1/`, 110 ZIP entries.


## 2026-10-17 — multi-source real-dataset benchmark candidate (PR #165)

- Source branch: feature/aurum-multi-source-benchmark; draft PR [#165](https://github.com/orionrayy/ai-determinism-engine/pull/165), based on feature/aurum-online-dataset-benchmark. This is an implementation candidate, not a release or production deployment.
- Hosted CI [run #39014942332](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39014942332) completed successfully on head 707f563c7b4382e5a35a18197d1a0ec4100fc305. Existing temporal-calibration tests (8), the single-source online benchmark and 22 Python unit/regression tests passed. Benchmark artifact #12094680550: 32,870 bytes; SHA-256 2cd809365080a3adb564ebd8c73fe8a3a48f635349aa18bec602e5ca9f3d8350; expires 2026-10-31 07:48:40 UTC.
- Multi-source run downloaded and audited three public community files separately. Dataset rows sum to 140,164, not deduplicated and not an independent sample count: GetData M5 35,954 rows / SHA-256 acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db; Sai community M5 17,371 rows / SHA-256 ee23e18f0ed77160893936d0312c008261e4f2ac16d82365898aba20ea26de9d; Sai community M1 86,839 rows / SHA-256 de74973a63c06ef43de56b616647eb2e380d50cb0ed22a388f158c0bcc828e13.
- Real-data diagnostic coverage: 50 chronology-spread cases × 5 fixed rules per source = 750 rule comparisons. Each dataset received full-history replay, chronological-tail evaluation and causal direction-probability baselines; aggregated scenario coverage was 2,160 synthetic market-path plus 720 software stress contracts. Synthetic cases test code robustness and are not additional observed market events.
- **Critical data-quality finding:** 11,509 exact UTC M5 timestamps overlap between the two community M5 datasets. Close-to-close direction agreement is 97.933%, but median absolute close-price difference is 63.515 price units (P95 94.445), or relative median 1.3580% and relative P95 2.0576%. Diagnostic compatibility limits are 0.25% median / 0.50% P95. Result is DIVERGENT. Do not pool these data, treat the series as interchangeable, or use cross-source confirmation until the discrepancy and original feed provenance are investigated.
- On each tested source, all four active fixed-rule baselines remained net-negative on full history under assumed costs; simple expanding/rolling rate forecasts did not improve Brier skill meaningfully versus constant 0.5. Costs are assumptions, not observed broker execution costs. These results do not validate profitability, exact TP-before-SL probabilities or world-leading accuracy.
- Pipeline status PASS means online acquisition, schema/chronology checks, benchmark execution, artifact writing, scenario assertions and report contracts passed. The separate source price-compatibility result is DIVERGENT; production_ready=false, cross_source_confirmation_allowed=false, predictive_model_validated=false, deployment_calibrated=false, and live signals/order execution remain disabled (NO_TRADE).
- A first hosted run exposed a report JSON trailing-literal-newline defect and CI failed with JSONDecodeError: Extra data. The writer now uses a shared report-output function and a regression test checks valid JSON plus Markdown newline semantics. A later run caught an outdated test key after the price-compatibility refactor; the test was updated. The current run is green, including all 22 Python unit/regression tests.
- Remaining gates: determine why the two M5 feeds have different price levels; independently verify source lineage, original timezone and intended data-use rights; source licensed broker Bid/Ask/tick history; measure costs/fill semantics; validate exact TP-before-SL calibration on independent regimes and untouched holdout; then paper trade and complete security/risk/Telegram/OANDA deployment validation.


## Final candidate checkpoint — bounded timestamp-probe CI PASS (2026-10-17)

- Latest candidate head: e9f73719bc9acfbd0f9e85c777a97d856b76e1b5; draft PR [#165](https://github.com/orionrayy/ai-determinism-engine/pull/165), still open and not merged.
- Final hosted run [#39015468873](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39015468873) succeeded on this exact head. Temporal-calibration suite: 8 tests passed. Multi-source/unit regression discovery: 22 tests passed. Both single-source and multi-source live dataset download/replay steps and final contract assertions passed.
- Final evidence artifact [#12094870433](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39015468873/artifacts/12094870433): 33,975 bytes, ZIP SHA-256 8a23015c04bcd27821d41be4abebcf408e1f57df40a9fd399e406b98bad26bb3, retention expires 2026-10-31 07:57:28 UTC. It contains the online single-source report plus the multi-source JSON/Markdown report; raw source CSVs are not uploaded.
- Final run coverage remained 3 community files, row-count sum 140,164 (not deduplicated and not an independent sample count), 750 real-data rule comparisons, 2,160 synthetic market-path contracts, and 720 software contracts. Multi-source runner runtime: 53,549.711 ms. Pipeline PASS is not a strategy-performance PASS.
- Cross-source M5 discrepancy remains critical: 11,509 exact timestamp matches, 97.933% direction agreement, but median absolute close delta 63.515 price units and median relative delta 1.3580%; P95 relative delta 2.0576%. Price-level compatibility remains DIVERGENT, so sources are not merged, pooled, or allowed to cross-confirm.
- The latest offset probe tested 13 candidate timestamp offsets (-25200, -10800, -7200, -3600, -600, -300, 0, 300, 600, 3600, 7200, 10800, 25200 seconds). Best observed offset was -25,200 seconds (-7h) but improved median relative close delta only from 1.358013% to 1.271836%, i.e. 6.3458%, below the configured 20% material-improvement threshold. Result: NO_OBVIOUS_BOUNDED_SHIFT_FIX, material_shift_candidate=false, candidate offset null. No source bars were shifted.
- Benchmark evidence again found all four active fixed-rule baselines net-negative over full history on all three datasets under assumed costs. Expanding/rolling rate baselines did not meaningfully improve Brier skill over constant 0.5. There is still no validated exact TP-before-SL model, no broker-observed execution-cost model, and no demonstrated alpha.
- Source rights, original timezone semantics, publisher/broker provenance, quote side and statistical independence remain unverified. Keep NO_TRADE, predictive_model_validated=false, deployment_calibrated=false, cross_source_confirmation_allowed=false, and live order execution disabled.
- Next gate is data discrepancy investigation, not new strategy optimization: verify bar timestamp/label conventions, compare sample candles, audit original feed lineage and usage rights, and obtain a licensed broker Bid/Ask/tick dataset before asserting independent multi-source price evidence. Then implement execution-aware costs and exact TP-before-SL calibration on independent chronological holdouts before paper trading.
