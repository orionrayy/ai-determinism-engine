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


## 2026-10-17 — P0 v1.4.0 evaluation + cross-source audit

- Added `evaluation_protocol_v1_4.py` and focused tests: next-open entry after signal-close, ATR TP/SL barriers, fixed horizon, same-bar barrier collision is ambiguous/excluded, time-ordered 70/10/20 partitions, matured-label prequential baseline, Platt calibration only on calibration rows whose outcomes mature before holdout start, and interval/price-jump data-quality profile.
- Added `source_concordance_v1_0.py` plus tests and a second public community feed (`Sai310421/xauusd-data`) to online CI. Datasets are matched on exact timestamps and kept separate; OHLC price differences, return correlation, overlap, signed offset and threshold flags are emitted to JSON/CSV/Markdown.
- Latest hosted run [#39022220573](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39022220573) PASS. Main source: 35,954 rows; 94,604 candidate events, 94,084 resolved binary labels, 520 ambiguous same-bar events. Secondary: 17,371 rows, 11,509 exact overlapping timestamps, zero exact OHLC matches, median close difference 134.885 bps, p95 203.661 bps, one-bar return correlation 0.99845. Result classified `SOURCE_DISAGREEMENT_REVIEW_REQUIRED` and never auto-corrected/merged.
- Artifact #12096704482, 2,556,935 bytes, SHA-256 `94d432018fa313db265bc01dbeba1e37df5efc0a3c90f65eaec432259877f7de`.
- This is a successful benchmark/evaluator run, not validation of source truth or trading performance. Keep `NO_TRADE`. Next: resolve source/time/price basis + rights; obtain licensed broker Bid/Ask/tick and measured costs; then rerun independent regimes and untouched holdout.
