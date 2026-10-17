> **Identity migration note (2026-10-17):** The current project identity is **Serevex Quant Research**, with canonical path `projects/serevex-quant/`. This file preserves historical experiment descriptions, source claims, CI run IDs, checksums, and provider-issued artifact names where changing them would damage auditability. Any prior branding in historical entries is archived metadata, not active branding.

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


## 2026-10-17 — v1.4.0 P0 cross-source price-basis diagnostics

- Added non-mutating diagnostics to `online-benchmark/source_concordance_v1_0.py`: signed OHLC price differences, descriptive OLS relationship (primary close against secondary close), fit residual distribution in price units and bps, chronological quartile summaries of signed close differences, and direct paired-return difference metrics.
- Added two regression tests for a known additive level offset and a known multiplicative scale difference. Both assert diagnostic reporting and explicitly forbid automatic correction/source promotion.
- Hosted CI run [#39024803960](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39024803960) passed on code/test head `6b5b2f04a674c81deee84f1a2cdaf5ca442663d5`: temporal calibration contracts, parser/baseline/source-concordance tests, online acquisition/benchmark, count assertions, and artifact upload. Artifact #12097644871, 2,563,258 bytes, SHA-256 `b2bc267bda4d067ed6baec07d196540f903e84545cf342f9938b17371f607415`.
- Live diagnostic result remains `SOURCE_DISAGREEMENT_REVIEW_REQUIRED`: 11,509 exact timestamp overlaps, zero exact OHLC matches, median/p95 absolute close discrepancy 134.885/203.661 bps, return correlation 0.99845. Level OLS diagnostic: slope 0.984434, intercept +130.351 price units, R² 0.940465, residual p95 128.848 bps; no simple affine basis relationship is established.
- Direct consecutive-bar return-difference median/p95 is 0.264/0.982 bps. The timestamp-offset scan's best candidate is +12 bars (the scan boundary), reduces median close discrepancy only 2.37%, and its one-bar return correlation is -0.0146. Do not treat this as a timestamp correction; extend source methodology investigation instead.
- Synthetic source fixtures and the full online benchmark pass are software/evaluator verification only. They do not validate an authoritative market feed, predictive alpha, calibration quality, profitability or live execution. Maintain `NO_TRADE`.
- Next: inspect upstream timestamp/bar-label, price basis, revision/provenance and data-rights semantics; independently verify the source with broker-authorized quotes/ticks and measured costs; keep source-specific replay separate until these gates pass.


## 2026-10-17 — third-source tick Bid/Ask and lineage-audit checkpoint

- Added `online-benchmark/tick_bidask_benchmark_v1_0.py` with bounded data acquisition from `CarlosSilva1/xauusd-ticks` (CC-BY-4.0 attribution recorded), strict bid/ask validation, UTC M5 aggregation for Bid/Mid/Ask, observed spread summaries, and exact-timestamp comparisons with both community OHLCV feeds. Only one month is fetched; the 58,806,458 bytes are not redistributed in the report artifact.
- Added unit tests for spread correctness, crossed/nonpositive quotes, exact M5 overlap, and lineage classification; extended workflow gates to require results for both OHLCV sources, no auto-correction/promotion, and explicit lineage-status output.
- Hosted GitHub Actions [run #39025937658](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39025937658) **passed** on code/test head `c6f5062fab2a919b4d3795304c5cbee83b67618a`; all stages completed, including live dataset benchmark, tick Bid/Ask audit, lineage assertions and artifact upload. Artifact #12098756760, 2,595,322 bytes, SHA-256 `54a3905b3fb3a3909070245da80a9089d2051b029dd9f45bab5332ed260d7aa8`, retention through 2026-10-31.
- May-2026 selected Parquet parts: 4 files, 58,806,458 bytes total, SHA-256 recorded per part in `tick_bidask_benchmark.json`; 6,685,266 ticks; observed timestamps 2026-05-01 00:00:00.086 UTC to 2026-05-22 20:59:59.132 UTC; 4,392 UTC M5 quote bars; zero missing/non-finite/nonpositive/crossed quote rows. Same-millisecond duplicates are counted, not silently removed.
- Spread distribution in this reconstructed feed: median 0.58 price units, P95 0.827, P99 0.96; median/P95 spread as bps of mid 1.2555/1.7884. Not measured cost for any user's broker/account.
- Tick-mid vs GetData primary: 4,380 exact timestamp overlaps, close-difference absolute median/P95 123.229/138.379 bps; median signed primary-minus-tick-mid difference +56.325 price units. No exact OHLC matches.
- Tick Bid vs Sai310421 secondary: 4,392 exact timestamp overlaps and **4,391 exact OHLC matches (99.9772%)**, although the secondary README describes its feed as mid-quote. Classifier returns `POTENTIAL_SHARED_LINEAGE_OR_PRICE_BASIS_MISLABEL_REVIEW_REQUIRED`. This does not establish cause, but the secondary feed and tick feed cannot be counted as independent confirmations until source lineage/quote basis is explained.
- Overall online benchmark status PASS means data retrieval, tests, evaluation and assertions worked. It does not mean the sources agree, that provenance is authoritative, or that any trading model has predictive skill. `NO_TRADE`, `validated=false`, `calibrated_model=false`, and live signal execution remains disabled.
- Remaining P0: verify primary/secondary/tick provider provenance, timestamp and price/quote basis, licensing scope, and whether the secondary series is derived from the tick series; investigate primary-vs-tick median level discrepancy; acquire broker-authorized quotes and measured execution costs; then evaluate on independent regimes and untouched holdout. Do not add model complexity or order routing before these gates.

- Follow-up validation on the later PR head `65958a8e68f0c3feed22693996f881b86c9d495f` also passed [workflow #39026245471](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39026245471); artifact #12098043414, SHA-256 `184ff507423f4902dd8d9f86647d0e6ca5fe3fd73513de4d604f1b3823bc16aa`. This confirms the full online/tick/lineage workflow after checkpoint/status docs updates; no trading-validation state changed.


## 2026-10-17 — large-scale real Bid/Ask benchmark + fair base-rate calibration comparison

- Latest end-to-end hosted CI run [#39028110673](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39028110673) **passed** on PR head `375aa2e3c225394943f177a5243d5c18ccd04fd9`. Artifact #12099177720, 2,613,340 bytes, SHA-256 `5b5129edebd988ca885e0e06f69b8430a9db01ab807fcdb8e11c6d3b0470891b`, retention until 2026-10-31.
- Real historical data: 1,313,280 paired M1 Bid/Ask candles across 30 months (2023-01 through 2025-06), 131,316,007 bytes acquired. 937,440 weekday rows and **870,500 weekday non-flat candles**; the last figure, not weekend/calendar padding, passes the 700,000-active-candle gate. Zero-range Mid OHLC fraction is 32.7459% and is reported, not silently discarded.
- Scenario gate: 20,000 distinct chronological timestamp anchors; all 20,000 have valid ATR; 875,611 eligible contiguous windows with valid ATR; 120,000 TP-before-SL barrier-path scenarios (20,000 anchors × two directions × three TP distances). 52,883 directional rule events scored; 10,577 scored events in the final 20% chronological holdout. Ambiguous same-bar events are excluded and counted.
- Probability comparison strengthened: besides the constant-0.5 baseline, the report now compares rule/side Laplace probabilities against a causal global matured-label prevalence baseline. Both baselines use only outcomes that matured before the decision; priors are frozen at the final-test boundary. A deterministic moving-block bootstrap reports a 95% interval for Brier improvement against each baseline.
- **Negative/no-edge result:** model Brier 0.140102; global matured-label baseline Brier 0.140015; global base rate 0.168126. Brier skill vs weak constant-0.5 is +0.439593, but this is not a valid standalone edge claim for an imbalanced target. Against the more appropriate global causal base-rate baseline, aggregate skill is **-0.000616**, with bootstrap 95% interval for Brier improvement [-0.0001737, -0.0000091] (entirely negative). Final holdout model Brier 0.163455 vs global baseline 0.163549 (skill +0.000576) and ECE 0.043313; the tiny positive final difference is not evidence of robust predictive edge given the negative aggregate comparison and residual open data/provenance gates.
- Added active-candle and ATR-valid-anchor gates; fixed CI contract naming for `eligible_real_contiguous_windows_with_valid_atr`. Latest workflow assertions and all tests passed. Report is still a research benchmark, not a live-execution or profit claim.
- Dataset source: `kevingtlin/Market-Data-Lab`, reporting Dukascopy via `dukascopy-node`; source/provider lineage and upstream rights are still not independently verified. Raw monthly M1 candle prices are not redistributed by the benchmark artifact.
- Deployment remains `NO_TRADE`; `deployment_calibrated=false`, `validated_model=false`, `live_signal_enabled=false`. P0 still requires provenance/rights/price-basis verification, broker-specific measured execution costs, independent market regimes and an untouched final holdout. No world-leading-accuracy claim is supported by these metrics.


## 2026-10-17 — point-in-time external context design specification

- Wrote `docs/superpowers/specs/2026-10-17-aurum-point-in-time-external-context-design.md` on `feature/aurum-online-dataset-benchmark` (spec commit `5672fa167377aa23129a8f8cca4c21d0fddde1c6`). Design includes typed point-in-time records, CFTC COT positioning, historical GDELT GKG/news metadata, BLS publication-time constraints, optional FRED/ALFRED vintages, as-of joins, same-row ablations, data rights/provenance and no-leakage tests.
- Source-contract review used the official CFTC historical compressed/release schedule, FRED vintage/API requirements, BLS API limits, and GDELT archival-versus-rolling-search documentation. The design explicitly prevents Tuesday COT report dates, revised latest macro values, or backdated news publication timestamps from leaking into earlier predictions.
- No feature implementation was started. The architectural review gate now is the written spec; user approval of the in-chat direction is not treated as approval of an unseen implementation plan. Next action after spec approval is to create/review the detailed plan, then obtain execution-method confirmation before implementation.
- Latest known full benchmark run `39028110673` passed on code/test head `375aa2e3c225394943f177a5243d5c18ccd04fd9`. The latest PR head `576c1de66e9a7b91da3ba38782235d03732a87dd` had run `39028331604` end as `startup_failure` with zero jobs created; this is an Actions startup issue, not a test result. Do not claim the latest PR head is green until a fresh run executes and passes.
- Current trade decision remains `NO_TRADE`; the existing COT/news/macro design does not clear price-source provenance, rights, broker execution cost, calibration, or risk gates.


## 2026-10-17 — point-in-time external context plan ready for review

- Design spec: [docs/superpowers/specs/2026-10-17-aurum-point-in-time-external-context-design.md](https://github.com/orionrayy/ai-determinism-engine/blob/feature/aurum-online-dataset-benchmark/docs/superpowers/specs/2026-10-17-aurum-point-in-time-external-context-design.md). Spec correction commit: `5dd1fbb96f6c1c1965c3558964ed6047e75a2ce0`; tightened the COT fallback from an unsafe interpretation of “Wednesday after report date” to 00:00 America/New_York on the Wednesday of the following calendar week (report date + 8 days for normal Tuesday reports), otherwise quarantine ambiguous holiday calendars.
- Implementation plan: [docs/superpowers/plans/2026-10-17-aurum-point-in-time-external-context.md](https://github.com/orionrayy/ai-determinism-engine/blob/feature/aurum-online-dataset-benchmark/docs/superpowers/plans/2026-10-17-aurum-point-in-time-external-context.md), latest plan commit `65ac5c1888390c6010102bb66da2e398daec4e7f`. Eight tasks cover shared schema, four source adapters, point-in-time joins/leakage tests, same-row ablation/calibration benchmark, CI and evidence checkpoint.
- The plan fixes a deterministic L2 logistic baseline (lambda=1.0, learning rate 0.05, max 200 full-batch steps, zero initialization), fits normalization/model on development only, reuses the existing temporal calibration diagnostics for Platt fitting on mature calibration labels, purges immature boundary labels, and reserves final 20% chronological tail for scoring only.
- Real-data scale target is at least 100,000 GDELT/archive observations. If the real source sample cannot reach the threshold under explicit download/runtime caps, status must be `NOT_MET` with true counts. No synthetic fill. Existing ≥700k active M1 and ≥20k real anchor/scenario gates remain.
- No feature code has been implemented yet; the written plan is waiting for user review and selection of execution method. Keep `NO_TRADE`. Last complete price benchmark remains run #39028110673, success on tested code/test head `375aa2e3c225394943f177a5243d5c18ccd04fd9`; the later PR-head Actions run #39028331604 ended `startup_failure` with zero jobs, so it is not a test result.
## 2026-10-17 — GDELT temporal-coverage safety, report-status separation, and retry closure

- Fixed GDELT budget control so a per-archive size rejection no longer masquerades as exhaustion of the aggregate download budget. Oversized archives are skipped with explicit warnings; subsequent sampled archives continue. Raw record count no longer triggers an early stop before cross-snapshot deduplication.
- Added manifest-derived 15-minute archive coverage intervals and a UTC coverage summarizer. Sample completion, source acquisition, complete historical-window coverage, and eligibility of temporal features are separate statuses. The benchmark can report `PASS` for successful evaluation/scale gates while the archive universe remains `PARTIAL`; this is intentional and documented in the output.
- `build_asof_features` receives the actual acquired GDELT archive intervals. News numeric features are emitted only when the required 1h/24h interval is fully covered by already-visible archive batches. Gaps are `incomplete_source_window`, not zero-news. Future/contiguous intervals cannot leak into earlier decisions or obscure availability boundaries.
- Added regression coverage for per-archive versus aggregate limits, continuing after oversized archives, sample completion versus full archive completeness, interval merging/deduplication, untrusted manifest hosts, sparse windows, true zero-event fully covered windows, future interval leakage, and visible prior contiguous archive windows. Current full CI run reports **115 unit/regression tests passed** and 240/240 temporal-calibration contract scenarios passed.
- A failed intermediate run [#39035356832](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39035356832) is preserved. Root cause was a `NameError`: the source completeness flag created inside `_acquire_sources()` was referenced outside that scope in the report builder. Fixed by reading `source_details["gdelt_gkg"]["archive_window_complete"]`; contract test covers the correct scope. It is not counted as a pass.
- Latest end-to-end CI: [#39035732248](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39035732248), success on head `f3eaa644a1f73c0d4f9521870e1ba5c460cfaf69`. Artifact #12102745058, 2,624,921 bytes, SHA-256 `b1710712519fd48e3a5f29acf1d532707997821bba1948e49b093004fd7b81f0`. Download: https://github.com/orionrayy/ai-determinism-engine/actions/runs/39035732248/artifacts/12102745058
- Latest external context benchmark: 141,020 unique real GDELT records against the 100,000 scale target (no synthetic or duplicated records added), 100/100 selected archives processed from 85,809 discoverable archives; acquired bytes 701,820,752. The selected temporal sample covers only **1,500 / 1,313,279 minutes = 0.114218%** of the requested historical window. `source_acquisition_status=PASS`, `source_completeness_status=PARTIAL`; these fields are deliberately not interchangeable.
- Feature eligibility across 19,937 mature unambiguous price labels: COT 19,739 rows (99.0069%, `PARTIAL`), macro 0 rows (`NOT_MET`), news 0 rows (`NOT_MET`). BLS has 119 descriptive records but no release-time calendar, so they are not causal features. FRED/ALFRED was skipped because `FRED_API_KEY_NOT_CONFIGURED`. All 141,020 GDELT rows are not evidence of continuous news-window coverage.
- Same-row final holdout ablations scored 3,987 rows each: price-only Brier 0.1609893; price+COT/combined Brier 0.1609044. Incremental Brier improvement of the COT ablation versus price-only is only +0.00008494 and its moving-block 95% interval [-0.00014777, +0.00032012] crosses zero. The result does **not** establish robust incremental predictive value; no best-model/alpha claim is made. `price_news` and `price_macro` remain `NOT_ELIGIBLE_NO_CONTEXT_FEATURES_IN_DEVELOPMENT`.
- The M1 benchmark in the same CI still passes 1,313,280 paired Bid/Ask candles, 20,000 valid ATR anchors and 120,000 TP/SL path scenarios; 19,937 mature labels were scored in the context benchmark. Historical source lineage and intended-use rights remain unresolved, and source price-basis mismatch persists. App remains v1.3.1 research/UX prototype; evaluator v1.4.0 P0 remains draft PR #164, unmerged. `NO_TRADE`, `deployment_calibrated=false`, `validated_model=false`, `validated_for_profitability=false`, `live_signal_enabled=false`.
- Next gates: verify upstream rights/lineage/quote/timestamp semantics; obtain broker-authorized Bid/Ask/tick and measured execution costs; supply point-in-time macro release/vintage sources; acquire temporally adequate news archives or keep news features ineligible; rerun independent-regime locked holdout diagnostics and paper trading before any separate risk/security review.


## 2026-10-17 — evidence-backed source qualification gate (verified)

- Added modular qualification logic requiring referenced evidence for upstream provenance, intended-use rights, timestamp semantics, bar convention and quote basis; execution readiness separately requires broker authorization, measured costs and verified fill semantics. The helper rejects truthy non-booleans and malformed evidence refs; even complete evidence never automatically promotes a feed or authorizes live execution.
- Integrated the gate into online community OHLCV, tick Bid/Ask and hosted workflow assertions. All three feeds currently report `REFERENCE_ONLY` for research and `BLOCKED` for execution.
- Hosted workflow [#39038107206](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39038107206) passed on code/test head `d2621fcda08456984ec3adef9015f91d3d432ddd`: 125 unit/regression tests, 240/240 temporal-calibration cases, current community-data acquisition, tick audit, M1/ATR/scenario gates, external-context benchmark and artifact upload.
- Artifact #12102563009, 2,626,545 bytes; SHA-256 `993c27155e00c7d57d87aba5c475c1751d7026fcbfce4345b784996215bc065c`. Downloaded ZIP hash matches GitHub's digest.
- Dataset concordance remains unresolved: 35,954 primary and 17,371 secondary M5 rows; 11,509 common timestamps, zero exact OHLC matches, 134.885/203.661 bps median/P95 close discrepancy. A separate tick audit finds 4,391/4,392 secondary bars exactly equal to reconstructed tick Bid bars despite reported mid-quote semantics.
- GDELT scale reached 141,020 real unique records, but the 100 sampled archives cover only 0.114218% of the preregistered window; news features remain ineligible. Macro features remain ineligible. COT incremental Brier improvement versus price-only is 0.00008494 and its 95% interval crosses zero.
- Maintain `NO_TRADE`, `deployment_calibrated=false`, `validated_model=false`, `validated_for_profitability=false`, `live_signal_enabled=false`. Draft PR #164 remains unmerged.
