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
