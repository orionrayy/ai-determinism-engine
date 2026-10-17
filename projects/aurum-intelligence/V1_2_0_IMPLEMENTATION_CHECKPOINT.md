# Aurum Intelligence v1.2.0 — implementation checkpoint

**Checkpoint:** 2026-10-16, Asia/Jakarta. **Release state:** research/UX prototype; not production-ready. The source ZIP is a downloadable artifact attached to the project work, not a binary committed to this repository.

## Implemented in v1.2.0

- Added a visible, accessible **Sederhana / Tingkat lanjut** selector to the Mini App. The default simple mode keeps core navigation, data availability and NO TRADE state prominent; advanced mode reveals more indicators, cross-asset context, methodology cautions and personalization controls.
- Added the same mode selection to Telegram preferences. The SQLite preference has an in-place backward-compatible migration, enum validation, and regression tests preserving existing timeframe preferences.
- The Mini App's view choice is stored in browser local storage. It is not advertised as synchronized with the server. Mode changes alter display density only; they do not alter model probabilities, risk rules or signal eligibility.
- Updated benchmark/version labels to derive from the authoritative VERSION file rather than stale hard-coded 1.1.0 literals.
- Preserved the hosted online benchmark outputs as small provenance/evidence files; the ~3 MB raw market CSV is not committed or packaged.

## Verified local release run

Release: aurum-intelligence-v1.2.0.zip, 260,781 bytes, SHA-256 19912b690e74af802a7170316703d40048a67731a1eb26d806c9929f048d2db3.

| Verification | Observed result |
|---|---:|
| pytest | 176 passed |
| unittest | 109 passed |
| compileall | PASS |
| Mini App inline JavaScript syntax | PASS |
| HTML ID audit | 48 unique IDs; 0 duplicates |
| ZIP integrity / clean extraction | PASS; extracted archive reran 176 pytest tests |
| Bundled dataset | 60 public community XAUUSD M5 rows; 0 hard audit issues, timezone warning |
| Direct historical cases | 50 timestamps × 5 rules = 250 rows |
| Local market-path smoke | 1,200/1,200 scenarios passed |
| Local software robustness smoke | 240/240 scenarios passed |
| Forecast model OOS predictions | 0; insufficient history |
| Live trade decision | NO_TRADE |

The current smoke dataset is one short, clustered community sample and cannot validate a profitable strategy or stable calibration. Scenario pass counts validate code contracts, not prediction skill.

## Hosted public-data evidence

GitHub Actions run [#38851629049](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851629049) passed the fixed-source public dataset benchmark on the v1.1.0 online-benchmark branch. Its artifact was independently retrieved and checked for ZIP integrity in the project workspace.

- Source: [getdata-finance/XAUUSD 5m OHLCV sample](https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data).
- Observed CSV: 35,954 rows, 3,070,226 bytes, SHA-256 acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db.
- Sample timestamps span 2026-03-26 02:30 UTC through 2026-09-25 20:40 UTC; 300-second median interval, 136 intervals longer than 600 seconds. It is historical input, not a live feed, and was about three weeks behind this checkpoint.
- Repository publishes an MIT license. Data rights/provenance, broker/quote-side semantics, actual fills and intended redistribution scope still require review. The raw CSV is intentionally not redistributed here.
- Example: prior20_range_breakout posted +2.37 price units on only seven trades under assumed costs; that small observation is not sufficient evidence of alpha. Other active baselines were net negative. No forecast model is validated.

See the preserved report under projects/aurum-intelligence/benchmarks/live_dataset_benchmark_2026-10-16.md and the machine-readable JSON artifact in this project folder.

## Remaining gates

1. Verify the source's data-specific licensing/provenance and timestamp semantics; acquire separate licensed broker-specific bid/ask/tick data.
2. Add multiple non-overlapping market regimes and independent data sources; use chronological walk-forward folds, outcome-maturity purging, and an untouched holdout.
3. Train and calibrate TP-before-SL probabilities with Brier/log loss, reliability curves, ECE/MCE and uncertainty intervals; require meaningful sample size before publishing probabilities.
4. Include real spread, commission, slippage, swap/rollover, gaps and fill assumptions; compare net edge against a no-trade baseline and address multiple testing.
5. Run browser/mobile accessibility checks and authorized read-only Telegram/provider integration tests. Live order execution remains out of scope until separate approval, risk, security and paper-trading gates pass.

**Do not claim world-leading accuracy or superiority from these results.** Such claims require a clearly specified task, comparable public baselines, time-matched out-of-sample data, statistical uncertainty and reproducible replication.
