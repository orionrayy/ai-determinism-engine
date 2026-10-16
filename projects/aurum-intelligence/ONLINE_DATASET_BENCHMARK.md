# Aurum v1.1.0 online dataset benchmark

**Goal.** Obtain fresh public community OHLCV data from a fixed upstream endpoint on a hosted runner, validate its structure, chronology and provenance, and run reproducible diagnostics without storing raw upstream data in Git. This is a research gate, not a trading recommendation.

## Fixed source and limits

- Repository: https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data
- CSV: https://raw.githubusercontent.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data/main/XAUUSD_5m.csv
- License reference: https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data/blob/main/LICENSE
- The runner is pinned to this HTTPS endpoint, caps downloads at 16 MiB, verifies a declared content size where provided, reads no more than cap plus one byte, and retries transient transport errors up to three total attempts.
- It rejects invalid UTF-8, missing/duplicate headers, missing/nonnumeric/non-finite fields, invalid OHLC geometry, negative volume, mixed timezone-aware/naive timestamps, duplicate/backwards timestamps, and fewer rows than the configured minimum.
- It does not sort timestamps, impute candles, join source offsets, or commit online raw data. Each run records SHA-256, payload size, observed time range and audit statistics.
- Intended commercial/derivative use, timezone, broker, quote side, volume semantics, spreads, commissions, fills and slippage are not independently verified. A public link or repository license badge does not automatically settle every downstream use.

## Evaluation protocol

- Historical cases: 50 timestamp-spread decision points across the valid chronology, each evaluated against five fixed non-trained rules (last return sign, EMA5-vs-EMA10, 10-bar mean reversion, prior-20 range breakout, and flat/no-trade) = 250 comparison rows. The next candle is used only as the outcome.
- Market-path stress: five real-data anchor windows × 12 synthetic transformation archetypes × 20 variations = 1,200 counterfactual test paths. This checks both valid transformations and deliberately invalid timestamp/OHLC cases; these are not observed market episodes.
- Software input stress: 20 malformed/edge-case categories × 12 variations = 240 cases.
- Cost assumption: 0.30 price units spread plus 0.10 slippage per fill. These are test assumptions, not measured broker costs. The output is diagnostic, not a promise of profitability.
- The runner must leave the decision output at NO_TRADE, predictive_model_validated=false, and live_trading_enabled=false.

## Run locally with a CSV

From projects/aurum-intelligence/online-benchmark:

    python -m unittest -v
    python ci_benchmark.py --input path/to/xauusd.csv --min-rows 1000 --cases 50 --scenario-anchors 5 --output-dir benchmark-output

Omitting --input downloads the allow-listed current CSV. Direct online access from the current local runtime was unavailable, so only the offline smoke path has been verified locally. The first hosted workflow execution is the authority for the current upstream row count and hash; until then DS51 remains reference-only.

## Hosted CI

Workflow: .github/workflows/aurum-online-dataset-benchmark.yml

It unit-tests parser and scenario contracts, downloads current upstream data, enforces minimum row/case/evaluation/simulation counts and the NO_TRADE safety gate, writes a job summary, and uploads JSON/CSV/Markdown artifacts with 14-day retention. No repository write permission is required and raw online data is not committed.

## Latest local evidence (2026-10-16)

- Full source package: aurum-intelligence-v1.1.0.zip, 245,370 bytes, SHA-256 259103a2bf406498048e31c8ea97970c9f1ecd6bbf7636bd3bed07d7ca0ecb6e.
- Package release verification: 169 pytest passed; 102 unittest passed; compileall and all release checks passed; fresh ZIP extraction and pytest re-run passed.
- Bundled-data smoke: 60 existing public OHLCV rows; 50 decision cases × 5 rules = 250 output rows; 1,200 generated market paths passed; 240 software cases passed.
- Portable online-runner smoke against the bundled sample: 30 cases = 150 fixed-rule rows; 480 paths over two anchors; 240 software stress cases; all expected contracts passed.
- Still pending: a live online CI run against the larger upstream dataset. No live-source size, hash, runtime, or predictive-quality claim is made before its report is inspected.

## Scope boundary

Scenario pass counts show only that assertions and validators behave as expected. They do not establish forecast accuracy, model calibration, net-of-cost alpha, independent regime coverage, or readiness for live trading. Keep broker connections and order execution disabled until a separately reviewed data/model/risk gate is satisfied.
