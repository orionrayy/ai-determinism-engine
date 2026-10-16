# Benchmarks and evidence ledger

**Source snapshot:** v0.9.0. **Dataset:** `data/xauusd_5m_public_sample.csv`, 60 rows. **SHA-256:** `dae535a16e13cf49f123231b1ee72e19787259e7a5f943b017a013346f36a93b`. Source mirror: `ZombitX64/xauusd-gold-price-historical-data-2004-2025` on Hugging Face, attributed by the mirror to a Kaggle source. Independent original provenance, broker, licensing for all intended uses, timezone, Bid/Ask and fills remain unverified.

## Reproduced report counts (v0.9.0 report)
- `python -m pytest -q`: 126 passed.
- `python -m unittest discover -s tests -q`: 92 passed.
- `python -m compileall -q app scripts tests`: PASS.
- Inline Mini App `node --check`: PASS (syntax only; not complete browser/Telegram test).
- Registered references: 50; sampled/ingested: 1; reference-only: 49.
- Next-candle case replay: 50 decision timestamps × 5 rules = 250 strategy/decision rows.
- Chronological tail holdout: 19 observations.
- TP-before-SL: 18 decision windows × 5 rules = 90 rows; target 1.5× ATR, stop 1× ATR, max horizon 12 bars, next-bar-open entry, assumed 0.50 price-unit round-trip cost.
- Software fault/stress suite: 240/240 pass (20 categories × 12 variations).
- Synthetic market-path suite: 240/240 pass (12 archetypes × 20 variations).

## Selected holdout results
Assumed cost: 0.30 round-trip spread plus 0.10 slippage per fill; not observed broker cost.

| Fixed rule | Trades | Direction accuracy | Net PnL (price units) | Max drawdown | Profit factor |
|---|---:|---:|---:|---:|---:|
| last_return_sign | 19 | 0.421 | -11.50 | 13.36 | 0.248 |
| ema20_trend | 19 | 0.632 | -2.54 | 4.40 | 0.728 |
| mean_reversion_z1 | 16 | 0.438 | -13.53 | 15.23 | 0.135 |
| prior20_range_breakout | 3 | 0.333 | -1.69 | 1.69 | 0.091 |
| flat_no_trade | 0 | n/a | 0.00 | 0.00 | n/a |

This tiny holdout shows why directional accuracy is not equivalent to profitable trading: EMA20 has 63.2% direction accuracy but negative net PnL. It is descriptive only and is not evidence about the gold market generally.

## Models and label policy
- Direction logistic candidate: `insufficient_history`, zero OOS predictions; default minimum history 463 bars; target is candle-body direction only.
- TP-before-SL logistic candidate: `insufficient_history`, zero OOS predictions, raw/un-calibrated and not connected to live decision gate. Train only on trade outcome labels that matured by the forecast index.
- Same-bar TP/SL collisions default stop-first because OHLC does not reveal intrabar order; `target_first`/ `exclude` policies are sensitivity options. Gap-through-open is only a rough fill proxy.
- Cost sensitivity uses assumed spreads 0, 0.1, 0.3, 0.5, 1.0 plus 0.1 slippage per fill; these are assumptions, not actual broker observations.

## Data expansion risk
Remote dataset server advertises 8,887,001 rows, but widely separated indices inspected had non-global chronological ordering. A contiguous 500-row region from 8,886,400–8,886,899 was chronologically ordered, with overnight/session gaps, but was not imported. Downloader has strict validation and should remain fail-closed. Do not join separated offsets or silently sort away provenance anomalies.

## Interpretation
Satisfying software scenarios validates assertion behavior only; it does not validate predictive skill. A 60-bar one-session sample cannot establish stable alpha or probability calibration. No ML model is validated, no live BUY/SELL output is enabled, live provider integrations and broker-specific replay are untested, and deployment wasn't verified. Production status remains **research/UX prototype; decision gate = `NO_TRADE`**.

## v1.1.0 release verification and online dataset candidate — 2026-10-16

### Reproduced package evidence
- Release archive: `aurum-intelligence-v1.1.0.zip`; 223,353 bytes; SHA-256 `4f3aafa682d1777a1dc6d220ad65b4b931b76760dafa8700818f8d1c0c18e623`.
- Base release-builder report: pytest 153 passed, unittest 92 passed, compileall PASS, audit no hard failures (unknown timezone warning only), baseline benchmark PASS, external-data benchmark smoke PASS, release ZIP extraction/retest PASS.
- Bundled-sample hash remains `dae535a16e13cf49f123231b1ee72e19787259e7a5f943b017a013346f36a93b`, 60 M5 bars. The package smoke benchmark chooses 50 valid decisions from this one excerpt and writes 250 comparison rows; clustered times remain statistically dependent.
- Portable online-runner smoke (local fixture only): 30 cases × 5 baselines = 150 rule comparisons, 480 synthetic market path scenarios over two anchors, and 240 software input cases; 100% of expected-valid/expected-invalid contracts detected. Runtime observed about 26 ms on this local environment. This is not performance evidence for the upstream market data or for strategy profitability.
- GitHub Actions benchmark contract (new branch): the hosted job must fetch current upstream CSV, record its SHA-256 and timestamp range, run 50 spread-out cases = 250 baseline comparisons, at least 200 scenario cases, and preserve CSV/JSON/Markdown artifacts. The live dataset result is still pending; do not claim an observed upstream row count/hash until its run artifacts are verified.

### New public-source candidate DS51
- Repository: `getdata-finance/xauusd-5m-ohlcv-metals-historical-data`, raw CSV URL recorded in `data_source_register.csv`.
- The source describes community-aggregated OHLCV. Current broker, timezone, price-side and actual fill semantics are not independently verified. A public repository link or MIT license badge does not automatically settle licensing for commercial derivatives or redistribution.
- DS51 remains `reference_only` until a live runner succeeds and a separate data-rights/provenance review approves its intended use. Raw OHLCV is not committed into the repository artifact.
- The generated 1,200 paths are counterfactuals. The 240 software cases are input-validation checks. Neither set increases the number of historical observations or independent market regimes.
- Decision output remains `NO_TRADE`; no live signal, model validation, or order execution was enabled by this change.


## Final v1.1.0 rebuild — verified 2026-10-16

After adding the source-register entry and the portable online benchmark suite, the release was rebuilt from a clean verification path. Final artifact SHA-256 is 259103a2bf406498048e31c8ea97970c9f1ecd6bbf7636bd3bed07d7ca0ecb6e (245,370 bytes; 98 files in ZIP; manifest lists 51 sources). Final verification: pytest 169 passed; unittest discovery 102 passed; compileall PASS; source dataset audit has zero hard failures and one expected unknown-timezone warning; baseline and online-smoke benchmark entry points PASS; HTML IDs/inline JS checks PASS; extraction test reran pytest and passed. The source remains a research prototype, and remote acquisition of DS51 is pending the hosted GitHub Actions run.


## Final hosted verification — 2026-10-16, run 38851436990

The GitHub Actions run is **green**, including unit tests, live acquisition, dataset/count assertions, job summary and artifact upload. Run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990
Artifact: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990/artifacts/12025522071
Artifact SHA-256: `7c661df1afafc372fd2816f2074ceb05aa7af0f8c7a388e64f07aa6b4bc43ef9` (9,208 bytes; retention expires 2026-10-30 04:35:53Z).

Fresh online sample: 35,954 rows, 3,070,226 bytes, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`; timestamp range `2026-03-26T02:30:00+00:00`–`2026-09-25T20:40:00+00:00`; median interval 300 seconds; 136 intervals longer than 600 seconds. The remote report ran 50 timestamp cases / 250 rule comparisons, 1,200/1,200 market-path contracts and 240/240 software robustness contracts. Runtime 458.588 ms. All counts/assertions passed; decision output remains `NO_TRADE`.

The +00:00 offset is the representation seen in the file, not independent proof of the provider's intended timezone. Broker, quote-side, license scope and fill semantics remain unverified. DS51 remains `reference_only`. The positive PnL of the breakout baseline comes from only seven trades and cannot establish alpha. See `benchmarks/live_dataset_benchmark_2026-10-16.md` and its JSON companion.

## v1.2.0 current release — UX plus latest hosted online evidence

Local build date: 2026-10-16. Release archive v1.2.0: 260,781 bytes, SHA-256 19912b690e74af802a7170316703d40048a67731a1eb26d806c9929f048d2db3. Verification: 176 pytest, 109 unittest, compileall pass; inline Mini App JavaScript syntax pass; 48 IDs, no duplicates; ZIP integrity and extraction/retest pass.

Local packaged dataset remains 60 community bars with unknown timezone and broker provenance. This run evaluates 50 timestamps × five fixed baselines = 250 output rows; 1,200 generated market-path contracts and 240 software stress cases passed. The barrier candidate reports insufficient_history and zero OOS predictions: it has no calibrated predictive probability. Output stays NO_TRADE.

Separately, hosted GitHub Actions run [#38851629049](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851629049) validated one public community dataset: 35,954 rows; SHA-256 acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db; range 2026-03-26 through 2026-09-25; 50 decision cases / 250 rule evaluations; 1,200/1,200 synthetic paths; 240/240 software cases. The repository declares MIT but data-rights scope/provenance still need review, timestamps/broker quote semantics and actual costs are unverified, and this 2026-09-25 sample is not current live data. prior20_range_breakout netted +2.37 units on only seven trades under assumed costs; not sufficient evidence for alpha.

The v1.2.0 interface offers Sederhana/Tingkat lanjut modes; mode selection is presentation only. The Mini App stores the choice locally; Telegram mode settings persist in SQLite with migration. No model accuracy, win rate, or world-leading superiority claim is made.

## v1.3.0 full-history hosted benchmark — 2026-10-16

Run [#38888666520](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38888666520) passed on source commit d790435b9f09d623a795aa6ab7ee4d1732515aa5. Detailed Markdown and JSON reports: projects/aurum-intelligence/benchmarks/full_history_replay_2026-10-16.md and .json.

- Dataset: 35,954 rows, SHA-256 acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db, dates 2026-03-26 through 2026-09-25.
- Full replay after 60-bar warm-up: 35,893 opportunities per rule. Five disjoint fold sizes 7,175 / 7,176 / 7,176 / 7,176 / 7,190; fold 5 is exactly the final 20% tail.
- Assumed round-trip costs: 0.30 spread plus 0.10 slippage each fill, total 0.50 price units/trade; not measured broker costs.
- Full / tail net PnL: last_return_sign -19,531.435 / -3,766.090; ema5_vs_ema10 -17,497.475 / -3,577.310; mean_reversion_z10 -8,710.560 / -1,762.260; prior20_range_breakout -2,290.290 / -480.350. All active rules lose in both.
- The selected-case breakout +2.37 across seven trades is not robust; full history is -2,290.29 across 4,035 trades.
- Prequential baselines, 35,893 forecasts: constant Brier .250000, expanding .250019, rolling-288 .250749. Expanding-vs-constant Brier improvement interval [-0.0000639, +0.0000412] spans zero.
- 1,200/1,200 market-path and 240/240 software checks pass. They validate runner contracts, not predictive skill.
- Keep NO_TRADE. No calibrated TP-before-SL forecast or broker-executable backtest is validated.

## v1.3.1 hosted calibration CI and replay verified

Run [#38897894594](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38897894594) PASS on commit `6017eec3141fb803907583abeca006a230e90be4`. Artifact #12044756186, SHA-256 `a2561de0161e79dbaac794988b5545cd8f32fe2cf55d38c875e8aed91e99d9b3`.

The artifact confirms 35,954 public XAUUSD M5 rows, 35,893 sequential opportunities per baseline over five chronological folds, 50 point-case inputs / 250 baseline comparisons, 1,200/1,200 market-path scenarios, 240/240 software robustness cases, and 240/240 temporal-calibration synthetic contracts. All hosted validation/count gates passed. Calibration scenario data are synthetic and not forecast-skill evidence. The source CSV period ends 2026-09-25 and source-rights/broker/timezone semantics remain unverified. The current decision remains `NO_TRADE`.
