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
