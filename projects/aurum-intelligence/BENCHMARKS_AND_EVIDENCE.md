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


## v1.4.0 P0 evaluator and cross-source disagreement — 2026-10-17

Hosted run [#39022220573](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39022220573) passed on branch head `fe6ed83b0ec242e2d8e81317c2c798dac00f93bb`. Artifact #12096704482: 2,556,935 bytes, SHA-256 `94d432018fa313db265bc01dbeba1e37df5efc0a3c90f65eaec432259877f7de`.

Primary community XAUUSD M5 source: 35,954 rows, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`. Secondary `Sai310421/xauusd-data`: 17,371 rows, SHA-256 `ee23e18f0ed77160893936d0312c008261e4f2ac16d82365898aba20ea26de9d`. Across 11,509 exact timestamp overlaps, exact OHLC match count is zero; absolute close difference median/p95/p99/max is 134.885/203.661/217.617/220.412 bps. Signed primary-minus-secondary close difference median is +63.515 price units (p05 +0.295, p95 +94.445). One-bar return correlation is 0.99845. Classification: `SOURCE_DISAGREEMENT_REVIEW_REQUIRED`; this is not a clean source-consistency result. Both sources remain reference-only and must not be merged/shifted automatically.

The evaluator produced 94,604 candidate signal events, 94,084 binary-scored outcomes and 520 same-bar ambiguous collisions (excluded). It reports 70/10/20 chronological segments and keeps the final 20% holdout scoring-only for the Platt calibrator. Raw vs calibrated holdout metrics vary by rule/side; calibration is not uniformly better. P0 is still blocked on source/time/data rights and broker-specific execution evidence. `NO_TRADE` remains active.

## GDELT coverage-integrity correction and external-context rerun — 2026-10-17

Hosted end-to-end run [#39035732248](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39035732248) **PASS** on tested head `f3eaa644a1f73c0d4f9521870e1ba5c460cfaf69`. Artifact #12102745058, 2,624,921 bytes, SHA-256 `b1710712519fd48e3a5f29acf1d532707997821bba1948e49b093004fd7b81f0`. Unit/regression tests: 115 passed; temporal calibration contract scenarios: 240/240 passed. The same run passed the 1,313,280 paired Bid/Ask M1 candle gate, 20,000 valid ATR anchors and 120,000 TP/SL path scenarios.

### GDELT acquisition versus temporal coverage

- 141,020 unique real GDELT GKG observations from 100 selected archives; 85,809 archive entries were discovered. All 100 selected archives were processed. Total acquired bytes: 701,820,752; no synthetic/duplicated observations were added.
- Requested historical window: 1,313,279 minutes. Actually covered by acquired archive intervals: 1,500 minutes, **0.114218%**. The 100-archive sample is a bounded time-spread sample, not a full-window archive census.
- Status fields are intentionally distinct: pipeline/unique-observation scale `PASS`; source acquisition `PASS`; full source completeness `PARTIAL`; temporal news-feature coverage `NOT_MET` (0 eligible of 19,937 mature unambiguous labels). Because required 1h/24h windows are not covered, news values are null/ineligible rather than fabricated zero-news observations.
- COT temporal feature coverage is 19,739/19,937 rows (99.0069%, `PARTIAL`). BLS yielded 119 descriptive observations but no authoritative publication-time calendar, so they do not count as causal macro features. FRED/ALFRED returned `SKIPPED_CREDENTIAL_REQUIRED` because `FRED_API_KEY_NOT_CONFIGURED`. News and macro ablations remain `NOT_ELIGIBLE_NO_CONTEXT_FEATURES_IN_DEVELOPMENT`.

### Same-row model comparison (diagnostic only)

- Price-only and price+COT/combined ablations each scored the same 3,987 chronological holdout rows. Price-only Brier: 0.1609893; price+COT/combined Brier: 0.1609044.
- Incremental Brier improvement for price+COT versus price-only: +0.00008494; paired moving-block 95% interval [-0.00014777, +0.00032012] crosses zero. This is inconclusive evidence about incremental value, not a robust edge. No best-model or alpha claim is made.
- Decision output remains `NO_TRADE`; model deployment calibration, validation for profitability and live signals remain false. Data provenance/rights, source lineage/quote-basis mismatch, measured broker costs, independent market regimes, untouched holdout and paper-trading gates remain open.

### Failure retained and corrected

- Run [#39035356832](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39035356832) failed at the external-context runner after earlier test and price gates passed. The report used an acquisition-local `gdelt_archive_window_complete` variable outside its scope. The runner now reads the value from `source_details["gdelt_gkg"]["archive_window_complete"]`, with a contract test. The failed run remains in the audit trail and is not treated as successful evidence.


## 2026-10-17 — evidence-backed source qualification CI

- Hosted run [#39038107206](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39038107206) **success** on code/test head `d2621fcda08456984ec3adef9015f91d3d432ddd`; unit/regression suite 125/125, temporal-calibration contracts 240/240. Community-source acquisition, Bid/Ask audit, large M1 scenario gates, point-in-time context evaluation, and artifact upload all succeeded.
- Reproducible artifact: [#12102563009](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39038107206), 2,626,545 bytes, SHA-256 `993c27155e00c7d57d87aba5c475c1751d7026fcbfce4345b784996215bc065c`. Local downloaded ZIP matched the published digest.
- New source qualification helper separates research evidence (provenance, rights, timestamp semantics, bar convention, quote basis) from execution evidence (broker authorization, measured costs, fill semantics). The CI asserts no source can be promoted automatically and that every source without evidence stays `REFERENCE_ONLY`/`BLOCKED`.
- Data scale: 1,313,280 paired M1 Bid/Ask candles (30 months), 20,000 valid ATR anchors, 875,611 eligible contiguous windows, and 120,000 TP/SL scenarios. The output is still a research diagnostic, not a profitability claim.
- Price-data discrepancy remains material: primary/secondary M5 feeds have 11,509 overlapping timestamps but no exact OHLC matches; absolute close difference median/P95 134.885/203.661 bps. Tick audit identifies 4,391 exact Bid-OHLC matches out of 4,392 bars against the secondary source whose README claims mid-quote; likely lineage/basis review is required before any independence claim.
- Context benchmark: 141,020 unique GDELT observations reached the numeric count target, but 100 selected archives cover only 0.114218% of the historical window. COT supports 19,739/19,937 rows; macro/news are ineligible. Incremental COT-vs-price-only Brier change +0.00008494; paired 95% interval [-0.00014777,+0.00032012] crosses zero.
- Decision remains `NO_TRADE`; deployment/model validation/live signal flags remain false. The draft PR is not merged.
