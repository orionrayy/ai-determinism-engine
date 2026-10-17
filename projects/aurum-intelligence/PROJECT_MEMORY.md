# Project memory

**Product:** Aurum Intelligence, a modular XAUUSD research, analysis, charting, risk-gating and Telegram/Mini App assistant. **Checkpoint date:** 2026-10-16, Asia/Jakarta. **Goal:** a rigorous decision-support product—not merely confident-sounding BUY/SELL messages.

## Product principles
1. Abstain as `NO_TRADE` when source quality, history length, calibration, execution cost, regime applicability, or model evidence is insufficient.
2. Keep different objectives separate: next-candle direction, next-candle return, and TP-before-SL are not interchangeable labels or probabilities.
3. Use chronological walk-forward evaluation and an untouched final holdout; no random shuffle and no label leakage from trade outcomes that have not matured.
4. Include broker-relevant costs, spread, slippage, gaps, intrabar barrier ambiguity, and multiple-testing controls before claiming an edge.
5. Preserve a reproducible evidence ledger: source provenance, data hash, model/code version, test commands, metrics, decision and failure reason.

## Architecture in the v0.9.0 source snapshot
- Telegram bot/webhook app and routing: `app/main.py`, `app/telegram/api.py`.
- Settings and secret templates: `app/config.py`, `.env.example`.
- Technical analysis / decisions: `app/engine/analysis.py`, `decision.py`, `cross_asset.py`, `chart.py`.
- Data validation and market features: `app/engine/market_data.py`.
- Fixed-rule replay and diagnostics: `backtest.py`, `real_data_cases.py`, `research_benchmark.py`, `logistic_research.py`.
- TP-before-SL research: `barrier_benchmark.py`, `barrier_probability.py`.
- Scenario labs: `scenario_lab.py`, `market_scenario_lab.py`.
- Market/news adapters: `app/providers/oanda.py`, `news.py`; persistence: `app/infra/database.py`.
- Mini App: `webapp/index.html`.
- Scripts: `scripts/download_public_xauusd.py`, `run_research_benchmark.py`, `run_validation_benchmark.py`, `build_release_manifest.py`, `set_webhook.py`.
- Data: `data/xauusd_5m_public_sample.csv`, `data/data_source_register.csv`, `data/DATASETS.md`.
- Evidence: `benchmarks/`; tests: `tests/`.

## Evidence boundary
The source release’s validation report records 126 pytest passes, 92 unittest passes, `compileall` success and a Mini App JavaScript syntax check. Those are software/fixture checks; no live Telegram, OANDA account, MT5, cTrader or production deployment was validated. The complete code artifact available at checkpoint was the v0.9.0 ZIP; later v1.0.0 notes must not be treated as present in that source unless reconciled and rerun.

## Data status
The canonical sample has 60 public community OHLCV bars from a short M5 XAUUSD session. Timezone and independent broker provenance are unknown; there are no actual fills or broker Bid/Ask/tick data. The source register has 50 candidate references, only 1 sampled and 49 reference-only. Direct downloading of an expanded dataset did not complete in the local runtime because outbound DNS/network was unavailable. A contiguous 500-row web-inspected slice was not added to the canonical data; other distant row-index points showed non-global chronological ordering, so do not concatenate separated offsets.

## Provider and account boundaries
OANDA code is a market-data adapter, not evidence of a connected brokerage account. Telegram live integration was not tested due missing authorized bot credentials. MT5 and cTrader personal-account connectors are not implemented. News/macro adapters exist, but there is no validated point-in-time event-impact model. TradingView alerts do not equal a personal-account integration.

## Safety and security
Do not commit tokens, API keys, OAuth refresh tokens, account identifiers, .env secrets or private logs. Any later order-routing work requires explicit authorization, least-privilege credentials, risk caps, idempotency, audit logs, sandbox/paper trading, and a separate security review. No promise of profit or world-leading performance is evidenced.

## Latest checkpoint override — v1.1.0 (2026-10-16, Asia/Jakarta)

The previous paragraphs describe the v0.9.0-era checkpoint. They are retained as historical evidence; the verified latest source artifact is now **Aurum Intelligence v1.1.0**, built from the reconciled v1.0.0 archive plus the current v1.1.0 changes. The new ZIP is a downloadable conversation artifact, not a binary committed to this repository.

- Release ZIP: `aurum-intelligence-v1.1.0.zip`, 223,353 bytes, SHA-256 `4f3aafa682d1777a1dc6d220ad65b4b931b76760dafa8700818f8d1c0c18e623`.
- Fresh release builder checks: pytest **153 passed** at the base v1.1 build stage; unittest **92 passed**; compileall, dataset audit, bundled benchmark, online-benchmark smoke, HTML/JS checks and extracted-ZIP retest passed. The full pytest count increases once the additional online-runner test module is included; see latest progress evidence for the final rerun.
- Primary bundled source remains 60 public community OHLCV bars, SHA-256 `dae535a16e13cf49f123231b1ee72e19787259e7a5f943b017a013346f36a93b`. Source timezone and broker/executable quote semantics remain unknown.
- Source acquisition register is extended to 51 entries; DS51 is a new public GetData Finance OHLCV candidate and stays `reference_only` until online CI actually fetches and validates the current upstream CSV.
- Added stdlib-only online runner and hosted CI contract: 50 chronology-spread real timestamp cases × 5 fixed rules = 250 evaluations; 5 source anchors × 12 scenario archetypes × 20 variants = 1,200 generated market paths; 20 software input categories × 12 variations = 240 software checks.
- Local tests cover the runner with bundled/fixture data. The hosted live network run has not yet completed at this checkpoint; only its first successful workflow artifact can establish the observed upstream row count and current source hash.
- Current decision gate remains `NO_TRADE`. Scenario validation is a software-contract measure, not evidence of alpha; there are no calibrated or validated live forecasts and no live order routing.


## Final v1.1.0 rebuild — verified 2026-10-16

After adding the source-register entry and the portable online benchmark suite, the release was rebuilt from a clean verification path. Final artifact SHA-256 is 259103a2bf406498048e31c8ea97970c9f1ecd6bbf7636bd3bed07d7ca0ecb6e (245,370 bytes; 98 files in ZIP; manifest lists 51 sources). Final verification: pytest 169 passed; unittest discovery 102 passed; compileall PASS; source dataset audit has zero hard failures and one expected unknown-timezone warning; baseline and online-smoke benchmark entry points PASS; HTML IDs/inline JS checks PASS; extraction test reran pytest and passed. The source remains a research prototype, and remote acquisition of DS51 is pending the hosted GitHub Actions run.


## Final hosted verification — 2026-10-16, run 38851436990

The GitHub Actions run is **green**, including unit tests, live acquisition, dataset/count assertions, job summary and artifact upload. Run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990
Artifact: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990/artifacts/12025522071
Artifact SHA-256: `7c661df1afafc372fd2816f2074ceb05aa7af0f8c7a388e64f07aa6b4bc43ef9` (9,208 bytes; retention expires 2026-10-30 04:35:53Z).

Fresh online sample: 35,954 rows, 3,070,226 bytes, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`; timestamp range `2026-03-26T02:30:00+00:00`–`2026-09-25T20:40:00+00:00`; median interval 300 seconds; 136 intervals longer than 600 seconds. The remote report ran 50 timestamp cases / 250 rule comparisons, 1,200/1,200 market-path contracts and 240/240 software robustness contracts. Runtime 458.588 ms. All counts/assertions passed; decision output remains `NO_TRADE`.

The +00:00 offset is the representation seen in the file, not independent proof of the provider's intended timezone. Broker, quote-side, license scope and fill semantics remain unverified. DS51 remains `reference_only`. The positive PnL of the breakout baseline comes from only seven trades and cannot establish alpha. See `benchmarks/live_dataset_benchmark_2026-10-16.md` and its JSON companion.

## Latest checkpoint override — v1.2.0 (2026-10-16, Asia/Jakarta)

This section supersedes older version-specific statements above where they differ. Those are retained as historical records.

- Verified source artifact: aurum-intelligence-v1.2.0.zip, 260,781 bytes, SHA-256 19912b690e74af802a7170316703d40048a67731a1eb26d806c9929f048d2db3. The archive is a downloadable project artifact; the complete ZIP is not a binary tracked in this repository.
- Clean source release checks: pytest 176 passed, unittest 109 passed, compileall passed, inline Mini App JS syntax passed, 48 HTML IDs with no duplicates, ZIP integrity passed, and clean extraction reran 176 pytest tests.
- v1.2.0 implements Sederhana/Tingkat lanjut progressive disclosure in the Mini App, the same mode selector in Telegram preferences, SQLite storage with an in-place legacy migration and invalid-value rejection. Mini App choice is browser-local; no server synchronization is claimed. Mode selection is presentation only.
- Local benchmark still uses one 60-bar community sample. 50 observed decisions × 5 fixed baselines = 250 rows; 1,200 market-path contracts and 240 software stress contracts pass. TP-before-SL model has zero OOS predictions and remains insufficient-history/uncalibrated.
- Hosted online benchmark run #38851629049 passed on the online-benchmark branch: 35,954 public XAUUSD rows with SHA-256 acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db, 50 timestamp cases / 250 rule evaluations, 1,200/1,200 synthetic paths and 240/240 software stress checks. The file period ends 2026-09-25 and is not a live market feed.
- Upstream repository advertises an MIT license, but data rights scope/provenance and broker/execution semantics remain under review; raw market data is not redistributed. Example breakout PnL of +2.37 price units occurred on only seven trades under assumed costs and is not evidence of stable alpha.
- User-experience modes do not change the model, confidence or risk rules. Production remains blocked on independent multi-regime walk-forward calibration, broker-specific costs, paper trading, security review and authorized provider integration. Do not claim world-leading accuracy.

## Latest checkpoint override — v1.3.0 hosted full-history run (2026-10-16)

- Release aurum-intelligence-v1.3.0.zip: 278,402 bytes, SHA-256 53dabb5e86c8ce74691790e4c865d2624667cf531e572818bc571527bd4c2b67; clean build passes 181 pytest, 109 unittest, 11 benchmark tests, compileall, Mini App static checks and ZIP extraction.
- Hosted run #38888666520 succeeded on commit d790435b9f09d623a795aa6ab7ee4d1732515aa5. Dataset: 35,954 rows, SHA-256 acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db, 2026-03-26 through 2026-09-25.
- Full replay is 35,893 sequential opportunities per fixed rule; five disjoint folds contain 7,175 / 7,176 / 7,176 / 7,176 / 7,190 observations, final fold equals last 20%.
- All active fixed baselines are net negative in full and tail under assumed costs. The selected-case breakout positive does not survive complete replay.
- Prequential direction baselines do not establish Brier improvement over constant 0.5; no calibrated TP-before-SL model is validated. Decision remains NO_TRADE.
- Machine-readable evidence and detailed report are under projects/aurum-intelligence/benchmarks/full_history_replay_2026-10-16.json/.md. Raw OHLCV is not redistributed.
- Next: review rights/provenance/timezone, acquire licensed broker Bid/Ask/tick data and measured costs, validate independent chronological regimes and external holdout, calibrate exact target, and paper trade before production.

## Latest v1.3.1 hosted CI checkpoint — 2026-10-16

- Release ZIP v1.3.1: 276,128 bytes; SHA-256 `b63af5ab3412e7cb0bac14deffdb0d8cf07e024b0e46364d6fe5e1329d831d5b`; local release passes 185 pytest, 109 unittest, compileall, JS/HTML audits and clean extraction retest.
- Hosted run [#38897894594](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38897894594) passed on code commit `6017eec3141fb803907583abeca006a230e90be4`. The Platt calibration unit tests and 240/240 synthetic contract cases passed; online dataset acquisition, full-history counts, 1,200 market paths, 240 software robustness scenarios and evidence upload also passed.
- New artifact #12044756186, SHA-256 `a2561de0161e79dbaac794988b5545cd8f32fe2cf55d38c875e8aed91e99d9b3`. Full dataset remains 35,954 community rows ending 2026-09-25. Active fixed-rule baselines remain net-negative under assumed costs; no production-calibrated forecast exists.
- Synthetic 720-row calibration illustration is strictly a code/metrics check, not financial accuracy evidence. Actual bundled market sample still yields zero OOS TP-before-SL predictions; decision stays `NO_TRADE`.


## Latest implementation checkpoint — multi-source benchmark and eligibility gates (2026-10-17)

### Source branches and release boundaries
- Canonical implementation remains this repository and the Aurum working branch feature/aurum-multi-source-benchmark.
- Draft PR #165: https://github.com/orionrayy/ai-determinism-engine/pull/165 (base feature/aurum-online-dataset-benchmark). It is open, intentionally draft, and unmerged.
- Tested code head: 8db8bd90830b78b0007a6904a104a80c851f3464. Latest hosted CI run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/39016857786.
- Final CI artifact #12095247324 is 38,308 bytes, SHA-256 eadd77b94dffa51408c34b6aa8da63406070b6ac01da0056ba6fee6cb283503a, expires 2026-10-31 08:19:17 UTC. Local snapshot is /mnt/data/aurum-readiness-gates-final.zip. It includes derived JSON/Markdown evidence and test reports, never raw source CSVs.
- Test evidence on the tested head: 8 temporal-calibration unit tests; 25 Python unit/regression tests; single-source and multi-source online benchmark downloads, report validation, scenario assertions and artifact upload all pass.
- Coverage on the multi-source run: three data files; 140,164 sum of rows, explicitly not deduplicated/independent; 750 actual-dataset fixed-rule comparisons; 2,160 synthetic market-path contracts; 720 synthetic software robustness contracts; about 37.52 s runtime.

### Current architecture increment
- Added projects/aurum-intelligence/online-benchmark/multi_source_benchmark.py as a separate runner, not a replacement for the older v1.3.1 benchmark entry point.
- Added source-level SHA-256, row counts, time ranges, median/P95/max interval/gap indicators, timezone-policy disclosure, quote-side/provenance/rights claims and a source-eligibility object.
- Added exact-UTC timestamp M5 overlap, absolute and relative price differences, signed OHLC bias, monthly buckets, five paired-candle examples and a bounded timestamp-offset probe. It never shifts timestamps, fills gaps, or pools source bars automatically.
- Added explicit readiness gates that separate diagnostic pipeline PASS from data-use and production approval. The code requires source-price compatibility, independent lineage, verified broker/timezone/quote-side semantics, underlying-data rights and commercial-use approval before it could mark downstream gates eligible. Live order execution remains hard-blocked.
- Regression tests prove unverified quote-side blocks cross-source confirmation and production readiness even if other hypothetical fixture gates are green.

### Results from three public community data files
1. GetData XAUUSD M5: 35,954 rows, SHA-256 acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db, observed 2026-03-26 02:30 UTC through 2026-09-25 20:40 UTC.
2. Sai community XAUUSD M5: 17,371 rows, SHA-256 ee23e18f0ed77160893936d0312c008261e4f2ac16d82365898aba20ea26de9d, observed 2026-02-25 through 2026-05-26 UTC. Timestamps are naive in CSV and assumed UTC from the README; not independently verified.
3. Sai community XAUUSD M1: 86,839 rows, SHA-256 de74973a63c06ef43de56b616647eb2e380d50cb0ed22a388f158c0bcc828e13, observed 2026-02-25 through 2026-05-26 UTC. Also timestamp-naive and assumed UTC from README.

### Blocking M5 disagreement
- The two M5 files share 11,509 exact UTC timestamps and close-to-close direction agrees for 97.933% of eligible sequential pairs, but price levels are not interchangeable: median absolute close delta 63.515 price units, P95 94.445, median relative delta 1.3580% and P95 2.0576%. Thresholds for a basic comparable status are median relative delta <=0.25% and P95 <=0.50%; status is DIVERGENT.
- Signed B−A close delta median is -63.515 and mean is about -57.864; B closes below A in 95.10% of matched timestamps. Monthly median signed delta varies significantly: March -94.105, April -68.200, May -56.710 price units. At the end overlap sample 2026-05-26 00:00 UTC the paired close delta is only -0.235. This pattern is not a stable additive offset, but the source's root cause is not proven.
- Bounded offsets tested: -25,200, -10,800, -7,200, -3,600, -600, -300, 0, +300, +600, +3,600, +7,200, +10,800, +25,200 seconds. The best observed offset is -7h, but improves median relative delta by only 6.3458%, below the configured 20% material-candidate criterion. Diagnosis NO_OBVIOUS_BOUNDED_SHIFT_FIX. No timestamp shift is approved or applied.
- As a result: source comparability BLOCKED; cross-source confirmation BLOCKED; no source-level inference should combine these series as if equivalent independent feeds.

### Rights and lineage decision
- GetData's GitHub sample advertises MIT; GetData provider documentation describes purchaser-only standard data terms and restricts embedding raw CSV/ZIP files in paid/commercial services absent separate written permission. The relationship of the GitHub sample MIT declaration to those provider terms is not resolved. Do not interpret this as a blanket finding that derived signals are banned; the exact rights scope for this dataset and intended paid model-output service needs written clarification.
- Sai's GitHub repository advertises MIT and says the Q1-Q2 bars are aggregated from public broker mid-quote feeds; exact upstream feed identities, quote construction, underlying data rights and original timezone are not independently verified. File-specific history inspected for the M1/M5 CSVs shows them present in initial commit 01a9d1c1544eeea9632a29c1b3cde7ea0d13afe6 (2026-06-30 UTC) without a visible per-month source checksum/provenance trail. Weak auditability is a caveat, not proof of fabricated data.
- A public GitHub mirror of Dukascopy XAUUSD M1 BID/ASK files was identified as a diagnostic candidate. It lacks a clear repository license; it was therefore not incorporated into CI or production and its data was not redistributed. Only revisit after rights and provenance are established or a licensed direct feed is acquired.
- Machine-readable gates in IMPLEMENTATION_STATUS.json now report pipeline and test success separately from blocked comparability, commercial rights, commercial model training, and production readiness. Current values: NO_TRADE; predictive_model_validated=false; deployment_calibrated=false; cross_source_confirmation_allowed=false; live_signal_enabled=false; live_order_execution_enabled=false; production_ready=false.

### Next required sequence
1. Obtain written clarification for GetData's public GitHub sample and its relationship to provider terms; verify Sai feed lineage and quote calculation.
2. Obtain a licensed, provenance-documented broker/provider source with Bid/Ask/tick semantics, timestamps/timezone and commercial rights appropriate to the intended service.
3. Cross-check exact M1/M5 candles/price sides against that licensed feed; retain independent series rather than normalizing away the current discrepancy.
4. Implement execution-aware costs/fill handling and evaluate exact TP-before-SL target on matured outcomes, independent chronological regimes and an untouched final holdout.
5. Only after evidence gates pass, proceed to paper trading, end-to-end Telegram/provider acceptance, security/risk review and a separate decision on production order capability.


## Target-specific TP-before-SL OOS benchmark — 2026-10-17

- New candidate implementation is on stacked draft PR #165 branch `feature/aurum-multi-source-benchmark`: `online-benchmark/barrier_oos_benchmark.py`, tests, runner integration, report and CI assertions.
- The diagnostic target is TP before SL within 12 bars with ATR(14), TP=1.5 ATR, SL=1 ATR, next-bar-open entry, conservative stop-first same-bar ambiguity, gap-open checks, and timeout failure labels. Long and short are separate targets.
- Chronology contract: train/calibration/final holdout = 60/20/20; scaler/model fit on training only; training labels must mature before calibration starts; calibrator fit uses middle-period OOS forecasts whose outcomes mature before final holdout; no final-holdout model selection. Platt helper and proper-score/reliability/block-bootstrap diagnostics are reused from `temporal-calibration-v1.3`.
- This remains diagnostic only. Data source rows have community OHLCV semantics; provenance, rights, broker quote side and actual fills are not verified. Do not promote or claim calibrated trading skill from it; maintain `NO_TRADE`, `deployment_calibrated=false`, `validated=false`, `live_signal_enabled=false`, and `live_order_execution_enabled=false`.
- New tests and CI acceptance rules were committed, but a fresh hosted run and artifact review are still required before this checkpoint may be marked green. See `ONLINE_MULTI_SOURCE_BENCHMARK.md` and the dated entry in `PROGRESS_LOG.md`.


## Gap audit and prioritization — 2026-10-17

- Added `GAP_AUDIT_AND_COMPETITIVE_POSITION_2026-10-17.md` to the project’s audit trail. It converts the broad 100-weakness / 100-comparison inventory into risk-weighted priorities, acceptance criteria, evidence classes and free-first trade-offs.
- Correct distinction: the bundled TP-before-SL candidate has 60 bars and zero OOS forecasts; a separate hosted historical replay consumed 35,954 community XAUUSD M5 rows. The online dataset is not live and its rights, independent provenance, timezone, quote side and actual fills remain unverified.
- The full-history replay did not establish an edge under assumed costs; calibration diagnostics are not deployment validation. Keep `NO_TRADE`; do not claim global superiority without reproducible, independently validated OOS evidence.


## Verified TP-before-SL baseline and paired uncertainty — 2026-10-17

- Verified hosted run #39019576117 passed on `8e1dabf70e0dd1df816424717c4d0efe56c90a0d`. Artifact #12096301869, SHA-256 `69ac2c75a670fd033d43d0f242bb585a00996135cbc71e425f549e5e548b0727`, 49,841 bytes; pipeline runtime 141,795.762 ms. Persisted summary: `benchmarks/tp_before_sl_oos_2026-10-17_baseline.md/.json`.
- In that artifact's exact TP-before-SL holdout, calibration point estimates are mixed; e.g. GetData M5 long Brier improved by 0.000543 while GetData M5 short worsened by 0.000193. Other results are near zero or negative. Raw skill vs event-rate reference is small and sometimes negative. Marginal CIs in that old artifact use 40 replicates and do not quantify paired calibration gain.
- New unverified candidate adds paired 250-replicate moving-block bootstrap for raw-minus-calibrated Brier/log-loss gains. Positives mean the calibrator reduced proper-score loss; if a 95% interval includes zero, do not claim a reliable gain from that comparison. Final holdout remains for evaluation, not model/threshold selection.
- New regression test mutates only holdout bars and requires identical training parameter hash and calibration fit parameters, directly checking separation of future data from training/calibration. Workflow acceptance also verifies bootstrap count, interval ordering, probability-score consistency and unchanged `NO_TRADE` safety flags.
- Fresh hosted CI/artifact review is pending. Source rights/provenance/timezone/quote-side, broker costs/fills, source independence, predictive validity, deployment calibration and live order routing remain blocked. Keep `NO_TRADE` and all live-signal/order flags false.


## Verified paired OOS and bid/ask tick probe — 2026-10-17

- Latest hosted run [#39021204557](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39021204557) passed all 13 workflow steps on code commit `f2706642ed8da7c7727b53881ec7e045473ae06c`. Artifact #12096921783: 55,004 bytes, SHA-256 `cca5e4e8c735b24261bfafc336a749a5422d97c3cd7472196dd4bc8dcd22ab27`.
- Multi-source pipeline passed: 3 community/public OHLCV files, 140,164 rows summed without deduplication, 750 fixed-rule comparisons, 2,160 synthetic market-path contracts, 720 software robustness contracts and 55,996 source/side TP-before-SL holdout predictions. These totals do not represent independent market observations. Multi-source runner runtime was 135,372.536 ms.
- TP-before-SL calibration remains diagnostic: all 12 paired 95% intervals for Brier/log-loss raw-minus-calibrated gains (six source/side pairs; 250 paired moving-block bootstrap replicates) include zero. Calibration is not shown to improve reliably; do not tune or select a model using this final holdout.
- Bid/ask probe read 3,000 quote records from three disjoint 1,000-row windows in the 286,823,542-row `CarlosSilva1/xauusd-ticks` dataset via 30 API pages and 614,982 response bytes. Within these sampled windows there were zero timestamp-order violations, crossed quotes or locked/zero-spread quotes. Only aggregate statistics and per-window hashes are retained; raw quotes were not committed/artifacted.
- Sampled median spread in price units was 0.417 (head, 2021-05-24), 0.377 (middle, 2024-07-10) and 1.040 (tail, 2026-05-22); P95 was 0.447, 0.41415 and 1.504 respectively. This is quote data, not measured fills/slippage or a full-period spread model.
- Dataset card claims CC-BY-4.0, but upstream rights, broker lineage, timestamp/quote semantics and commercial approval are unverified. Production cost calibration and commercial training remain blocked. Detailed reports: `benchmarks/tick_quote_probe_2026-10-17.md/.json` and `benchmarks/tp_before_sl_oos_2026-10-17_paired_ci.md/.json`.
- No application ZIP was rebuilt; v1.3.1 remains the latest documented app artifact. PR #165 remains draft. Safety remains `NO_TRADE`, with model validation/deployment calibration/live signals/live order execution false.
