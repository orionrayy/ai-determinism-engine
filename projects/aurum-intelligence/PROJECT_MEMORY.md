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


## Gap audit and prioritization — 2026-10-17

- Added `GAP_AUDIT_AND_COMPETITIVE_POSITION_2026-10-17.md` on the existing draft PR branch. It consolidates the supplied 100-weakness/100-comparison inventory into evidence classes, risk-weighted remediation priorities, acceptance gates, category-appropriate competitor benchmarking, and free-first constraints.
- Correct distinction: the bundled TP-before-SL candidate still has 60 bars and zero OOS predictions; a separate hosted historical replay used 35,954 community XAUUSD M5 rows. Both facts can be true. The online dataset remains historical, not a live feed; licensing/intended-use rights, independent provenance, intended timezone, broker/quote-side semantics and actual fills remain unresolved.
- The current full-history replay does not establish predictive edge: active fixed-rule baselines lost under assumed costs and direction-only baselines did not show meaningful Brier skill over a constant reference. Synthetic scenario contracts remain software/stress evidence, not market observations.
- Keep `NO_TRADE`; no live execution or claim of validated/calibrated trading performance is authorized. The audit is documentary and does not modify application behavior.
