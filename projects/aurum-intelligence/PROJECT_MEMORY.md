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
