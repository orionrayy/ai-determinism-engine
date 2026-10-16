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