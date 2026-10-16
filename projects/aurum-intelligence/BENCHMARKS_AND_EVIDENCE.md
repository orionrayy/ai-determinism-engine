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