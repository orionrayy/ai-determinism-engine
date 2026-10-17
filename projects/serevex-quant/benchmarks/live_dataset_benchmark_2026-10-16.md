# Aurum live online dataset benchmark — 2026-10-16

**Verified result: PASS.** Hosted GitHub Actions [run #38851436990](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990) fetched the live community XAUUSD M5 CSV, passed its six unit tests and data/count assertions, and uploaded the evidence artifact [#12025522071](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990/artifacts/12025522071).

## Data identity and quality

- Upstream: [getdata-finance/xauusd-5m-ohlcv-metals-historical-data](https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data)
- Observed file size: 3,070,226 bytes.
- Rows: **35,954**.
- SHA-256: `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`.
- Timestamp range reported by the data: 2026-03-26 02:30 UTC to 2026-09-25 20:40 UTC.
- Median interval: 300 seconds; 136 intervals exceed 600 seconds.
- OHLCV schema, finite values, valid candle geometry, nonnegative volume, and strictly increasing timestamps passed. The raw data is not committed to Git.

The rows expose a `+00:00` offset, but the source's intended timezone, broker origin, quote side, volume semantics and licensing scope have not been independently verified. DS51 therefore remains `reference_only` pending provenance and rights review.

## Historical diagnostic replay

Fifty timestamps spread through the available chronology were evaluated with five fixed, untrained rules, creating 250 comparison rows. The replay assumes 0.30 price units spread plus 0.10 slippage for active trades; these are not broker-observed costs.

| Baseline | Trades | Direction accuracy | Net PnL units | Max drawdown units |
|---|---:|---:|---:|---:|
| Last-return sign | 50 | 52.00% | -19.12 | 24.79 |
| EMA5 vs EMA10 | 50 | 44.00% | -52.10 | 62.61 |
| Mean reversion z10 | 30 | 53.33% | -5.21 | 13.51 |
| Prior-20 range breakout | 7 | 71.43% | +2.37 | 6.41 |
| Flat / no trade | 0 | n/a | 0.00 | 0.00 |

The breakout result is only seven trades; it is far too small to establish an edge. The 50 decision cases are diagnostic, sparse, and not a calibrated untouched holdout. This is not a broker-executable backtest.

## Simulation and software robustness

- **1,200/1,200 market-path scenario contracts passed**: 5 real-data anchor windows × 12 archetypes × 20 variants. These are synthetic counterfactual perturbations, not additional observed market events.
- **240/240 software stress scenarios passed** across 20 input categories × 12 variations.
- Hosted benchmark runtime: approximately 458.6 ms.
- All test/assertion steps and artifact upload passed on run #38851436990.

## Decision boundary

The result demonstrates that the current ingestion, chronology, validation, fixed-baseline and report-generation path handled this upstream file. It does **not** demonstrate predictive skill, calibrated TP/SL probabilities, positive net-of-cost alpha or live-trading readiness. Production decision output remains `NO_TRADE`; predictive model validation and live order routing remain disabled.

Machine-readable record: [live_dataset_benchmark_2026-10-16.json](./live_dataset_benchmark_2026-10-16.json). The actual run artifact contains five JSON/CSV/Markdown evidence files and has a 14-day retention window.
