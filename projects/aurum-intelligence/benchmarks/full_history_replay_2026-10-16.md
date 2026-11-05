# Aurum v1.3.0 — full-history benchmark evidence

**Hosted run:** [#38888666520](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38888666520) — success on source commit d790435b9f09d623a795aa6ab7ee4d1732515aa5.  
**Artifact:** ID 12040248097, 14,319 bytes, SHA-256 8a21fda87e8031b52b7ab2bada70b2dff58b5cefe402728a1afcea121a397dbe.

## Dataset and evaluation coverage

- Source: [GetData Finance XAUUSD M5 community CSV](https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data).
- 35,954 rows, 3,070,226 bytes; CSV SHA-256 acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db.
- Observed timestamps 2026-03-26 02:30 UTC to 2026-09-25 20:40 UTC; median interval 300 seconds; 136 intervals exceed 600 seconds. Historical input, not current live data.
- All 35,893 eligible next-candle opportunities per fixed rule were replayed after a 60-bar warm-up. Five contiguous/disjoint folds contain 7,175 / 7,176 / 7,176 / 7,176 / 7,190 observations; fold 5 is exactly the final 20% tail holdout.
- Separate selected-case output: 50 timestamps × 5 fixed rules = 250 comparisons.
- Synthetic counterfactual market paths 1,200/1,200 pass; software robustness cases 240/240 pass.
- Hosted runtime 14.57 seconds, environment-specific rather than a latency SLO.

## Full-history versus chronological tail

Costs are assumptions: 0.30 price units round-trip spread plus 0.10 slippage per fill, total 0.50 units/trade. Not broker-observed costs or actual fills; PnL is price units, not account-currency profit.

| Fixed rule | Full trades | Full accuracy | Full net PnL | Full PF | Tail trades | Tail accuracy | Tail net PnL | Tail PF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| last_return_sign | 35,825 | 48.80% | -19,531.435 | 0.667 | 7,182 | 49.94% | -3,766.090 | 0.666 |
| ema5_vs_ema10 | 35,893 | 48.63% | -17,497.475 | 0.697 | 7,190 | 48.79% | -3,577.310 | 0.681 |
| mean_reversion_z10 | 18,881 | 52.08% | -8,710.560 | 0.719 | 3,869 | 51.54% | -1,762.260 | 0.712 |
| prior20_range_breakout | 4,035 | 47.36% | -2,290.290 | 0.701 | 897 | 49.28% | -480.350 | 0.689 |
| flat_no_trade | 0 | n/a | 0.000 | n/a | 0 | n/a | 0.000 | n/a |

All four active baselines lose in the full replay and final tail under the cost assumptions. The older selected-case breakout was +2.37 units on seven trades; full history is -2,290.29 across 4,035 trades. The small positive subset is not evidence of alpha.

## Prequential probability diagnostics

Target is next-candle candle-body up vs not-up. Predictions use only outcomes observed through each decision close. These are historical-rate baselines, not a trained model, TP-before-SL probability, or calibrated trading signal.

| Baseline | N | Brier | Log loss | Brier skill vs 0.5 | ECE descriptive | MCE descriptive |
|---|---:|---:|---:|---:|---:|---:|
| constant_50pct | 35,893 | 0.250000 | 0.693147 | 0.000000 | 0.005335 | 0.005335 |
| expanding_laplace_base_rate | 35,893 | 0.250019 | 0.693186 | -0.000076 | 0.001107 | 0.007291 |
| rolling_288_bar_laplace_base_rate | 35,893 | 0.250749 | 0.694651 | -0.002995 | 0.024608 | 0.027133 |

Approximate moving-block-bootstrap 95% interval for mean Brier improvement vs constant 0.5:
- Expanding rate: estimate -0.000019; interval [-0.0000639, +0.0000412], includes zero — no improvement established.
- Rolling-288 rate: estimate -0.000749; interval [-0.0009196, -0.0005358], worse than constant 0.5 in this sample.

ECE/MCE and intervals are descriptive/approximate. The folds are segments of one dataset, not independent sources or regimes.

## Decision and limitations

- Mechanical benchmark PASS; predictive-model validation NO; trading gate NO_TRADE.
- The source repository advertises an MIT license, but intended-use rights scope and source provenance need review. Timestamp offset is not independent proof of provider timezone semantics.
- Broker identity, quote-side, Bid/Ask/tick history, measured costs, actual fills, swaps and latency/no-fill are unavailable.
- Do not claim world-leading accuracy, stable alpha or profitability. Next gates: licensed broker data, independent regimes, external holdout, exact-target calibration and paper trading.
