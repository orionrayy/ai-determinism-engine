# Aurum Intelligence v1.3.0 — implementation checkpoint

**Status at commit time:** local source release cleanly verified; the updated live-dataset CI run is pending for this new benchmark protocol. Trading decisions remain NO_TRADE and no live order execution is enabled.

## Engineering increment

- Full-history rule replay evaluates every eligible chronological next-bar opportunity after a 60-bar warm-up, rather than using only 50 timestamp-spread cases.
- Five disjoint chronological folds, a descriptive full-history summary, an early 80% segment and the final chronological 20% tail are reported for each fixed baseline. Rule definitions are fixed; no fitting or tuning is performed.
- Three prequential probability baselines are evaluated for next-candle body up versus not-up: constant 0.5, expanding Laplace base-rate, and rolling 288-bar Laplace base-rate.
- Metrics: Brier score, log loss, Brier skill, 10-bin reliability, descriptive ECE/MCE, and deterministic moving-block bootstrap intervals for paired Brier improvement.
- Benchmark runner is optimized to avoid rescanning every prior bar for each decision, which would be quadratic on a 35k-row dataset.
- CI gates assert decision-row count, fold partition coverage, probability forecast count, reliability metrics and continued absence of a calibrated or live model.

## Local v1.3.0 release verification

- Archive: aurum-intelligence-v1.3.0.zip, 267,854 bytes, SHA-256 51a82f710033eee0cb7723c8940721d8c3bcd64565159650ba90e6235945ff0b.
- pytest: 181 passed.
- unittest (application suite): 109 passed.
- Dedicated online benchmark suite: 11 passed.
- Python compileall, Mini App JavaScript syntax, 48 unique HTML IDs, ZIP integrity and clean-extraction pytest: PASS.
- Bundled smoke data stays 60 bars; 50 selected cases / 250 comparisons; 1,200 synthetic market paths and 240 software cases pass.
- Synthetic throughput validation exercised 35,954 generated rows: 35,893 sequential decisions per rule, five chronological folds and 35,893 prequential probability forecasts; 1,200/1,200 market paths and 240/240 software robustness cases passed. This validates throughput/contracts only, not market prediction.

## Evidence guardrails

- Prequential probabilities are historical-rate baselines, not a trained/calibrated model and not TP-before-SL probabilities.
- Brier/log-loss/ECE/MCE on a single aggregated OHLCV source are descriptive, and block-bootstrap uncertainty is approximate.
- Full-history rule metrics are descriptive; five folds show variability but do not create five independent data sources.
- Data license, source rights for intended use, timezone provenance, broker Bid/Ask, measured costs and actual fills remain unverified.
- Maintain NO_TRADE and do not claim world-leading accuracy or superiority without replicated out-of-sample evidence and comparable baselines.
