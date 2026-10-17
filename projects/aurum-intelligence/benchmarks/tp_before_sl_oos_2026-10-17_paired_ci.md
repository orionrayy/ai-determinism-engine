# Aurum TP-before-SL OOS — paired calibration uncertainty

**Verification:** [GitHub Actions run #39020164520](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39020164520) passed all 13 recorded workflow steps on code commit `d44f500c629ef80309bbf66da259412691b6925d`. Artifact #12095778983 (51,437 bytes), SHA-256 `0130fc4319c7d6243ca294125baf47343fad0645f22a007a02a1ef95e86fc11a`. Report runtime: 175729.18 ms. This verifies reproducible execution and contracts; it is not a production model validation.

## Evaluation protocol

Target: take-profit before stop-loss within 12 forward bars, long and short separately. ATR(14), TP=1.5 ATR, SL=1 ATR, entry next bar open, seven causal OHLC features. Chronological 60/20/20 train/calibration/final holdout. Labels that cross a split boundary without having matured are purged. A fixed L2 logistic model is trained only in the first period; Platt calibration fits only from matured middle-period OOS forecasts. Same-bar barrier collisions resolve stop-first, and timeout labels are counted separately.

The new paired moving-block bootstrap uses the *same* sampled 12-bar blocks for raw and calibrated probabilities, with 250 deterministic replicates and seed 20261019. Positive raw-minus-calibrated score gain favors calibration.

## Observed results

Brier score and log loss are lower-is-better. The gains below are raw score minus calibrated score. All confidence intervals are paired 95% percentile bootstrap intervals.

| Dataset | Side | Train | Calibration | Holdout | TP event rate | Raw Brier | Cal Brier | Brier gain 95% CI | Raw log loss | Cal log loss | Log-loss gain 95% CI | Positive Brier gain replicates |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|
| getdata_xauusd_m5 | long | 21546 | 7190 | 7179 | 0.3563 | 0.230024 | 0.229481 | [-0.000710, 0.001658] | 0.652767 | 0.651526 | [-0.001523, 0.003709] | 81.6% |
| getdata_xauusd_m5 | short | 21546 | 7190 | 7179 | 0.3760 | 0.234666 | 0.234859 | [-0.001058, 0.000600] | 0.662155 | 0.662575 | [-0.002281, 0.001286] | 34.0% |
| sai_community_xauusd_m5 | long | 10392 | 3471 | 3463 | 0.3341 | 0.222111 | 0.222110 | [-0.000074, 0.000079] | 0.636223 | 0.636221 | [-0.000168, 0.000175] | 49.6% |
| sai_community_xauusd_m5 | short | 10391 | 3471 | 3463 | 0.3722 | 0.233999 | 0.234180 | [-0.000469, 0.000120] | 0.660811 | 0.661191 | [-0.000986, 0.000253] | 18.0% |
| sai_community_xauusd_m1 | long | 52078 | 17357 | 17356 | 0.3539 | 0.228472 | 0.228498 | [-0.000067, 0.000012] | 0.649431 | 0.649488 | [-0.000145, 0.000027] | 10.0% |
| sai_community_xauusd_m1 | short | 52079 | 17363 | 17356 | 0.3911 | 0.238133 | 0.238161 | [-0.000090, 0.000022] | 0.669222 | 0.669283 | [-0.000190, 0.000048] | 15.2% |

**Interpretation:** all 12 paired 95% intervals (Brier and log loss across six source-side pairs) include zero. Positive point estimates appear only in selected cases; calibration is not a consistent gain. No deployment calibration or predictive edge is established.

## Data and safety boundary

The three OHLCV inputs sum to 140,164 rows without deduplication; this is not an independent sample count. Existing source-overlap diagnostics remain divergent: 11,509 matched timestamps, 97.93% close direction agreement but 1.358% median relative close-level difference (0.25% threshold), P95 2.058% (0.50% threshold). Cross-source confirmation stays blocked.

The final holdout has now been inspected for this fixed protocol. Do not use it for model selection or tune toward these results; subsequent changes require a new, predeclared independent holdout. Upstream data rights, provenance, intended time zone, quote semantics and actual broker costs/fills remain unverified.

Current gate: NO_TRADE. Predictive model validated = false; deployment calibrated = false; live signals = false; live order execution = false; production approval = blocked.
