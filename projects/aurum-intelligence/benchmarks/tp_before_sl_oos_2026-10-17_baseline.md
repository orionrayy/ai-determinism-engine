# Aurum TP-before-SL OOS benchmark — baseline results

**State:** CI_PASS_DIAGNOSTIC_ONLY. This is the verified output of hosted run [#39019576117](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39019576117) on source commit `8e1dabf70e0dd1df816424717c4d0efe56c90a0d`. Artifact ID `12096301869`, SHA-256 `69ac2c75a670fd033d43d0f242bb585a00996135cbc71e425f549e5e548b0727`, 49841 bytes; observed report runtime 141795.762 ms.

## Scope and protocol

The workflow downloaded and processed three community OHLCV files separately. Row counts sum to **140,164 without deduplication**; do not treat them as independent observations. The same run had 750 historical fixed-rule comparisons, 2,160 synthetic market-path contracts, and 720 software robustness contracts.

The diagnostic target is TP before SL within 12 future bars, tested separately for long and short. It uses ATR(14), TP at 1.5 ATR, SL at 1 ATR, next-bar-open entry, seven fixed OHLC-derived causal features, a fixed L2 logistic model, and chronological 60/20/20 train/calibration/untouched holdout. Unmatured labels are purged at period boundaries. Platt scaling is fit on middle-period OOS predictions with outcomes matured before the final holdout. If TP and SL touch within the same candle, the label is conservatively stop-first. Timeouts are label 0 but are reported separately.

## Results

Brier and log loss lower is better. The reference is the event-rate probability estimated from the calibration period only. Values below are descriptive holdout metrics, not live performance.

| Dataset | Side | Train | Calibration | Holdout | TP event rate | Raw Brier | Calibrated Brier | Brier gain (raw − calibrated) | Raw log loss | Calibrated log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| getdata_xauusd_m5 | long | 21546 | 7190 | 7179 | 0.3563 | 0.230024 | 0.229481 | 0.000543 | 0.652767 | 0.651526 |
| getdata_xauusd_m5 | short | 21546 | 7190 | 7179 | 0.3760 | 0.234666 | 0.234859 | -0.000193 | 0.662155 | 0.662575 |
| sai_community_xauusd_m5 | long | 10392 | 3471 | 3463 | 0.3341 | 0.222111 | 0.222110 | 0.000001 | 0.636223 | 0.636221 |
| sai_community_xauusd_m5 | short | 10391 | 3471 | 3463 | 0.3722 | 0.233999 | 0.234180 | -0.000181 | 0.660811 | 0.661191 |
| sai_community_xauusd_m1 | long | 52078 | 17357 | 17356 | 0.3539 | 0.228472 | 0.228498 | -0.000026 | 0.649431 | 0.649488 |
| sai_community_xauusd_m1 | short | 52079 | 17363 | 17356 | 0.3911 | 0.238133 | 0.238161 | -0.000029 | 0.669222 | 0.669283 |

The six source-side holdouts contain **55,996 label predictions across different source/timeframe/side combinations**, not 55,996 independent market observations.

## Interpretation

- Calibration improvement is **not consistent**. The GetData M5 long result improves modestly; GetData M5 short worsens. Sai M5 gains are essentially zero on long and worsen on short. Both Sai M1 directions show slight deterioration in Brier and log loss after calibration.
- Raw Brier skill against the calibration-period event-rate reference is small; the GetData M5 long result is negative (-0.00237). No durable predictive edge has been established.
- The first baseline artifact included 40-replicate marginal bootstrap intervals, but no paired confidence interval for the raw-minus-calibrated gains. A follow-on implementation adds a 250-replicate paired moving-block bootstrap and holdout-mutation invariance tests. Those are not considered passed until the next hosted workflow and artifact are verified.
- Cross-source M5 comparison remains **DIVERGENT**: 11,509 exact timestamps, direction agreement 97.93%, median relative close-price difference 1.358% (limit 0.25%), P95 2.058% (limit 0.50%). Cross-source confirmation remains blocked.
- Dataset rights, commercial-use permission, original provenance, time-zone semantics, broker quote side, source independence, spread, commission, slippage and actual fills remain unverified/unmeasured.

**Status remains:** `NO_TRADE`; `validated=false`; `deployment_calibrated=false`; `live_signal_enabled=false`; `live_order_execution_enabled=false`. The benchmark does not authorize production or support a “#1 in the world” claim.
