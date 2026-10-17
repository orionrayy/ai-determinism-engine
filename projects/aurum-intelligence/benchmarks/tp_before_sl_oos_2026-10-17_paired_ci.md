# TP-before-SL OOS — paired calibration uncertainty, latest hosted run

Run [#39021204557](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39021204557) passed all 13 steps on commit `f2706642ed8da7c7727b53881ec7e045473ae06c`. Protocol: long/short TP-before-SL within 12 bars, ATR(14), TP 1.5 ATR, SL 1 ATR, next-open entry, chronological 60/20/20 split, matured-label purging, fixed L2 logistic model and calibration-only Platt fitting. Paired moving-block bootstrap: 250 replicates, block size 12, seed 20261019; positive raw-minus-calibrated gain favors calibration.

| Dataset | Side | Train / calibration / holdout | Event rate | Raw Brier | Cal Brier | Brier gain 95% CI | Raw log loss | Cal log loss | Log-loss gain 95% CI | Reps with Brier gain |
|---|---|---:|---:|---:|---:|---|---:|---:|---|---:|
| getdata_xauusd_m5 | long | 21546 / 7190 / 7179 | 0.3563 | 0.230024 | 0.229481 | [-0.000710, 0.001658] | 0.652767 | 0.651526 | [-0.001523, 0.003709] | 81.6% |
| getdata_xauusd_m5 | short | 21546 / 7190 / 7179 | 0.3760 | 0.234666 | 0.234859 | [-0.001058, 0.000600] | 0.662155 | 0.662575 | [-0.002281, 0.001286] | 34.0% |
| sai_community_xauusd_m5 | long | 10392 / 3471 / 3463 | 0.3341 | 0.222111 | 0.222110 | [-0.000074, 0.000079] | 0.636223 | 0.636221 | [-0.000168, 0.000175] | 49.6% |
| sai_community_xauusd_m5 | short | 10391 / 3471 / 3463 | 0.3722 | 0.233999 | 0.234180 | [-0.000469, 0.000120] | 0.660811 | 0.661191 | [-0.000986, 0.000253] | 18.0% |
| sai_community_xauusd_m1 | long | 52078 / 17357 / 17356 | 0.3539 | 0.228472 | 0.228498 | [-0.000067, 0.000012] | 0.649431 | 0.649488 | [-0.000145, 0.000027] | 10.0% |
| sai_community_xauusd_m1 | short | 52079 / 17363 / 17356 | 0.3911 | 0.238133 | 0.238161 | [-0.000090, 0.000022] | 0.669222 | 0.669283 | [-0.000190, 0.000048] | 15.2% |

All 12 paired 95% intervals (Brier and log-loss across six source/side pairs) include zero. Calibration changes are small and inconsistent. No reliable calibration gain or validated edge is established. Do not tune or select a model based on the final holdout. Cross-source price levels remain divergent and cross-source confirmation is disabled.

Gate remains `NO_TRADE`; model validation and deployment calibration false; live signals and live orders disabled.
