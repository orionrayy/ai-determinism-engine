# Serevex Baseline Advantage — Real-Data Findings v0.1–v5.0

**Evidence status:** full GitHub Actions run #336 succeeded; this memo records the completed real-data research result. It does not authorize trades or model promotion.

## 1. Verified result

- Run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/39368843774 — all 20 workflow job steps succeeded.
- Isolated method tests: https://github.com/orionrayy/ai-determinism-engine/actions/runs/39368843795 — 73 tests passed.
- Input: 35,954 reference OHLCV rows; 17,747 forecast ledger decisions; 3,573 final-holdout scored rows.
- Dataset SHA-256: `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`.
- Ledger SHA-256: `c039987a51efd6915fa637e3ca705a8547925c922f28226309a3a0ab5abbcabf`.
- Scored-cohort SHA-256: `4648f1a1b0388a377b3d1d705cf52d73e3d2a27a255eedc466ca2dee42c84bd6`.
- Metrics SHA-256: `bc7030a9a2cebbb83de1510643b544d629fe77b3f08476997eca82704f216ac6`; report SHA-256: `8f974950285109ee72798661fcadd56a42b701c713e1757420a5101073da7f5e`.

## 2. Exact mechanism that makes the baseline hard to beat

The producer's v0.1 probability is the Laplace-smoothed expanding prevalence of previously matured labels from the same strategy-rule/side cohort:

`p_t = (positive_matured_labels_before_t + 1) / (matured_labels_before_t + 2)`.

The information set is filtered by `label_available_index < decision_index`; target labels use the registered TP-before-SL definition. This gives a low-variance reference probability that adapts as outcomes mature without using model features whose historical availability is not independently verified.

The specialist rates are derived from that same target stream, not from independent evidence: a rolling window of the last 288 matured events, a regime subset and a volatility subset. The conditional estimators use Laplace smoothing and fall back to the global prior below a 20-row support threshold. The tested v0.2–v4.9 policies then blend correlated rates with the baseline using fixed, hand-specified weights. Such mixtures can improve a point estimate, but they also introduce estimation variance and can follow short-lived subgroup noise.

## 3. Fifty-checkpoint holdout comparison

Baseline v0.1 on the final holdout: Brier `0.23005422`, log loss `0.65275465`, ECE `0.01502580`.

The best descriptive point-estimate result was v4.5: Brier `0.22988638` and log loss `0.65239310`. Absolute gains were only `0.00016784` Brier and `0.00036155` log loss. Its paired 95% source-time-block bootstrap intervals were Brier `[-0.00107018, 0.00071876]` and log loss `[-0.00219877, 0.00153087]`; both include zero.

Across the 48 non-baseline candidates, 19 had both point losses lower, but 0/48 had both paired interval lower bounds strictly above zero. All 48 had a lower descriptive ECE than v0.1, yet some had materially worse Brier/log loss. ECE is binning-dependent and cannot replace proper scoring rules or independent replication.

v5.0 selected v0.1 for all 17,747 decisions: 419 decisions had inadequate matured block support and 17,328 failed the prequential dual-score gate. The selected count is not a tie or a runtime error; it is the intended fallback result when no specialist passes the evidence gate.

## 4. Why the alpha candidate failed

The frozen calibrated alpha candidate scored worse than v0.1 on this cohort. In the integrated v4.0 reference path its Brier was `0.23200446` versus baseline `0.23005422`, log loss `0.65656360` versus `0.65275465`, and ECE `0.04853555` versus `0.01502580`. The calibrated alpha model's separate unguarded Brier was `0.23233952` and ECE `0.04901466`.

The uncalibrated/raw alpha forecast had a slightly better Brier point estimate (`0.22975551`) but worse ECE (`0.02615875`) and no confirmed paired-block Brier advantage. Its guarded Brier was `0.22969494`; the paired interval still crossed zero. The alpha shrinkage governor selected weight `0.0`, retaining the baseline.

This points to two distinct effects: (a) the raw feature model did not establish reliable incremental information against the strong prevalence prior; (b) calibration reduced reliability on this chronological cohort, consistent with calibration/distribution-shift risk. The artifact does not include enough evidence to attribute the calibration deterioration to one coefficient or one feature alone.

## 5. Temporal/data caveats

The result is real-data computation but not fully source-qualified market evidence. The workflow explicitly keeps false flags for provider provenance, data rights, true source timestamp semantics, historical feature availability and executable broker quote equivalence. Current-candle OHLCV-derived features use bars at or before the decision index, but actual bar-close availability is unverified. A simple matured-label prior is less dependent on those feature-time assumptions than the alpha feature path, which may contribute to its practical robustness on the reference dataset.

The 12-bar scoring window/12-bar embargo, same-cohort hashes, strict maturity cutoff, sequential alpha-spending and no-live-authority checks passed. Bootstrap intervals remain descriptive and are not finite-sample guarantees.

## 6. Next experiment — preregister before any new holdout

Do not choose v4.5 because it topped this holdout. Do not tune weights against these 3,573 rows. The next experiment should test **support-adaptive hierarchical shrinkage** on development/model-selection/calibration partitions only:

1. Keep the exact expanding Laplace v0.1 as the immutable reference.
2. Replace fixed mixture weights with a support-adaptive residual: `p_conditional = p_global + w(n) * (p_subgroup - p_global)`, where `w(n)=n/(n+k)` and `k` belongs to a small preregistered grid.
3. Estimate `k` only in the development partition; freeze the candidate after model selection, and use the chronological calibration partition solely for a preregistered dual-score/stability/calibration gate.
4. Permit only one final evaluation on a new untouched chronological cohort. Require both Brier and log-loss conservative lower screens above zero, positive-block share at least 70%, adequate independent block support, and no material calibration degradation.
5. If the candidate fails, keep v0.1. If it passes, replicate on another untouched period/asset and on qualified executable bid/ask data before considering any downstream decision use.

## 7. Authority

`NO_TRADE` remains the output. Model-validated and deployment-calibrated are false; live signal and live order execution are disabled. A score improvement is not proof of profitability.