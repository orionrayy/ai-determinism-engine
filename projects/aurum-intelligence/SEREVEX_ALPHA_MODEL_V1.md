# Serevex Alpha Model v4 — Embargoed Dual-Score and Out-of-Domain Governed Probabilities — Fit, Selection, Calibration and Holdout Contract

**Status:** implemented on the research branch; hosted validation is required.  
**Runtime:** Python standard library only.  
**Authority:** research only; output is always `NO_TRADE`; live signals and orders remain disabled.

## Purpose

This component adds a genuine trained forecaster to Serevex's stage 1–3 ledger. It is a regularized logistic classifier over eight causal OHLCV-derived features: ATR fraction, trend efficiency, one- and five-bar ATR-scaled momentum, candle-body/range ratio, close location in range, bar-range/ATR ratio and log volume versus its prior-20-bar median. Feature definitions and transform hashes are embedded in the ledger. All source times and bar-close availability remain unverified; the features are reference-only.

## Chronological fit protocol

The split points are derived from eligible source-bar indices before filtering to the selected rule/side event cohort. Labels must be binary and mature, and `label_available_index < cutoff_index` is required at each fit/calibration boundary.

1. **Development:** the alpha expert emits 0.5 cold-start forecasts; at the end of this block it fits a scout logistic model using only development labels matured before model selection starts.
2. **Model selection:** the scout's forecasts are made with fixed default regularization. Candidate L2 strengths (0.01, 0.1, 1.0) are compared using labels in model selection that mature before calibration starts. This selection metric is not presented as final-holdout evidence.
3. **Calibration:** a refit uses only development + model-selection samples whose labels mature before calibration starts. The frozen refit predicts calibration observations. A Platt-style logistic calibrator is fitted only on calibration labels that mature strictly before final holdout starts. Insufficient support falls back to identity calibration and is recorded.
4. **Final holdout:** the refitted model and calibrator are frozen. Final-holdout labels never participate in model fitting or calibration. Outcomes are read only to calculate evaluation metrics after predictions have been generated.

The trainer hashes model parameters, training-label identity and calibration-label identity. The ledger records the model artifact hash, per-row prediction stage, model-selection candidate metrics, sample counts, calibration status, reliability bins and paired alpha-versus-causal-baseline Brier/log-loss metrics with a deterministic paired moving-block bootstrap interval. Holdout-label mutations cannot alter the model/governor artifact hash or holdout predictions.

## Limits

A trained model is not automatically a predictive edge. A positive Brier difference in one historical dataset is not independent replication, net profitability, or evidence of world-first mathematics. The fixed rule/side cohort, reconstructed timestamps, unresolved source rights and lack of independently verified broker quotes remain material limitations. The existing atlas can mark a result for independent replication only; the delayed-feedback mixer also remains diagnostic-only. Every stage keeps `model_validated=false`, `deployment_calibrated=false`, `live_signal_enabled=false`, `live_order_execution_enabled=false`, and `decision_output=NO_TRADE`.

## Tests

From `projects/aurum-intelligence/online-benchmark`:

    python -m unittest -v test_serevex_alpha_model_v1.py
    python -m unittest -v test_serevex_forecast_ledger_producer_v1.py


## Probability governor v2 update — 2026-10-19

The alpha forecast now has a conservative baseline-shrinkage stage. The calibration partition is divided chronologically: an early portion fits Platt scaling, and a later disjoint portion selects a fixed convex blend weight from `{0, 0.1, 0.25, 0.5, 0.75, 1}`. Selection compares paired Brier loss against the causal baseline using non-overlapping chronological block means and a conservative 2.58-standard-error screen across nonzero candidates. This is an engineering guardrail, not a formal finite-sample significance guarantee.

If calibration support is inadequate or no alpha blend shows a positive conservative lower screen, the selected weight is exactly zero: the final-holdout alpha expert then emits the row-specific causal baseline, not an unsupported alpha estimate. When selected, the weight and calibrator are frozen before holdout; the baseline input remains point-in-time for each decision. The model artifact hash includes the calibration-fit and blend-selection label hashes and selected weight, but excludes final-holdout labels and metrics.

Final-holdout reporting now includes raw alpha, calibrated alpha and governed blend Brier scores; log loss; a deterministic paired moving-block bootstrap interval for baseline-minus-governed-alpha Brier improvement; and a binned reliability/ECE diagnostic. The interval and ECE are descriptive diagnostics and do not replace independent replication or selection-bias correction.


## Probability hardening v3

The governor requires candidate alpha/baseline weights to pass both a conservative Brier-improvement screen and a log-loss-improvement screen. It also requires at least 70% of chronological non-overlapping calibration blocks to improve on the causal baseline for each score. An alpha candidate that wins one proper score but loses the other, or wins only through a small number of blocks, is rejected. The zero-weight baseline remains the fail-closed default.

At final holdout, a training-scaler-only out-of-domain (OOD) diagnostic flags feature vectors with at least 25% of features beyond four training standard deviations (missing/invalid values also count). Flagged rows use the exact row-specific causal baseline regardless of the selected blend weight. The OOD gate uses features and training-only scaling, not holdout outcomes. Its holdout fallback rate and per-row diagnostics are reported.

This improves conservatism but cannot prove statistical validity under arbitrary dependence or distribution shift; the block screen is heuristic. Final holdout remains read-only for post-prediction evaluation.


## Temporal-block independence hardening v4

Governor uncertainty is no longer estimated by grouping consecutive *event rows*, which can have overlapping targets. Rows are grouped by absolute source decision index into 12-source-bar scoring windows followed by 12-source-bar embargo windows. Only non-empty scoring windows are used for the dual-score screen; the block-level mean improvement gives each source-time block equal weight. With the fixed 12-bar maximum target horizon, a scored block's outcomes mature before the next scored block begins. The same embargoed block design is used for the paired final-holdout Brier bootstrap diagnostic.

The policy now requires at least eight non-empty source-time blocks and sufficient scored rows. It records span and embargo parameters in the model artifact hash. This is a conservative dependence-control heuristic; irregular sampling, data-provider timestamp semantics, residual dependence and selection bias still require independent scrutiny.
