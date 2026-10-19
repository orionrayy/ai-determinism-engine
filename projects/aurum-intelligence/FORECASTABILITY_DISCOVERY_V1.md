# Serevex Causal Forecastability Discovery — v1 Research Prototype

**Status:** isolated research prototype; synthetic contract tests only; not integrated into a trading path; not a claim of original-in-the-world mathematics or predictive alpha.

## Research question

Can a model/regime competence gate that relies exclusively on matured out-of-sample forecasts identify which registered experts provide incremental probability skill over a causal baseline, while suppressing unsupported or unstable experts?

This is a candidate mechanism to test against simpler alternatives, not a claim of novelty. Online expert weighting, proper scoring rules, conditional forecasting and multiple-testing controls all have prior art. The potentially differentiating hypothesis is the exact combination of Serevex's as-of regime contract, matured-label ledger, time-gap/session-bounded bootstrap, fixed model × regime hypothesis family and fail-closed output. Prior-art review is required before making an originality claim.

## Mathematical contract

For each forecast made at decision index t, model probability is p[k,t], causal baseline probability is b[t], and binary outcome y[t] only becomes usable after label_available_index[t].

Incremental Brier skill is:

`d[k,t] = (b[t] - y[t])^2 - (p[k,t] - y[t])^2`

Positive skill favors the model. At as-of index T, only rows satisfying both decision_index < T and label_available_index < T are scored. Regime identifiers must have been available no later than their own forecast decision. All registered models must have aligned decision IDs and identical labels/target-maturity metadata and baseline probabilities to prevent cherry-picked model coverage.

For every preregistered pair (model k, regime r), estimate the mean of d[k,t]. Obtain uncertainty from deterministic moving-block resampling, splitting blocks at missing decision indices, changes of regime, and declared segment/session boundaries. A Bonferroni-adjusted one-sided percentile lower bound is computed across the fixed registered model × regime family. A pair is only ELIGIBLE_FOR_INDEPENDENT_REPLICATION if it clears predeclared sample/block support floors and its corrected lower skill bound exceeds minimum_skill.

Eligible pair weights are normalized positive lower-bound skill within the requested regime. Weights do not establish calibrated probabilities; they only form an experimental blend to score in a later, untouched cohort.

## Invariants

- No training, calibration, selection or reporting path may read an outcome before its maturity index.
- Regime features must be point-in-time available; post-hoc labels must not be presented as as-of regimes.
- Registered experts are paired on the same decision/outcome cohort, including the same baseline probabilities.
- Blocks never cross time gaps or segment/session boundaries.
- All model/regime comparisons count toward a predeclared multiplicity family.
- An unsupported model is not treated as a zero-skill model; the distinction is reported.
- A positive synthetic result can only validate code behavior, never promote a market model.
- Output remains decision_output=NO_TRADE, model_validated=false, deployment_calibrated=false, live_signal_enabled=false and live_order_execution_enabled=false.

## Required comparisons before integration

1. Global matured-label base rate.
2. Static equal-weight expert blend.
3. Best single expert selected only on the earlier development/selection partition.
4. A causal rolling/prequential skill gate without regime conditioning.
5. The proposed regime-conditioned gate.
6. The same model without multiplicity correction, shown only as a diagnostic negative control.

The final holdout must not be used to choose regime definitions, model family, minimum sample, block length, alpha, or weights. Use paired same-row Brier/log-loss comparisons, block-bootstrap intervals, calibration diagnostics, regime-level support, turnover/cost-adjusted strategy metrics, and an untouched final cohort. For financial performance, the binary probability outcome and net portfolio return are separate endpoints.

## Connection points

- point_in_time_features_v1.py: as-of availability and external context state.
- external_context_benchmark_v1.py: current chronological development/calibration/holdout protocol and paired Brier diagnostics.
- temporal_semantic_contract_v1.py: semantic and temporal evidence contracts.
- selection_bias_guard_v1.py and portfolio_panel_selection_adapter_v1.py: trial-family and portfolio-level inference gates.

This v1 module is intentionally not called automatically by the current benchmark. Integration requires an out-of-sample forecast ledger for all registered candidates and a preregistered selection partition distinct from calibration and the final holdout. The correct next experiment is to run the gate side-by-side on development/selection forecasts, freeze the selection and weights, then evaluate the frozen gate on later independent data without retuning.

## Literature/prior-art starting points

- Makridakis, Spiliotis & Assimakopoulos (2018), *The M4 Competition: Results, findings, conclusion and way forward*, International Journal of Forecasting, DOI 10.1016/j.ijforecast.2018.06.001 — forecast combinations and hybrid approaches.
- White (2000), *A Reality Check for Data Snooping*, Econometrica, DOI 10.1111/1468-0262.00152 — inference under tested model/strategy families.
- Dawid (1984), *Present Position and Potential Developments: Some Personal Views: Statistical Theory: The Prequential Approach*, Journal of the Royal Statistical Society A — sequential evaluation and prediction assessment.
- Yao et al. (2004), *Measure of predictability*, Physical Review E, DOI 10.1103/PhysRevE.69.066121 — historical work on forecastability measures.

These citations establish neighboring ideas, not an exhaustive novelty search. A claim that the proposed combination has never existed requires a wider paper, code and patent review.
