# Serevex baseline refinement v0.1a–v5.0a

**Status:** research-only; the exact causal baseline is still the safety reference and fallback. `NO_TRADE` remains enforced.

## Version reset and intent

The new a-series resets the hypothesis labels. **v0.1a is the old v4.5 deterministic blend recipe**, not a declaration that v4.5 was validated or should replace the causal baseline. The legacy v4.5 recipe was selected after inspection of the prior final-holdout results; comparisons with v0.1a on that same cohort are post-selection and descriptive, not independent confirmation. Its earlier paired source-time-block intervals crossed zero. The original expanding matured-label baseline remains separately represented as `CAUSAL_BASELINE` and is always the fallback.

## Focused 50-checkpoint sequence

| Range | Controlled change | Reason to test |
|---|---|---|
| v0.1a | Legacy v4.5 blend recipe, exact alias | Reproduce the prior hypothesis under new naming. |
| v0.2a–v1.0a | Support-prior sweep (`k=20…180`) | Reduce noisy expert influence when causal support is limited. |
| v1.1a–v2.0a | Specialist-mass cap (`0.20…0.65`) | Control variance and preserve residual mass on the causal reference. |
| v2.1a–v3.0a | Rolling/regime/volatility contribution ablation | Identify which expert family adds incremental information. |
| v3.1a–v4.0a | Expert-disagreement guard (`0.04…0.22`) | Shrink blended influence when specialist forecasts conflict. |
| v4.1a–v4.9a | Joint lower-mass/support/disagreement sweep | Test conservative combinations instead of blindly maximizing complexity. |
| v5.0a | Prequential governor | Select only when support, dual-reference/dual-score, block-stability, and calibration gates pass; otherwise use the exact causal baseline. |

This is a controlled set of candidate policies, not fifty claims of increasing skill. Version numbers represent preregistered hypothesis families, not quality rankings.

## Checks and governance

- **Causal integrity:** labels affect later decisions only when `label_available_index < current_decision_index`; duplicate indices, malformed maturity flags, non-finite probabilities, and malformed support counts fail closed.
- **Selection:** 48 candidates require lower screening bounds above zero on both Brier and log-loss versus both the exact causal baseline and the v0.1a anchor. At least 70% of source-time blocks must be positive. Repeated selection looks use summable alpha-spending; the governor re-checks only every eight new closed blocks to reduce redundant work.
- **Calibration:** reliability bins, ECE, maximum bin gaps and regime/volatility slices are audited. ECE against a constant global-prevalence baseline is reported but is not the sole acceptance gate because a condition-aware probability model can improve proper scoring rules while being less globally flat. ECE must not degrade beyond tolerance against v0.1a, and dual-score superiority is still required against both references.
- **Uncertainty:** paired 12-bar source spans with 12-bar embargo are bootstrapped on the same eligible source-window rows for each reference/candidate comparison. The full-holdout point estimate uses all scored holdout rows, while the bootstrap interval excludes embargo-window rows and equally weights eligible source blocks; those estimates can differ, even in sign. The report now emits both estimands, row/block counts, and uses identical resampled block indices for Brier and log-loss. Intervals remain diagnostic rather than universal finite-sample guarantees.
- **Reproducibility:** dataset/ledger cohort, forecast manifests, metric aggregates, selection trace and reports receive SHA-256 hashes.
- **No-trade authority:** the gate cannot authorize deployment, live signals, or order execution. No static parameter is selected from aggregate final-holdout scores; only matured prior observations can affect a later prequential decision.

## Audit correction (v5.0a.3)

The first full-data audit revealed that full-holdout point estimates and source-window bootstrap means could have opposite signs because embargo-window rows were excluded only from the bootstrap. That was an estimand-alignment/reporting defect, not proof that either estimate is correct. v5.0a.1 emitted full-cohort and source-window/block-weighted gains separately and shared resampled block indices across Brier and log-loss. A second cross-check found that the online calibration gate was still pooling embargo-window labels while its Brier/log-loss evidence excluded those rows. v5.0a.2 computed calibration evidence for online selection from the same matured non-embargo source-window cohort and failed closed if calibration and score-gate row counts diverged. A third cross-check hardened the end-of-ledger contract: expected support now excludes labels unavailable before the final simulated decision, and expected versus observed source-window label counts must reconcile. It also verifies exact row-wise equality between the causal baseline and every decision where the governor says it selected the baseline fallback; the full governed probability stream is cross-checked when all decisions select baseline. The workflow also explicitly asserts that model validation, deployment calibration, live signals, and order execution remain disabled. The final-holdout calibration table remains a full-cohort descriptive diagnostic. Promotion remains blocked unless interval-aligned lower bounds, data qualification, and independent replication gates pass.

## Interpretation boundary

The v0.1a alias preserves the **recipe**, not the old winning holdout score. For each a-series candidate the report shows gains and paired intervals against both references. If the gate remains on `CAUSAL_BASELINE`, that is the correct evidence-governed outcome. A green workflow confirms code contracts and execution, not predictive edge, data provenance/use rights, broker-feed equivalence, or profitability after costs.

## Acceptance condition

Do not promote a specialist unless it clears the registered gates and is replicated on a new, later untouched chronological cohort, after source provenance, point-in-time feature availability, and broker-specific execution assumptions are qualified. Keep `NO_TRADE` until those independent research, calibration, economic, and operational gates pass.
