# Serevex Forecastability Discovery v2 — research design

**Status:** experimental candidate; isolated draft PR; not integrated into the application or live decision path. The design does not claim never-before-seen mathematics, predictive superiority, or tradable alpha.

## Why v2 exists

Version 1 uses one skill estimate per model/regime pair, Bonferroni-adjusted percentile block-bootstrap bounds, and weights based on a corrected lower bound. It is useful as a minimum viable evidence gate, but it has three gaps:

1. It does not explicitly require skill to persist over predeclared horizons / recent windows.
2. Its separate model-regime intervals do not explicitly exploit synchronized paired resamples to preserve cross-model loss dependence.
3. A score that wins in a long historical average can remain misleading if recent performance has deteriorated.

Version 2 adds a fixed multi-window contract, synchronized block resampling within each window/block-length, a studentized max-|t| bound across the registered supported model × regime × window family, and a conservative minimum-bound weighting rule. A descriptive recent-versus-long-window change signal is reported, but no adaptive threshold is fitted on the final holdout.

## Mathematical object

For decision time t, model k's pre-generated probability is p[k,t], the causal baseline is b[t], and binary outcome y[t] becomes observable only at label-availability index a[t].

The incremental Brier skill is

**d[k,t] = (b[t] − y[t])² − (p[k,t] − y[t])².**

Positive d favors the model. Brier skill is bounded in [-1, 1]. The evaluation at as-of index T includes only forecasts where decision_index < T, label_available_index < T, and the outcome is explicitly mature.

For regime r and decision-window size W, define

**S[k,r,W] = mean(d[k,t] | regime[t] = r, T − W < decision_index[t] < T, label matured).**

The window is measured in decision bars/index units, not the number of observations in that regime. Sparse regimes therefore can fail support even if their total lifetime count is large.

### Synchronized moving-block uncertainty

For each predeclared block length L, rows are resampled in contiguous blocks. A block cannot cross a missing decision index or a declared session/segment boundary, but it may contain regime transitions so the dependence of regime transitions remains represented within the sampled history. Within each draw, all models and regimes use the same sampled row positions, preserving the paired structure of predictions and outcomes.

For each supported hypothesis h=(model, regime, window), let mu_hat[h] be the observed skill, mu_star[b,h] the bootstrap mean and se[h] its bootstrap standard deviation. The maximum studentized deviation per resample is

**M_star = max_h |(mu_star[h] − mu_hat[h]) / se[h]|.**

The (1-alpha) empirical quantile of this maximum gives an approximate simultaneous interval

**mu_hat[h] ± q_(1-alpha)(M_star) × se[h].**

The implementation computes a maximum separately for each predeclared block length and aggregates maximum statistics across windows using deterministic substreams. It reports the method as **approximate**: cross-window draws are not an exact joint stationary bootstrap and there is no valid finite-sample guarantee under arbitrary market dependence, arbitrary structural breaks, or wrong segmentation. The method is a conservative research screen, not a theorem of universal validity.

A model-regime pair is eligible for independent replication only if:
- every predeclared window has the required number of matured observations;
- each window has the required count of non-overlapping complete blocks that include the regime for every declared block length;
- the required bootstrap support is stable and non-degenerate;
- every simultaneous lower bound across the declared windows and block sizes exceeds the preregistered minimum skill.

The combination weight is proportional to the positive worst-case lower bound across the tested windows and block lengths. This is a candidate-selection weight, not a calibrated posterior probability and not a live trading weight. It must be frozen and assessed on later untouched forecasts.

### Descriptive drift warning

**drift_delta = S[k,r,W_recent] − S[k,r,W_long].**

A warning is raised when recent mean skill is nonpositive or has fallen behind long-window mean skill by a preregistered margin. This is deliberately descriptive. It is not presented as a calibrated change-point p-value and cannot justify retuning after inspecting the final holdout.

## Why these choices — literature comparison

- Forecast combination is a mature field. Wang, Hyndman, Li and Kang's review covers fixed, time-varying, nonlinear, probabilistic and cross-learning combinations, and discusses structural changes and estimation error: [Forecast combinations: an over 50-year review](https://arxiv.org/html/2205.04216v2).
- Regime-switching forecast-combination weights predate this project: Elliott and Timmermann's [Optimal Forecast Combination Under Regime Switching](https://cepr.org/publications/dp4649) explicitly lets weights vary with a latent regime.
- Qian et al.'s [Combining forecasts for universally optimal performance](https://doi.org/10.1016/j.ijforecast.2021.05.004) distinguishes adaptation to the best forecaster from combinations intended to improve on the best. Serevex must benchmark both goals instead of assuming that selection or blending always wins.
- Diebold and Mariano's [Comparing Predictive Accuracy](https://doi.org/10.1080/07350015.1995.10524599) formalizes paired forecast comparison with serially correlated forecast errors. The exact sampling design and loss differential still matter.
- Hansen's [A Test for Superior Predictive Ability](https://doi.org/10.1198/073500105000000063) and White's [A Reality Check for Data Snooping](https://doi.org/10.1111/1468-0262.00152) address searching across forecast/strategy families. This prototype's max-|t| block-bootstrap screen is not a replacement for Hansen SPA, White's full Reality Check, or a complete registered trading-strategy family test.
- Bailey et al.'s [The Probability of Backtest Overfitting](https://doi.org/10.21314/jcf.2016.322) motivates evaluating the model-selection process itself, rather than only the selected model.
- Bifet and Gavaldà's [ADWIN](https://doi.org/10.1137/1.9781611972771.42) offers a dedicated adaptive-window drift detector with its own guarantees. Version 2 does not reproduce those guarantees. Its recent-versus-long delta is only a transparent alarm feature.
- Howard et al.'s [time-uniform confidence sequences](https://doi.org/10.1214/20-AOS1991) offer sequential coverage under explicit probabilistic conditions. They should not be naively asserted to hold for arbitrary autocorrelated financial skill differences; a future sequential gate must state and defend its conditional/martingale assumptions.

**Novelty conclusion:** Version 2 is a Serevex-specific evidence-control hypothesis composed of established ideas. A claim of scientific originality would require a systematic prior-art review and a defensible new theorem or algorithmic contribution. The design alone is not such proof.

## Required experiment protocol

1. Generate prequential/out-of-sample forecasts for each registered expert before its outcome exists. All experts share the same decision IDs, target maturity, labels and causal baseline.
2. Freeze the target (for example, the current XAUUSD TP-before-SL label) and cost/decision assumptions before evaluating candidate skill. Do not mix direction, returns and TP-before-SL labels.
3. Treat a model-selection cohort separately from probability-calibration data and the final holdout. The v2 gate may select weights only from its earlier registered cohort.
4. Freeze gate configuration, candidates, windows, block lengths, alpha, support floors, and weights before the later final holdout is revealed.
5. Compare against causal global/rolling base rate, equal weights, a best expert selected on the selection partition, the v1 gate, and price-only / context ablations on identical final rows.
6. Report Brier and log loss, calibration, model/regime support, skill intervals, stability by time window, and all failure/abstention counts. Separately report net portfolio returns after verified executable quote-side costs and dependence-aware inference; probability skill does not imply economic utility.
7. Keep the complete trial registry and preserve failed models/configurations. Where the actual question is whether any strategy in a tested family beats a benchmark, use an appropriate portfolio-panel test such as SPA/Reality Check with complete family registration; do not substitute the forecastability screen.
8. Re-run on an independent feed and later period after data provenance, time zones, quote-side semantics, licensing and execution cost are qualified.

## Known limitations

- Bootstrap intervals are approximate and can misbehave under structural change, sparse regimes, dependent censoring, and nonstationary windows.
- Max-|t| across supported hypotheses does not repair unregistered experiments or researcher degrees of freedom.
- Sparse/unobserved regimes are marked unsupported, not imputed.
- Weights reuse the selection cohort and must not be reported as out-of-sample performance.
- A positive Brier skill is not equivalent to positive net PnL, profitable trading or superior return forecasts.
- Synthetic fixture passes demonstrate code contracts only.

## Integration points

Current source modules include point_in_time_features_v1.py, external_context_benchmark_v1.py, temporal_semantic_contract_v1.py, selection_bias_guard_v1.py and portfolio_panel_selection_adapter_v1.py. Version 2 deliberately remains a stand-alone consumer of a causal forecast ledger until that ledger exposes complete paired OOS forecasts and a separate selection partition.

## Acceptance gates

- Unit tests prove strict maturity/as-of semantics, paired cohort invariants, block/session boundary preservation, regime-conditional rejection, degenerate-bootstrap rejection and immutable NO_TRADE authority.
- Hosted CI passes all repository tests and benchmark workflow steps.
- The candidate is not called a winner from its own selection cohort.
- A later untouched holdout shows a preregistered improvement over multiple causal baselines, with uncertainty intervals and calibration diagnostics.
- Any trading claim additionally survives registered-family selection correction, independent data, realistic costs and a reproducible portfolio return ledger.
