# Serevex Forecastability Discovery v3 — delayed-feedback expert selector

**Status:** isolated research prototype; 13 synthetic contract tests pass locally. It consumes pre-generated forecasts; it is not a trained price model, not market-validated, not integrated into Telegram/live signals, and it never authorizes trading.

## Why v3 is a different layer from v1/v2

- **v1** audits whether registered OOS experts beat a causal baseline under one-window, regime-conditioned skill checks.
- **v2** requires skill to persist over preregistered windows and moving-block sensitivities, with approximate simultaneous inference. It remains an evidence/selection gate.
- **v3** is an actual sequential forecast-mixture mechanism. It outputs a probability by combining registered expert probabilities and the causal baseline, then updates its weights only after the relevant outcome has matured.

The three layers can be compared; they should not be silently conflated. v3 is a candidate generator/selector, while v1/v2 are evidence controls. The candidate still must be evaluated on a later untouched cohort.

## Update rule

At decision index t, the expert probabilities are p[k,t], the baseline probability is b[t], and the outcome y[t] is available at index a[t]. The candidate records the forecast and its current weights before any future label is revealed. It may update weights at decision index t only for prior forecasts satisfying a[i] < t.

For binary target and Brier loss:

\[
\ell_{k,i}=(p_{k,i}-y_i)^2,\quad \ell_{0,i}=(b_i-y_i)^2,\quad d_{k,i}=\ell_{k,i}-\ell_{0,i}.
\]

The centered loss d is in [-1,1]. The multiplicative weights update for a matured batch B is

\[
\tilde w_k \propto w_k\exp\left(-\eta\,|B|^{-1}\sum_{i\in B}d_{k,i}\right).
\]

After normalization, fixed-share redistribution gives

\[
w'_k=(1-\gamma)\tilde w_k+\gamma/K,
\]

where K includes every registered forecasting expert and an explicit baseline expert. The baseline term is part of the mixture, not an after-the-fact comparator. When all candidates underperform relative to the baseline, the baseline's weight can grow; fixed share leaves nonzero mass on candidates so the selector can recover if conditions change.

The prototype keeps global weights and separate per-regime weights. A regime-specific pool is used only after its declared matured-label support floor is reached. Before that, it falls back to global weights after sufficient global history; otherwise the current forecast is exactly the baseline probability.

Labels with the same availability index are updated as one batch so arbitrary label order does not alter the weight update. If several maturity batches become observable between two recorded decisions, they are applied in maturity-index order. All update events record when the batch was applied and the weight hashes before/after.

## What this does and does not claim

Fixed-share/exponential expert weighting and adaptive expert tracking are established topics. This prototype does **not** claim a new regret theorem or guarantee under arbitrary delayed feedback. Published fixed-share/online-learning regret results depend on assumptions and feedback timing; applying a theorem to variable-horizon financial labels without matching assumptions would be incorrect.

The research hypothesis is that the combination of strict label-maturity semantics, regime-scoped pools with a global fallback, explicit baseline inclusion, hashed update lineage, and maturity-batch order may be useful for Serevex. It must be compared experimentally with static equal weights, baseline-only, v1 selection, v2 selection, and the best expert chosen only on an earlier selection cohort.

A hindsight best fixed expert and a dynamic-programming best-switching expert are computed only as diagnostics. They use outcomes across the scored cohort, so they must never be used as deployed weights or as evidence that the online method could have known the future.

## Literature / prior art

- Wang, Hyndman, Li & Kang, *Forecast combinations: an over 50-year review*: https://doi.org/10.1016/j.ijforecast.2022.11.005
- Elliott & Timmermann, *Optimal Forecast Combination Under Regime Switching*: https://cepr.org/publications/dp4649
- Giacomini & White, *Tests of Conditional Predictive Ability*, Econometrica 74(6), 1545–1578 (2006), DOI https://doi.org/10.1111/j.1468-0262.2006.00718.x. Their framework studies conditional out-of-sample forecast selection under heterogeneity; it motivates testing whether performance varies with the regime, rather than merely splitting a sample after the fact.
- Hansen, *A Test for Superior Predictive Ability*: https://doi.org/10.1198/073500105000000063
- Diebold & Mariano, *Comparing Predictive Accuracy*: https://doi.org/10.1080/07350015.1995.10524599
- Mourtada & Maillard, *Efficient tracking of a growing number of experts*, PMLR 76 (2017): https://proceedings.mlr.press/v76/mourtada17a.html
- Baby & Wang, *Optimal Dynamic Regret in Proper Online Learning with Strongly Convex Losses*, PMLR 151 (2022): https://proceedings.mlr.press/v151/baby22a.html
- *Optimal Tracking in Prediction with Expert Advice*: https://arxiv.org/abs/2208.03708
- Bifet & Gavaldà, ADWIN: https://doi.org/10.1137/1.9781611972771.42

These sources describe neighboring established methods, not an exhaustive paper/patent/code novelty search. Any claim of original mathematics would need a precisely stated new contribution and broader prior-art review.

## Data and evidence contract

The input forecast ledger must use one target definition and the same decision cohort for all experts. Every regime classification must have been available at the decision time; each label must have a later maturity index. The optional manifest records the target identifier, target-definition hash, dataset hash, source-lineage hash, forecast-ledger hash, and a hash for every registered model artifact. Code validates the manifest's shape only; this is not independent proof of source authenticity, rights, or model-fit chronology.

For a real experiment, hold out chronological periods as follows: model fitting, expert/gate selection, probability calibration, then a final locked holdout. Freeze the expert pool, eta, fixed-share rate, regime definitions, support floors, and target before revealing the final cohort. Evaluate Brier/log loss and calibration on identical matured labels, plus uncertainty intervals. Re-test on an independent feed/regime and measure net executable portfolio returns separately; lower forecast loss does not establish profit.

## Current empirical backdrop

The verified v2 benchmark run #259 passed all workflow steps, including 1.313 million paired XAUUSD M1 Bid/Ask candles, 20,000 real anchor windows, tick-feed audit, quote-side replay contracts, external-context benchmark and artifact upload. The tested prequential probability candidate still has Brier 0.1747186248 versus matured-label baseline 0.1745785974; baseline-minus-model skill is -0.0001400274, with a 95% block interval entirely negative [-0.0002451349, -0.0000518749]. COT/news incremental comparisons remain inconclusive, and point-in-time news/macro support remains insufficient. These results do not evaluate the new v3 mixer; v3 has only synthetic contract tests so far.

## Safety and acceptance

- Output is always NO_TRADE; model_validated=false; deployment_calibrated=false; live signals/orders disabled.
- Passing synthetic tests validates code contracts, not market skill.
- The online mixer must beat causal baseline, equal-weight blend, and the best expert selected on a separate selection cohort on the same later untouched holdout before its candidate skill can be discussed.
- Report delayed-label coverage, update delay, regime fallback rates, weight concentration, expert turnover, Brier/log loss, calibration, and regret diagnostics. No hindsight comparator may drive decisions.
- Before merging, run full hosted CI and code review. Before integration, require model/data provenance, a frozen selection/calibration/holdout protocol, independent replication, and separate portfolio-level selection-bias control.
