# Serevex Baseline-Advantage Evolution v0.1–v5.0

**Status:** research-only; baseline retained as the exact champion/fallback. This proposal does not authorize trades.

## Objective

The preceding v4.0–v9.0 real-data benchmark showed that the alpha candidate did not clear a conservative paired Brier + log-loss gate. The proper next step is not to relax the gate. This evolution tests whether the properties of the causal Laplace baseline—maturity-safe updates, prevalence tracking, low-variance shrinkage, support-aware conditioning and abstention when evidence is insufficient—can be improved in small, falsifiable steps.

The versions are deterministic evaluation policies over the existing forecast ledger. They do not modify the target definition, signal routing, or live trading path.

## Version ladder

| Version | Hypothesis under test | Policy |
|---|---|---|
| v0.1 | The reference forecast is hard to beat when specialist evidence is weak. | Exact row-wise causal expanding Laplace baseline; always retained. |
| v0.2 | Recent matured outcomes may add a small amount of adaptive information without discarding the global prior. | Bounded blend towards the causal 288-matured-label rolling-rate expert; the rolling influence increases gradually after a minimum support threshold. |
| v0.3 | Regime and volatility conditioning may help only when those conditional cohorts have enough matured observations. | Shrink regime and volatility rates towards v0.1 using support-dependent weights; specialist influence is capped. |
| v0.4 | A conservative combination may stabilize specialist estimates. | Baseline-anchored combination of v0.1, v0.2 and v0.3. |
| v0.5 | A specialist should be used only after evidence from earlier, matured observations supports it. | Prequential governor compares candidates with v0.1 on source-time blocks, requires both Brier and log-loss improvements, block stability, adequate support and bounded ECE degradation; otherwise it emits the exact row-wise v0.1 forecast. |

These versions are a controlled research ladder, not five claims of increasing accuracy. v0.2–v0.4 weights are initial, explicit engineering hypotheses and remain subject to ablation. If they do not improve on the same-cohort benchmark, do not tune them against the final holdout. Design another preregistered experiment on development/calibration data.

## Multi-check, multi-audit and multi-calibration

1. **Input contract:** validate indices, unique decision positions, binary matured labels, finite probabilities and valid expert mappings. Reject malformed inputs rather than silently repairing them.
2. **Point-in-time boundary:** an outcome may update the selector only if label_available_index is strictly less than the current decision index. Its own decision can never use its own future outcome.
3. **Same-cohort audit:** score v0.1–v0.5 on exactly the same final-holdout decision identities and labels; persist prediction-manifest hashes.
4. **Maturity/block audit:** score 12 source bars, then embargo 12 bars for uncertainty summaries. Count independent source-time blocks, not adjacent event rows as independent evidence.
5. **Dual-score gate:** require the conservative block-mean lower screens for both Brier improvement and log-loss improvement to be positive; require at least 70% positive blocks for both. Inadequate support forces v0.1.
6. **Calibration audit:** report equal-width ten-bin ECE and maximum bin gap, plus subgroup Brier/log loss/reliability by the preregistered regime and volatility bucket. Keep subgroup results descriptive when support is low. A candidate may not worsen accumulated prequential ECE by more than 0.015 relative to v0.1 to pass the gate.
7. **Uncertainty:** compute paired source-time-block bootstrap intervals with a fixed seed and report block count and replicate count. Intervals are diagnostic and are not exact finite-sample guarantees.
8. **Causal regression tests:** test deterministic hashes, missing-support fallback, harmful-specialist rejection, acceptance of a controlled synthetic informative specialist, duplicates/malformed inputs failing closed, and invariance of earlier complete block-selection hashes when labels after the cutoff are mutated.
9. **Authority guard:** maintain NO_TRADE; validated-model/deployment-calibration statuses remain false, and live signal/order execution remains disabled.

## Reading results

- Positive Brier/log-loss improvement means the candidate's loss is lower than the baseline's loss under the report's sign convention.
- A nonpositive lower confidence screen, unstable block share or excess ECE degradation means the candidate is not eligible to replace v0.1.
- If v0.5 selects v0.1 for every decision, that is a correct safety outcome but **not** an improvement claim.
- A lower holdout point score without a suitable paired interval and an untouched independent replication is not sufficient evidence of skill.
- Better probability scores do not establish positive net returns. Costs, executable broker quotes, fills, portfolio exposure, source provenance, source rights and historical feature availability require separate qualification.

## Experiment scope and limitations

This module evaluates the supplied ledger and depends on the upstream producer's stated reference-only OHLCV assumptions. It cannot independently establish the truth of source timestamps, market-data rights, source independence or broker quote equivalence. The online governor is a prequential research comparison; a final-holdout aggregate still needs untouched replication before it supports a scientific or economic claim.

**Acceptance condition:** retain v0.1 unless a candidate passes the preregistered dual-score, block-stability, support and calibration gates, and then confirms that advantage on a new untouched chronological cohort. No live trading authorization is part of this experiment.
