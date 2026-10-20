# Serevex Baseline-Advantage Evolution v0.1–v5.0

**Status:** research-only; v0.1 remains the exact causal champion/fallback. This evolution does not authorize trades.

## Objective

The real-data v4.0–v9.0 benchmark showed that the alpha candidate did not clear the paired Brier + log-loss gate; the blend governor therefore selected weight 0.0. The correct follow-up is not to relax that gate. This experiment isolates the baseline's durable advantages—maturity-safe prevalence tracking, low-variance probability shrinkage, support-aware conditioning, stable probabilities and abstention when evidence is insufficient—and tests them through a reproducible ladder of **50 checkpoints**, from v0.1 to v5.0.

The numbered variants are deterministic policies over the same causal forecast ledger. They are hypotheses for controlled ablation, not fifty claims of increasing accuracy or fifty separately trained models. They do not modify the target definition, Telegram decision path, broker routing or live-trading authority.

## Fifty-checkpoint ladder

| Range | Hypothesis and policy family |
|---|---|
| **v0.1** | Exact row-wise causal expanding Laplace baseline. Its probabilities are the unmodified reference and must reproduce exactly. |
| **v0.2–v2.0** | Nineteen progressive variants increase the bounded contribution of the causal rolling 288-matured-label rate, from a small blend toward a capped contribution. |
| **v2.1–v3.0** | Ten variants gradually introduce regime-conditioned matured rates while retaining a global/rolling component and shrinking on limited regime support. |
| **v3.1–v4.0** | Ten variants gradually introduce volatility-conditioned rates; their combined specialist weights are bounded so the baseline remains the residual prior. |
| **v4.1–v4.9** | Nine robust-shrinkage variants progressively reduce specialist mass again, explicitly testing whether a more conservative mixture is more stable than a heavily conditioned one. |
| **v5.0** | The final prequential governor. At each decision it may select one of v0.2–v4.9 only from evidence already matured before that decision. Otherwise it emits the exact v0.1 forecast. |

These are deterministic policy-family checkpoints. Passing a later version number does not imply a better model. If no variant beats v0.1 under the preregistered gates, v5.0 retaining v0.1 is the intended correct outcome—not a performance improvement.

## Multi-check, multi-audit and multi-calibration

1. **Input contract:** validate unique decision indices, integer maturity indices that do not precede decisions, binary labels only when matured, finite probabilities and valid expert mappings. Reject malformed input rather than silently repairing it.
2. **Point-in-time boundary:** an outcome may update the governor only if its label-availability index is strictly less than the current decision index. A forecast cannot use its own future outcome.
3. **Same-cohort audit:** score all 50 checkpoints on identical final-holdout decision identities and mature labels; persist per-version forecast-manifest hashes, a scored-cohort hash, a metrics hash and a report hash that binds cohort labels and measured results.
4. **Maturity/block audit:** summarize 12 source bars, followed by a 12-bar embargo, to reduce overlap between target horizons. Treat source-time blocks—not adjacent event rows—as the units for uncertainty screens.
5. **Multiple-candidate and repeated-look correction:** v5.0 evaluates 48 candidate policies and two loss endpoints. It spends familywise alpha across selection looks using alpha_look(k)=0.05/(k*(k+1)), then divides each look's alpha across 96 one-sided candidate/score tests and converts that tail probability into a normal critical threshold. Its thresholds become stricter at later looks.
6. **Block-cadenced selection:** re-evaluate candidate eligibility only when a new completed source-time block adds evidence; repeated decisions on an unchanged evidence snapshot reuse the cached gate. This reduces repeated-look inflation and compute load.
7. **Dual-score gate:** both the Brier-improvement and log-loss-improvement lower screens must be positive; at least 70% of eligible blocks must be positive for each metric; at least 200 matured scoring-window rows and eight blocks are required. If no candidate passes, fall back to v0.1.
8. **Calibration audit:** report ten-bin ECE and maximum bin gap plus subgroup Brier, log loss and reliability diagnostics by regime and volatility bucket. A candidate cannot pass if its accumulated prequential ECE is more than 0.015 worse than v0.1. Sparse subgroup values remain descriptive, not proof of conditional calibration.
9. **Uncertainty and manifests:** compute paired source-time-block bootstrap intervals with fixed per-version seeds. Hash prediction manifests, the cohort including labels, block-level selection traces and the resulting metrics for reproducibility. Intervals are diagnostic; they are not exact finite-sample guarantees.
10. **Causal regression tests:** test exact baseline reproduction, specialist rejection under harmful predictions, acceptance of a genuinely informative synthetic specialist, strict maturity, future-label mutation invariance, report-integrity hashes, duplicate/malformed input rejection and the permanent NO_TRADE boundary.
11. **Authority guard:** NO_TRADE remains enforced; validated-model/deployment-calibration statuses stay false, and live signal/order execution stays disabled.

## Reading results

- Positive Brier or log-loss improvement means the candidate's loss is lower than the baseline under the report's sign convention.
- A nonpositive conservative lower screen, unstable block share, insufficient support or excess ECE degradation makes a candidate ineligible.
- If v5.0 selects v0.1 on every decision, it has correctly preserved the baseline; that result must not be reported as specialist skill.
- An improved holdout point estimate without paired block evidence and independent untouched replication is not sufficient evidence of predictive skill.
- Better probability scores do not establish positive net returns. Costs, executable broker quotes, fills, portfolio exposure, provenance, data-use rights and historical feature availability require separate qualification.

## Experiment scope and limitations

This evaluator consumes the upstream forecast ledger and inherits its reference-only OHLCV assumptions. It cannot independently prove source authenticity, data rights, source independence, source timestamp semantics, historical feature availability or broker quote equivalence. The v5.0 online governor is a prequential policy evaluation: it may use a prior label only after that label has matured, but the aggregate holdout result still needs replication on a new untouched chronological cohort.

**Acceptance condition:** retain v0.1 until a variant clears the preregistered support, multi-candidate dual-score, stability and calibration gates and replicates on a later independent cohort. No live-trading authorization is included.
