# Serevex readiness findings — 21 October 2026

## Verified state

The latest checked hierarchical-shrinkage v2.0b hosted benchmark is [run #39466436358](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39466436358), from draft [PR #172](https://github.com/orionrayy/ai-determinism-engine/pull/172). The workflow completed and uploaded artifacts. Its separate probability-method tests passed in [run #39466436373](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39466436373).

A green CI run means the evaluation executed and code contracts held; it does not mean the candidate beat the baseline.

## Model-selection failure

On the source-time-block-purged model-selection partition:

| Metric | Causal baseline | Candidate 4.9b | Assessment |
|---|---:|---:|---|
| Brier score | 0.21727059 | 0.21727555 | Candidate slightly worse |
| Log loss | 0.62622395 | 0.62623229 | Candidate slightly worse |
| Positive block share | — | 50% | Below the 70% required gate |

The gate correctly returned DUAL_SCORE_GATE_FAILED. Calibration was not run. The final policy remained 0.1b, the immutable expanding matured-label Laplace baseline. Final-holdout candidate-shadow estimates are descriptive and cannot reverse the earlier failed gate.

## Data qualification blockers

- Price datasets remain REFERENCE_ONLY. Provenance, data rights, timestamp semantics, historical feature availability, broker quote equivalence, and actual execution semantics are not independently established.
- The tick diagnostic found a potential shared-lineage or price-basis-label issue: 4,391 of 4,392 overlapping tick-derived Bid bars exactly match a secondary repository described as mid-quote. Source independence is not verified; no automatic source promotion or price correction is allowed.
- The primary-versus-secondary comparison remains SOURCE_DISAGREEMENT_REVIEW_REQUIRED. No timestamp shift, rescale, or source substitution is applied automatically.
- External context is not ready for predictive use: around 205/19,894 rows have full 1-hour news coverage, 100/19,894 have 24-hour news coverage, and 0/19,894 have macro coverage.
- Execution replay remains diagnostic; profitability has not been validated using authorized broker quotes, measured costs, and independently reviewed fill semantics.

## Implemented hardening

The readiness-audit module added in this branch parses actual benchmark artifacts and emits JSON/Markdown status, input hashes, blockers by area, and integrity violations. It checks that a failed model-selection or calibration gate retains the exact baseline metrics; final holdout cannot affect candidate selection; missing evidence fails closed; and no research report can enable live signals or order routing.

This is governance and diagnostic hardening, not evidence of improved predictive accuracy. Statistical thresholds remain unchanged and the final holdout is not used for retuning.
