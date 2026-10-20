# Serevex Failure-Closure Plan — 21 October 2026

## Objective

Close every software defect under repository control; preserve strict fail-closed behavior for scientific, data-rights, and execution failures that cannot be repaired by code alone. Success means Serevex can distinguish a verified model improvement from missing, contradictory, or insufficient evidence. It does not mean every candidate can be forced to pass.

## Current evidence and failure classification

### F1 — Candidate skill is not established (model/statistics)

The latest source-time-block-purged hierarchical-shrinkage v2.0b run selected candidate 4.9b during development, but failed the later model-selection gate:

- Brier: baseline 0.2172705916; candidate 0.2172755504.
- Log loss: baseline 0.6262239541; candidate 0.6262322919.
- Positive block share: 50%; required floor: 70%.
- Calibration: correctly not run after selection failure.
- Final policy: immutable causal baseline 0.1b.

**Resolution:** retain 0.1b. Do not tune on the same final holdout, change the gate, or claim an edge. Any next candidate must be preregistered on development data, pass both Brier and log-loss paired block screens, meet the positive-block support rule, pass the independent later calibration gate, then be tested once on an untouched future chronological cohort. A failed candidate is rejected; this is the intended safe outcome, not an implementation defect.

### F2 — Price source identity and semantics are unqualified (data integrity)

Current diagnostics report:

- Primary community OHLCV: 35,954 rows, but publisher provenance, intended-use rights, timezone/bar semantics and broker-quote equivalence are unverified.
- A secondary feed has 11,509 exact timestamp overlaps, no exact OHLC matches, median absolute close discrepancy about 134.9 bps and p95 about 203.7 bps.
- In a bounded tick sample, 4,391 of 4,392 overlapping tick-derived Bid bars exactly match a repository described as mid-quote OHLC. Independent lineage is unverified.
- Historical Bid/Ask archive rights, broker identity, timestamp semantics and measured execution costs remain unverified.

**Resolution:** do not concatenate feeds, align them by a chosen offset, rescale prices, promote sources, or select a truth feed from correlation alone. Preserve source-specific checksums and reports. Resolve each feed with provider/readme/owner evidence and the intended-use license; compare quote basis and aggregation convention; then repeat the same preregistered evaluation on the qualified source. If upstream evidence is unavailable, leave the feed as REFERENCE_ONLY and execution as BLOCKED.

### F3 — External context lacks continuous historical coverage (feature eligibility)

The observed report shows only 205 of 19,894 price-label rows with fully observed 1-hour news coverage, 100 of 19,894 with 24-hour news coverage, and zero with eligible macro coverage. The acquired archive sample covers roughly 0.114% of the requested historical window. A high raw observation count does not repair gaps in the decision-time lookback windows.

**Resolution:** keep news/macro features ineligible unless the feature-level acquisition report returns its registered PASS status and adequate chronological support. Do not turn uncovered periods into zero-news or forward-fill macro values. If increasing the bounded archive budget is not feasible, explicitly run a smaller coverage-aligned matched cohort: price-only and context candidate scores must use exactly the same decision rows, labels, source-time blocks and purging protocol; report the effective cohort and its calendar span. If that cohort lacks enough blocks/support, return NOT_ELIGIBLE and do not include context in the decision model. COT can be assessed only on its own point-in-time availability and rights lineage.

### F4 — Economic edge and production execution are unverified

Public historical replay currently has validated_model=false, validated_for_profitability=false, and live_execution_authorized=false. The tested public price archive is not independently qualified as an authorized broker feed, and the spread is not the user's measured all-in cost.

**Resolution:** keep order routing off. Before any promotion, run a frozen strategy over broker-authorized historical Bid/Ask/tick data with measured commission, spread, swap, slippage, gaps, rejections, latency and fill semantics; test sensitivity and drawdown; then paper trade under explicit risk caps and a reviewed rollback/security plan. Model calibration, model skill, and net profitability are separate gates.

### F5 — Evidence orchestration must not hide failure or mix unrelated runs

Reports can be incomplete, corrupt, or accidentally refer to different source snapshots. A successful workflow is not a model-skill claim.

**Resolution implemented in draft PR #173:**

- Parse the benchmark report formats actually emitted by the pipeline.
- Hash every readable input artifact and bind the model ledger's dataset hash to the online dataset; bind the secondary source hash across reports.
- Require explicit gates for primary, secondary and tick sources.
- Mark missing/malformed required artifacts as input errors, emit JSON/Markdown blockers, and exit non-zero for those integrity errors.
- Preserve exact baseline fallback metrics when the model-selection/calibration gates fail.
- Detect an attempt to change the candidate using final holdout, promote a policy after failed gates, or enable live authority.
- Run the readiness audit even if an earlier benchmark stage fails so the uploaded artifact can show the blockers.
- Add regression cases for fallback identity, hash mismatch, malformed/missing artifacts, unqualified source, context ineligibility, and unauthorized live state.

## Execution DAG and acceptance gates

| Stage | Work | Acceptance criteria | Current disposition |
|---|---|---|---|
| A. Software integrity | Audit module, schema-aware parsing, unit/regression tests | Tests pass; malformed/missing JSON reported; digest mismatches are integrity violations | Implemented on draft PR; latest hosted run must be checked |
| B. Hosted end-to-end | Real acquisition → source audit → ledger/stages 1–3 → large Bid/Ask replay → external context → readiness artifact | Workflow completes; artifact includes all required reports and hashes; blocker summary is visible | Re-run in progress/queued; do not infer pass until completed |
| C. Model selection | Compare frozen candidate to causal baseline on purged chronological blocks | Both scores pass conservative paired screens, support floors pass, no material calibration failure | **Blocked:** current candidate fails; baseline remains selected |
| D. Source qualification | Independently verify publisher, intended-use rights, timestamp and quote/bar basis | Evidence is linked and versioned; independent crosscheck no longer contradicts the source declaration | **Blocked externally:** no source-owner/broker evidence |
| E. Context features | Achieve feature-level temporal coverage and same-cohort comparisons | News 1h/24h and macro each meet their registered eligibility/support gates on identical scored rows | **Blocked:** observed coverage is partial/not met |
| F. Economic validation | Broker-equivalent costs, adverse scenarios and paper trading | Positive net-of-cost results with uncertainty bounds across independent regimes; separate risk/security review | **Blocked:** no validated executable edge |
| G. Production | Explicit authorization and rollback | All prior gates pass and a separate approval is recorded | **Disabled by design** |

## Required invariants

1. decision_output = NO_TRADE while any required source, model or execution gate is unresolved.
2. The 0.1b causal baseline and its score trace remain immutable for each experiment.
3. The final holdout never selects hyperparameters, calibrators, regimes, sources or thresholds.
4. No synthetic fixture, raw observation count, green CI, or selected point estimate is reported as market predictive skill.
5. Source mismatch stays visible. There is no automatic timestamp correction, price rescaling, merging, source promotion, model deployment or order execution.
6. Any subsequent PR must include its exact commit SHA, workflow run, artifact hash, candidate-vs-baseline metrics, block uncertainty, blocker list and an explicit acceptance/rejection decision.

## Definition of complete closure

The code-controlled parts are complete only when the hosted audit and all regression tests pass and a final artifact is uploaded. Research is complete only when a preregistered candidate beats the baseline under the stated gates and independently replicates. Production is complete only after source qualification, net-of-cost validation, paper trading, security/risk review, and explicit approval. These are intentionally separate milestones.
