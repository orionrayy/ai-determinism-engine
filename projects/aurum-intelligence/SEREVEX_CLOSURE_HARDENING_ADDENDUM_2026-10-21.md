# Serevex Closure Hardening Addendum — 21 October 2026

## Scope

This change closes additional code-controlled evidence-integrity gaps. It does not upgrade the research candidate, qualify third-party data, or authorize production use.

## Implemented controls

1. **Context feature support gate.** Each requested context feature must have at least 30 point-in-time eligible observations and at least 80% eligible coverage in each of development, calibration, and final holdout. An ablation requiring a family with no feature meeting both requirements is \`NOT_ELIGIBLE_REQUIRED_CONTEXT_FAMILY_MISSING\`. The threshold is recorded in the report protocol. Raw source observation volume does not substitute for feature-level temporal coverage.
2. **Holdout score integrity.** Baseline and policy metric bundles are validated for required fields, finite values, proper ranges, positive integer row counts, same cohort size, and identical observed label rate. This applies even when both model selection gates pass.
3. **Temporal coverage reconciliation.** The readiness audit recomputes each coverage fraction from eligible/total counts, rejects invalid counts and inconsistent denominators, and blocks features that do not meet the 80% policy floor or report status \`PASS\` below that floor.
4. **Ablation reproducibility.** Every scored context ablation must state its effective families, show feature support above the registered per-split floor, provide valid paired moving-block confidence intervals against \`price_only\`, and share exact row/label hashes with the price-only control.
5. **Authority invariants.** The audit detects contradictory source-lineage claims and fails on missing/unsafe context or execution authority state. Research outputs remain \`NO_TRADE\`; source gates cannot claim live execution authority.

## Expected disposition on current data

The known news coverage (~1.03% for 1-hour windows and ~0.50% for 24-hour windows) and zero macro coverage do not meet the feature floor. These context ablations must be marked ineligible. COT may remain a diagnostic context ablation only when its own per-split feature support passes; upstream rights, lineage, and source evidence remain independent blockers.

## Verification gate

The change is accepted only if the full online-benchmark unit-test suite and the hosted end-to-end workflow pass, the generated readiness artifact contains no integrity violations, and current missing evidence remains visible as blockers. A successful workflow confirms the audit ran correctly; it does not confirm predictive skill or profitability.

## Still blocked by evidence outside code control

- Independently verifiable license/intended-use rights, publisher provenance, time-zone/bar conventions, and broker quote equivalence for each market dataset.
- Resolution of the potential shared lineage / mislabeled quote-basis issue in 4,391 of 4,392 overlapping Bid bars.
- A qualifying historical news archive covering the required decision-time windows, plus verified point-in-time macro release/availability history.
- A candidate that passes preregistered paired Brier/log-loss and block-support gates, then calibration on a separate later cohort and a final untouched chronological test.
- Broker-authorized historical quotes, measured commission/spread/slippage/swap and fill semantics, adverse-regime stress tests, paper trading, risk/security review, and explicit independent approval.

The production state therefore remains disabled, with \`NO_TRADE\` as the mandatory output.
