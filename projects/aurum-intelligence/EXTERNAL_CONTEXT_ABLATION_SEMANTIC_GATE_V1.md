# External Context Ablation Semantic Gate v1.0.0

## Scope

This contract prevents model labels from overstating which context families actually contributed to a scored model. It applies to the offline point-in-time external-context ablation benchmark. It does not enable signals or trading.

## Required report fields

Every ablation reports:
- requested_context_families: the families prescribed in ABLATIONS.
- effective_context_families: requested families with eligible fitted features and sufficient as-of-eligible support in development, calibration and holdout.
- omitted_context_families: requested families that failed one or more family gates.
- requested_context_feature_keys, effective_context_feature_keys and omitted_context_feature_keys.
- context_family_support_by_split: eligible row counts, row fractions, and per-feature eligible-value counts for development/calibration/holdout.
- missing_context_family_reasons for a non-eligible family.

The compatibility field context_families reports effective families, never merely the requested list.

## Family eligibility

A context family requires all of the following:
1. At least one feature from that family fits on matured development data.
2. At least 30 rows with an as-of-eligible value from that family in each of development, calibration and holdout.
3. At least one feature key from that family must itself have at least 30 eligible values in each split. Features below the per-key floor are omitted from the fitted model and listed explicitly. These thresholds are engineering support floors, not statistical-power guarantees.
4. If source_statuses explicitly declares the family status, the status must be PASS or PARTIAL. UNKNOWN, FAILED, NOT_MET, skipped, or other states block use.
5. Feature availability/time gates still apply row-by-row; missing/late values are not counted as eligible support.

A multi-family ablation is NOT_ELIGIBLE_REQUIRED_CONTEXT_FAMILY_MISSING if any requested family fails. It must not quietly degrade into the remaining subset while retaining the multi-family label. Single-family ablations are similarly blocked if their requested family is missing. price_only has no context-family requirement.

## GDELT batch-coverage semantics v1

News availability is assessed at the archive-batch cadence rather than requiring a
continuous interval to terminate at an arbitrary M1 decision minute. GDELT GKG files
are treated as 15-minute batches, available only at each batch's end with a conservative
availability proxy. Each scheduled batch whose availability falls inside a rolling
feature window must be present; aggregate counts do not fill archive gaps.

The one-hour and 24-hour windows are qualified separately:
- `feature_news_count_1h` requires the corresponding complete one-hour batch window.
- Every 24-hour news key (counts, themes, and tone) requires its full 24-hour batch window.
- `coverage_news` remains a compatibility field with its prior 24-hour interpretation;
  `coverage_news_1h` and `coverage_news_24h` expose the separate contracts.

The CI archive sampler chooses bounded contiguous blocks around centers in the development,
calibration, and final-holdout time segments. Centers come from the fixed chronological
price-row cohort, not context values. Blocks are downloaded in round-robin split order
so a byte-limited sample does not silently consume only the earliest segment. This is
a coverage-focused sample, not a complete archive census; overall time coverage and
row-level eligibility remain separate metrics.

## What this does not imply

Passing these semantic checks only establishes that the report accurately names the requested and effective feature families and that they meet a minimal support floor. It does not establish causal source provenance, statistical superiority, calibration, positive net PnL, independent source lineage, or deployability. All decisions remain NO_TRADE.
