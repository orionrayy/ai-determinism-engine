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
2. At least 30 rows with an as-of-eligible value from that family in each of development, calibration and holdout. This fixed contract threshold is an engineering support floor, not a statistical-power guarantee.
3. If source_statuses explicitly declares the family status, the status must be PASS or PARTIAL. UNKNOWN, FAILED, NOT_MET, skipped, or other states block use.
4. Feature availability/time gates still apply row-by-row; missing/late values are not counted as eligible support.

A multi-family ablation is NOT_ELIGIBLE_REQUIRED_CONTEXT_FAMILY_MISSING if any requested family fails. It must not quietly degrade into the remaining subset while retaining the multi-family label. Single-family ablations are similarly blocked if their requested family is missing. price_only has no context-family requirement.

## What this does not imply

Passing these semantic checks only establishes that the report accurately names the requested and effective feature families and that they meet a minimal support floor. It does not establish causal source provenance, statistical superiority, calibration, positive net PnL, independent source lineage, or deployability. All decisions remain NO_TRADE.
