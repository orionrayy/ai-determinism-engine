# Serevex Temporal-Semantic Contract v1.0.0

Status: research contract / not connected to live signal generation.

## Why this gate exists

A numeric value is not a usable trading feature merely because it parses. The system must know what the value means, which instrument and quote side it describes, which transformation created it, which revision it represents, and—most importantly—when that exact version became usable. Current-state macro data, event time, provider publication time and local ingestion time are different facts. Conflating them can create look-ahead leakage while leaving ordinary unit tests green.

This contract is deliberately stricter than a generic JSON schema. It is a decision-time admissibility screen, not an authentication service or alpha model.

## API surface

The module is temporal_semantic_contract_v1.py.

- validate_feature_fact(...) validates one immutable feature observation against a decision timestamp.
- validate_feature_batch(...) validates a complete declared feature snapshot, requires specified semantic slots, blocks exact duplicate source-record/revision identities, and incorporates the lineage audit.
- validate_feature_lineage(...) resolves every declared parent to an exact in-snapshot record identity, verifies parent/child temporal ordering, and rejects cycles.
- validate_ohlc_candle(...) checks an individual OHLC bar's time, quote basis, price geometry and optional volume.
- validate_candle_series(...) checks order, identity, duplicate bars, overlaps, within-session continuity, and explicit session boundaries.

Every returned report includes a diagnostic-only promotion state and preserves NO_TRADE, live_signal_enabled=false, and live_order_execution_enabled=false. Invalid arguments that make the audit itself undefined raise ValueError; invalid facts that can be meaningfully diagnosed return a BLOCKED report. No price repair, missing-bar synthesis, timezone inference or source promotion occurs.

## Temporal fields must not be conflated

| Field | Meaning | Required rule |
|---|---|---|
| event_time | The market event, observation period, reporting period, or scheduled event to which the fact refers | Its interpretation is explicit through event_time_role; it is not a proxy for when the system knew the value |
| published_at | Timestamp the upstream publisher made this version public | Must have an explicitly verified timezone and must not be after the decision time |
| available_at | Earliest defensible time that this exact source record/revision was available for use | Must be no later than the decision time; this is the principal point-in-time gate |
| observed_at | Time the ingestion/measurement system actually observed the record | Must not precede available_at and must not be after the decision time |
| source_revision | Provider revision/version identity | Must be explicit; revision semantics must be verified |
| source_sha256 | Digest of captured source content or immutable source artifact | Must be a 64-character SHA-256 hex digest |
| transform_id, transform_version, transform_sha256 | Reproducible identity and digest of the transformation used to derive the feature | Must be explicit; output is not reproducible merely because the code has a name |

The validator rejects timezone-naive timestamps. Do not replace an unknown timezone with UTC and then describe it as verified. A diagnostic UTC assumption may be useful for source-comparison work, but it is not the same as source timestamp qualification.

### Event-time roles

An explicit event_time_role prevents the system from assigning universal meaning to one timestamp:

- bar_close, past_market_event, observation_period and reporting_period: must not lie after the decision time.
- observation_period and reporting_period describe historical/reference periods; they are not a way to label known-ahead calendar events.

- scheduled_future_event: permits a known-ahead event timestamp only when the publisher/availability timestamps show the information was already available.
- effective_from: records an effective-time concept and requires a domain-specific interpretation before it can be admitted to a strategy.
- not_applicable: only for facts whose semantics truly have no event-time concept.

The contract also checks that common semantic types use compatible roles. For example, a market bar cannot be labelled as a scheduled future event to evade the closed-bar requirement.

## Point-in-time eligibility and fail-closed reasons

A feature fact can be eligible for a research feature snapshot only if:

1. Required semantic fields are present and typed.
2. The semantic type, event-time role, unit and quote basis are unambiguous.
3. published_at <= available_at <= observed_at and all three timestamps are no later than the decision time.
4. Every timestamp has explicit timezone semantics.
5. The exact source content and transform hashes are syntactically valid.
6. Provenance, rights, revision semantics and data quality are marked verified/passed.
7. Optional staleness thresholds declared by the caller are satisfied.
8. All required feature slots are present, and no exact source-record/revision identity is duplicated.
9. Every derived feature has parent references that resolve to exact records in the same feature snapshot. Parent records must be available before the derived feature is available, and must already have been observed by the ingestion system.
10. Dependency edges form an acyclic graph. A missing parent, changed source digest, duplicate logical identity, self-reference, or dependency cycle blocks the entire snapshot.
11. Any fitted transform declares a causal fit scope (expanding/rolling past-only), a timezone-aware fit cutoff no later than decision time and feature availability, a fit-data SHA-256, and a stable fit-manifest reference. Global/full-history fit scopes are blocked; stateless transforms do not claim a fitted window.

The `validate_feature_fact` function checks a single fact's local metadata. The `validate_feature_lineage` function must also pass before a derived feature set is considered complete; `validate_feature_batch` includes that graph check automatically. Parent references are exact and include feature ID, instrument, source ID, record ID, revision, and source SHA-256.

The validation flags (VERIFIED, PASS, semantics booleans) are claims made by the calling pipeline. They are not self-authenticating. A separate evidence review must verify the underlying provider/source, timestamp behavior, data license, revision history, and hash manifest. Therefore the contract's strongest result is ELIGIBLE_FOR_RESEARCH_FEATURE_SET, never “live-ready.”

Staleness is policy-driven. The module does not invent one global maximum age for prices, news, positioning, macro data or a scheduled event. Caller-declared thresholds must be specific to the feature and domain; a threshold selected after viewing holdout results is itself a selection decision that needs registration.

## OHLC and session integrity

The candle contract enforces:

- finite, positive open/high/low/close prices;
- high >= max(open, close, low) and low <= min(open, close, high);
- exact declared bar duration and an aware timestamp for start/end;
- candle close no later than the decision timestamp;
- explicit source/instrument/session identifiers and quote basis;
- finite, nonnegative volume if supplied;
- strictly increasing, unique bars without overlaps;
- exact continuity inside one explicitly identified session;
- gaps allowed only when a session boundary is explicit; session IDs may not recur after a boundary.
- a full snapshot lineage graph has no cycles and no parent/child availability violations;
- derived transformations identify whether they are stateless or fitted on expanding/rolling past-only data; global/full-history fit scope is blocked.

These are structural invariants, not proof that the upstream OHLC is true or licensed for use. A bid candle, ask candle, midpoint candle and last-trade candle are not interchangeable; quote basis is retained rather than silently normalized.

## Test coverage in the same CI suite

The new test module covers valid and unavailable-at-decision features, late ingestion, known-ahead scheduled events, future market-bar and future observation-period leakage, timezone-naive inputs, timestamp order, staleness, source/rights/hash blockers, typed values and quote basis, semantic-role conflicts, malformed enum types, required feature slots, duplicate source revisions, exact parent resolution, parent digest mismatch, parent availability/ingestion ordering, dependency cycles, fitted-transform cutoff and manifest rules, closed-candle checks, OHLC geometry, volume, session gaps, overlaps, and recurring session IDs.

CI passing establishes that these specified contracts and regression tests execute successfully. It does not establish real-world provenance, profitable expected return, broker-specific net PnL, or the absence of every possible leakage path. The module is not yet wired into the Telegram decision path and must not be described as doing runtime gating until that integration is separately reviewed and tested.

## Research and primary documentation

1. St. Louis Fed, FRED Real-Time Periods: https://fred.stlouisfed.org/docs/api/fred/realtime_period.html  
   Explains real-time periods and the fact that economic observations can be revised; a present-day history is not automatically the information set available at a past decision time.
2. St. Louis Fed, FRED versus ALFRED: https://fred.stlouisfed.org/docs/api/fred/fred_vs_alfred.html  
   Describes the distinction between information available today and historical-vintage information.
3. St. Louis Fed, ALFRED download data help: https://alfred.stlouisfed.org/help/downloaddata  
   Describes vintage-specific observations and revision history.
4. scikit-learn, TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html  
   Documents why time-ordered validation is needed to avoid training on future observations and testing on the past.
5. scikit-learn, common pitfalls / data leakage: https://scikit-learn.org/stable/common_pitfalls.html  
   Explains leakage as use of information unavailable at prediction time and warns against test-set-driven fitting/selection.
6. White, H. (2000), “A Reality Check for Data Snooping,” Econometrica 68(5), 1097–1126: https://doi.org/10.1111/1468-0262.00152  
   Motivation for keeping a complete registered candidate family and treating repeated reuse of a history as an inference risk.
7. Sullivan, R., Timmermann, A., and White, H. (1999), “Data-Snooping, Technical Trading Rule Performance, and the Bootstrap,” Journal of Finance 54, 1647–1691: https://doi.org/10.1111/0022-1082.00163  
   Demonstrates the relevance of data-snooping adjustments when a broad universe of trading rules is explored.

## Explicit remaining gaps

- No third-party auditor has independently authenticated all data-provider identities, licensing rights, timestamp semantics, or source snapshots.
- A digest proves file identity only after the trusted digest and collection procedure are established; it does not prove the source itself is correct.
- The graph validator only checks *declared* parent links. It cannot inspect arbitrary transformation code to discover an omitted dependency; independent review of transform code/manifests remains necessary.
- Parent facts must be included in the validated snapshot for exact in-snapshot resolution. This may require an explicit, hashed dependency manifest for large rolling features rather than silently omitting inputs.
- A fit cutoff and manifest reference are assertions, not proof that the training dataset truly stopped there. The fit manifest and data lineage must be independently audited.
- Feature causality also depends on every data join, label revision, fitted scaler, imputation rule and hyperparameter search. Global preprocessing or a revised label can reintroduce leakage.
- This module does not define a full event ontology for every possible domain. Unrecognized semantics should stay blocked until a versioned mapping and tests are reviewed.
- This is not a learned model, context understanding benchmark, profitability test, or live order safety certificate.
