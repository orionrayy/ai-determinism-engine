# Aurum Point-in-Time External Context Research — Design Specification

**Date:** 2026-10-17 (Asia/Jakarta)  
**Status:** Proposed design; awaiting user review  
**Repository:** `orionrayy/ai-determinism-engine`  
**Implementation branch:** `feature/aurum-online-dataset-benchmark`  
**Related work:** Draft PR #164, Aurum online multi-source/large-scale benchmark  
**Decision gate:** `NO_TRADE`

## 1. Purpose and success definition

Build a modular research-data layer that tests whether independently sourced public information improves probabilistic forecasts of a precisely defined XAUUSD target beyond a strong price-only reference. The first supported target stays the existing matured-label TP-before-SL outcome under the documented entry, ATR barrier, horizon, ambiguity exclusion, and chronological split contract. This project is not an order router and does not enable live signals.

“World-leading” is a research ambition, not an acceptance assertion. The system must make it possible to disprove that ambition: any added data family must beat preregistered, appropriate baselines on unseen chronological data, survive uncertainty and cost sensitivity, and replicate across separate regimes before it may be considered useful. Passing data-contract CI demonstrates reproducible acquisition and evaluation, not predictive edge or profitability.

### Scope

Implement one coherent research slice comprising a common point-in-time event schema, independently testable adapters for CFTC positioning, GDELT historical news metadata, BLS release observations/calendar, and optional FRED/ALFRED macro series; a deterministic merge/as-of feature builder; data-lineage and licensing metadata; and an ablation benchmark alongside existing XAUUSD candle/path benchmarks. Each external adapter may be skipped for a missing credential or unavailable/unauthorized source, but it must report an explicit reason; a skipped source never counts as a passed live-data acquisition gate.

Non-goals: broker integration, orders, trade recommendations, neural/LLM sentiment as a production feature, training a large model before feature-value tests, copying paywalled article text, silently merging price feeds, or distributing raw third-party datasets in the report artifact.

## 2. Verified source constraints and implications

1. **CFTC COT:** CFTC exposes annual historical compressed reports, including Disaggregated Futures Only reports from September 2009 and legacy Futures Only history from 1986. COT position date is not the same as public release date. CFTC describes reports as usually released Friday at 3:30 p.m. Eastern using positions as of the preceding Tuesday, with holiday delays. The adapter must retain both dates and may not make Tuesday positions visible on Tuesday. If exact historical release time cannot be verified from an authoritative release record, use a documented conservative availability proxy no earlier than the Wednesday after the report date, and mark the timing quality as `conservative_proxy`; never label this as the actual release timestamp. Sources: https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm and https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm.

2. **GDELT news metadata:** The GDELT project describes broad multilingual news coverage and historical event archives reaching back to 1979; the GDELT DOC API has a limited rolling search horizon (normally three months, with the documented API extension up to one year). Therefore a rolling DOC query must not be represented as a multi-year historical news corpus. For historical point-in-time tests, prefer downloadable timestamped GDELT 2.0 GKG/archive snapshots and retain the archive timestamp/first-seen evidence. The `available_at` used by a replay is the first observed archive snapshot time, not an article's possibly backdated publication time. If the chosen GDELT product cannot provide reproducible archive-time evidence, that slice is descriptive only and is excluded from causal model features. Avoid retaining or redistributing article body text; preserve URL/domain hash and allowed metadata only. Sources: https://www.gdeltproject.org/ and https://blog.gdeltproject.org/doc-geo-2-0-api-updates-full-year-searching-and-more/.

3. **BLS:** Unregistered Public Data API v1 supports fewer requests/series/years per request than registered v2. The adapter must enforce the documented limits, batch requests, cache validated responses, and expose `rate_limited`/credential states rather than silently dropping data. Observation period is not publication time: CPI and employment values may only become usable when an authoritative release schedule or a conservatively delayed, documented availability rule permits them. If actual-versus-consensus surprise data is absent, do not invent consensus or call a value a surprise. Source: https://www.bls.gov/developers/api_faqs.htm.

4. **FRED/ALFRED:** FRED API requests require a user-issued API key. Real-time/vintage parameters exist to reconstruct values known in a historical period; latest revised values must not be backfilled as if they were known then. The adapter is optional in default CI and must consume a secret/environment variable only, never a committed key. Where applicable, user-facing integration must honor the FRED API Terms of Use, attribution notice, terms link, and applicable third-party data terms. Sources: https://fred.stlouisfed.org/docs/api/fred/series_observations.html, https://fred.stlouisfed.org/docs/api/fred/realtime_period.html, https://fred.stlouisfed.org/docs/api/terms_of_use.html.

All dataset licences, redistribution rights, original provenance, provider semantics, and intended-use scope must be recorded independently. A public URL or repository license label alone does not establish the right to redistribute an upstream dataset.

## 3. Architecture and module boundaries

New modules stay under `projects/aurum-intelligence/online-benchmark/`; existing price/cross-source/tick modules remain separate and are consumed through reports/contracts rather than merged into one large script.

- `external_context_schema_v1.py`: canonical typed event, source manifest, and data-quality status contract; UTC-aware timestamps and schema validation.
- `cftc_cot_adapter_v1.py`: bounded annual COT acquisition/parsing, gold contract filtering, stable record IDs, report date vs availability, positioning fields and source hashes.
- `gdelt_gkg_adapter_v1.py`: allow-listed HTTPS acquisition of bounded historical archive snapshots, parser for relevant GKG metadata, first-seen archive time, URL/title de-duplication keys, and byte/record limits. DOC API is an optional recent-window smoke source only.
- `bls_release_adapter_v1.py`: bounded BLS observation acquisition plus explicit association with release-calendar timestamps where available; response/rate-limit validation.
- `fred_alfred_adapter_v1.py`: optional credential-gated observation/vintage retrieval and cache; missing key produces a typed `SKIPPED_CREDENTIAL_REQUIRED` result, not a false green acquisition.
- `point_in_time_features_v1.py`: deterministic as-of join/aggregation for event counts, lagged COT net positioning/changes, available macro levels/changes, and GDELT themes/tone/counts. No future event can enter an earlier decision.
- `external_context_benchmark_v1.py`: joins eligible features to price-only scenario anchors; performs preregistered ablations and writes machine-readable report plus compact Markdown summary.
- Corresponding focused `test_*.py` files; existing benchmark scripts are not rewritten unless a specific shared contract requires a minimal change.

Adapter interface: `acquire(config) -> AcquisitionResult`; `parse(payload, source_manifest) -> list[ExternalObservation]`; `validate(observations) -> SourceAudit`. A source-specific adapter must not own the model or the final evaluation policy. Network I/O is bounded and tested separately from deterministic parsers/feature builders.

## 4. Canonical point-in-time data contract

Every accepted observation carries these fields (nullable only when contract says so):

- `source_id`, `source_record_id`, `dataset_or_series_id`, `schema_version`
- `event_time`: when the underlying event or economic reference period occurred
- `published_at`: source-reported publication time, if supplied
- `available_at`: earliest defensible time the strategy could have observed this record
- `availability_quality`: `verified`, `conservative_proxy`, `unknown`, or `not_applicable`
- `retrieved_at`, `source_url`, `payload_sha256`, `source_revision`/vintage if applicable
- `value` or normalized metadata; `unit`, `currency`, `contract` where applicable
- `rights_status`, `redistribution_status`, `provenance_status`, `quality_flags`

All timestamps are UTC ISO-8601 internally; original timezone and raw timestamp are retained for audit. Naive timestamps, ambiguous DST times, invalid calendar values, duplicate conflicting IDs, non-finite numeric values, future `available_at`, and negative/illogical positioning values are rejected or explicitly quarantined. Unknown time is never silently equated to UTC.

At decision time (t), a feature builder may only use observations satisfying `available_at <= t` and the feature's required availability quality. Unknown-timing source rows may appear in descriptive reports but must not be included in causal scored features. Conservatively lagged rows may be separately evaluated and clearly labeled; do not combine them invisibly with verified-timing results.

GDELT de-duplication must be stable across repeated snapshots. Count an item no earlier than the earliest archive in which the record appears. Store hashed canonical URL (not article text) plus publication/source metadata where permitted. Do not use a backdated published timestamp alone to imply the article was available to Aurum at that time.

## 5. Feature families and preregistered ablations

Use simple, interpretable features first. Version each family independently:
- Price-only: existing price/ATR features and present price-only baseline.
- COT positioning: commercial/non-commercial category net position and week-to-week change, optionally normalized by open interest; only if the selected COT table semantics support those fields.
- Macro series: available values and changes for predeclared series such as U.S. 10-year Treasury yield and broad dollar measures, using ALFRED vintages when available. A current-revised-only FRED series cannot claim strict historical point-in-time validation.
- Release events: time-to/after CPI/employment/FOMC events and observed released values. “Surprise” features require separately verified consensus snapshots; otherwise omit.
- News context: volume/intensity counts and selected source-provided themes/tone, grouped into fixed windows before each decision. Separate raw record volume from deduplicated-event volume and do not assume tone is a calibrated probability.
- Combined contexts: price+COT, price+macro, price+news, and only then combinations supported by individual ablations.

Every ablation is evaluated on the same eligible rows and same target labels as the baseline. Report coverage and attrition per feature family; do not compare a feature's Brier score on an easier subset to a baseline on another subset. At least one `feature_family_off` control and a label/permutation sanity test must exist. Hyperparameter/model selection must be confined to the development/calibration slices, never the untouched final holdout.

## 6. Evaluation protocol and minimum acceptance gates

- Reuse existing `evaluation_protocol_v1_4.py` target construction, matured-label constraints, entry/barrier/horizon semantics, same-bar ambiguity policy, and chronology contract; do not silently alter labels to improve results.
- Keep the existing ≥700,000 paired valid Bid/Ask M1 candle gate, ≥700,000 weekday active non-flat bar gate, ≥20,000 unique valid-ATR real time anchors, and ≥20,000 scenario minimums from the current benchmark.
- Add a context benchmark with a bounded live acquisition smoke and a large historical replay mode. The report must distinguish acquired records, valid parsed records, deduplicated records, usable point-in-time records, and anchor rows with full/partial context coverage.
- Prefer ≥100,000 real news/event observations across a clearly declared multi-year or preregistered event-window archive sample for the expanded benchmark; if upstream size/capture limits prevent this, mark the scale threshold `NOT_MET` rather than manufacturing rows or counting repeat copies as independent observations. COT and macro observations are reported separately and are not expected to match news record counts.
- Split data chronologically into development/calibration/final holdout, with a final tail holdout untouched until protocol, feature list, and thresholds are fixed. Use embargo/purge at least as long as the maximum forecast label horizon and feature windows where they can overlap target outcomes.
- Score exact matured TP-before-SL forecasts with Brier score, log loss, ECE/MCE and reliability bins, plus a strong constant/base-rate or causal prequential baseline. Include bootstrap uncertainty (moving blocks to respect temporal dependence) and per-regime/per-feature-coverage reporting.
- Evaluate net economics only with source-grounded or explicitly assumed costs. The current community/tick spread is not any user's broker fill cost. Do not claim alpha from a direction metric, a positive small trade sample, Brier skill against weak 0.5 alone, or synthetic stress-test passes.
- A feature family is eligible for promotion only if preregistered out-of-sample improvement versus the appropriate baseline is positive with uncertainty analysis, does not materially harm critical regimes, is robust to coverage/cost sensitivities, and can be reproduced on data whose source lineage is independent enough for the claim. The final holdout may be used once for final assessment, not repeatedly tuned.
- Until all existing data source, rights, execution-cost, calibration, independent-validation, risk/security and paper-trading gates pass: `NO_TRADE`, `deployment_calibrated=false`, `validated=false`, `live_signal_enabled=false`.

## 7. Acquisition safety, repeatability and output rules

- Use HTTPS, an allow-list per provider, strict URL/redirect checks, finite timeouts, bounded retries, max response bytes/record counts, deterministic backoff and provider-specific rate limits.
- Never include credentials in source, logs, reports, artifact ZIPs, error traces, Git history, URLs or query strings. FRED/BLS keys must arrive through environment/secrets. Missing credentials must be observable.
- Hash each downloaded payload and record URL, acquisition time, bytes, row counts, response/schema version, and parse/audit summary. Do not claim provenance independently verified solely because a hash exists.
- Avoid writing raw high-volume market/news data into Git or the report artifact. Keep only permitted compact summaries, manifests, hashes, and small clearly-labeled samples. Check upstream terms before persisting article identifiers or redistributing any metadata.
- Re-running the same fixture, manifest and configuration produces identical deterministic features/metrics except explicitly recorded runtime/provider timestamps.
- CI unit tests use frozen fixture payloads and need no secrets. Live acquisition job should use bounded public-source smoke requests; optional secret-gated providers report a skip. A separate manually initiated scale job may fetch the preregistered archive interval under strict caps. The job summary must not conflate unit-test pass, acquisition pass, data-scale pass, predictive-evidence pass, or deployment readiness.
- All external data anomalies fail closed or quarantine rows; source disagreement is reported, not repaired/promoted automatically.

## 8. Required tests

1. UTC conversion, DST ambiguity, naive timestamps, malformed/duplicate identifiers and deterministic manifest hashing.
2. `available_at > decision_time` rows never contribute to features; changing future rows never alters past feature rows (mutation/leakage test).
3. CFTC Tuesday report date cannot appear before conservative/verified publication availability; holiday/date parsing and gold contract filtering are tested.
4. GDELT archive timestamp beats backdated publication timestamp; duplicate URL records across snapshots are counted only from first seen; missing archive timestamps make rows non-causal/descriptive-only.
5. BLS observation date is not treated as release time; rate-limit/error responses are typed; no consensus means no “surprise” value.
6. FRED ALFRED vintage changes do not rewrite old as-known-as-of features; missing credential is reported as a skip; no secrets in output.
7. Same-row ablation comparison, coverage parity, final holdout isolation, matured-label chronology, block-bootstrap determinism and low-count calibration warnings.
8. Data adversarial cases: oversized payload, invalid redirect/domain, truncated content, row/byte cap, non-finite values, negative/illogical OHLC or positioning, duplicate/conflicting snapshot, partial source outage and stale snapshots.
9. Benchmark gates for the existing M1/anchor/scenario minima, external context observation counts, source rights/provenance status, and explicit `NO_TRADE`.
10. End-to-end hosted CI with reports/artifact hashes, no redistributing raw large-scale data, and evidence linking source/code/config hashes.

## 9. Work sequencing

The implementation plan, after this spec is approved, will stage work so later modules depend on frozen interfaces rather than tangled changes:
1. Common schema, manifest, data audit, tests and point-in-time/leakage contract.
2. CFTC adapter and parser with conservative release-time policy.
3. GDELT historical archive adapter, bounded download, deduplication, and event-window/record-volume audit.
4. BLS release observations and optional FRED/ALFRED vintage adapters with explicit credential gates.
5. Deterministic as-of joins, feature coverage and ablation runner; frozen-baseline fixture tests first.
6. Hosted multi-source benchmark workflow, real-data count thresholds, artifact/report summary and gate assertions.
7. Reconcile project status/memory, inspect workflow results, review all diff/tests, and retain `NO_TRADE`.

Each implementation task must begin with a failing targeted test, run the relevant complete suite after its change, record the exact test output and code/source hashes, and commit in focused units. If a public provider is down or data rights are unclear, report the failed/blocked source truthfully and do not replace it with synthetic observations.

## 10. Source references

- CFTC Historical Compressed COT reports: https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm
- CFTC release schedule and typical timing: https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm
- CFTC overview and COT methodology: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
- GDELT public news/event project: https://www.gdeltproject.org/
- GDELT DOC API horizon notes: https://blog.gdeltproject.org/doc-geo-2-0-api-updates-full-year-searching-and-more/
- BLS API FAQ/limits: https://www.bls.gov/developers/api_faqs.htm
- FRED observations and vintage parameters: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- FRED real-time periods: https://fred.stlouisfed.org/docs/api/fred/realtime_period.html
- FRED API terms: https://fred.stlouisfed.org/docs/api/terms_of_use.html

## 11. Review checklist / unresolved risks

- [ ] Confirm this is the right first research slice: PIT data contract + CFTC + historical GDELT + BLS/FRED adapters + same-row ablation.
- [ ] Confirm no FRED key is available in repository code; if none, CI exercises fixture/parser and reports the live connector as credential-gated.
- [ ] Confirm a public GDELT archive acquisition cap that is sufficient for useful historical coverage but fits GitHub-hosted time/network limits; if not, keep full-scale mode manual and mark scale not met.
- [ ] Confirm final holdout remains untouched and no external feature is eligible for live deployment simply from CI passing.
- [ ] Open P0 issues (upstream rights/provenance, candle time/price semantics, broker costs) remain blockers and cannot be bypassed by adding macro/news features.
