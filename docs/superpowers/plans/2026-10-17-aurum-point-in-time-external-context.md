# Aurum Point-in-Time External Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add a reproducible point-in-time context layer and real-data ablation benchmark for CFTC gold positioning, GDELT news archives, BLS macro releases, and optional FRED/ALFRED vintages—without weakening Aurum’s data gates or claiming unproven trading skill.

**Architecture:** Define a canonical schema/audit contract first. Isolated read-only provider adapters emit observations into that contract; a deterministic as-of feature builder joins only historically available observations to existing price/label anchors. The benchmark compares same-row ablations against causal price-only baselines, reports coverage and uncertainty, and remains separate from trading/deployment state.

**Tech Stack:** Python 3.12; standard library unless a bounded-format reader is necessary; unittest; existing evaluation_protocol_v1_4.py; GitHub Actions.

**Spec:** docs/superpowers/specs/2026-10-17-aurum-point-in-time-external-context-design.md

## Global Constraints

- Preserve the existing price benchmark gates: ≥700,000 paired valid Bid/Ask M1 candles, ≥700,000 weekday active non-flat candles, ≥20,000 unique valid-ATR timestamps, and ≥20,000 real barrier scenarios.
- Target ≥100,000 real GDELT/archive observations in a preregistered multi-year/event-window sample. If byte/provider/runtime limits prevent this, report NOT_MET with real counts; never synthesize or clone observations.
- A scored feature must satisfy available_at ≤ decision_time. Unknown-timing rows are descriptive-only. COT fallback availability is 00:00 America/New_York on the Wednesday of the following calendar week (report date + 8 days for usual Tuesday reports), unless the actual verified release time is later/better evidenced; ambiguous holiday calendars are quarantined.
- Do not backfill revised latest macro values into past knowledge. FRED/ALFRED is optional and returns SKIPPED_CREDENTIAL_REQUIRED if no runtime API key is provided.
- Public URLs/repository labels do not prove data rights, lineage or broker-feed equivalence. Carry rights/provenance flags in reports.
- Never commit/log/report credentials, raw high-volume price rows, or article body text. Use bounded HTTPS, strict host/path checks, finite timeouts, bounded retries, and byte/record caps.
- Preserve source-disagreement warnings, NO_TRADE, deployment_calibrated=false, validated=false, live_signal_enabled=false. Do not auto-shift, merge, promote sources or route orders.
- Pipeline CI success is not evidence of alpha, profitability, independent provenance, completed scale gates or deployment readiness.

## Review Focus

1. COT Tuesday position date used as availability, leaking before release: Task 2 tests verified-time/following-week-Wednesday policy.
2. Backdated publication date used ahead of GDELT archive batch time: Task 3 tests availability time.
3. Latest revised macro value rewrites old known-as-of features: Tasks 4–5 test release availability and vintage isolation.
4. Partial/outage source appears as successful zero-event dataset: Tasks 2–5 and 7 require typed statuses and separate counts.
5. Context model is scored on easier timestamps or holdout is tuned: Tasks 6–7 require same-row ablation, immutable split and mutation tests.

## File Map and Frozen Interfaces

Create under projects/aurum-intelligence/online-benchmark/:
- external_context_schema_v1.py / test_external_context_schema_v1.py: schema, source manifest, result statuses and audit.
- cftc_cot_adapter_v1.py / test_cftc_cot_adapter_v1.py: official COT download/parsing and release availability.
- gdelt_gkg_adapter_v1.py / test_gdelt_gkg_adapter_v1.py: bounded historical GKG archive parsing, first-seen time and de-duplication.
- bls_release_adapter_v1.py / test_bls_release_adapter_v1.py: public BLS observations and release mapping.
- fred_alfred_adapter_v1.py / test_fred_alfred_adapter_v1.py: optional key-gated vintage series.
- point_in_time_features_v1.py / test_point_in_time_features_v1.py: causal as-of features and coverage.
- external_context_benchmark_v1.py / test_external_context_benchmark_v1.py: same-row ablations and report.

Modify only after tests require integration: .github/workflows/aurum-online-dataset-benchmark.yml, projects/aurum-intelligence/PROGRESS_LOG.md, IMPLEMENTATION_STATUS.json, V1_4_1_IMPLEMENTATION_CHECKPOINT.md, and BENCHMARKS_AND_EVIDENCE.md. Keep existing evaluation_protocol_v1_4.py, source_concordance_v1_0.py, tick_bidask_benchmark_v1_0.py and large_scale_bidask_benchmark_v1_0.py behavior unchanged unless a contract-level issue requires a minimal tested change.

Schema module interfaces:
- ExternalObservation: frozen dataclass with source_id, source_record_id, dataset_or_series_id, schema_version, event_time, published_at, available_at, availability_quality, retrieved_at, source_url, payload_sha256, source_revision, value, unit, attributes, rights_status, provenance_status, quality_flags.
- SourceManifest: source_id, dataset_or_series_id, source_url, retrieved_at, payload_sha256, payload_bytes, row_count_raw, row_count_valid, provider_version, rights_status, provenance_status.
- AcquisitionResult: status (PASS, PARTIAL, SKIPPED_CREDENTIAL_REQUIRED, SKIPPED_POLICY, NOT_MET, FAILED), observations, manifests, warnings, error_code.
- SourceAudit: stable totals and invalid/duplicate/conflict counts.
- parse_utc_timestamp(value: str) -> datetime; validate_observations(observations) -> SourceAudit; manifest_sha256(payload: bytes) -> str.
- Each adapter exposes acquire(config) -> AcquisitionResult and parse(payload, manifest) -> list[ExternalObservation]. Parsing is deterministic and separated from network I/O.

Feature builder interfaces:
- visible_as_of(observations, decision_time, allow_conservative_proxy=True) -> list[ExternalObservation].
- build_asof_features(decision_times, observations, config) -> list[dict[str, object]] including features, per-family coverage/counts and compact lineage.
- Future and unknown-availability rows never contribute to causal features.

Benchmark interface:
- run_context_benchmark(price_rows, observations, config) -> dict[str, object].
- fit_logistic_baseline(train_features, train_labels, feature_names) -> dict[str, object]; predict_logistic_probability(model, features) -> numpy.ndarray. Model uses deterministic L2 full-batch gradient descent, lambda=1.0, learning rate=0.05, maximum 200 steps, zero initialization and stable sigmoid; normalization is fitted on development data only.
- Price-only and augmented models use matching decision rows, labels, splits and metrics; report source coverage/status, ablations, Brier/log loss/ECE/MCE, fair causal base-rate baseline, moving-block uncertainty, regime breakdown and scale_status. decision_output always NO_TRADE. Reuse temporal-calibration-v1.3/calibration_diagnostics.py for Platt fitting and probability metrics. Fit Platt only on labels matured in the calibration slice, purge non-mature boundary events, and score the untouched final 20% tail once.

---

### Task 1: Canonical observation schema and fail-closed audit

**Files:** Create external_context_schema_v1.py and test_external_context_schema_v1.py.

- [ ] Step 1 — Write failing tests: test_rejects_naive_datetime, test_rejects_nonfinite_values, test_conflicting_duplicate_id_is_counted_not_silently_overwritten, test_manifest_hash_is_deterministic, test_future_available_at_is_a_quality_error, test_unknown_availability_is_flagged_in_audit.
- [ ] Step 2 — Run RED: from the online-benchmark directory, run python -m unittest -v test_external_context_schema_v1.py. Expected: missing-module/function failures.
- [ ] Step 3 — Implement frozen dataclasses, validated statuses and stable audits. Do not mutate inputs or silently assume naive timestamps are UTC.
- [ ] Step 4 — Run focused test and python -m unittest -v for all existing/new tests in the directory; expected all pass.
- [ ] Step 5 — Commit: feat(aurum): add point-in-time observation contracts.

### Task 2: CFTC Gold COT adapter

**Files:** Create cftc_cot_adapter_v1.py and test_cftc_cot_adapter_v1.py; consumes Task 1.

**Interfaces:** parse_cot_csv(payload: bytes, manifest: SourceManifest) -> list[ExternalObservation]; conservative_cot_available_at(report_date: date) -> datetime; resolve_cot_available_at(report_date: date, verified_release_at: datetime | None = None, ambiguous_calendar: bool = False) -> tuple[datetime | None, str]; select_gold_futures(observations) -> list[ExternalObservation].

- [ ] Step 1 — Write fixture tests: test_tuesday_report_is_not_available_on_tuesday_or_friday, test_conservative_proxy_is_next_week_wednesday, test_verified_release_timestamp_overrides_proxy_only_when_valid, test_gold_filter_excludes_other_contracts_and_options_combined, test_invalid_positions_are_quarantined, test_truncated_or_oversized_zip_fails_closed. Use tiny inline CSV/ZIP fixtures only.
- [ ] Step 2 — Run python -m unittest -v test_cftc_cot_adapter_v1.py; expected RED.
- [ ] Step 3 — Implement bounded acquisition from official CFTC Disaggregated Futures Only archives, strict official HTTPS host/path, byte/retry/timeout limits, source hashes, exact field/report-date preservation and Gold-only filtering. Declare years and coverage; one-year smoke does not count as complete historical coverage.
- [ ] Step 4 — Run focused tests and complete online-benchmark unittest suite; expected GREEN independent of network.
- [ ] Step 5 — Commit: feat(aurum): add causal CFTC gold positioning adapter.

### Task 3: GDELT historical GKG archive adapter

**Files:** Create gdelt_gkg_adapter_v1.py and test_gdelt_gkg_adapter_v1.py; consumes Task 1.

**Interfaces:** parse_gkg_tsv(payload: bytes, manifest: SourceManifest) -> list[ExternalObservation]; archive_available_at(record_id: str) -> datetime; deduplicate_snapshots(observations) -> list[ExternalObservation].

- [ ] Step 1 — Write fixture tests: test_backdated_publication_is_not_available_before_archive_batch, test_duplicate_article_across_snapshots_keeps_first_seen, test_conflicting_duplicate_is_audit_warning, test_bad_record_id_and_oversized_payload_fail_closed, test_doc_api_rolling_window_is_not_labeled_long_history, test_no_article_body_is_persisted.
- [ ] Step 2 — Run python -m unittest -v test_gdelt_gkg_adapter_v1.py; expected RED.
- [ ] Step 3 — Implement allow-listed bounded acquisition using official GDELT archive/master lists and a preregistered multi-year event-window schedule. DOC rolling search is recent-window smoke only. Archive/batch timestamp is the availability evidence; never substitute a backdated publication timestamp. Cap file and aggregate bytes; preserve only permitted metadata, URL/domain hashes and selected GKG themes/tone; never retain article body.
- [ ] Step 4 — Run focused tests and full unittest suite. If real data scale is below 100,000, record scale_status=NOT_MET and actual counts; never duplicate rows.
- [ ] Step 5 — Commit: feat(aurum): add bounded historical GDELT context adapter.

### Task 4: BLS observation and publication-time adapter

**Files:** Create bls_release_adapter_v1.py and test_bls_release_adapter_v1.py; consumes Task 1.

**Interfaces:** parse_bls_api_response(payload: bytes, manifest: SourceManifest) -> list[ExternalObservation]; map_release_availability(series_id: str, observation_period: str, release_calendar: Mapping[str, datetime]) -> datetime | None.

- [ ] Step 1 — Write inline fixture tests: test_observation_period_is_not_published_at, test_public_api_v1_limits_are_enforced, test_error_status_and_rate_limit_are_typed, test_missing_release_calendar_means_descriptive_only, test_consensus_is_not_invented.
- [ ] Step 2 — Run python -m unittest -v test_bls_release_adapter_v1.py; expected RED.
- [ ] Step 3 — Implement bounded public API v1 request path and release association; enforce 25 queries/day, at most 25 series/request and 10 years/request. Observation period is not publication time. Use authoritative release time or a later safe proxy; omit “surprise” features without an independent consensus snapshot.
- [ ] Step 4 — Run focused and whole suite; typed rate/partial/error states must not become successful empty datasets.
- [ ] Step 5 — Commit: feat(aurum): add point-in-time BLS macro release adapter.

### Task 5: Optional FRED/ALFRED vintage adapter

**Files:** Create fred_alfred_adapter_v1.py and test_fred_alfred_adapter_v1.py; consumes Task 1.

**Interfaces:** parse_fred_observations(payload: bytes, manifest: SourceManifest) -> list[ExternalObservation]; acquire_fred(config) -> AcquisitionResult; observation_as_of_vintage(observations, as_of_date) -> list[ExternalObservation].

- [ ] Step 1 — Write fixture tests: test_missing_key_returns_typed_skip, test_key_never_appears_in_report_or_error, test_latest_revision_does_not_rewrite_older_vintage, test_nonfinite_and_missing_values_are_quarantined, test_pagination_and_result_caps_are_enforced.
- [ ] Step 2 — Run python -m unittest -v test_fred_alfred_adapter_v1.py; expected RED.
- [ ] Step 3 — Implement official-host-only parsing/acquisition, explicit vintage/real-time parameters and bounded pagination. Key is runtime environment only; absent key returns SKIPPED_CREDENTIAL_REQUIRED. No key is assumed to exist in GitHub secrets.
- [ ] Step 4 — Run focused + whole test suite; no-credential CI reports a skip, not fake acquisition success.
- [ ] Step 5 — Commit: feat(aurum): add optional FRED ALFRED vintage source.

### Task 6: Deterministic as-of features and leakage tests

**Files:** Create point_in_time_features_v1.py and test_point_in_time_features_v1.py; consumes Tasks 1–5.

- [ ] Step 1 — Write tests: test_future_observation_never_changes_past_feature, test_unknown_availability_is_excluded_from_causal_features, test_conservative_proxy_is_separately_counted, test_deduplicated_news_counts_first_available_time_only, test_family_coverage_is_reported_without_dropping_baseline_rows, test_output_is_deterministic_and_sorted, test_mixed_timezone_inputs_are_rejected.
- [ ] Step 2 — Run python -m unittest -v test_point_in_time_features_v1.py; expected RED.
- [ ] Step 3 — Implement visible_as_of and build_asof_features. Build lagged/normalized COT, macro level/change, fixed-window news event-count/theme/tone features, family coverage and compact lineage. Retain decision rows with absent feature families; distinguish zero events from data unavailable.
- [ ] Step 4 — Run mutation tests plus full suite; adding/changing observations where available_at > decision_time must leave past feature rows unchanged.
- [ ] Step 5 — Commit: feat(aurum): add deterministic point-in-time feature builder.

### Task 7: Same-row ablation benchmark

**Files:** Create external_context_benchmark_v1.py and test_external_context_benchmark_v1.py; consumes Task 6 and evaluation_protocol_v1_4.py.

- [ ] Step 1 — Write tests: test_all_ablations_use_same_decision_rows_and_labels, test_final_holdout_is_not_used_for_feature_or_threshold_selection, test_matured_labels_only_update_causal_baseline, test_future_feature_mutation_does_not_change_holdout_predictions, test_block_bootstrap_is_deterministic, test_low_count_calibration_is_flagged, test_context_scale_below_100k_is_NOT_MET_not_synthetic_pass, test_report_never_enables_trading.
- [ ] Step 2 — Run python -m unittest -v test_external_context_benchmark_v1.py; expected RED.
- [ ] Step 3 — Implement deterministic L2 logistic regression for each ablation: zero initialization, stable sigmoid, full-batch gradient descent with lambda=1.0, learning rate=0.05 and max 200 steps. Fit normalization and model only on development rows; fit Platt calibration using only mature calibration-slice labels by reusing temporal-calibration-v1.3/calibration_diagnostics.py (minimum 100 calibration samples). Purge split-boundary examples whose labels are not mature at the next segment start. The final 20% tail is scoring-only and never used for feature, normalization, model or calibrator selection. Compare price-only to price+COT, price+macro, price+news and eligible combined context on identical timestamps/labels. Report causal base-rate/prequential baseline, Brier/log loss/ECE/MCE/reliability, moving-block CI, coverage/attrition, regimes, provider states and source caveats.
- [ ] Step 4 — Run bounded real historical acquisition/benchmark; produce compact JSON/Markdown and manifest hashes only, retain existing 700k/20k gates, and report failure/partial/NOT_MET honestly.
- [ ] Step 5 — Run focused and complete tests; commit: feat(aurum): benchmark external context ablations.

### Task 8: CI integration, evidence checkpoint and final review

**Files:** Modify .github/workflows/aurum-online-dataset-benchmark.yml, PROGRESS_LOG.md, IMPLEMENTATION_STATUS.json, V1_4_1_IMPLEMENTATION_CHECKPOINT.md and BENCHMARKS_AND_EVIDENCE.md.

- [ ] Step 1 — Add static workflow contract tests before edits: new paths trigger CI, credentials remain optional/secret-only, NO_TRADE and price benchmark gates remain, and NOT_MET cannot be displayed as scale-pass.
- [ ] Step 2 — Confirm RED, then update workflow to run all schema/adapter tests before bounded acquisition, cap all downloads, allow FRED skip, and exclude raw datasets/secrets from artifacts.
- [ ] Step 3 — Run hosted CI and inspect job logs/artifacts. Existing large price benchmark must still clear gates. Each source status is separately reported PASS/PARTIAL/SKIPPED_CREDENTIAL_REQUIRED/NOT_MET/FAILED. Pipeline may pass while reporting missing scale, but the scale gate itself cannot be marked passed.
- [ ] Step 4 — Update status and project memory only from observed evidence: tested source SHA, run ID, artifact SHA-256/expiry, exact counts, URLs/versions, skips, tests, metrics and blockers. No “accuracy #1” or validated-alpha assertion.
- [ ] Step 5 — Review full diff/status; verify no raw dataset/key, auto-correction/merge, target-semantic changes, altered trade gate, or claim that an untested head passed. Keep PR #164 draft/unmerged absent a separate merge authorization.
- [ ] Step 6 — Commit verified checkpoint: docs(aurum): checkpoint external context benchmark; record deferred minors and NOT_MET scales.

## Acceptance Checklist

- [x] Each provider parser has deterministic fixture tests independent of network.
- [x] At least one public-source acquisition is verified by payload hash and caps; missing data cannot masquerade as success.
- [x] COT availability cannot precede verified/conservative public availability.
- [x] GDELT archive-time availability and de-duplication are deterministic.
- [x] BLS period/publication time remain distinct; FRED vintages never rewrite old as-known-as-of rows.
- [x] Future-row mutation tests leave past features and holdout predictions unchanged.
- [x] Ablations use identical rows/labels and report coverage.
- [x] ≥100,000 real event records met or explicitly NOT_MET with actual count/reason.
- [x] Existing ≥700,000 active-candle and ≥20,000 real scenario gates remain.
- [x] Calibration uses fair causal baselines and uncertainty intervals; no profitability implication.
- [x] Latest tested commit has an actual completed green hosted run; startup_failure/zero-job runs are not tests.
- [x] Artifacts exclude raw large datasets and credentials.
- [x] Decision remains NO_TRADE; deployment flags remain false.

## Official References

- CFTC historical compressed COT files: https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm
- CFTC timing: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
- GDELT archive/master lists and update cycle: https://gdeltproject.org/data.html
- GDELT GKG record/batch timestamps: https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf
- BLS API limits: https://www.bls.gov/developers/api_faqs.htm
- FRED observations/vintage parameters: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- FRED real-time/vintage semantics: https://fred.stlouisfed.org/docs/api/fred/realtime_period.html
- FRED terms: https://fred.stlouisfed.org/docs/api/terms_of_use.html


## Actual Execution Outcome — 2026-10-17

**State:** Implementation and hosted verification complete on the P0 research branch; this is not a production release. The PR remains draft/unmerged.

- Verified run: [GitHub Actions #39035732248](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39035732248), conclusion `success`, tested source head `f3eaa644a1f73c0d4f9521870e1ba5c460cfaf69`.
- Unit/regression tests: 115 passed; temporal-calibration contracts: 240/240 passed. The existing large Bid/Ask benchmark passed with 1,313,280 paired M1 candles, 20,000 valid ATR anchors, and 120,000 TP/SL path scenarios.
- Artifact #12102745058, 2,624,921 bytes, SHA-256 `b1710712519fd48e3a5f29acf1d532707997821bba1948e49b093004fd7b81f0`; downloadable from https://github.com/orionrayy/ai-determinism-engine/actions/runs/39035732248/artifacts/12102745058.
- GDELT scale gate: 141,020 unique real observations >= 100,000, no synthetic or duplicated records added, 100 selected archives processed out of 85,809 discovered. The archive set covers only 1,500 of 1,313,279 minutes (0.114218%), so `source_acquisition_status=PASS` but `source_completeness_status=PARTIAL`; this is not full historical news coverage.
- Feature eligibility across 19,937 mature price labels: COT 19,739 rows (99.0069%, `PARTIAL`), macro 0 (`NOT_MET`), news 0 (`NOT_MET`). BLS values are descriptive without a verified publication-time calendar; FRED/ALFRED skipped without the optional API key. Price+news and price+macro remain ineligible.
- Same-row holdout n=3,987: price-only Brier 0.1609893 versus price+COT/combined Brier 0.1609044. Incremental Brier improvement +0.00008494, paired moving-block 95% interval [-0.00014777, +0.00032012] crossing zero. No robust incremental COT value, alpha, best-model, or world-leading-accuracy claim is established.
- A preceding run [#39035356832](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39035356832) failed after unit/price gates due to a report-scope `NameError`. It is preserved as a failed diagnostic. The scope was corrected and the full pipeline was rerun successfully; don't retroactively classify the failed run as green.
- Application remains v1.3.1 `research-ux-prototype`; evaluation protocol v1.4.0 P0. `NO_TRADE`, deployment calibration false, validated model false, validated profitability false, and live signal/order flags false. PR #164 remains draft/unmerged.
- Remaining independent blockers: data rights/provenance and quote/time semantics; a material primary-vs-tick price-level discrepancy and suspected shared lineage of the secondary OHLCV/tick Bid feed; broker-authorized quotes and measured execution costs; adequate continuous news coverage and authoritative macro vintages; out-of-sample validation across independent regimes and paper trading.

Note: the checkbox checklist above records current acceptance outcomes. It does not assert that every planned historical RED-phase step was separately logged in chronological TDD order; the evidence of completion is the current code/test head and hosted run cited here.
