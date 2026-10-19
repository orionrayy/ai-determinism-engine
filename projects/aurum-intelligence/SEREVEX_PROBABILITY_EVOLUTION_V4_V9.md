# Serevex Probability Evolution v4.0–v9.0

**Checkpoint:** 20 October 2026 (Asia/Jakarta)  
**Branch:** \`research/serevex-forecastability-discovery-v1\`  
**Implementation:** \`online-benchmark/serevex_probability_evolution_v4_v9.py\`  
**Tests:** \`online-benchmark/test_serevex_probability_evolution_v4_v9.py\`  
**State:** isolated research / offline governance comparison, now invoked by the hosted real-OHLCV benchmark workflow; not a production or Telegram/broker integration.

## Objective and evidence boundary

The stack takes an already-generated probability series from the current Serevex alpha expert plus a row-specific causal baseline. It compares increasingly conservative probability-governance policies on the same chronological cohort. It does not claim to train a better forecaster by itself. The input series must have been generated causally and must not have been tuned on the outcomes used to evaluate it.

The earlier XAUUSD M1 benchmark reported candidate Brier 0.1747186248 versus the causal matured-label baseline 0.1745785974, with baseline-minus-candidate improvement -0.0001400274 and a 95% moving-block interval fully below zero. That is negative evidence against that tested probability method, not a result for this new v4.0–v9.0 comparison. Do not carry a positive result from synthetic fixtures over to market data.

## Version ladder

| Version | Increment | Fail-closed behavior |
|---|---|---|
| **v4.0** | Reference policy: pass through the supplied, already-governed v4 candidate probability. Establishes a reproducible comparison anchor. | Invalid/missing candidate uses the exact row baseline. |
| **v4.1** | Strict cohort identity, unique decision IDs/indices, valid baseline range, candidate-input checks, future-feature rejection, and regime-availability checks. | Invalid/future feature candidates fall back to baseline; future/unusable regime metadata becomes \`__UNKNOWN__\`. |
| **v4.2** | Adds paired Brier and log-loss evidence against the same row-level baseline, using source-index blocks separated by the configured horizon embargo. | Without sufficient matured history, adequate independent blocks, positive conservative lower screens, and a positive-block share of at least 70% for both metrics, fall back to baseline. |
| **v5.0** | Searches a preregistered convex blend grid \`{0, 0.1, 0.25, 0.5, 0.75, 1}\` on an earlier selection segment, then confirms the chosen blend on a later, disjoint chronological gate. | The blend remains at zero alpha weight if selection or the later gate fails. |
| **v6.0** | Adds regime-conditioned blend estimates with shrinkage toward the global estimate. A regime must be available at decision time and supported by enough matured history. | Unknown/sparse regime or failed local gate uses the global estimate; failed global evidence uses the exact baseline. |
| **v7.0** | Raises the later confirmation gate to at least eight non-empty independent source-index blocks, while retaining dual-score and positive-block-stability requirements. | Unsupported confidence means baseline fallback; no probability is described as “high confidence” merely from its numerical extremity. |
| **v8.0** | Adds a prequential Brier-gain deterioration alarm based only on previously issued forecasts whose labels have matured. A detected adverse shift triggers baseline fallback. | Drift is a heuristic warning, not a calibrated change-point test; unknown history is explicitly labelled unknown. |
| **v9.0** | Combines the preceding gates, bounded historical memory, deterministic weight-update buckets, source-ledger adapter, metrics, reason codes, cohort comparison, and SHA-256 forecast/evaluation manifests. | No automatic promotion. \`NO_TRADE\`, validation false, deployment calibration false, and live signal/order flags false remain invariants. |

These are software protocol versions, not claims of mathematically novel methods. The blocks, conservative standard-error screens, score metrics, regime shrinkage, and drift alarms are engineering choices that require ablation and independent testing. The uncertainty screen is not a formal finite-sample guarantee.

## Chronology and scoring rules

- Forecasts are processed in ascending source decision index and only for rows with `decision_index < as_of_index`. The exclusive cutoff constrains emitted forecasts as well as scored outcomes; rows at or beyond it are excluded from the forecast manifest, fallback statistics, and evaluation metrics.
- An outcome is available for policy updates only when \`label_mature == true\`, the label is binary, and \`label_available_index < current_decision_index\`. Labels cannot mature at or before their own decision index.
- Source-index blocks score 12 bars followed by a 12-bar embargo by default. Block eligibility is anchored to absolute decision indices, not filtered row rank.
- The selection segment and confirmation segment are split at an absolute source-index block-period boundary (default period 24 bars), never by filtered-event rank. Their scored source-block IDs are exposed in the policy trace and checked for disjointness. v7–v9 require at least eight eligible blocks in the confirmation segment.
- Weight decisions update in deterministic 12-bar buckets by default and use a bounded 720-record matured-history window. This is a compute/recency policy and must be frozen before final testing.
- Model metrics are compared with the same baseline and same scored rows: Brier, log loss, ten-bin expected calibration error (ECE), base rate, forecast coverage, and fallback rate. Directional accuracy is not the primary promotion metric.
- The output includes an identity hash for emitted forecasts and a separate evaluation-report hash. Hashes establish reproducible identity, not data authenticity, legal rights, or source-time truth.

## Existing repositories/components reused

The implementation intentionally depends only on the Python standard library and uses Serevex's current forecast-ledger contract rather than replacing it. It is compatible with the existing producer output \`serevex.forecast-ledger.v1\`, selecting the \`alpha_logistic_v1\` expert by default.

Relevant existing assets:
- \`online-benchmark/serevex_alpha_model_v1.py\`: existing trained alpha v4 forecaster and baseline governor.
- \`online-benchmark/serevex_forecast_ledger_producer_v1.py\`: point-in-time forecast-ledger producer for a fixed registered rule/side cohort.
- \`online-benchmark/serevex_stage123_integrated_v1.py\`: ledger contract, predictability atlas, complementarity audit, delayed-feedback mixer.
- \`temporal-calibration-v1.3/calibration_diagnostics.py\`: established calibration metric contracts and synthetic contract tests.
- \`projects/serevex-quant/\` in the rebrand/snapshot repository: project documentation and earlier temporal-calibration / benchmark components. It is a separate snapshot; the newer alpha and evolution modules are still on the research branch and are not automatically synchronized there.
- \`project-memory-and-database\` contains release snapshots and earlier evaluation modules; these are reference material, not a substitute for testing the active branch.

## Run against an existing forecast ledger

From \`projects/aurum-intelligence/online-benchmark/\`:

\`\`\`bash
python -m py_compile serevex_probability_evolution_v4_v9.py test_serevex_probability_evolution_v4_v9.py
python -m unittest -v test_serevex_probability_evolution_v4_v9.py
python serevex_probability_evolution_v4_v9.py \\
  --input /path/to/serevex-ledger.json \\
  --output /path/to/serevex-probability-v4-v9-report.json \\
  --as-of-index 100000 \\
  --candidate-expert alpha_logistic_v1
\`\`\`

Use the ledger's true exclusive source decision-bar cutoff for \`--as-of-index\`; do not use the number of filtered forecast events as a substitute. The data source must be authorized for the intended use. This runner preserves the ledger's source-qualification flags instead of promoting them.

## Acceptance gates for any future promotion

1. Isolated tests pass on the exact commit and the full hosted workflow passes after the versioned module is added to the workflow.
2. The real producer emits the forecast ledger for a fixed target/rule/side, and adapter output hashes and row IDs reconcile exactly with the ledger.
3. A preregistered later untouched cohort compares v4.0, v4.1, v4.2, v5.0, v6.0, v7.0, v8.0 and v9.0 against the same causal baseline on identical eligible rows, including all fallback rows.
4. Any claimed improvement has positive baseline-minus-model Brier and log-loss gains under the predefined dependence-aware procedure, adequate coverage and support, no unresolved time leakage, and calibration diagnostics.
5. Repeat the frozen protocol on a later independent window/feed. Then, separately, evaluate cost-aware net utility and operational readiness.
6. Review runtime, memory, history truncation, missing outcomes, sparse regimes, drift false alarms, and selection/multiplicity effects. If any requirement fails, do not promote.
7. Keep \`model_validated=false\`, \`deployment_calibrated=false\`, \`live_signal_enabled=false\`, \`live_order_execution_enabled=false\`, and \`decision_output=NO_TRADE\`.

## Integration status and current limitation

The hosted `aurum-online-dataset-benchmark.yml` workflow now creates a canonical `serevex.forecast-ledger.v1` from the real OHLCV diagnostic cohort, adapts it through `evaluate_forecast_ledger_versions`, evaluates all eight registered versions at the same exclusive as-of boundary, asserts a common scored-row count and research-only authority flags, and writes both `probability_evolution_v4_v9_report.json` and a compact summary into the benchmark artifact directory.

This is an **offline evaluation connection**, not a production prediction-path change: it does not replace the alpha predictions emitted by the producer or connect to Telegram, brokers, or live execution. Hosted CI passing verifies code execution and contracts, not source authenticity, legal data rights, point-in-time publisher availability, predictive superiority, or profitability. Exact ordered source-ledger decision-ID reconciliation remains an additional check before treating the adapter/producer join as fully audited.
