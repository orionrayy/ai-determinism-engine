# Serevex Integrated Stages 1-3 — Contract v1

**Status:** integrated research prototype; hosted run for the latest hardened revision is pending. No market-performance claim.  
**Scope:** canonical ledger contract, causal predictability atlas, expert-complementarity audit, delayed-feedback expert mixer.  
**Authority:** research only; NO_TRADE; no live signals or broker order routing.

## Integration model

This increment consumes pre-generated probabilities from registered experts. It does not yet fit/train market forecasters and is not a claim of predictive alpha.

1. Stage 1 validates a point-in-time ledger: one chronological XAUUSD series per ledger; timezone-aware decision, regime, state-tag and label-availability timestamps; index/time agreement wherever a corresponding decision-time anchor exists; immutable target/data/lineage/model hashes; explicit data-source verification flags; one shared baseline and expert probability vector per decision; and feature-fact availability/observation chronology. Unknown state values must be explicitly encoded as UNKNOWN, not silently dropped.
2. Stage 2 measures paired Brier and log-loss improvement over the row-specific causal baseline in preregistered single-axis slices: regime, session, volatility bucket, spread bucket and event-risk bucket. Exact category lists for every supported axis are required in metadata under atlas_category_registry before evaluation; the atlas emits an insufficient-support result even for a registered category absent from holdout, instead of shrinking the hypothesis family after observing holdout categories. It reports sample support, block support, deterministic block-bootstrap intervals and a Bonferroni-adjusted per-cell level. A cell can be eligible only for independent replication. The atlas is descriptive/evidence triage, not a production router and not permission to choose features from the final holdout.
3. Stage 2 also audits residual correlation and equal-weight pairwise blend performance on a common matured-label cohort. It reports possible complementarity without selecting experts.
4. Stage 3 computes causal mixtures with the explicit baseline included as an expert. Weight updates use only previously recorded labels with label_available_index < current_decision_index; equal-maturity labels are processed deterministically as a batch. Global weights are used after minimum global support; regime weights require local support; otherwise the exact baseline probability is emitted. The output stores hashes for the forecast ledger and weight updates.

## Input contract

The input object is JSON-compatible and follows serevex.forecast-ledger.v1. Metadata contains:

- series_id, target_id, target_definition_sha256, dataset_sha256, source_lineage_sha256
- model_artifact_sha256_by_expert, registered_experts, registered_regimes, atlas_category_registry
- evidence_kind, source_verification, decision_time_basis, label_definition

Each record has decision_id, strictly increasing decision_index and timezone-aware decision_time; chronological partition; strictly later label_available_index and timezone-aware label_available_time; label/maturity state; regime id plus its availability index/time; session; baseline/expert probabilities; state tags plus availability index/time; and a list of feature facts. Each fact has source and transform identities/hashes, semantic type, quote basis, quality status, available_at, and observed_at. published_at, event_time and numeric value are optional.

Current partitions must appear chronologically in this order: development, model_selection, calibration, final_holdout. A missing bucket is represented explicitly with a state value such as UNKNOWN. This code validates shape and temporal ordering; it does not independently prove a hash, provider identity, license, point-in-time source capture, model-fit chronology, or broker-feed match.

## Output contract

- Stage 1: canonical ledger SHA-256, record/partition/fact counts, explicit source verification states.
- Stage 2: cell-level paired skill, log-loss improvement, support and intervals, hypothesis-family size and replication-only eligibility; pairwise residual-complementarity diagnostics.
- Stage 3: prequential probabilities, maturity-batched update ledger, final weights and hashes, matured-support counts and baseline-relative Brier/log loss.
- All stages retain model_validated=false, deployment_calibrated=false, live_signal_enabled=false, live_order_execution_enabled=false, and decision_output=NO_TRADE.

## Execution

From projects/aurum-intelligence/online-benchmark:

    python -m unittest -v test_serevex_stage123_integrated_v1.py
    python serevex_stage123_integrated_v1.py --input forecast-ledger.json --output stage123-report.json --as-of-index 50000

Or invoke run_serevex_stage123(ledger, as_of_index=..., score_partition="final_holdout") on a pre-generated forecast ledger. The output is a diagnostic report, not a signal. as_of_index is exclusive: records at or after that decision index are not processed by the mixer, and a label with label_available_index == as_of_index is never considered matured for scoring.

## Acceptance gates before future integration

1. Unit and hosted workflow passes on the exact branch head.
2. A real ledger producer emits complete point-in-time forecasts for every registered expert on the same decision cohort.
3. Model selection and calibration are frozen before the final holdout is inspected; an online mixer's pre-registered delayed feedback is part of the algorithm, not permission for human retuning.
4. Independent replication demonstrates corrected incremental skill versus baseline, equal-weight forecast combination and an earlier-cohort selected expert.
5. Coverage, calibration, regime fallback rate, weight concentration, drift and missingness are reported.
6. Economic validation remains separate: quote-side portfolio replay, transaction costs, slippage/fill assumptions and out-of-sample net risk-adjusted outcomes.
7. Data provenance, usage rights, target/source semantics and broker representativeness are independently reviewed.

No stage claims world-first mathematics. The research hypothesis is the integrated control loop (causal admissibility -> registered state-conditioned evidence atlas -> complementary expert audit -> delay-aware mixture with baseline fallback), to be judged by out-of-sample results rather than novelty language. The current implementation still consumes pre-generated expert probabilities: a qualified real-feed forecast-ledger producer and full independent source/time-rights verification remain separate blocking work.
