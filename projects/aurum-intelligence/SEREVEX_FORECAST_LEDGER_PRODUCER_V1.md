# Serevex Forecast-Ledger Producer v1 — Contract

**State:** research adapter implemented; hosted validation required.  
**Input:** existing parsed historical OHLCV rows (source acquisition is separate).  
**Output:** serevex.forecast-ledger.v1 JSON plus one integrated stage 1–3 diagnostic report.  
**Authority:** reference-only historical source, NO_TRADE, no live signals/orders.

## What this producer does

It adapts the existing evaluation_protocol_v1_4.label_tp_before_sl and ci_benchmark.get_signal pipeline to a single fixed rule + side cohort. This is intentionally one series at a time, matching the integrated ledger's unique decision-index contract. It generates labels for TP-before-SL using the existing fixed rule, next-bar-open entry, ATR(14), TP 1.5 ATR, SL 1.0 ATR and 12-bar horizon defaults. Same-bar TP/SL collisions remain unresolved and are excluded from binary label scoring.

For each signal event, it emits a baseline Laplace rate over prior labels strictly matured before that decision plus four registered probability experts:

- constant_50pct
- rolling_matured_rate_288
- regime_matured_rate with baseline fallback below 20 matured labels in the state
- volatility_matured_rate with baseline fallback below 20 matured labels in the state

The conditional estimators are deliberately small causal comparators, not learned alpha models. A model update or forecast can only use labels whose target_exit_index is less than the current decision_index. The underlying protocol's prequential predictions remain available in the original benchmark output; this adapter creates a stricter and hashable cohort contract for the integrated stages.

## Temporal and source semantics

The producer fixes the directional rule and side before scoring: no rules are selected from final-holdout outcomes. It partitions eligible bar indices chronologically into development (first 50%), model selection (50–70%), calibration (70–80%), and final holdout (last 20%) before filtering down to the matching signal cohort. Missing observations in each cohort are not used to shift split boundaries.

Timezone-aware source timestamps are normalized to UTC but their provider semantics are not independently verified. Naive timestamps are rejected unless the operator explicitly passes --assume-naive-utc-for-diagnostic-only. That opt-in is recorded in the ledger and must never be treated as timezone verification. Historical feature availability and bar-close convention also remain unverified, so source/data/broker verification flags remain false and feature quality remains REFERENCE_ONLY.

The ledger retains source URI, the caller-supplied exact-byte dataset SHA-256, source-lineage digest, target-definition digest, estimator manifest hashes, target exit index, label resolution, and conservative feature facts. Hashes provide integrity identity for the supplied artifacts, not independent proof of source authenticity, data rights, live availability or broker representativeness.

## Run

From projects/aurum-intelligence/online-benchmark:

    python -m unittest -v test_serevex_forecast_ledger_producer_v1.py

Produce a ledger and run all three integrated stages in one invocation:

    python serevex_forecast_ledger_producer_v1.py \
      --input XAUUSD_5m.csv \
      --source-uri https://example.invalid/data.csv \
      --rule last_return_sign --side long \
      --as-of-index 35000 \
      --ledger-output forecast-ledger.json \
      --report-output stage123-report.json

Supply the real source URI and the actual dataset; the placeholder URI above is illustrative only. The as-of-index is an exclusive original bar index, not the row number in the filtered event cohort. If the CSV uses naive timestamps, explicitly add --assume-naive-utc-for-diagnostic-only for a diagnostic-only run. Data license and source qualification are not verified by this program.

## Acceptance conditions

1. Hosted test pass on the exact branch head.
2. Review of target/source timestamp semantics and independent data provenance/rights.
3. Paired out-of-sample comparisons on one exact rule/side cohort: causal expanding baseline, constant-50% expert, rolling expert, conditional experts, and pre-registered equal-weight mixture.
4. A later untouched cohort or an independent price feed for replication.
5. If economic claims are attempted, separate quote-side/tick evaluation with measured costs, execution uncertainty, overlap/exposure management, and a portfolio-level selection-bias correction.

A contract pass means only the code produced a schema-valid diagnostic. It does not prove predictive skill, calibration, originality, positive net PnL, or deployment readiness.
