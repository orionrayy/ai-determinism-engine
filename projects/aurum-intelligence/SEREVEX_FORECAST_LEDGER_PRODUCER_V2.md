# Serevex Forecast-Ledger Producer v2 — Alpha-Integrated Contract

**Status:** research adapter plus trained alpha forecaster implemented; hosted validation required.  
**Output:** `serevex.forecast-ledger.v1` JSON and integrated stage 1–3 diagnostic report.  
**Authority:** reference-only historical OHLCV; `NO_TRADE`; no live signals/orders.

## Event and target construction

The adapter uses the existing `evaluation_protocol_v1_4.label_tp_before_sl` and `ci_benchmark.get_signal` implementations for one fixed directional rule + side cohort. Defaults remain ATR(14), next-bar-open entry, TP 1.5 ATR, SL 1.0 ATR, 12-bar horizon; same-bar TP/SL collisions stay unresolved and are excluded from binary scoring. The chronological partition boundaries are based on source-bar indices before the event cohort is filtered.

## Experts and trained alpha model

The producer emits one causal Laplace baseline and five registered experts:

- `alpha_logistic_v1`: standard-library regularized logistic model with eight causal OHLCV features, development fit, model-selection L2 choice, refit before calibration, and calibrator trained only on mature calibration labels.
- `constant_50pct`
- `rolling_matured_rate_288`
- `regime_matured_rate`, falling back to the causal expanding baseline below 20 matured labels.
- `volatility_matured_rate`, falling back to the causal expanding baseline below 20 matured labels.

The alpha expert emits a 0.5 cold-start probability in development, a development-fit scout prediction during model selection, a pre-calibration refit during calibration, and a frozen calibrated prediction throughout final holdout. Hyperparameter selection uses model-selection observations whose labels mature before calibration begins. Model refit uses only development/model-selection labels mature before calibration begins. Calibration uses only calibration labels mature before final holdout begins. Holdout labels never enter alpha fitting or calibration. Artifact hash, transform hashes, split cutoffs, counts, calibration status and holdout metrics are included in the ledger.

The predictions are model outputs, not trade signals. Statistical improvement remains unproven until independently replicated.

## Data-time and source limitations

Timezone-aware timestamps are normalized to UTC, but provider semantics and historical bar-close availability are not independently verified. Naive timestamps are rejected unless an explicit diagnostic-only UTC assumption is requested. Source provenance, rights and broker-feed matching flags remain false. Feature facts are marked reference-only. The program's successful contract checks do not establish source authenticity, data rights, calibrated predictive skill or profitability.

## Run

From `projects/aurum-intelligence/online-benchmark`:

    python -m unittest -v
    python serevex_forecast_ledger_producer_v1.py \
      --input XAUUSD_5m.csv \
      --source-uri https://example.invalid/data.csv \
      --rule last_return_sign --side long \
      --as-of-index 35000 \
      --ledger-output forecast-ledger.json \
      --report-output stage123-report.json

The example URI is illustrative. Supply the exact dataset, correct source URL and license reference. If timestamps are naive, explicitly request the diagnostic-only policy. The as-of index refers to original source-bar indices, not filtered event rows.

## Acceptance gates

1. Hosted unit/integration tests pass on the exact branch head.
2. Alpha fit/calibration cutoffs are audited for strict label maturity and holdout isolation.
3. Report alpha and mixer forecasts against identical causal baseline labels; retain negative and inconclusive findings.
4. Replicate on an independent feed and a later untouched cohort.
5. Separately validate executable quote-side economics, costs, slippage, overlap/exposure and portfolio-level selection bias.
6. Independently review source provenance, usage rights, timestamps and broker representativeness.

All output continues to enforce `NO_TRADE` and disabled live-signal/order flags.
