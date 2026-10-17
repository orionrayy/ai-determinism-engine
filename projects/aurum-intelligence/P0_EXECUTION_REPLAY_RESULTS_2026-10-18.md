# P0 execution replay — verified benchmark results

**Checkpoint:** 2026-10-18, Asia/Jakarta  
**Research commit:** `2cb6c3dc1eb1d48a2bcdbf8e0792f6e80bdec151`  
**PR:** [#164 — feat(aurum): P0 temporal evaluation and source-concordance gates](https://github.com/orionrayy/ai-determinism-engine/pull/164)  
**Hosted CI:** [run #39057955872 — success](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39057955872)  
**Artifact:** `12110540720`, 2,635,151 bytes, SHA-256 `7e192a620b75bf61c32d4d9daa4d3528dcdb8264b9c8e0b4a23a7ad05f10a7fe`; expires 2026-10-31 18:28:37Z.  
**Decision gate:** `NO_TRADE`; profitability validation false; live execution authorization false.

## Pipeline verification

- 153 unit/regression tests passed.
- Temporal-calibration test module: 8 tests passed; 240/240 synthetic calibration contracts passed.
- The online-data workflow re-acquired the public sources, ran M1 Bid/Ask replay, performed the cross-source/tick audits, evaluated point-in-time external context, and uploaded the artifact.
- These are software/data-contract and diagnostic checks, not independent verification of public-feed provenance or proof of profitability.

## Dataset and entry geometry

- Dataset label: `XAUUSD-DUKASCOPY-M1-BID-ASK`; 1,313,280 paired M1 Bid/Ask candles over 30 calendar months, 2023-01 through 2025-06; 131,316,007 bytes fetched.
- Weekday active non-flat candles: 870,500.
- Provider/broker provenance and intended-use rights remain unverified. This is reference/research data, not verified execution data from the intended broker.
- Across 875,611 contiguous, ATR-valid candidate windows, 217,544 had entry spread greater than or equal to the 1×ATR stop distance: **24.8448%**. Median entry spread was 0.36 price units, P95 0.68, P99 0.957; median ATR was approximately 0.5535.
- The sampler filters that geometry before choosing 20,000 unique anchors. The selected set contained zero invalid-stop-geometry paths. This is a sample-quality gate; it does not mean the market rarely has wide spreads.

## Quote-side path replay

Protocol: signal after completed anchor close; next M1 open entry; long Ask/Bid exit; short Bid/Ask exit; TP = 1×ATR; SL = 1×ATR; 60-bar horizon. Gaps use the observed executable-side open. Same-bar TP/SL collisions retain pessimistic/optimistic gross-PnL bounds. Threshold construction follows the legacy evaluator's floating-point operation order to prevent exact-price threshold rounding from changing labels; reported PnL remains Decimal-based.

- Unique anchors: 20,000.
- Bid/Ask direction paths: 40,000.
- Legacy-label cross-checks: 40,000; mismatches: 0.
- Strategy signal events across the four fixed rules: 53,129; final chronological holdout events: 10,637.
- Path outcomes: 30,721 stop losses; 8,891 take profits; 207 same-bar ambiguous outcomes; 124 adverse stop-at-open exits; 48 take-profit-at-open exits; 9 horizon timeouts.
- Same-bar ambiguity is retained as bounds. Metrics are overlapping per-signal events, not a capital-aware portfolio/equity simulation.

## Economic result of the tested fixed rules

The values below are **gross quote-side R per signal**, before unmeasured commission, swap, slippage, impact, latency, rejection, partial fills, and user-broker costs. Lower/upper means represent the pessimistic/optimistic treatment of same-bar ambiguity. The illustrative extra 0.05R cost sensitivity is a hypothetical assumption, not an observed execution cost.

| Fixed rule | All events | Final-holdout events | Final-holdout gross mean R bounds | Final-holdout PF bounds | Mean R bounds with +0.05R assumed extra cost |
|---|---:|---:|---:|---:|---:|
| Momentum (5-bar) | 19,975 | 3,995 | -0.5094 to -0.5003 | 0.325 to 0.333 | -0.5594 to -0.5503 |
| EMA 10 vs EMA 30 | 20,000 | 4,000 | -0.4922 to -0.4812 | 0.340 to 0.350 | -0.5422 to -0.5312 |
| Mean reversion z20 | 10,616 | 2,141 | -0.5296 to -0.5203 | 0.308 to 0.316 | -0.5796 to -0.5703 |
| Range breakout 20 | 2,538 | 501 | -0.4723 to -0.4523 | 0.358 to 0.377 | -0.5223 to -0.5023 |

**Result:** all four tested baselines have negative mean gross R and profit factor well below 1 in the final holdout, before unmeasured execution costs. The tested baselines therefore fail this research gate and must not be promoted as candidates for production. This is not a claim that all possible trading models fail; it is a rejection of these fixed rules under this data/protocol.

## Probability and context diagnostics

- Large-scale causal probability benchmark: aggregate Brier 0.174719 against the matured-label baseline 0.174579; skill versus that baseline -0.000802. The final holdout has a tiny positive Brier skill of +0.000315, which does not reverse the negative aggregate result or independently establish an edge.
- Point-in-time external-context acquisition processed 141,020 unique observations, exceeding its bounded 100,000 observation software gate without adding synthetic/duplicated observations. Overall source acquisition passed, but source-window completeness remains `PARTIAL`.
- COT was eligible on 19,688 of 19,894 rows (98.9645%). Macro and news features were eligible on 0 rows in the evaluation cohort; their ablations correctly remained `NOT_ELIGIBLE`.
- On the same 3,978-row holdout, price+COT Brier was 0.186169 versus price-only 0.186287. Paired moving-block 95% interval for the Brier improvement was [-0.000381, +0.000675], crossing zero. The incremental COT contribution is therefore inconclusive; the available macro/news observations are not evidence of added value.

## Remaining blockers

1. Obtain independently licensed, provenance-verified Bid/Ask/tick data with documented timestamp semantics, quote basis, contract specification, and rights for the intended use.
2. Keep cross-source disagreement/lineage warnings unresolved; no automatic time shift, rescaling, merge or source promotion.
3. Add instrument/venue-specific commission, financing/swap, slippage, latency, no-fill/rejection and stop-gap execution evidence before calculating net account-currency PnL.
4. Build portfolio-level replay with realistic overlap, position sizing, exposure, equity, margin and tail-risk accounting.
5. The final holdout metrics in this report have now been inspected. Do not tune the model against this same holdout and continue to call it untouched; any further model selection requires a new locked chronological test period or truly independent source/regime.
6. Preserve `NO_TRADE`, `validated_for_profitability=false`, `deployment_calibrated=false`, and `live_execution_authorized=false`.

CI success means the declared software and data-processing contracts passed. It does not validate source rights, broker equivalence, execution fidelity, model calibration for production, or profitability.
