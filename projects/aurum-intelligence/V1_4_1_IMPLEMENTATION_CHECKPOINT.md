# Aurum Intelligence — follow-up implementation checkpoint

**Checkpoint:** 2026-10-17, Asia/Jakarta  
**Application package:** v1.3.1  
**Research evaluator:** v1.4.0 P0  
**State:** research/UX prototype; P0 remains open; trading decision gate `NO_TRADE`.

## Implemented in this continuation

- Added non-mutating price-basis diagnostics to the cross-source comparison: signed deltas for OHLC, descriptive OLS price-level relationship, residual distribution in price units and basis points, four chronological quartile summaries of signed close differences, and direct paired consecutive-bar return differences.
- Added regression tests for additive level offset and multiplicative scale discrepancy. The tests require the report to expose the relationship while ensuring that no price correction, timestamp shift, merge or source promotion occurs.
- Source implementation commit: `5ec99e2fe4e478ddda60a8ebbef264226c47a076`.
- Regression tests commit / code-test head: `6b5b2f04a674c81deee84f1a2cdaf5ca442663d5`.

## Hosted verification

- [GitHub Actions run #39024803960](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39024803960): **success** on code/test head `6b5b2f04a674c81deee84f1a2cdaf5ca442663d5`.
- Passed workflow stages: temporal-calibration contracts, parser/baseline/source-concordance unit tests, current public dataset acquisition, benchmark execution, count/assertion gates, and artifact upload.
- Artifact #12097644871, 2,563,258 bytes; SHA-256 `b2bc267bda4d067ed6baec07d196540f903e84545cf342f9938b17371f607415`; retention through 2026-10-31.
- The workflow can run on GitHub-hosted infrastructure even though the current local shell cannot resolve `github.com`.

## Online dataset evidence and new diagnostics

Primary dataset: 35,954 community XAUUSD M5 OHLCV rows, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`.

Secondary dataset: 17,371 community XAUUSD M5 OHLCV rows, SHA-256 `ee23e18f0ed77160893936d0312c008261e4f2ac16d82365898aba20ea26de9d`.

- Exact common timestamps: 11,509; exact OHLC matches: 0.
- Absolute close discrepancy median/p95: 134.885 / 203.661 bps.
- One-bar return correlation: 0.9984495.
- OLS relationship (primary close ~= intercept + slope × secondary close): intercept +130.351 price units; slope 0.984434; level R² 0.940465; residual absolute p95 128.848 bps. A simple affine mapping does not explain the discrepancy adequately.
- Across the four chronological overlap quartiles, median signed close differences are +28.55, +72.535, +55.945 and +60.5135 price units; the discrepancy is not a stable constant offset.
- Consecutive 5-minute return absolute difference median/p95: 0.264 / 0.982 bps. Strong return correlation does not establish price-level or execution-price equivalence.
- The best timestamp-offset candidate is +12 bars (+60 minutes), exactly at the scan boundary; it reduces median discrepancy only 2.37% and its one-bar return correlation falls to -0.0146. This is not evidence to shift timestamps.

The source status stays `SOURCE_DISAGREEMENT_REVIEW_REQUIRED`. No data are corrected, rescaled, joined into a synthetic continuous series, or promoted. CI PASS here means only that acquisition, evaluator, diagnostics, assertions and artifacts function as designed.

## Open P0 gates

1. Resolve provider timestamp/candle-label semantics, price-basis methodology, revision behavior and source provenance using independent upstream evidence.
2. Verify intended-use rights for both upstream datasets; repository license labels alone are insufficient proof of the underlying data rights.
3. Obtain broker-authorized Bid/Ask or tick history and measured spread, commission, swap, slippage and fill semantics.
4. Keep source-specific backtests separate until the data discrepancy is understood; validate multiple genuinely independent regimes and an untouched holdout after the data gate passes.
5. Validate calibration for the exact matured-label TP-before-SL target with sufficient out-of-sample events and uncertainty intervals; then perform authorized paper trading and separate security/risk review.

Do not add more model complexity, enable live signals, or route orders before these gates. `deployment_calibrated=false`, `validated=false`, `live_signal_enabled=false`, `decision_output=NO_TRADE`.

## Artifact references

- PR: https://github.com/orionrayy/ai-determinism-engine/pull/164 (remains draft; not merged).
- Benchmark report/artifact: https://github.com/orionrayy/ai-determinism-engine/actions/runs/39024803960
- Latest follow-up CI should be rechecked after this documentation checkpoint updates the PR branch.
