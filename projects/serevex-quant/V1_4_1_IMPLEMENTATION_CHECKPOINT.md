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
- Latest code/test head passed workflow #39025937658. A later PR head after status/checkpoint documentation also passed workflow #39026245471: https://github.com/orionrayy/ai-determinism-engine/actions/runs/39026245471; artifact #12098043414, SHA-256 `184ff507423f4902dd8d9f86647d0e6ca5fe3fd73513de4d604f1b3823bc16aa`. The latest full pipeline is green; all source/provenance/calibration gates remain open.


## 2026-10-17 — third-source tick Bid/Ask and lineage-audit checkpoint

- Added `online-benchmark/tick_bidask_benchmark_v1_0.py` with bounded data acquisition from `CarlosSilva1/xauusd-ticks` (CC-BY-4.0 attribution recorded), strict bid/ask validation, UTC M5 aggregation for Bid/Mid/Ask, observed spread summaries, and exact-timestamp comparisons with both community OHLCV feeds. Only one month is fetched; the 58,806,458 bytes are not redistributed in the report artifact.
- Added unit tests for spread correctness, crossed/nonpositive quotes, exact M5 overlap, and lineage classification; extended workflow gates to require results for both OHLCV sources, no auto-correction/promotion, and explicit lineage-status output.
- Hosted GitHub Actions [run #39025937658](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39025937658) **passed** on code/test head `c6f5062fab2a919b4d3795304c5cbee83b67618a`; all stages completed, including live dataset benchmark, tick Bid/Ask audit, lineage assertions and artifact upload. Artifact #12098756760, 2,595,322 bytes, SHA-256 `54a3905b3fb3a3909070245da80a9089d2051b029dd9f45bab5332ed260d7aa8`, retention through 2026-10-31.
- May-2026 selected Parquet parts: 4 files, 58,806,458 bytes total, SHA-256 recorded per part in `tick_bidask_benchmark.json`; 6,685,266 ticks; observed timestamps 2026-05-01 00:00:00.086 UTC to 2026-05-22 20:59:59.132 UTC; 4,392 UTC M5 quote bars; zero missing/non-finite/nonpositive/crossed quote rows. Same-millisecond duplicates are counted, not silently removed.
- Spread distribution in this reconstructed feed: median 0.58 price units, P95 0.827, P99 0.96; median/P95 spread as bps of mid 1.2555/1.7884. Not measured cost for any user's broker/account.
- Tick-mid vs GetData primary: 4,380 exact timestamp overlaps, close-difference absolute median/P95 123.229/138.379 bps; median signed primary-minus-tick-mid difference +56.325 price units. No exact OHLC matches.
- Tick Bid vs Sai310421 secondary: 4,392 exact timestamp overlaps and **4,391 exact OHLC matches (99.9772%)**, although the secondary README describes its feed as mid-quote. Classifier returns `POTENTIAL_SHARED_LINEAGE_OR_PRICE_BASIS_MISLABEL_REVIEW_REQUIRED`. This does not establish cause, but the secondary feed and tick feed cannot be counted as independent confirmations until source lineage/quote basis is explained.
- Overall online benchmark status PASS means data retrieval, tests, evaluation and assertions worked. It does not mean the sources agree, that provenance is authoritative, or that any trading model has predictive skill. `NO_TRADE`, `validated=false`, `calibrated_model=false`, and live signal execution remains disabled.
- Remaining P0: verify primary/secondary/tick provider provenance, timestamp and price/quote basis, licensing scope, and whether the secondary series is derived from the tick series; investigate primary-vs-tick median level discrepancy; acquire broker-authorized quotes and measured execution costs; then evaluate on independent regimes and untouched holdout. Do not add model complexity or order routing before these gates.


## Large-scale M1 Bid/Ask benchmark — 2026-10-17

- Latest full CI: [run #39028110673](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39028110673), PASS on head `375aa2e3c225394943f177a5243d5c18ccd04fd9`. Artifact #12099177720 (2,613,340 bytes), SHA-256 `5b5129edebd988ca885e0e06f69b8430a9db01ab807fcdb8e11c6d3b0470891b`.
- Verified dataset: 1,313,280 paired XAUUSD M1 Bid/Ask candles across 30 months, 2023-01 through 2025-06, 131,316,007 bytes fetched. Weekday-active non-flat candles: 870,500, passing the strict >=700,000 gate.
- 20,000 distinct real timestamp anchors, all ATR-valid; 875,611 eligible contiguous windows; 120,000 TP/SL path scenarios; 52,883 directional rule events; 10,577 final-holdout events.
- Baseline comparison: event Brier 0.140102 vs global matured-label baseline Brier 0.140015; base rate 0.168126; skill against that causal base-rate baseline -0.000616. Moving-block 95% interval for Brier improvement [-0.0001737, -0.0000091], so the overall result does **not** show improvement over the fair baseline. The Brier skill vs constant 0.5 is misleadingly high (+0.4396) because 0.5 is a weak comparator for this imbalanced label.
- Final holdout: model Brier 0.163455 vs global baseline 0.163549, skill +0.000576, ECE 0.043313. This small positive difference is not enough for an edge claim, particularly with aggregate comparison negative and source provenance/data rights/execution-cost gates open.
- Code changes add a causal global matured-label baseline, a block-bootstrap interval against that baseline, valid ATR filtering for all anchors, and a strict >=700,000 weekday-active non-flat candle gate. CI assertion contracts passed after a field-name fix.
- Decision remains `NO_TRADE`. This is evidence of a functioning large-scale benchmark and an honest negative aggregate calibration comparison; it is not live trading validation or a #1 accuracy result.
## 2026-10-17 — GDELT coverage-status correction and latest hosted revalidation

- Corrected GDELT download-budget semantics, added archive interval/time-coverage accounting, and wired real acquired archive intervals into the point-in-time feature builder. News counts/tone/themes are null unless the relevant historical window is fully covered by already-available observed archives; gaps in a selected sample cannot be interpreted as zero news.
- Corrected report generation to keep software/scale-gate status, source acquisition status, source-completeness status and feature temporal coverage as independent fields. A successful sample acquisition is not represented as complete historical coverage.
- Hosted CI [#39035356832](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39035356832) failed after tests and price gates because of a report-scope `NameError`; the failure and artifact remain preserved. Corrected the out-of-scope variable reference and the contract test, then re-ran.
- Latest end-to-end CI [#39035732248](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39035732248): **success**, tested head `f3eaa644a1f73c0d4f9521870e1ba5c460cfaf69`, 115 unit/regression tests and 240/240 temporal-calibration contracts. Artifact #12102745058, 2,624,921 bytes, SHA-256 `b1710712519fd48e3a5f29acf1d532707997821bba1948e49b093004fd7b81f0`.
- Real external-context acquisition: 141,020 unique GDELT observations, 100 sampled archives processed out of 85,809 known archive entries, 701,820,752 bytes. Archive sample is fully processed, but its intervals cover only 1,500 of 1,313,279 minutes (0.114218%). Final report correctly says source acquisition `PASS`, full-window source completeness `PARTIAL`, and news feature eligibility `NOT_MET` (0/19,937 rows).
- COT is partially eligible on 19,739/19,937 rows (99.0069%); BLS observations remain descriptive because publication times are not provided; FRED/ALFRED is skipped without the optional API key. `price_only`, `price_cot`, and `price_combined` are scored on 3,987 identical holdout rows. Price+COT Brier 0.1609044 vs price-only 0.1609893; incremental difference +0.00008494 with 95% interval [-0.00014777,+0.00032012] crossing zero, so no robust added value is established. News/macro ablations are ineligible.
- P0 remains open: original data rights/provider lineage and the primary-vs-tick price discrepancy are unresolved; no measured broker execution-cost series; no valid macro publication/vintage feature cohort; no continuous news-window cohort; aggregate and independent-regime prediction edge not established. Keep `NO_TRADE`; do not merge the draft PR or enable live execution.


## 2026-10-17 — evidence-backed source qualification gate

- Implemented `online-benchmark/source_qualification_gate_v1.py`. Research evidence is split into independent provenance, intended-use rights, timestamp semantics, bar convention, and quote basis. Execution readiness additionally requires broker authorization, measured costs, and verified fill semantics.
- Every declared `True` evidence claim requires a valid, stable reference. Truthy strings/numbers and malformed references fail closed. Complete evidence allows review eligibility only; `source_promotion_allowed=false` and `live_execution_authorized=false` remain invariant.
- Wired the gate into the primary/secondary community OHLCV benchmark, the tick Bid/Ask audit, and the hosted CI assertion layer. Current reports intentionally mark all three sources `REFERENCE_ONLY` and `BLOCKED`; no independent research evidence or execution-evidence references are supplied.
- Regression tests: hosted unit suite **125 tests passed**; temporal calibration contract matrix **240/240 passed**. Full hosted workflow [#39038107206](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39038107206) **success** at head `d2621fcda08456984ec3adef9015f91d3d432ddd`. This included online public-data acquisition, tick audit, 1.3M+ M1 candles/20k anchor gate, external-context benchmark and artifact upload.
- Artifact #12102563009: `aurum-online-dataset-benchmark-39038107206.zip`, 2,626,545 bytes, SHA-256 `993c27155e00c7d57d87aba5c475c1751d7026fcbfce4345b784996215bc065c`; locally downloaded file hash matches GitHub's published digest.
- Community M5 cross-check: 35,954 primary rows, 17,371 secondary rows, 11,509 exact timestamp overlaps, zero exact OHLC matches, absolute close discrepancy median/P95 134.885/203.661 bps. Tick source has 4,391/4,392 exact secondary-Bid OHLC matches despite the community source's stated mid-quote basis. Keep the lineage/price-basis anomaly unresolved pending owner evidence.
- External-context status: 141,020 unique GDELT observations from 100 sampled archives cover only 1,500/1,313,279 requested minutes (0.114218%); news remains ineligible. COT is eligible for 19,739/19,937 labels; macro/news ablations remain ineligible. COT incremental Brier improvement vs price-only 0.00008494; paired 95% interval [-0.00014777,+0.00032012] crosses zero.
- P0 remains open. Keep research/UX prototype; `NO_TRADE`; `deployment_calibrated=false`; `validated_model=false`; `validated_for_profitability=false`; `live_signal_enabled=false`. PR #164 remains draft and unmerged.
