# Multi-source real-dataset benchmark — implementation candidate

## Purpose

The existing online benchmark uses one community M5 CSV. This candidate adds a separate read-only runner that audits three public community OHLCV datasets, runs each dataset independently, measures exact-timestamp agreement between the two M5 series, and stores only hashes and derived summaries in the CI artifact.

It is a data-coverage/measurement increment, not an assertion that the underlying data is independent, licensed for every commercial use, broker-verified, or suitable for live execution.

## Sources in the fixed allowlist

| ID | Repository/file | Timeframe | Approximate size noted by source repository | Provenance/semantics caveat |
|---|---|---:|---:|---|
| getdata_xauusd_m5 | [GetData Finance XAUUSD M5 CSV](https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data) | M5 | 35k+ rows in the previous hosted checkpoint | Community OHLCV; broker provenance, quote side, intended timezone and data-use rights remain unverified |
| sai_community_xauusd_m5 | [Sai310421 XAUUSD M5 CSV](https://github.com/Sai310421/xauusd-data/tree/main/csv/XAUUSD) | M5 | About 17k rows per repository README | Repository README describes community mid-quote data and claims UTC; raw datetime values are timezone-naive; upstream broker data lineage remains unverified |
| sai_community_xauusd_m1 | [Sai310421 XAUUSD M1 CSV](https://github.com/Sai310421/xauusd-data/tree/main/csv/XAUUSD) | M1 | About 86k rows per repository README | Same publisher as the second source; it provides higher-frequency bars, not an independent publisher |

These are sources published online by community repositories, not three independently verified market feeds. The repo-level MIT licenses are not treated as proof that the original price-data rights or intended commercial use are unrestricted. The runner downloads the current files within strict byte limits and does not commit or upload the raw datasets.

## Benchmark protocol

1. Each exact URL is pinned in code and constrained to its source repository host/path; retry count, timeout and byte ceilings are bounded.
2. Each CSV is parsed using the established OHLCV validator. Rows must contain finite, valid OHLCV values and strictly increasing timestamps.
3. Explicit offsets are normalized to UTC. A naive timestamp is accepted only for sources whose catalog policy explicitly says UTC, and this interpretation is reported as an unverified assumption.
4. M1 and M5 datasets are evaluated separately. No data is concatenated, resampled or filled to create a larger synthetic history.
5. Each source receives 50 chronology-spread decision timestamps, five fixed-rule comparisons per timestamp, full-history replay, five disjoint chronological folds with a final tail segment, and causal prequential probability-baseline diagnostics.
6. Each source anchors 720 synthetic counterfactual market-path contracts and 240 software-input contracts. These test software handling; they do not count as additional observed market events.
7. The two M5 datasets are compared at exact shared timestamps and with a bounded diagnostic probe of candidate timestamp offsets (±1/±2 bars and several hour-level shifts). The report also adds signed B−A close bias, OHLC signed differences, percentage of timestamps above/below, monthly discrepancy buckets, and five deterministic paired-candle samples for source investigation. The best observed offset is reported separately from a material shift candidate. A non-zero offset becomes a candidate only if it improves median relative close-price delta by at least 20%; the probe never shifts or merges production data. Relative price-level compatibility is flagged DIVERGENT if median delta exceeds 0.25% of matched price level or P95 exceeds 0.50%. These are data-comparability thresholds, not trading-performance targets. Source independence remains unverified.
8. CI enforces a minimum aggregate of 100,000 downloaded rows (reported as a sum, not a deduplicated or independent sample count), a minimum exact-timestamp overlap, and fixed rule/scenario counts.
9. Reports include data hashes, UTC observation ranges, interval/gap diagnostics, benchmark metrics, licensing/provenance caveats and a hard-coded NO_TRADE decision boundary.

## Data-use and production eligibility

Public accessibility and a repository-level license badge do not automatically verify rights to the original underlying broker data, the right to train/serve a commercial model, or permission to embed data in a paid service.

- The GetData GitHub sample repository advertises MIT, while GetData's own published documentation says the standard purchased-data license is purchaser-only, allows the buyer's own research/backtesting and derived outputs under restrictions, and does not allow raw CSV/ZIP embedding in commercial products or paid services without separate written permission. The exact relationship between those provider terms and this GitHub evaluation sample has not been resolved. See [repository license](https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data/blob/main/LICENSE), [provider documentation](https://www.getdata.finance/documentation), and [XAUUSD product page](https://getdata.finance/datasets/xauusd).
- Sai's repository advertises MIT and describes 2026 Q1–Q2 bars as aggregated from public broker mid-quote feeds, but the individual upstream publisher/feed lineage, upstream data-use terms, and quote construction have not been independently verified. See [repository README](https://github.com/Sai310421/xauusd-data).
- The inspected file-specific commit history for Sai's M1/M5 Q1–Q2 CSVs shows them present in the repository's initial commit on 2026-06-30, without an observed per-month provenance/checksum trail. This weakens auditability but is not proof of fabricated data.
- A public GitHub mirror of Dukascopy XAUUSD M1 BID/ASK files was identified as a possible diagnostic candidate, but the mirror does not expose a clear repository license. It was not added to automated CI or production and its data was not redistributed.
- As a result, CI uses these sources only for diagnostic benchmark execution and does not upload the raw CSVs. This is a technical handling choice, not a legal conclusion or approval for commercial use.
- Current machine-readable gates intentionally hold model_training_for_commercial_service, commercial_product_use, production_feed_approved, cross-source confirmation, and production readiness at BLOCKED until rights scope and provenance are verified.
- For the actual production system, prefer a broker/provider relationship whose contract explicitly permits the intended internal model use, paid service, storage, derived output and required redistribution (if any), with documented source identity and quote semantics. A public BID/ASK mirror with no clear license is not a substitute for that approval.

## Interpretation rules

- PASS means the acquisition, structural validation, benchmark execution and scenario contracts met their software requirements. It does not mean a strategy is profitable.
- A positive metric from one source must not be advertised as an independent holdout result when source lineage, overlap, timestamp semantics or quote side are unresolved.
- High directional agreement does not make datasets interchangeable when their price levels diverge; `DIVERGENT` disables cross-source confirmation and blocks production readiness.
- A non-zero best timestamp offset is only a diagnostic clue, not permission to align sources or use shifted bars. A best-observed offset that improves the median relative price delta by less than 20% is reported without promoting it to a material shift candidate.
- Existing cost assumptions are not measured broker Bid/Ask, tick, commission, swap, slippage or fill data.
- Current direction baselines are not the same target as TP-before-SL probability. Deployment calibration remains false.
- NO_TRADE, validated=false, and live execution disabled remain mandatory until the research gates in ROADMAP.md are met.

## Acceptance conditions

- Unit tests pass for source allowlisting, timezone normalization, gap reporting, overlap computation, mismatch fail-closed behavior and the decision gate.
- Hosted CI retrieves the online files and checks actual source hashes/counts and all benchmark contracts.
- Aggregate rows are at least 100,000 without pooling sources; at least 1,000 M5 timestamps overlap.
- The complete report distinguishes real source data from synthetic scenarios and records independent-source/provenance status as unverified.
- The run is green before this candidate can be considered integrated. The hosted report and exact hashes must be read back before recording an observed success.

## Still blocked after this increment

Licensed broker-specific Bid/Ask/tick history; independently verified source lineage and intended-use rights; measured execution costs/fills; additional independent market regimes; exact TP-before-SL calibration with adequate mature labels; paper trading; security review; and live Telegram/OANDA validation. These require separate evidence and are not solved by multi-source CI alone.


## TP-before-SL target-specific OOS diagnostic — 2026-10-17

A modular diagnostic candidate now evaluates the actual research target separately from the existing next-candle direction baseline. The new module is `online-benchmark/barrier_oos_benchmark.py`; the multi-source runner calls it after each CSV is validated and timezone-normalized. This increment has not yet passed its fresh hosted CI run at the time of this note.

- Target: whether TP is touched before SL within 12 forward bars, for long and short separately. ATR(14) is computed only through the decision close; entry is the next bar open; TP = 1.5×ATR and SL = 1.0×ATR.
- Feature contract: return over 1/3/12 bars, EMA5−EMA10, candle body, candle range and 5/20 range ratio; features and scaling use only bars available by the decision close. The first 60% is training, the middle 20% is OOS calibration, and the final 20% is the untouched chronological holdout. Samples with barrier outcomes not matured by the next split boundary are purged.
- A simple stdlib-only L2 logistic model is an explicitly fixed-protocol research baseline, not a claim of alpha. Platt calibration uses only OOS predictions/outcomes in the middle period whose labels matured before the final holdout. Holdout reports Brier score, log loss, reliability/ECE/MCE, reference-rate comparisons and moving-block-bootstrap intervals.
- OHLC ambiguity is resolved conservatively: stop-first when both barriers are hit within a bar; open-gap checks occur before intrabar high/low checks; timeout outcomes remain separately counted. Because OHLC does not expose tick sequence, these are research labels, not actual fill outcomes. Spread, commissions, swaps, slippage and liquidity are not measured by this dataset.
- Automated checks cover feature causality, matured-label purging, stop-first collision policy, gap-through-barrier handling, chronological holdout sizes, calibration output shape and permanent NO_TRADE/no-live-execution invariants.
- The benchmark may report DIAGNOSTIC_ONLY, but the report intentionally hard-codes model validation, deployment calibration, production approval and live order execution to false/blocked. Any calibration improvement on this one final holdout is evidence for further research only; it must not be reused for model selection or advertised as world-leading accuracy.

Acceptance after implementation: fresh hosted workflow must parse every current allowlisted dataset; pass all existing source/overlap gates and new unit tests; report matured train/calibration/holdout counts for long and short; preserve all execution/data-rights caveats; and maintain `NO_TRADE`. Any failing invariant blocks promotion and requires correction before further feature expansion.


## Paired calibration-gain uncertainty and verified baseline results — 2026-10-17

Hosted run [#39019576117](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39019576117) passed on commit `8e1dabf70e0dd1df816424717c4d0efe56c90a0d`. Its uploaded artifact was 49,841 bytes, SHA-256 `69ac2c75a670fd033d43d0f242bb585a00996135cbc71e425f549e5e548b0727`; report runtime was 141,795.762 ms. Summarized baseline results are persisted in `benchmarks/tp_before_sl_oos_2026-10-17_baseline.md` and its JSON companion.

The exact TP-before-SL holdout ran on three community OHLCV inputs, for six source/side metric sets and 55,996 reported label predictions across those sets (not independent market observations). Train/calibration/holdout sizes were 21,546/7,190/7,179 (GetData M5, both sides), 10,392/3,471/3,463 (Sai M5 long), 10,391/3,471/3,463 (Sai M5 short), 52,078/17,357/17,356 (Sai M1 long), and 52,079/17,363/17,356 (Sai M1 short). All labels are separate from the earlier next-candle direction target.

Calibration impact was mixed, not a consistent gain: GetData M5 long Brier changed 0.230024→0.229481; GetData M5 short worsened 0.234666→0.234859; Sai M5 long was virtually unchanged 0.222111→0.222110; Sai M5 short worsened 0.233999→0.234180; Sai M1 long changed 0.228472→0.228498; Sai M1 short changed 0.238133→0.238161. Raw Brier skill versus calibration-period event rate was small, and GetData M5 long was negative. These are point estimates only; the previous artifact's marginal 40-replicate intervals were not paired uncertainty intervals for gains.

Follow-on candidate implementation adds a deterministic 250-replicate paired moving-block bootstrap directly on raw-vs-calibrated predictions/outcomes and a holdout-mutation invariance test to prove that modifying only final-period bars cannot alter the fitted training-model parameter hash or calibration parameters. The next hosted run is required before these added assertions and intervals are marked green. The data/rights gate, cross-source comparability block and `NO_TRADE` remain unchanged.
