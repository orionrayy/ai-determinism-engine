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
7. The two M5 datasets are compared at exact shared timestamps and with a bounded diagnostic probe of candidate timestamp offsets (±1/±2 bars and several hour-level shifts). The probe reports the best observed offset only for investigation; it never shifts or merges production data automatically. The report includes overlap size, absolute/relative close-price differences, and close-to-close direction agreement. Relative price-level compatibility is flagged DIVERGENT if median delta exceeds 0.25% of the matched price level or P95 delta exceeds 0.50%. This is a conservative data-comparability gate, not a trading-performance target. Source independence remains unverified.
8. CI enforces a minimum aggregate of 100,000 downloaded rows (reported as a sum, not a deduplicated or independent sample count), a minimum exact-timestamp overlap, and fixed rule/scenario counts.
9. Reports include data hashes, UTC observation ranges, interval/gap diagnostics, benchmark metrics, licensing/provenance caveats and a hard-coded NO_TRADE decision boundary.

## Interpretation rules

- PASS means the acquisition, structural validation, benchmark execution and scenario contracts met their software requirements. It does not mean a strategy is profitable.
- A positive metric from one source must not be advertised as an independent holdout result when source lineage, overlap, timestamp semantics or quote side are unresolved.
- High directional agreement does not make datasets interchangeable when their price levels diverge; `DIVERGENT` disables cross-source confirmation and blocks production readiness.
- A non-zero best timestamp offset is only a diagnostic clue, not permission to align sources or use shifted bars for backtests.
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
