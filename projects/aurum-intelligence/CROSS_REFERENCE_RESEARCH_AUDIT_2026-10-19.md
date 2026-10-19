# Serevex Quant: Cross-Reference Research Audit and Priority Map
Date: 2026-10-19
Repository: orionrayy/ai-determinism-engine
Research branch: research/evidence-gates-2026-10-19
Last fully validated head before this audit increment: 5183b5d0c574b9e82a3d0404f09de0acfd5ddb36
State: research-only; never authorizes signals or orders

## 1. Executive conclusion

The project's most valuable next step is not to add more indicators, increase model complexity, or optimize a single backtest score. It is to eliminate ambiguity in the measured object and produce a defensible portfolio-level outcome ledger. The current evidence does not establish a positive, executable XAUUSD edge.

The last fully completed CI run (run 241, head 5183b5d0c574b9e82a3d0404f09de0acfd5ddb36) passed the online benchmark workflow, including 195 unit tests, price-source comparison, tick-data contract, 1,313,280 paired M1 bid/ask candles, 20,000 unique anchor timestamps, 120,000 TP/SL path scenarios, 52,824 strategy events, and external-context acquisition/ablation. These are software and research-protocol acceptance results; they do not verify upstream data rights, broker equivalence, profitable fills, or predictive skill.

### High-confidence empirical stop signals

- Cross-source M5 comparison: 11,509 matched timestamps; zero exact OHLC matches; median and P95 absolute close difference 134.885 and 203.661 basis points; one-bar return correlation 0.998450. High return correlation does not resolve the material price-level/quote-basis disagreement.
- Offset diagnostic: the best tested offset lies at the +12-bar search boundary, but the median-discrepancy reduction is only about 2.37%. That is not positive evidence for a correction. A boundary optimum means the searched range did not bracket the candidate minimum; the proper conclusion is indeterminate within this scan, not a proven mismatch and not proof of no mismatch.
- Third-source lineage diagnostic: 4,392 overlap bars against the community M5 source, 4,391 exact OHLC matches (99.9772%) in the Bid-convention comparison, while the repository labels its bars as midpoint. Source independence remains unverified. This strongly warrants a source-owner, transformation, or price-basis audit before the feed is treated as an independent confirmation.
- M1 prequential probability diagnostic: aggregate Brier 0.174719 versus global matured-label baseline 0.174579; Brier skill is -0.000802. The configured paired moving-block interval for baseline-minus-model improvement is approximately [-0.000245, -0.000052]. Under this evaluation, the probability method is slightly worse than the causal base-rate benchmark.
- External-context ablation: COT-augmented Brier is 0.186169 versus price-only 0.186287, a point improvement of only 0.000119. Its paired moving-block 95% interval is [-0.000381, +0.000675], crossing zero. The interval is inconclusive; no incremental value is established.
- The ablation named price_combined uses the same feature keys and Brier as price_cot. Its effective columns are price features plus two COT fields; macro and news context are absent. The current report records requested context families for that model, not only the families that actually contributed. It must not be represented as a demonstrated price+COT+macro+news model.
- GDELT: 141,020 unique observations passed the scale-count gate, but the acquired archive covered only about 0.114218% of the requested historical time window. News feature eligibility was 0/19,894 labels. Observation count and temporal coverage are different acceptance criteria.
- Macro feature eligibility was 0/19,894 labels. COT was eligible on 19,688/19,894 rows (about 98.9645%), but its incremental Brier improvement is inconclusive.
- M1 spread/stop geometry: 217,544 of 875,611 otherwise ATR-valid contiguous windows (24.8448%) were infeasible under the rule that entry spread must be less than stop distance. Median spread/ATR was 0.689, P95 2.026, and P99 4.287. This is a feasibility constraint, not just a cost scalar.
- The four simple M1 rule baselines had negative final-holdout gross mean R and profit factors far below 1, even before broker-observed commission, swap, slippage, latency, rejected/partial fills, or account-currency conversion. A separate M5 TP-before-SL label protocol is not interchangeable with the M1 bid/ask economic replay.
- No portfolio ledger exists yet to aggregate mutually competing/overlapping entry opportunities into capital, exposure, cash, margin, and realized equity over a regular decision clock. Sampled path outcomes are not a portfolio return series and must not be fed to the White Reality Check guard.

Conclusion: keep NO_TRADE. Fix information-set and lineage semantics, source qualification, economic realism, and portfolio accounting before evaluating a more complex model.

## 2. Evidence classes: do not collapse different meanings of PASS

Use these separate state dimensions in every summary:

1. Software execution: did a test/process complete successfully?
2. Data acquisition: were bytes downloaded and parsed?
3. Source completeness: does the archive cover the intended time/instrument/quote universe?
4. Source qualification: are provider provenance, rights, timezone, timestamp labels, quote basis, and revisions evidenced?
5. Point-in-time eligibility: could each exact feature version have been known at decision time?
6. Label validity: are targets causal, mature, and correctly handling ambiguity/overlap?
7. Statistical evidence: does a pre-registered out-of-sample comparison support improvement versus a relevant baseline?
8. Economic validity: does the portfolio ledger include actual or evidenced spread, fee, swap, slippage, fills, sizing, margin and currency conversion?
9. Operational fitness: can the deployed runtime reproduce the offline contract, monitor faults, and fail closed?
10. Authority state: is live use separately reviewed and explicitly authorized?

A PASS at level 1, 2, or 3 never promotes levels 4–10. A source can download successfully while remaining incomplete; a model can score without being calibrated; a calibrated probability can still have negative expected net PnL.

## 3. Local artefact inventory and cross-reference matrix

| Topic | Local/repository evidence | External standard/reference | Gap or conflict | Required disposition |
|---|---|---|---|---|
| Historical macro revisions | ExternalObservation has event_time, published_at, available_at, retrieved_at and availability quality; unknown availability is excluded. | FRED/ALFRED distinguishes today's revised view of history from information available on a historical date. https://fred.stlouisfed.org/docs/api/fred/fred_vs_alfred.html and https://fred.stlouisfed.org/docs/api/fred/realtime_period.html | A timestamp or Boolean flag is not proof that every adapter has the true release/version time. | Store actual release/vintage and first availability evidence per record; keep revision history immutable. |
| Feature-store point-in-time correctness | Temporal semantic contract and lineage graph exist as separately tested research gates; not wired into Telegram decision runtime. | Feast warns that event-time-only joins can still leak values backfilled/corrected after the entity timestamp unless created time is filtered. https://docs.feast.dev/getting-started/concepts/point-in-time-joins | Local validators inspect declared parents; hidden dependencies in code cannot be discovered from metadata alone. | Bind feature transforms to hashed dependency manifests and enforce the gate in the offline feature builder before any future runtime integration. |
| COT availability | The external-context pipeline reports COT eligibility on 98.9645% of matured labels; adapter uses a conservative availability proxy in the absence of verified release evidence. | CFTC says COT reports are generally published Fridays at 3:30 p.m. Eastern using the preceding Tuesday's positions; holidays and interruptions can alter schedules. https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm and https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm | A generic weekday proxy may be conservative, but it is not the actual release-time history and may discard valid information or misplace availability around exceptional releases. | Build a versioned release calendar with official actual dates, timezones, holiday exceptions, and source hashes. Unknown actual publication remains unknown/proxy, never silently promoted. |
| News archive acquisition | 141,020 unique GDELT records; only 0.114218% time coverage of target historical window; 0 news-feature-eligible scored rows. | GDELT documentation says GDELT 2.0 updates event/GKG feeds every 15 minutes; update cadence is not a guarantee that a particular downloaded archive covers every requested historical window. https://gdeltproject.org/data.html | Count-based scale PASS was achieved even though temporal completeness is PARTIAL and news features are NOT_MET. | Require interval-union coverage to meet a declared threshold per feature window (1h/24h) before counting an absence as zero news. Keep count gate and coverage gate separate. |
| XAUUSD M5 source identity | Primary community source has 35,954 bars; reported timestamps are not independently verified; 136 intervals exceed 600 seconds. | White/data-snooping literature cannot repair source uncertainty; source correctness must be qualified separately from statistical tests. https://doi.org/10.1111/1468-0262.00152 | The history covers only approximately six months and the timezone semantics and rights remain unverified. | Do not use as an execution-price truth. Pin immutable source hashes, rights evidence, UTC/candle boundary semantics, and missing-session calendar. |
| Second M5 source | 11,509 overlap; zero exact OHLC matches; median/P95 close difference 134.885/203.661 bps; return correlation 0.998450. | Source comparisons require exact instrument, price basis, timestamp convention and rollover alignment; a high correlation is not equivalence. | Price level/basis disagreement is material; scan's best offset is at +12 bars but reduces median discrepancy by only 2.37%. | Maintain separate source IDs and pricing bases. Boundary scan must be labelled indeterminate; never auto-shift/rescale or merge sources. |
| Third tick source and lineage | 6,685,266 ticks from a CC-BY-labelled dataset partition; 4,392 M5 overlap bars; 4,391 exact OHLC matches against a repository claiming midpoint data; source independence false. | A dataset's repository license and source description do not authenticate upstream broker origin or whether transformation/redistribution rights apply to upstream observations. | Near-exact Bid identity with nominal midpoint bars suggests shared lineage or a price-basis label problem; neither explanation is yet established. | Escalate provenance and quote-basis audit before counting this as independent replication. A third data file is not automatically a third independent source. |
| OHLC/quote semantics | M1 replay uses separate Bid/Ask OHLC and crosschecks 40,000 path labels with zero mismatches to a reference evaluator. | Quote-aware execution requires mapping buy entries to Ask, exits to Bid (and vice versa for shorts) and separately accounting for fees/slippage. QuantConnect's reality-modeling documentation explicitly separates brokerage, fees, fills, slippage and portfolio models. https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/key-concepts | Label agreement against another evaluator validates implementation consistency, not truth of the feed or broker-equivalent fills. | Retain bid/ask-side fills and add evidence-backed latency, partial-fill, slippage, commission, swap and currency conversion models. |
| Entry geometry | 24.8448% of ATR-valid windows rejected because observed entry spread >= 1 ATR stop. | Quote-side cost is state-dependent; a constant cost in R cannot represent spread-to-stop feasibility, volatility regimes, or entry timing. | Geometry can invalidate a setup before it is scored. This is not adequately modelled by adding one flat cost later. | Preserve the pre-entry feasibility gate and report rejection fractions by hour, volatility, session, side and spread/ATR bucket. |
| Probability forecasts | Prequential M1 Brier 0.174719 vs global matured-label baseline 0.174579; skill -0.000802 and block interval for improvement is fully below zero. | Proper scoring rules assess probabilistic forecasts, not trading profitability. A model should be compared against meaningful causal baselines. Gneiting & Raftery: https://doi.org/10.1198/016214506000001437 | Current method does not beat the simple causal benchmark on this holdout. | Do not promote or relabel it as an alpha model. Investigate target/base-rate calibration, then require frozen OOS and independent replication. |
| External context value | price_cot Brier 0.186169; price_only 0.186287. Paired price-only-minus-COT improvement 0.000119 with 95% interval [-0.000381, +0.000675]. | Same-row paired evaluation is the right comparison for incremental context value, but uncertainty must be carried through to the claim. | Interval crosses zero. price_combined and price_cot have the same scores and effective columns; macro/news are absent. | Report requested versus effective families, feature counts, actual coverage and paired incremental interval. No combined-context claim until macro/news features exist and are eligible. |
| Strategy selection | White Reality Check helper exists for regular, synchronized portfolio-return panels, with trial registry and evidence gates, but is not connected to sampled event anchors. | White (2000) addresses data snooping; DSR corrects selection bias and non-normality; PBO/CSCV estimates selection overfit. https://doi.org/10.1111/1468-0262.00152 https://doi.org/10.3905/jpm.2014.40.5.094 https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 | Sampled overlapping paths are not a regular portfolio-level return panel. Applying WRC directly to them would be a category error. | Build the portfolio ledger first. Register every model/feature/parameter attempt; then run WRC or SPA with sensitivity analysis, DSR/PBO where assumptions and trial history permit. |
| Portfolio outcomes | No capital/exposure-aware ledger, no broker-observed cost, no account-currency net PnL in the replay; four M1 rule baselines have negative gross holdout mean R and PF well below 1. | Execution research systems separate portfolio constraints, fills, fees, slippage and live-vs-backtest reconciliation. https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/key-concepts and https://www.quantconnect.com/docs/v2/writing-algorithms/live-trading/reconciliation | Event-level R and target-hit rates do not account for overlapping positions, rejected signals, equity, margin, leverage or broker-specific execution. | The next milestone must produce one regular-clock portfolio return vector, including zero-return periods without exposure and net costs. |

## 4. Highest-priority issue map

### P0 — Source and price semantics (blocks further inference)
1. Freeze the two M5 sources as separate datasets with SHA-256, retrieval timestamp, declared provider, licensing evidence, timezone, candle start/close convention, instrument alias, price basis, rollover and session calendar.
2. Keep exact-time source concordance as the primary result. Offset scans are exploratory and must not modify source timestamps.
3. The +12-bar boundary candidate must be classified INDETERMINATE_SCAN_BOUNDARY. A boundary optimum means the scan does not cover both sides of a possible minimum; it does not justify an automatic offset.
4. Ask whether the third tick source and midpoint repository share upstream history or whether the quote-basis tag is wrong. Do not call the source independent until independently evidenced.

Acceptance: provenance/rights and quote basis have stable evidence; at least two truly independent sources have compatible candle semantics; discrepancy metrics fall within thresholds declared before inspection, or source disagreement remains an explicit hard block.

### P0 — Portfolio ledger and economic accounting
Create an immutable ledger at every regular decision interval. Each row must contain timestamp, signal/state, desired quantity, accepted/rejected reason, bid/ask snapshot ID, order intent, simulated/observed fill, fee, slippage, swap, conversion, equity, margin, exposure and resulting net account-currency return. No position means zero return for that interval, not a missing row.

Do not use 120,000 simulated TP/SL paths as if they were 120,000 trades. Many outcomes share overlapping price windows and compete for the same capital. Portfolio exposure and entry scheduling decide which trades actually coexist.

Acceptance: re-run is deterministic; every cost field is observed or clearly tagged as assumed; missing material costs block NET_PNL; each portfolio-return interval can be traced to decisions and execution events; the ledger's digest and manifest are persisted.

### P0 — No present evidence of alpha
1. Keep NO_TRADE. Current prequential Brier is worse than the global matured-label baseline.
2. Do not select the COT-augmented model from a point estimate whose paired interval crosses zero.
3. Do not describe macro/news context as active in the combined model when those families have no eligible development features.
4. Stop treating threshold hits as expected net return; evaluate net returns with portfolio constraints.

Acceptance: an economically meaningful, predeclared primary metric beats a strong causal baseline on a locked OOS cohort, the uncertainty interval excludes zero after multiplicity adjustment, cost stress cases remain viable, and an independent forward period replicates the finding.

### P1 — Archive coverage and point-in-time context
- Convert news coverage into a time-union audit at the exact feature windows. A row can mean zero events only if the whole window was observed and eligible.
- Store provider release/available timestamps and historical revisions for macro/COT; keep publication timestamp separate from observation/reference period.
- Expose family coverage per split and by regime, not only total unique event count.

Acceptance: target-window completeness and feature coverage thresholds pass independently of the count threshold. The report lists effective, omitted and unavailable feature families explicitly.

### P1 — Statistical selection, labels and validation
- Freeze an experiment registry before using a holdout. Record model family, features, thresholds, transformations, cost profile, label definition, trial timestamp, data hashes and rejection reason.
- Keep development, calibration and final scoring distinct. Purge training labels that overlap the test boundary; enforce an embargo if residual dependence requires it.
- Use causal baselines, proper scoring rules for probabilities, and portfolio metrics for trading economics. Do not substitute one metric for another.
- Apply WRC/SPA to a complete regular portfolio return panel and the full trial family. Add DSR/PBO as additional diagnostics only after defining their assumptions and registered trial universe.
- Predeclare moving-block lengths and show sensitivity. Two hundred fifty bootstrap replicates are useful engineering diagnostics, not a universally sufficient basis for final inference.

Acceptance: no test/holdout fitting or feature selection; complete registry of attempts; purged boundary tests; corrected inference and clear limitations; final holdout is never recycled into another round of design.

### P2 — Operational reliability and reproducibility
- Generate a manifest with source URLs, retrieval times, bytes, SHA-256 digests, code commit, environment, parameters, labels, output hashes and artifact policy.
- Separate software PASS, data PASS, inferential PASS, economic PASS and operational PASS in summaries.
- Once the offline evidence passes, shadow/paper-run on the chosen broker's actual quotes and fills. Reconcile backtest vs forward execution; investigate every material deviation before considering deployment.
- Do not merge or deploy automatically simply because CI is green.

## 5. The “#1” objective: turn an unmeasurable ambition into acceptance criteria

There is no verifiable universal “#1 accuracy” target for XAUUSD signals. Accuracy can be increased by changing the target base rate, suppressing coverage, or overfitting a historical period. For this project, the aggressive-but-defensible objective should be:

- Source truth: no unresolved material timestamp/quote-basis/source-lineage disagreement.
- Prediction: pre-registered OOS improvement over a strong base rate and price-only baseline, evaluated on identical rows/labels, with uncertainty and multiple-testing correction.
- Economic outcome: positive net portfolio expectancy after observed or stress-tested costs, with drawdown/tail-risk and capacity constraints.
- Stability: improvement persists across time folds, volatility/session regimes, alternative data sources and a separate forward period.
- Operations: exact reproducibility, monitoring, fail-closed behavior, and paper execution reconciliation.
- Honesty: missing context, missing costs, incomplete archive coverage and unverified data rights remain visible blockers rather than synthetic zeros or successful status labels.

“More aggressive” should mean faster falsification, more precise registered comparisons, higher-quality data and better diagnostics—not looser gates or more untracked trials.

## 6. Recommended next implementation increment

1. Add an explicit alignment assessment enum for source timestamp offset scans: insufficient overlap, indeterminate boundary, candidate mismatch, or no strong offset evidence within the configured scan. Keep the original boolean for compatibility but never interpret false as proof of no mismatch.
2. In every ablation report separate requested_context_families from effective_context_families, omitted_context_families, effective feature keys and row-level coverage. This makes price_combined visibly equal to price_cot when macro/news columns are absent.
3. Add machine-checked invariants to the report: if a family has zero eligible development values, it cannot appear in effective_context_families; if all context families are absent, the combined ablation is NOT_ELIGIBLE; if effective feature keys and predictions equal price_cot, the report may show equivalence but must not claim a three-family experiment.
4. Include the paired price-only incremental interval in the human-readable report, not only the raw JSON. Explicitly state whether the interval crosses zero.
5. Keep all these changes research-only. Do not wire into Telegram or enable trades as a side effect of evidence-report code.

## 7. Primary references and their correct interpretation

- White (2000), “A Reality Check for Data Snooping,” Econometrica: https://doi.org/10.1111/1468-0262.00152. Supports accounting for re-use of the same history across a universe of trials; it does not turn event rows into portfolio returns.
- Sullivan, Timmermann & White (1999), “Data-Snooping, Technical Trading Rule Performance, and the Bootstrap,” Journal of Finance: https://doi.org/10.1111/0022-1082.00163. Supports evaluating the full rule universe rather than reporting only a chosen winner.
- Bailey & López de Prado (2014), “The Deflated Sharpe Ratio”: https://doi.org/10.3905/jpm.2014.40.5.094. Addresses selection bias and non-normality; the trial count/universe must be meaningful and documented.
- Bailey et al. (2015), “The Probability of Backtest Overfitting”: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253. Proposes CSCV/PBO for evaluating backtest-selection overfit; it is a supplement, not a replacement for forward chronological evaluation.
- Feast, point-in-time joins: https://docs.feast.dev/getting-started/concepts/point-in-time-joins. Explains why created/available time matters in addition to event time.
- FRED/ALFRED real-time periods: https://fred.stlouisfed.org/docs/api/fred/realtime_period.html and https://fred.stlouisfed.org/docs/api/fred/fred_vs_alfred.html. Historical values may be revised; today’s vintage is not necessarily yesterday’s information set.
- CFTC COT schedule/index: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm and https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm. COT positions are generally as of Tuesday and reports generally released Friday at 3:30 p.m. Eastern, subject to schedule changes and exceptions.
- GDELT data documentation: https://gdeltproject.org/data.html. GDELT 2.0’s feeds update every 15 minutes; cadence does not mean a bounded archive download is historically complete.
- QuantConnect reality models and reconciliation: https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/key-concepts and https://www.quantconnect.com/docs/v2/writing-algorithms/live-trading/reconciliation. Useful engineering comparison for separating fill, fee, slippage, portfolio, and live-vs-backtest discrepancies; not a claim that those default models are correct for this broker or gold product.

## 8. Verified pointers

- Latest completed full CI at the time of this research snapshot: run 241, head 5183b5d0c574b9e82a3d0404f09de0acfd5ddb36: https://github.com/orionrayy/ai-determinism-engine/actions/runs/39202072036
- Its immutable CI artifacts (run-scoped, retention-limited): https://github.com/orionrayy/ai-determinism-engine/actions/runs/39202072036/artifacts/12163414809
- Draft PR: https://github.com/orionrayy/ai-determinism-engine/pull/167
- Raw M5 public repository in the benchmark: https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data
- Second M5 repository: https://github.com/Sai310421/xauusd-data
- CC-BY tick dataset repository: https://huggingface.co/datasets/CarlosSilva1/xauusd-ticks
- M1 Bid/Ask data repository with claimed Dukascopy source: https://github.com/kevingtlin/Market-Data-Lab

This audit describes evidence available in the cited repository commit and CI artifacts. It does not independently authenticate those data providers, license grants, broker fills, or investment performance.
