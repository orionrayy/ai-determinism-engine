# SOL Sentinel — Economic Viability Audit

**Audit date:** 2026-10-30 17:50 WIB (10:50 UTC)  
**Repository base:** `f61038dc53826dc1d23d3f263bb2beaca00fcaf2` (`main`)  
**Research source:** SOL Sentinel Research run [#41037517595](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41037517595), event `workflow_dispatch`, source commit `5dcfccf0138620328a9850c892c36327d7715d5d`  
**Research artifact:** [#12967394940](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41037517595/artifacts/12967394940), ZIP SHA-256 `0f668a4a1380f5c13364d2a99061b155985966d156de05c201e87905d16944cb`  
**Scope:** read-only source, CI, workflow, state-branch metadata, dataset and research-artifact audit. No live order, account access, credential change, manual paper dispatch, or production mutation was performed.

## 1. Executive decision

**Primary decision: SIMPLIFY.** Keep the existing data capture, offline evaluator, risk controls and paper ledger. Freeze feature expansion and do not promote any candidate strategy. Use the current pipeline only for bounded research and verified read-only observations until a strategy passes an independent cost-adjusted OOS gate.

The project has substantial engineering capability, but no proven positive economic edge. The locked baseline generated zero candidates from 43,199 evaluations. Five alternative strategy families generated trades in the six-day OOS segment, but all five had zero winning trades and negative net P&L under every tested Pro-taker cost scenario. No candidate passed the project's promotion gate. The latest forward-paper state remains flat and no genuine scheduled Forward Paper run was found.

This is not a basis for live trading. It is also not yet a reason to delete the repository: the research/data/ledger components retain engineering value and can continue under a strict, low-cost evidence budget.

## 2. Repository and operations state

- Audited `main` at `f61038dc53826dc1d23d3f263bb2beaca00fcaf2`. PRs #190–#194 are integrated; PR #191 introduced the native multi-strategy research implementation and PR #194 recorded the v1.9 continuity checkpoint.
- Latest main SOL Sentinel Tests run [#41038862799](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41038862799) succeeded. The unit-test job reports **242 tests passed, 0 failed**. Its explicit active-state read-only audit job was **skipped**, so this run is not a fresh database audit.
- Research run #41037517595 completed successfully and uploaded artifact #12967394940. It was a manual `workflow_dispatch` research run, not a scheduled Forward Paper run.
- The Forward Paper workflow is on `main`, uses twelve five-minute UTC slots per hour, a bounded one-cycle execution, serialized state persistence and a SQLite integrity gate. The scheduler workflow and independent read-only canary are registered as active in the repository checkpoint.
- Repository-wide schedule history contained 153 scheduled events; none belonged to SOL Sentinel. They were AI Orchestrator runs. The Sentinel workflow therefore remains `FIRST_CRON_PENDING`; the exact platform cause is unresolved. Do not count a manual dispatch or a different workflow as proof of this scheduler.
- The dedicated `sol-sentinel-paper-state` branch is still at commit `c3f259242d450d64b4baeb01ae083c178fd4d8b0`, dated 2026-10-30T05:10:23Z. It represents the earlier flat-state checkpoint: Rp300,000 cash, 0 SOL, 0 orders, 0 fills and Rp0 realized P&L/fees. The last referenced read-only SQL audit found SQLite integrity `ok` and zero foreign-key violations; this audit did not reopen the database, so it does not claim a new SQL integrity pass.
- Physical Android/Termux validation remains `NOT_TESTED`. Ubuntu CI's Termux Doctor output is not handset evidence.

## 3. Data provenance and research methodology

The run captured **43,200 native INDODAX SOL/IDR one-minute candles** from 2026-09-30T10:01:00Z through 2026-10-30T10:00:59.999Z. No FX conversion was applied. Dataset SHA-256:

```
95e95e5600df8a2a21f7de34393321cf98226ace42d485d1da12c480b667656d
```

Metadata reports zero duplicate timestamps, zero missing intervals, zero invalid OHLC rows and zero negative-volume rows. It also reports 13,789 zero-volume rows (31.92% of the dataset). Those rows may represent genuine no-trade minutes; their semantics and effect on volume-based filters remain a validation item, not a confirmed data defect.

The research split is chronological and disjoint: development 25,920 bars (60%), validation 8,640 (20%), OOS 8,640 (20%, six days). The report declares prior-observation-only indicator warmup, parameters selected before the final OOS segment, no future-data features and no randomized parameter search. This is better controlled than an in-sample-only result, but one 30-day window and one six-day OOS period do not represent durable performance across market regimes.

Five declared families were assessed: `range_mean_reversion`, `regime_adaptive`, `trend_breakout`, `trend_pullback` and `volatility_expansion`. A locked baseline v1 was also replayed unchanged.

## 4. Quantitative results

### 4.1 Locked baseline

| Measure | Result |
|---|---:|
| Signal evaluations | 43,199 |
| Signal candidates | 0 |
| Order attempts / fills / trades | 0 / 0 / 0 |
| Net P&L | Rp0 (no position opened) |
| Expectancy / win rate / profit factor | Undefined |
| Readiness | `INSUFFICIENT_SAMPLE` |

The largest overlapping rejection counters were 1m breakout 40,499, relative-volume confirmation 35,890, 5m momentum 28,311, 15m trend 26,101, 5m RSI 23,381, 1m momentum exhaustion/missing 3,187 and net-edge gate 139. Counters overlap and are not unique-bar totals. The baseline's zero P&L is not a zero-risk profitable result; it is a no-trade result.

### 4.2 Alternative families — Pro taker base cost

The OOS segment begins with Rp300,000 in the simulator. All five alternatives lost on every OOS trade.

| Strategy | OOS trades | Wins | Win rate | Profit factor | Net P&L | Net return | Net expectancy / trade | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Range mean reversion | 23 | 0 | 0% | 0.00 | -Rp30,070.00 | -10.02% | -Rp1,307.39 | 10.02% |
| Regime adaptive | 22 | 0 | 0% | 0.00 | -Rp30,238.74 | -10.08% | -Rp1,374.49 | 10.08% |
| Trend breakout | 18 | 0 | 0% | 0.00 | -Rp24,539.38 | -8.18% | -Rp1,363.30 | 8.18% |
| Trend pullback | 20 | 0 | 0% | 0.00 | -Rp27,672.27 | -9.22% | -Rp1,383.61 | 9.38% |
| Volatility expansion | 16 | 0 | 0% | 0.00 | -Rp23,274.39 | -7.76% | -Rp1,454.65 | 7.76% |

For the base scenario, the measured simulator decomposition illustrates the cost problem. Range mean reversion produced only Rp1,243.61 gross P&L against Rp18,040.15 modeled fees and Rp13,273.46 spread/slippage friction, resulting in -Rp30,070 net. Volatility expansion produced -Rp2,007 gross P&L, Rp12,251.14 modeled fees and Rp9,016.25 friction, resulting in -Rp23,274.39 net.

Across the other three tested cost scenarios, every family remained negative with zero winning OOS trades. OOS losses ranged approximately from 7.53% to 10.30% of the segment's initial equity. No candidate was promoted.

### 4.3 Cost assumptions

| Scenario | Buy + sell taker fee | Round-trip spread input | Slippage assumption | Total modeled round-trip cost |
|---|---:|---:|---:|---:|
| Pro taker snapshot | 62.11 bps | 15.4835 bps | 5 bps/side | 87.5935 bps |
| Pro taker base | 62.11 bps | 25.6669 bps | 10 bps/side | 107.7769 bps |
| Pro taker adverse | 62.11 bps | 40 bps | 15 bps/side | 132.11 bps |
| Pro taker severe | 62.11 bps | 60 bps | 25 bps/side | 172.11 bps |

The modeled Pro taker fee components are 20.555 bps on buy and 41.555 bps on sell, totaling 62.11 bps. The base case costs Rp1,077.77 per Rp100,000 of round-trip notional. This is 71.9% of a Rp1,500 risk budget (0.5% of Rp300,000), as an illustrative cost-to-risk comparison. Actual account fee mode remains unverified. Spread inputs include a point-in-time snapshot or declared stress assumption; slippage is assumed, not calibrated from realized fills.

### 4.4 Statistical limits

- OOS sample sizes are 16–23 trades in the base case and 16–25 in the snapshot case. All are below the predeclared 30-trade minimum.
- OOS block-bootstrap 95% expectancy intervals are `null / INSUFFICIENT_SAMPLE`; no confidence interval or positive expectancy inference is available.
- Sharpe and Sortino were not reported by the research artifact. They should not be reverse-engineered from a handful of closed trades and treated as robust time-series estimates.
- Buy-and-hold benchmark excess return was not reported in this artifact. Benchmark comparison remains outstanding.
- Baseline expectancy, win rate and PF are undefined because it generated no trades. Alternative-family PF=0 and win rate=0% because each OOS trade lost under the model.
- These are offline simulated outcomes, not actual exchange fills or realized income. The 17 previously audited depth observations span only about 13m42s, have no exchange timestamps, and provide no queue position, trade-flow linkage or empirical fill probability.

## 5. Root causes ranked

| Rank | Finding | Confidence | Economic impact | Verification / action |
|---:|---|---|---|---|
| 1 | No demonstrated strategy edge: five candidate families all lost every OOS trade; baseline produced zero candidates. | High | Critical | Freeze candidates; require independent positive net expectancy before promotion. |
| 2 | Trading friction overwhelms the modeled gross edge: 87.59–172.11 bps round-trip. | High for modeled economics; medium for actual future costs | Critical | Verify personal fee tier and collect representative spread/depth/slippage observations; do not lower assumed costs without evidence. |
| 3 | Baseline filter stack suppresses all entries; breakout, relative volume, momentum, trend and RSI filters reject heavily. | High for zero signal output; medium for individual causal contribution | High | Use existing rejection telemetry and bounded ablation on development/validation only; never tune OOS. |
| 4 | Evidence horizon and market microstructure sample are inadequate: 30 days, six-day OOS, under 30 trades per family, 31.92% zero-volume rows and only sparse depth snapshots. | High | High | Extend independent data windows; validate volume semantics; collect time-distributed order-book snapshots with timestamps. |
| 5 | Scheduled Forward Paper operation is not proven: workflow is active and installed, but no Sentinel `schedule` event is present in the inspected history; state remains flat. | High | High for operational readiness | Keep `FIRST_CRON_PENDING`; require a genuine schedule event plus artifact and state checkpoint. Investigate separately without substituting manual dispatch. |

Infrastructure complexity is now greater than the demonstrated strategy value. That is an opportunity-cost problem, but it is not evidence of an exchange execution bug.

## 6. Financial target reality check

The target is Rp300,000 to Rp3,500,000, a factor of **11.6667x** and a required cumulative net return of **+1,066.67%**, before considering any additional capital contribution. If the previously recorded 24-hour horizon remains intended, this implies the same 11.67x growth inside one day. No research result supports that target or supplies a defensible probability.

Required compounded **net** return per trade, assuming no new deposits and full reinvestment:

| Number of completed trades | Required net return per trade |
|---:|---:|
| 10 | 27.85% |
| 25 | 10.33% |
| 50 | 5.04% |
| 100 | 2.49% |

Scenario interpretation:

- **Conservative:** do not place trades; preserve the Rp300,000 reference capital. Return is zero by design, not a profit forecast.
- **Base:** remain research-only and collect evidence. No candidate is promoted, so no positive return is assigned.
- **Optimistic:** only consider a return projection after an independently validated candidate exists. Current data cannot quantify this case.
- **Adverse diagnostic:** the tested candidate families lost approximately 7.5%–10.3% on their six-day OOS segment. This is a historical model result, not a forecast of future loss.
- **Failure boundary:** a 10% drawdown from Rp300,000 would leave Rp270,000. The drawdown gate is a control threshold, not a guarantee that realized losses cannot exceed it.

Probability of ruin, expected annual return and Monte Carlo target probability remain **not estimable**. The research engine intentionally returned `INSUFFICIENT_SAMPLE` and no probability estimate because no candidate passed the validation/OOS gate.

## 7. Strategic alternatives

| Alternative | Effort / cost | Economic value now | Decision |
|---|---|---|---|
| Continue infrastructure expansion | High effort; recurring CI/storage consumption | No direct evidence of higher net expectancy | Reject for now |
| Simplify to current data/evaluator/risk/ledger stack | Low incremental effort; free-first | Highest information value per unit effort | **Selected** |
| Add more strategy families immediately | Medium/high effort; overfitting risk | Five current families already fail the same OOS gate | Defer |
| Semi-automated or manual live orders | Human time and capital at risk | Cannot cure a negative edge; introduces execution risk | Reject |
| Divert some development time to a separately testable income activity | Depends on chosen activity | May reduce opportunity cost while Sentinel remains research-only | Consider; no income is assumed |

## 8. Recovery plan and acceptance gates

### P0 — by 2026-11-06

1. Freeze strategy parameter changes and nonessential infrastructure work.
2. Obtain evidence of one genuine scheduled Forward Paper event on `main`; require matching workflow identity, run event, source SHA, `result.json`, `status.json`, cycle ID and state-branch checkpoint. Do not use manual dispatch as cron proof. If no run appears, diagnose the GitHub schedule trigger independently and seek approval before workflow/settings changes.
3. Re-audit the paper SQLite state read-only; retain all existing data and verify integrity, foreign keys and ledger reconciliation.
4. Collect time-distributed public INDODAX order-book samples with observation timestamps. Do not claim realized slippage from static depth.

**P0 acceptance:** genuine scheduled run, durable artifact and state checkpoint are independently verifiable; no ledger inconsistency; no live order route.

### P1 — by 2026-11-13

1. Validate the 13,789 zero-volume rows against the source's documented meaning and examine whether volume-dependent filters are operating on valid observations.
2. Run bounded, predeclared signal ablations on development/validation only. Report candidates, rejections, gross move, fee hurdle and net expectancy per filter state.
3. Use additional chronological data if the source permits; otherwise extend forward observation. Freeze the parameters before any new OOS interval.
4. Verify the account's actual fee mode outside this research-only repository process; do not assume Pro rates are personal-account rates.

**P1 acceptance:** an evidence-backed explanation of the zero-candidate baseline and a calibrated cost range; no OOS retuning.

### P2 — by 2026-11-29

A strategy may become a paper candidate only if it satisfies all of these gates on a new, untouched OOS interval: positive net expectancy in validation and OOS; PF > 1.10; at least 30 closed OOS trades; at least two positive market regimes with ten or more trades each; max drawdown no greater than 10%; bootstrap expectancy interval that does not imply a negative edge; and cost assumptions supported by observed market data. Continue forward-paper validation for at least 30 calendar days with reconciled, non-flat evidence before any later readiness review.

**P2 failure condition:** if two separate frozen OOS windows fail these gates, pause SOL Sentinel strategy development and redirect most project effort elsewhere. Keep the repository archived/researchable; do not keep expanding the trading stack solely to justify past investment.

## 9. Evidence matrix

| Finding | Evidence | Status | Impact | Required action |
|---|---|---|---|---|
| Native INDODAX dataset quality | Run #69 artifact; 43,200 bars; SHA-256 above; 0 gaps/duplicates/invalid OHLC | Verified within source artifact | High | Preserve hash and revalidate next capture |
| Baseline produces no trade | 43,199 evaluations; 0 candidate/trade | Confirmed | Critical | Explain filter stack without forcing entries |
| Five alternatives have negative OOS | Run #69 `strategy_research.json`; 16–23 base OOS trades/family, zero wins | Confirmed model result | Critical | No candidate promotion |
| Modeled cost is substantial | Fee model + declared spread/slippage scenario table | Confirmed as assumptions, not realized costs | Critical | Calibrate only with observations |
| Forward Paper code/tests | Main CI #41038862799, 242 tests passed | Engineering CI verified | Medium | Keep CI regression-only |
| Recurring Sentinel schedule | 153 schedule events, none for Sentinel | Unverified/not observed | High | Genuine schedule run required |
| Paper account state | Last referenced DB audit #41035585382; state branch checkpoint unchanged | Historical read-only evidence | High | Fresh explicit read-only audit |
| Android/Termux portability | Ubuntu Doctor only | Not tested on device | Medium | Physical device report required |
| Account-specific fee tier and realized slippage | No authenticated account/order access or fills | Missing | High | Do not infer |

## 10. Actions performed in this audit

- Re-read the current repository continuity memory, forward-paper operations guide, research workflow, fee model, strategy research source and chronological evaluator.
- Inspected and unpacked research artifact #12967394940; independently recomputed OOS trade counts, per-trade expectancy, net P&L, fees and modeled execution friction from each closed-trade record.
- Reconciled current main CI, workflow schedule history and state-branch commit metadata.
- Made **no** strategy parameter change, code-runtime change, state reset, manual workflow dispatch, account query or live action.
- This document and the appended continuity checkpoint are documentation-only. The PR is intentionally not merged automatically.

## Final readiness

**`RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`**

Not `BACKTEST_VALIDATED`, `OUT_OF_SAMPLE_VALIDATED`, `FORWARD_PAPER_VALIDATED`, `OPERATIONALLY_READY` or `LIVE_TRADING_ELIGIBLE`. No live authorization is implied by any CI or research result.

---
