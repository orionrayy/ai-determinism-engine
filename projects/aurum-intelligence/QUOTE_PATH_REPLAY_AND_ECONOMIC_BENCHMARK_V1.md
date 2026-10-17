# Quote-side TP/SL replay and economic benchmark v1

Status: hosted CI validated as a research diagnostic in run #39057955872 on 2026-10-18; no profitability validation. Trading gate remains NO_TRADE.

## Accounting and path rules

- Long enters at Ask and closes at Bid; short enters at Bid and closes at Ask. Spread is embedded in gross quote-side results and must not be deducted twice.
- Signals are formed after the completed anchor close; hypothetical entry uses the next M1 open.
- TP/SL levels use ATR available at decision time. Gap-through outcomes use the observed executable-side open.
- If both barriers are touched in one OHLC candle, path order is unknown. The evaluator retains pessimistic/optimistic gross-PnL bounds; it neither guesses a favorable sequence nor deletes the observation.
- Missing bars inside the horizon fail closed. A no-hit horizon exits at the last executable-side close.
- PnL is per instrument unit and gross of unmeasured costs. The assumed additional-cost grid in R is sensitivity analysis, not a user-broker estimate. Net account-currency PnL is not produced.

## Benchmark

execution_replay_benchmark_v1.py runs the existing large-scale M1 Bid/Ask benchmark from one cached snapshot, then replays the same real timestamp anchors through the quote-side path engine. It requires at least 700,000 validated Bid/Ask bars, 20,000 anchors, 1,000 strategy events in the final chronological holdout, and exact outcome-label agreement with the prior evaluator. The original base benchmark report remains in place.

Outputs:
- large_scale_bidask_benchmark.json / .md: existing data and probability diagnostics.
- execution_replay_report.json / .md: gross R, ambiguity bounds, per-strategy chronological splits, and cost sensitivity.

## Interpretation boundary

- Passing parser, label-crosscheck and CI gates validates software contracts and repeatable processing, not strategy edge.
- Upstream provenance and intended-use rights remain unverified. The public data does not stand in for measured quotes from the user's broker.
- Touch fills are assumptions; partial fills, market impact, queue priority, rejection, latency, financing calendars, margin/liquidation, and venue-specific costs are not implemented.
- Event-level observations overlap. There is no capital-aware portfolio/equity simulation.
- Decision remains NO_TRADE; profitability validation and live execution remain false.


## Stop-distance versus entry-spread guard (P0 correction)

The candidate sampler now excludes an anchor if the observed entry spread is greater than or equal to the 1-ATR stop distance. At such an entry snapshot, a long's Bid is already at/below its Ask-anchored stop or a short's Ask is already at/above its Bid-anchored stop. A pre-crossed barrier is an invalid trade setup, not a conventional post-entry stop-out. The data audit reports the fraction of contiguous ATR-valid candidate windows rejected by this geometry rule. The quote-path primitive also checks this invariant independently for callers outside the guarded sampler.


## Barrier arithmetic parity

Barrier thresholds use the same floating-point evaluation order as the pre-existing price-path evaluator. Price and PnL values remain Decimal-based after the thresholds have been fixed. This prevents a float-versus-Decimal rounding difference at a threshold equality from silently changing a binary TP/SL label.


## Latest hosted result — 2026-10-18

Run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/39057955872  
Commit: `2cb6c3dc1eb1d48a2bcdbf8e0792f6e80bdec151`  
Artifact: `12110540720`, 2,635,151 bytes, SHA-256 `7e192a620b75bf61c32d4d9daa4d3528dcdb8264b9c8e0b4a23a7ad05f10a7fe`.

The 20,000-anchor / 40,000-path replay passed exact label agreement with the reference evaluator. With the entry-spread geometry correction, all four fixed strategies were negative in mean gross R on the final chronological holdout: momentum -0.5094 to -0.5003R; EMA10/30 -0.4922 to -0.4812R; mean-reversion z20 -0.5296 to -0.5203R; range-breakout 20 -0.4723 to -0.4523R. These are per-event quote-side diagnostics before unmeasured broker costs and without portfolio-level exposure. Do not promote these baselines.
