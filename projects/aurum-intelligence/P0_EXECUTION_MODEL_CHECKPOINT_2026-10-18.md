# P0 execution-accounting checkpoint — 2026-10-18

Status: research-only changes on PR #164 branch; not merged to main.
Decision gate: NO_TRADE; live order routing disabled.

## Accounting increment
- Added deterministic closed-trade accounting using entry/exit Bid/Ask snapshots.
- Long uses Ask/Bid; short uses Bid/Ask. Spread is exposed for reconciliation and not charged twice.
- Slippage, commission and signed swap are separate. Unknown material costs suppress net-PnL output; assumed costs remain estimates.
- Currency conversion fails closed. Evidence refs are syntax-checked only and cannot independently establish broker authorization.
- First hosted run exposed one assertion-format mismatch (-100 vs -100.0); assertion normalized on a new commit.

## Quote-side replay increment
- Added bracket-path replay for TP/SL outcomes using executable-side Bid/Ask OHLC.
- Same-bar TP/SL collisions retain pessimistic/optimistic PnL bounds rather than deleting ambiguous outcomes.
- Gap-through stops use the observed executable-side open; timeout exits use the executable-side close.
- A missing bar inside the horizon fails closed. The replay does not invent tick order.
- Added a large-scale runner that reuses a single acquired dataset snapshot, retains the prior benchmark report, cross-checks each directional outcome against the existing evaluator, and summarizes per-signal gross-R bounds across development/calibration/final chronology.
- Local accounting plus quote-path suites: 27 tests passed; staged modules compile.
- Hosted validation for this continuation is pending; no real-data replay result is yet claimed.

## Boundaries and next gates
- Hosted CI must pass on this commit, including the large real-data acquisition and 20,000 distinct timestamp path checks.
- Public-source provenance and intended-use rights remain unverified.
- Added-cost values in R are hypothetical; broker-specific commission, swap, slippage, latency and fill behavior remain absent.
- This is not a portfolio backtest. Overlapping rule events are not simulated as simultaneous capital-aware positions.
- No live signals, order routing, or profitability claims have been enabled.


## 2026-10-18 — stop geometry audit finding and corrective gate

- The first hosted full replay passed 40,000/40,000 legacy outcome-label cross-checks and reported 10,064 stop-loss-at-open paths. The report also showed feed close-spread median/P95 of 0.424/1.704 price units.
- Review of the entry-side logic revealed that the outcome loop checked the entry candle's opposite quote against an ATR-anchored stop at the same entry timestamp. Where entry spread >= stop distance, the barrier is already crossed at the quote snapshot; counting it as an ordinary later stop-out mixes invalid bracket geometry into strategy outcomes.
- Corrective change: exclude such anchors before sampling, add an explicit entry-spread/ATR population audit, and make the path primitive fail closed with invalid_stop_geometry_at_entry. Valid paths continue to require exact label agreement with the legacy evaluator; legacy false-valid immediate-stop labels are counted separately.
- This is a methodological correction, not a strategy improvement claim. The economic replay must be rerun on the changed eligible-anchor sample. No live order routing or profitability claim.
