# Execution model contract v1 — research-only P0 increment

Status: candidate implementation on the existing research PR branch; hosted CI pending.
Decision gate remains NO_TRADE. No broker adapter, order router, or production promotion is included.

## Accounting contract

A long uses Ask on entry and Bid on exit. A short uses Bid on entry and Ask on exit. Those executable quote sides already contain the spread; spread is itemized for audit, never deducted a second time.

Let Q = lots × contract units per lot and d = +1 for long, −1 for short.

- Midpoint gross PnL = d × (exit midpoint − entry midpoint) × Q.
- Spread drag for decomposition = Q × (entry spread + exit spread) / 2.
- Quote-side gross PnL = d × (exit quote-side price − entry quote-side price) × Q.
- Net PnL = quote-side gross PnL − adverse slippage − commission + signed swap PnL.

The code asserts that midpoint movement minus the spread decomposition equals quote-side gross PnL. That invariant guards against double-counting spread.

## Fail-closed rules

- Timestamps must include explicit timezone offsets; exit must follow entry.
- Bid/Ask must be finite, positive and non-crossed.
- Commission is per lot per side; slippage is an adverse price offset per fill; swap is signed PnL per lot per rollover.
- Unknown material costs yield a null net PnL. Assumptions may yield a numeric estimate only when explicitly labelled assumption-only.
- Currency conversion fails closed until a validated conversion module exists.
- Evidence references are syntax-checked, not independently verified. Even fully referenced inputs are eligible for manual review only.
- Live execution authorization is always false.

## Outside scope

No partial fills, order rejection, queue priority, limit-touch fill probability, market impact, stochastic latency, stop-gap execution, margin/liquidation, triple-swap calendar or currency conversion is implemented. This primitive is an accounting building block, not an execution simulator or proof of a tradable edge.

The standard-library unittest suite covers long/short signs, spread accounting, fee decomposition, swap, net R, unknown-cost failure, assumptions, evidence references and malformed input. Passing it proves arithmetic/contract behavior only, not data provenance, observed costs or profitability.
