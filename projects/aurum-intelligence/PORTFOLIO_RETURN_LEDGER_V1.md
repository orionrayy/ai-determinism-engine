# Portfolio Return Ledger v1.0.0

## Purpose and boundary

This pure-Python offline module validates a regular-clock, mark-to-market portfolio ledger and provides aligned return vectors for later selection-adjusted inference. It does not generate orders, connect to a broker, authenticate source data, or authorize deployment.

## Accounting convention

Only the explicit basis quote_side_after_spread is accepted. Gross PnL is marked at executable quote side: long inventory at Bid and short inventory at Ask. Therefore spread is already embedded in gross PnL. The spread_cost field is audit metadata and is not deducted a second time.

Net PnL for interval t is:

net_pnl = gross_pnl_quote_side - commission - slippage - other_fees - market_impact + financing_pnl

Equity must reconcile within an explicit tolerance: equity_after = equity_before + net_pnl. No untracked deposits or withdrawals are permitted. Each interval must link to the prior interval's equity.

## Fail-closed invariants

- Every clock interval must be present, including fully flat intervals with zero PnL, cost, exposure, turnover and return.
- Timestamps must be timezone-aware, strictly increasing, and regular within a session. Reused session identifiers after a session boundary are rejected.
- Every row has stable portfolio, strategy and trial IDs, a unique decision ID, source ID, market snapshot hash and decision-trace hash.
- A flat interval cannot contain PnL, cost, fill, turnover or exposure.
- Fills need stable evidence references and SHA-256; simulated fills never qualify as observed execution.
- Explicit costs cannot be negative. Financing PnL may be positive or negative.
- Unknown or assumed cost evidence blocks eligibility for market inference.
- Gross leverage has a caller-specified upper limit; equity must remain positive.
- Strategy ledgers must share the identical timestamps and session IDs. No unmatched rows can be dropped while building a panel.
- The panel builder recomputes each ledger SHA-256 over canonical normalized period rows and verifies the report-level timestamp/session/return vectors are exact projections of those same rows. Mutated rows or detached projection arrays block the panel.
- Each period net return is checked against net PnL divided by beginning equity within a tiny arithmetic tolerance.

## Evidence gate

selection_guard_input_eligible is true only if each gate flag is true and has a stable evidence reference: source data qualified, costs verified, portfolio replay verified, locked holdout unseen, and trial registry complete. Costs must be observed or not applicable with explicit reason; no material cost may be assumed or unknown. All trade fills must be marked observed.

These flags and references are not self-authenticating. They require independent review. A structurally valid ledger is not proof of alpha, positive net expectancy, data provenance, or deployability.

## Integration

Run the module tests from the online-benchmark directory with the existing unittest suite. Build one ledger report per preregistered strategy. Pass the full synchronized panel to selection_bias_guard_v1 only after source, execution-cost, replay, holdout, and trial-registry evidence is independently qualified. The module itself always returns NO_TRADE and never calls the selection guard.
