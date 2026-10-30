# MAGICUSDT Research v0.1 Acceptance Matrix

| ID | Gate | Required evidence | v0.1 state |
|---|---|---|---|
| DATA-01 | Correct instrument | Exchange metadata says active MAGICUSDT USDT perpetual | Implemented; live fetch not completed in this environment |
| DATA-02 | Contract sizing | Tick, lot, market lot, min/max notional respected | Implemented; current filters fetched only on scan |
| DATA-03 | Candle integrity | Completed 60+ bars, exact interval, strict order, no gaps/duplicates, finite OHLCV | Implemented + offline tests |
| DATA-04 | Freshness | Quote receipt age and candle age bounded | Implemented + fixture tests |
| DATA-05 | Clock integrity | Futures server time; >10s local offset blocks report | Implemented + fixture tests |
| DATA-06 | Market structure | Bid < ask, spread and mark/last basis bounded | Implemented + fixture tests |
| DERIV-01 | Funding | Current rate, next settlement and inferred funding interval | Implemented + 4h fixture test; live interval not fetched |
| DERIV-02 | Positioning | OI, historical OI, global/top-trader ratios, taker ratio | Optional context collectors; not signal authority |
| STRAT-01 | Multi-timeframe | 4h/1h/15m alignment; 5m break, volume and RSI | Implemented; not calibrated |
| COST-01 | User fee tier | Actual user-specific taker fee known | BLOCKED; no account connection |
| COST-02 | Slippage | Timestamped book walk / fill observations | BLOCKED; not calibrated |
| SIZE-01 | 1 USDT notional | Actual quantity rounded to current filters, never auto-increased | Implemented; live filters unverified |
| RISK-01 | 23x/25x/30x | Margin and stop-distance stress report | Implemented as analysis only |
| LEDGER-01 | Idempotency | Identical replay no-op; conflicting payload rejected | Implemented + regression tests |
| OPS-01 | Android acceptance | Public collector runs on user's actual Termux device | NOT RUN |
| PAPER-01 | Forward paper | >=30 calendar days and >=100 valid closed simulated trades | NOT STARTED |
| EDGE-01 | Strategy evidence | Positive net expectancy on untouched chronological OOS after costs | NOT ESTABLISHED |
| LIVE-01 | Live trading | Separate approval, credentials, risk/legal/security review | DISABLED / OUT OF SCOPE |

## Release disposition

`RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`. Not PAPER_READY, not LIVE_READY, and not a recommendation to enter a position.
