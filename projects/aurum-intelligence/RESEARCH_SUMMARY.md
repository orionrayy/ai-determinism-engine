# Research summary and design rationale

## Scientific feasibility
- Gold-price direction or OHLC prediction is not the same as tradeable alpha. A high direction hit-rate can still lose after spread, slippage, exits and tail risk.
- Targets must be explicit: next-candle body direction, numeric return, barrier outcome and actual trade PnL are distinct tasks.
- A valid protocol uses point-in-time features, chronological walk-forward, purged/matured labels, calibration only on held-out predictions, an untouched final holdout, realistic costs and baseline comparisons.
- Avoid look-ahead in macro/news features and data revisions. Model uncertainty and stale/missing provider data should trigger abstention.
- No performance claim is warranted by the current one-session, 60-bar sample.

## Competitive intelligence / product positioning
Relevant reference classes studied include TradingView, Autochartist, Trading Central, Myfxbook, GoldSignals.io, Verpips and institutional products such as LSEG/Bloomberg. The useful differentiation target is transparent methodology, probability calibration, reproducible outcome ledger, evidence citations, risk/abstention and measured net-of-cost results—not unsupported promises of accuracy.

## UX and service design
- Telegram is the low-friction entry point; use button-first navigation and progressive disclosure rather than dense command-heavy flows.
- The Mini App is the detail surface for charts, timeframe selection, source provenance and trade/forecast rationale.
- Every signal/report should display timestamp/timezone, instrument, horizon, invalidation, assumptions, risk status, data freshness, confidence/calibration status and why a decision abstained.
- Personalization should support alert cadence, instrument/timeframe, sensitivity and risk constraints without implying account connectivity.

## Operating architecture
Telegram/API -> data ingestion/quality gates -> evidence/provenance store -> price/cross-asset/news analyses -> forecast/calibration -> risk gate -> decision ledger -> Telegram/Mini App delivery. Provider adapters should be replaceable modules. Use a strict policy boundary between analytical signals and any eventual brokerage account/order layer.

## Current key blockers
1. Data breadth and independent broker/timezone provenance.
2. Direct reproducible data ingestion in the local runtime.
3. Uncalibrated, insufficient-history probability models.
4. No tick/Bid/Ask fill replay with observed broker costs.
5. No point-in-time historical news-impact model.
6. No live Telegram/OANDA test; MT5/cTrader account connectors not implemented.
7. Docker/production deployment and paper trading not validated.

See the original long-form science, competitive-intelligence, operating-architecture and deep-research notes in the research artifact bundle associated with this project checkpoint. Conclusions above summarize those reports; they do not imply all candidate sources are currently integrated.