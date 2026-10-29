# SOL Sentinel — Engineering Specification

Last updated: 2026-10-29 (Asia/Jakarta). Status: draft baseline for Phase 0.

## 1. Goal and non-goals

Goal: build a low-cost, auditable SOL market scanner and then evaluate a scalping strategy using actual net results after fees, spread, slippage, latency, and failed/partial execution assumptions.

Non-goals:
- Guaranteeing IDR 3,500,000 from IDR 300,000.
- Automatically maximizing leverage or risk to chase a deadline.
- Sending live orders in Phase 0.
- Storing Reku credentials or wallet secrets in GitHub, logs, or prompts.
- Using undocumented/private Reku endpoints to bypass its official controls.
- Treating Binance prices as executable Reku prices.

## 2. User-provided initial context

- The user reports IDR 300,000 has been deposited to Reku and is available for a purchase.
- The user showed Reku's SOL Pro screen with a limit price of IDR 1,820,000, amount IDR 150,000, estimated size 0.08241759 SOL, and displayed available balance IDR 300,000. The screen showed zero open orders. This is a screenshot snapshot, not current market state and not proof an order was placed.
- Aspirational goal: portfolio value IDR 3,500,000.
- The user wants a hyper-automated, smart, free-first workflow readable/auditable by the assistant through a connected repository.
- The user already completed KYC on Reku, so the primary fiat on-ramp candidate is their existing Reku account. The strategy should not make any assumption that KYC-free fiat on-ramp is safe or available.
- Candidate asset remains SOL for initial engineering fit with Solana/Jupiter. Asset selection must be revisited only with evidence and net-cost analysis, not because a high return is desired.

## 3. Architecture

1. Market data adapters: retrieve completed Binance SOLUSDT candles for broad technical context, and separately read Reku's documented public SOL/IDR price, bid/ask, and order book for venue-specific quote/spread/depth measurement. Timestamp each fetch and reject stale, invalid, crossed, or internally inconsistent data. Binance is never an executable Reku price.
2. Indicator layer: EMA 9/21, RSI 14, ATR 14, relative volume, prior-range breakout. Keep formulas deterministic and tested.
3. Regime filter: use 15m trend as higher-timeframe context, 5m as setup confirmation, 1m as entry trigger. A no-trade result is valid and often preferable.
4. Cost model: net expected edge = conservative gross movement assumption - round-trip fees - round-trip slippage - any separately configured transfer/priority costs. Every parameter must be explicit and recorded.
5. Risk gate: reject signals if inputs are stale, insufficient, costs exceed edge, volatility/data regime is outside tested bounds, or the daily loss lock is engaged.
6. Audit: append one JSONL record per scan with timestamp, source, candle timestamps, parameter values, metrics, outcome, and rejection reasons. Never store credentials.
7. Supervisory control: existing `ai-determinism-engine` remains responsible for bounded DAG execution, approvals, workflow identity, checkpoints, and failure recovery. This module must not bypass those contracts.
8. Later paper engine: simulated position state and fill model, including spread, fees, slippage, delay, failed orders, and no-fill limit orders.
9. Optional execution adapter: prohibited until an official venue/API and terms are verified, signer isolation is implemented, and explicit user approval is recorded.

## 4. DAG

```text
source_preflight
    -> fetch_1m ─┐
    -> fetch_5m ─┼-> freshness_and_integrity
    -> fetch_15m ┘             |
                               v
                         indicators
                               |
                               v
                         regime_filter
                               |
                               v
                       cost_profitability
                               |
                               v
                       deterministic_risk
                               |
                    ┌──────────┴──────────┐
                    v                     v
             paper_candidate         no_trade(reason)
                    |                     |
                    └──────────┬──────────┘
                               v
                         audit_jsonl
                               |
                               v
                    assistant-readable report
```

All source calls are read-only. Retry only bounded transient market-data failures; never retry a financial side effect automatically after an uncertain outcome.

## 5. Scanner contract

The initial worker:
- Uses only Python standard library.
- Fetches Binance public klines for SOLUSDT at 1m, 5m, and 15m.
- Discards the still-open candle.
- Validates minimum candle count, freshness, and numeric values.
- Computes deterministic indicators and a conservative candidate signal.
- Emits a human-readable summary and/or JSON.
- Appends JSONL audit records if a log path is configured.
- Never places orders and has no private-key handling code.

Binance is a *reference feed*. It cannot determine exact entry prices or fee schedules on Reku. A read-only Reku public market-data adapter has now been implemented from documented legacy endpoints; live endpoint availability and current API terms must be verified. This adapter does not place orders, and no authenticated order route is approved or assumed.

## 6. Initial strategy hypothesis (not validated)

A long candidate requires all conditions:
- 15m EMA(9) > EMA(21) and close above EMA(21);
- 5m EMA(9) > EMA(21), close above EMA(9), and RSI(14) in a non-extreme range;
- 1m close above the prior 20-candle high, volume ratio above a configurable threshold, and RSI below an exhaustion threshold;
- sufficient ATR-based conservative gross-move budget to exceed configured round-trip cost plus a net-edge buffer;
- all market data are fresh.

This is a testable hypothesis, not a claim of profitability. Thresholds must be tuned on training data and evaluated on unseen periods. No parameter may be selected solely to maximize past returns.

## 7. Acceptance gates

### Phase 0 — scanner
- Offline tests pass in CI.
- Malformed, empty, rate-limited, stale, and partial data fail closed.
- No network/API keys are required for public feed.
- Every evaluation emits source timestamps and rejection reasons.
- No order endpoint, private key, or transaction-signing functionality exists.

### Phase 1 — research
- Acquire enough historical 1m/5m/15m observations to include trending, sideways, and volatile regimes.
- Split chronologically into training, validation, and untouched out-of-sample periods.
- Include actual venue fees, spread, slippage, latency, and no-fill/failed-order assumptions.
- Report net expectancy, profit factor, max drawdown, return distribution, trade count, exposure, and sensitivity to costs.
- Keep parameter count and optimization constrained to limit overfitting.

### Phase 2 — paper trading
- Run for a predeclared minimum observation window and sufficient number of trades.
- Persist paper positions and fills durably; reconcile ledger totals.
- Compare estimated and observed quote movement; stop if cost model is materially biased.
- No real money.

### Phase 3 — live approval gate
All are mandatory:
- Written approval for this specific live deployment.
- Verified official trading route and terms; no undocumented endpoint.
- Wallet/credential isolated from logs and GitHub.
- Explicit maximum order size, aggregate exposure, loss cap, slippage, request rate, and session stop.
- Preflight balance/fees/withdrawal constraints.
- Idempotency and transaction reconciliation tested.
- Emergency stop tested and reachable without the assistant.
- Live mode is disabled by default and cannot be enabled by an LLM-generated strategy change.

## 8. Definition of success

Engineering success: reproducible data, correct risk gates, robust offline tests, and observable paper results.

Financial success cannot be promised. The IDR 3,500,000 target is tracked as an aspiration only. Report actual account equity only from verified sources and never infer a balance from strategy performance simulations.


## 9. Venue-specific market-data validation

The read-only module `reku_market.py` requests Reku coin metadata, prices, direct bid/ask, and order book from the legacy public API reference at https://api.reku.id/. It records spread (bps), best levels, top-five depth, and optional deviation from a contemporaneous-enough Binance reference converted via a dated USD/IDR rate. Because the documentation is old and the public endpoint responses may vary, the live snapshot is accepted only as market-data evidence when its status and warning fields are valid. It is not a trade signal and offers no order-placement capability. The historical backtest still uses Binance SOLUSDT candles converted with a single daily FX reference; replacing this with verified Reku SOL/IDR chart history is a separate task after the chart endpoint's semantics and continuity are validated.


## 10. First public Reku quote observation (not a live guarantee)

A read-only snapshot in Actions run `40916894414` observed last price Rp1,845,000, bid Rp1,840,000, ask Rp1,845,000 and spread around 27.137 bps. The orderbook endpoint returned HTTP 400 for the SOL coin ID and was therefore recorded as unavailable; no depth was inferred. The adapter now falls back to the documented `/v2/orderbookall` endpoint. This is a single snapshot with no guaranteed freshness after capture; use only a new `OK` snapshot for any quote-related research. It still does not establish a supported authenticated order API.
