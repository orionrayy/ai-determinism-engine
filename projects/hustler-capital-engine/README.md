# Hustler Capital Engine — MAGICUSDT Futures Research v0.1

**Status:** READ_ONLY_RESEARCH / NO_TRADE  
**Instrument:** Binance USDⓈ-M MAGICUSDT perpetual  
**Runtime:** Python 3.10+, standard library only; Termux-friendly  
**Order execution:** absent by design  
**Default size:** 1.00 USDT **notional** per candidate, not 1 USDT initial margin

This module is a read-only child project of `orionrayy/ai-determinism-engine`. It does not connect to private Binance account endpoints, read balances/positions, change leverage or margin mode, place/cancel orders, or transfer assets. The 7.94 USDT Futures available balance seen in the user's screenshot is a visual observation, not authenticated account state.

## Implemented

- Binance USDⓈ-M public REST collector: exchangeInfo/filters, exchange server time, last price, 24h ticker, mark price/funding, order book, 5m/15m/1h/4h candles, open interest, OI history, funding history, global and top-trader long/short ratios, and taker buy/sell ratios.
- Data integrity: active USDT perpetual validation; finite OHLCV; impossible OHLC rejection; exact close times; completed candles only; strict chronology; gap/duplicate rejection; server-clock skew, quote/candle freshness, crossed-book and spread gates; bounded HTTP responses/retries.
- Deterministic indicators: EMA20/EMA50, Wilder RSI14, ATR14, relative volume, 20-bar breakout/breakdown, multi-timeframe direction checks.
- Explicit costs: taker commission both sides, measured book spread, slippage both sides, current funding and expected funding events, plus net-edge buffer. Unknown fees/slippage/funding interval block eligibility.
- Exchange sizing: price tick, LOT_SIZE/MARKET_LOT_SIZE, quantity steps, min/max notional. The engine never silently increases a requested 1 USDT notional to satisfy exchange filters.
- 23x/25x/30x isolated-margin arithmetic and stop-distance stress. These are read-only scenarios; account settings are not changed.
- SQLite WAL evidence ledger. Identical report replay is a no-op; inconsistent same-identity payload fails closed.
- Offline unit tests and an optional free GitHub Actions read-only scan workflow. No Binance key or GitHub secret required.

## Termux

```sh
pkg update
pkg install python -y
mkdir -p ~/hustler-magic
cd ~/hustler-magic
# Copy magic_engine.py, test_magic_engine.py and config.example.json here.
python magic_engine.py self-test
python magic_engine.py scan --config config.example.json --db state/magic.sqlite3 --json
```

Analyze an existing snapshot and inspect the local ledger:

```sh
python magic_engine.py analyze --snapshot state/latest_magic_snapshot.json --config config.example.json --db state/magic.sqlite3 --json
python magic_engine.py status --db state/magic.sqlite3 --json
python -m unittest -v
```

Python's standard library only: `urllib`, `json`, `sqlite3`, `decimal`, etc. No NumPy, pandas, TA-Lib, Torch, browser automation, or large ML dependencies. Network access is needed only for `scan`; tests are offline. The collector checks Futures server time and blocks when local clock skew exceeds 10 seconds.

## Size semantics and uploaded screenshot

The user's “US$1 per entry” is interpreted as **1 USDT order notional**. At the screenshot's 0.08251 USDT price, approximate raw quantity is 12.1197 MAGIC. Initial-margin arithmetic for $1 notional is about $0.04348 at 23x, $0.04000 at 25x, and $0.03333 at 30x, before fees and maintenance margin.

The screenshot displayed available Futures balance 7.94 USDT, Cross 20x, no open positions and no open orders. It also displayed max capacity 152.01 USDT, which is not a trade target. This engine's risk scenarios are isolated-margin-only and do not change account settings. The balance is not verified as current.

The current contract's actual notional/quantity filters are fetched at scan time. If 1 USDT is below minimum, status remains NO_TRADE. No automatic size escalation is allowed. If the user intended $1 **initial margin**, notional exposure would become $23–30 at 23–30x and is a materially different risk decision.

## Decision and evidence boundaries

A directional research candidate needs aligned 4h, 1h and 15m bias plus a 5m 20-bar breakout/breakdown, relative volume >=1.2 and a non-exhausted RSI. Open interest and long/short ratios are context only; a single order-book imbalance is never treated as a forecast.

Status remains NO_TRADE in this release because paper and live execution are disabled. This is not a trade signal. Missing/stale/inconsistent data, contract filters, fee tier or slippage calibration blocks eligibility.

The screenshot's 0.00500% funding and apparent 4-hour interval are historical observations only. The collector reads current funding and infers the interval from settlement history instead of hard-coding an 8-hour assumption. User-specific taker commission must be confirmed through Binance's official account fee page/API documentation; do not paste API secrets into chat or the repository.

Scheduled GitHub Actions artifacts are short-retention diagnostics, not a durable cross-run database. Keep durable SQLite research history on Termux or in a separately reviewed state store.

## Deliberate non-goals

- No API key, signed endpoint, account-balance/position endpoint or credential storage.
- No order, leverage/margin configuration, cancel, TP/SL, transfer or withdrawal route.
- No forced size increase, averaging down, martingale, or forced LONG/SHORT.
- No model/LLM confidence as an authority for a trade.

## Next evidence gates

1. Execute read-only collector on the user's Android/Termux and reconcile results with Binance UI.
2. Preserve >=30 days of gap-free completed candles and timestamped order-book/funding/OI observations.
3. Freeze rules before chronological development/validation/untouched holdout evaluation with realistic fees, spread, slippage and funding.
4. Forward-paper for >=30 calendar days and >=100 valid closed simulated trades; extend the sample rather than weaken the threshold when too few trades occur.
5. Prove SQLite restart/replay, risk-limit, protective-order semantics, kill switch and uncertain-outcome reconciliation.
6. Only then consider a separate review and explicit approval for a distinct credentialed live-execution project.

Passing unit tests validates fixture behavior only. It does not prove positive market edge, profitable trading, production API availability or Android device acceptance.
