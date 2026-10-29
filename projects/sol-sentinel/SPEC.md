# SOL Sentinel Behavior Contract

## Modes

Only `paper` mode exists. The CLI uses public read-only market data. It cannot place, cancel, amend, or sign live orders.

## Data contract

The adapter tries the legacy Reku API in this order:
1. `GET /v2/coins` for a SOL coin identifier.
2. `GET /v2/price` for SOL/IDR last/high/low/open/volume.
3. `GET /v2/bidask` for current bid and ask.
4. `GET /v2/orderbook?id={id}` for depth (optional).
5. `GET /v2/chart?t=2&f=5&id={id}` for 5-minute OHLCV (required to authorize a fresh signal).

No key or credential is sent. If price/bid/ask are missing, return a failed snapshot. If chart data cannot be parsed, display market status but do not open a paper position. Do not fall back to stale values without marking them stale.

## Indicator contract

- EMA(9) and EMA(21) on completed 5-minute candles.
- RSI(14) with explicit handling of flat/up-only/down-only windows.
- ATR(14), as a volatility estimate.
- Relative volume = last completed candle volume / mean volume of up to 20 previous completed candles.
- Breakout = last completed close is above the maximum high of the previous 12 completed candles.
- Last candle is conservatively treated as incomplete until its 5-minute interval has ended.

## Initial signal eligibility

A simulated long entry can be considered only when all are true:
- Data adapter reports valid price and both sides of the market.
- Completed candle age <= 10 minutes.
- At least 30 valid completed candles.
- Spread <= configured maximum (default 35 bps).
- EMA9 > EMA21, close > EMA9, RSI in [50, 68].
- Relative volume >= 1.2.
- Breakout condition true.
- Expected reward exceeds modeled round-trip costs plus configured minimum net edge.
- Daily loss lock is not active.
- No paper position is already open.
- Current candle has not already generated a paper entry.

These thresholds are starting hypotheses, not calibrated edge. The bot may produce zero signals, which is acceptable.

## Paper trade sizing and accounting

- Initial paper equity defaults to Rp300,000.
- Maximum position notional defaults to 50% of current equity.
- Risk budget defaults to 0.5% of current equity.
- Stop distance uses max(1.5 * ATR, 0.8% of entry reference price).
- Reward/risk defaults to 2.0.
- Effective buy fill is ask increased by configured slippage.
- Effective exit fill is bid reduced by configured slippage.
- Fees are applied to both buy and sell notional according to configurable assumptions.
- Paper position size is capped by both the risk budget and max notional.
- When an exit condition is observed, the paper simulator closes at the modeled marketable price; it does not claim this would be a guaranteed real fill.
- Daily lock triggers when realized P&L plus current unrealized P&L reaches the daily loss threshold relative to day-start equity.

## Persistence

SQLite stores snapshots, event notes, paper state, and closed/open paper trade records. Database content is local by default. The CLI prints JSON status lines to stdout for simple piping/monitoring. SQLite writes use transactions.

## Required validation before next stage

1. All offline unit tests pass.
2. A live `--doctor --once` confirms current Reku endpoint shapes; if not, fix the adapter against official current data before using it.
3. Minimum 7 days paper collection (preferably more varied market regimes) without data-integrity gaps.
4. Backtest and forward paper results report net P&L after modeled fees/slippage, max drawdown, trade count, profit factor, average net trade, and signal frequency.
5. Validate assumptions against current Reku fees, bid/ask spreads, and order minimums.
6. Independent security review and explicit approval before any future real execution module.
