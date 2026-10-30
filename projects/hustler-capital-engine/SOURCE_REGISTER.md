# MAGICUSDT Research Source Register

Checked 2026-10-31 (Asia/Jakarta). Revalidate endpoint behavior, Binance filters and fee schedules before relying on results.

| Source | Use | Access/cost | Limitation |
|---|---|---|---|
| Binance USDⓈ-M `exchangeInfo` | Active contract, price tick, lot sizes and min/max notional | Public | Fetched every scan; never hard-code per-contract filters. Public schema sample shows MIN_NOTIONAL 5 USDT; live MAGIC filter must be queried. |
| Binance MAGICUSDT leverage-tier announcement (2026-03-07) | Public leverage bracket | Public | Lists 21–50x for 0–5,000 USDT and 11–20x for 5,000–10,000 USDT; current UI/account eligibility still needs verification. |
| Binance USDⓈ-M `/fapi/v1/time` | Exchange clock / freshness reference | Public | Local clock offset >10 seconds blocks analysis |
| Binance USDⓈ-M `/fapi/v1/klines` | 5m/15m/1h/4h OHLCV | Public | Completed, continuous candles only |
| Binance ticker/24hr and ticker/price | Last price and 24h context | Public | Indicative reference, not a guaranteed execution price |
| Binance `/fapi/v1/depth` | Spread and top-of-book depth | Public | Snapshot only; no queue position or future fill certainty |
| Binance `/fapi/v1/premiumIndex` | Mark price, funding rate and next funding | Public | Discrete settlement; payer depends on side and rate |
| Binance open interest and `openInterestHist` | OI context and recent change | Public | Statistics endpoints have bounded history |
| Binance funding history | Settlement timestamps/rates | Public | Interval is inferred from recent settlement timestamps; missing history blocks cost confidence |
| Binance global/top-trader long/short ratios | Positioning context | Public | Not a predictive directional signal |
| Binance taker long/short ratio | Flow context | Public | Not a standalone entry signal |
| Binance user commission info | User-specific taker/maker fee | Authenticated USER_DATA or account UI | Not implemented; unknown fee blocks eligibility |
| `ccxt/ccxt` | Possible future exchange-neutral adapter | Open-source MIT | Not included in v0.1; native endpoints keep Termux lightweight |
| CoinGecko API Demo | Optional broader spot cross-check / metadata | Public free-demo tier documented | Not integrated; aggregate spot price is not MAGICUSDT perpetual execution price |

## Endpoint inventory

Required:
- `GET /fapi/v1/exchangeInfo`
- `GET /fapi/v1/time`
- `GET /fapi/v1/ticker/price?symbol=MAGICUSDT`
- `GET /fapi/v1/ticker/24hr?symbol=MAGICUSDT`
- `GET /fapi/v1/premiumIndex?symbol=MAGICUSDT`
- `GET /fapi/v1/depth?symbol=MAGICUSDT&limit=20`
- `GET /fapi/v1/klines?symbol=MAGICUSDT&interval={5m,15m,1h,4h}&limit=200`

Optional evidence:
- `GET /fapi/v1/openInterest?symbol=MAGICUSDT`
- `GET /futures/data/openInterestHist?symbol=MAGICUSDT&period=15m&limit=10`
- `GET /futures/data/globalLongShortAccountRatio?symbol=MAGICUSDT&period=15m&limit=10`
- `GET /futures/data/topLongShortPositionRatio?symbol=MAGICUSDT&period=15m&limit=10`
- `GET /futures/data/takerlongshortRatio?symbol=MAGICUSDT&period=15m&limit=10`
- `GET /fapi/v1/fundingRate?symbol=MAGICUSDT&limit=10`

No private account, signed order or transaction endpoint exists in this implementation.

## References

- [MAGICUSDT leverage and margin tier update (2026-03-07)](https://www.binance.com/en/support/announcement/detail/a46249effc4645c89db2c18a198119d4)
- [Binance USDⓈ-M Futures market-data catalogue](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data)
- [Kline/Candlestick endpoint](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data)
- [Exchange information and filters](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information)
- [Futures server time](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Check-Server-Time)
- [Mark price / funding](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price)
- [Order book depth](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Order-Book)
- [Open interest](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest)
- [Open-interest statistics](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics)
- [Funding-rate history](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History)
- [Taker buy/sell volume](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-Buy-Sell-Volume)
- [Global long/short ratio](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Long-Short-Ratio)
- [Top-trader long/short ratio](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Top-Trader-Long-Short-Ratio)
- [CCXT (MIT)](https://github.com/ccxt/ccxt)
- [CoinGecko API pricing](https://www.coingecko.com/en/api/pricing)
