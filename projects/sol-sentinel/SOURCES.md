# Source Register

Checked: 2026-10-29. Links are references for the project, not a claim that each service is continuously available. Prices, API quotas, fees, and terms can change.

## Market data

| Source | Use | Link | Confidence / caveat |
|---|---|---|---|
| Reku legacy public API docs | SOL/IDR coin metadata, price, bid/ask, order book, chart | https://api.reku.id/ | Docs were crawled about 1.9 years ago. Current behavior was not reachable from this build environment; validate with `--doctor --once` before relying on it. |
| Reku price | All coin snapshots in legacy docs | https://api.reku.id/v2/price | Example fields: `cd`, `id`, `c`, `h`, `l`, `o`, `v`, `cp`. |
| Reku bid/ask | Current bid/ask list in legacy docs | https://api.reku.id/v2/bidask | Example fields: `code`, `bid`, `ask`. |
| Reku coins | Coin ID and transfer metadata in legacy docs | https://api.reku.id/v2/coins | Example `result[]` fields include `id`, `accountcode`, `minsend`, `sendfee_external`, `enablebuy`, `enablesell`. This project does not use transfer metadata to initiate withdrawals. |
| Reku order book | Depth and spread validation | https://api.reku.id/v2/orderbook?id={coin_id} | Example arrays `b` and `s`; row examples are IDR total, price, quantity. Validate schema before using. |
| Reku chart | 5-minute candle history | https://api.reku.id/v2/chart?t=2&f=5&id={coin_id} | Legacy example rows are `[timestamp, open, close, low, high, volume_coin, volume_idr]`; `t=2` is taken from the legacy docs and must be verified. |
| Binance Spot REST | Optional independent benchmark in a future adapter | https://developers.binance.com/en/docs/products/spot/rest-api | SOL/USDT is cross-market context only, not a Reku SOL/IDR fill price. Not used by the initial scanner. |

## Transaction cost and safety references

| Source | Use | Link | Notes |
|---|---|---|---|
| Reku fees | Verify current maker/taker and transaction costs | https://reku.id/biaya | Current in-app fee schedule and trade mode take priority over prior estimates. |
| Reku API docs | Public data versus authenticated/private API distinction | https://api.reku.id/ | This project implements public GET endpoints only. It does not implement order creation, signing, withdrawal, or account operations. |
| OJK PADK No. 3/2026 | Current Indonesian digital asset market regulation reference | https://www.ojk.go.id/id/regulasi/Pages/PADK-Nomor-3-Tahun-2026-Penyelenggaraan-Perdagangan-Aset-Keuangan-Digital-termasuk-Aset-Kripto.aspx | Effective 2026-09-01; the project is not legal advice. |
| OJK pocketbook 2.0 | Consumer-risk and crypto ecosystem background | https://ojk.go.id/id/Publikasi/Roadmap-dan-Pedoman/ITSK/Pages/Buku-Saku-Aset-Keuangan-Digital-dan-Aset-Kripto-2.aspx | Published 2026-08-26. |

## Free/open-source infrastructure considered

- Standard-library Python: https://docs.python.org/3/library/
- SQLite: https://sqlite.org/
- GitHub repository for project and audit trail: https://github.com/orionrayy/ai-determinism-engine

## Evidence limitations

- Build environment DNS access failed for `api.reku.id`, so a live endpoint response could not be independently observed during implementation.
- Published legacy schemas are parsed defensively. A malformed response or missing SOL market causes a fail-closed result.
- This source register does not claim strategy profitability, latency guarantees, or API/service SLA.