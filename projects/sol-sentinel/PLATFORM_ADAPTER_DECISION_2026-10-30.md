# SOL Sentinel — venue and adapter decision (2026-10-30)

## Decision summary

Reku is the initial SOL/IDR funding venue only. It is not the trading engine's required execution venue. The code now separates market-data, capital-source, custody, transfer planning, order execution, and ledger contracts; the only included order executor is deterministic in-memory paper execution. No wallet signer, withdrawal sender, or live exchange-order implementation exists.

Primary venue candidate for a future controlled paper integration: **INDODAX SOL/IDR Trade API V2**, subject to verifying the current account/network deposit screen and exact order/market filters before implementation. It has a directly quoted IDR market and documents separate View and Spot Trading permissions. Do not grant Deposit & Withdraw permission to an automation key for a trading-only process. This is a candidate, not a finding that transfer routes or personal-account access have already been verified.

Alternative: **Solana wallet + Jupiter Swap V2**. This offers multi-router swap quotes and on-chain execution, but needs isolated custody/signing, native SOL for transaction fees, route/price-impact controls, transaction expiry/reconciliation, and a valid fiat off-ramp. The current Sentinel probe is quote-only; it neither signs nor submits. At the current small reference capital, a Reku withdrawal fee and a later off-ramp/return transfer could dominate economics; exact SOL transfer fee/minimum must be read from the current official app preview and must not be guessed.

## Evidence-based comparison

| Route | Evidence confirmed from official source | Unknown or not certified here | Engineering decision |
|---|---|---|---|
| Reku buy -> keep trading on Reku | Reku publishes Pro/Lightning fee components and says external crypto sends incur variable network fees. https://reku.id/fees/crypto | Public API pages in this code are legacy; an authenticated order API was not verified. Native activity history route returned HTTP 404 in the previous read-only research run. | Keep as on-ramp only; do not make strategy/executor depend on Reku order APIs. |
| Reku -> INDODAX -> SOL/IDR Spot API | Official INDODAX page lists SOL/IDR as an active market. Official Trade API help documents View and Spot Trading permissions and API V2. https://indodax.com/trade/SOLIDR ; https://help.indodax.com/hc/en-us/articles/5315217255833-How-to-Integrate-the-INDODAX-API | Live SOL deposit network, minimums, confirmations, withdrawal fee, current fee tier, precise tick/step/min-notional, account API rate limits, and live quote/order semantics still require current official verification. Public market statistics are volatile and not a spread/depth guarantee. | Best first CEX adapter candidate because SOL/IDR avoids a USDT/USD conversion in the execution loop, and the API separates Spot Trading permission from withdrawals. Begin only with read-only market metadata/quotes and mock orders. |
| Reku -> self-custody Solana wallet -> Jupiter Swap V2 | Jupiter recommends Swap V2 Order & Execute; no taker yields a quote-only response. Current docs: https://developers.jup.ag/docs/swap/order-and-execute . Developer Platform changelog states Free is 1 RPS and keyless is 0.5 RPS: https://developers.jup.ag/changelog/developer-platform . Solana base fee is currently 5,000 lamports/signature plus optional priority fee: https://solana.com/docs/core/fees | Wallet safety/key isolation, route program/token support, exact output and price impact, fee/priority requirements, account rent where applicable, external SOL withdrawal costs, and end-to-end IDR off-ramp are not proven by a quote. API plan limits can change. | Maintain quote-only adapter. Do not build signer or transaction submitter until user separately approves and all readiness gates pass. |
| Binance spot | Current research candles came from public Binance SOLUSDT kline data. | Binance market feed is not proof of Indonesia account access, executable balances, fee tier, transfer route, or the chosen trading pair's current filters. | Research-only data source. Must not be conflated with execution price or venue. |

OJK's official September 2026 update describes Indonesia's licensed crypto-asset provider landscape; listing in a market page is not enough to certify every operational path. See https://ojk.go.id/en/berita-dan-kegiatan/siaran-pers/Pages/OJK-Board-of-Commissioners-Monthly-Meeting-September-2026.aspx . Verify the current legal provider list, exact asset/network availability, and account terms at the time any future deployment is considered.

## Fee facts versus assumptions

Reku's published page lists Pro Mode buy fee 0.1% + 0.00555% CFX, sell fee 0.1% + 0.21% PPh + 0.00555% CFX, and Lightning buy fee 0% + 0.00555% CFX, sell 0% + 0.21% PPh + 0.00555% CFX. External crypto sends incur a variable blockchain fee. The dynamic page did not expose a verifiable SOL-specific withdrawal amount/minimum in this source read. Recheck the current app's SOL withdrawal preview rather than using a hard-coded constant. https://reku.id/fees/crypto

Jupiter/Solana costs cannot be equated to a single 5,000-lamport fee for every route. The Solana base fee is 5,000 lamports per signature; priority fees may be charged, and failed transactions still incur fees. Other token accounts/rent and router/application fees may apply. https://solana.com/docs/core/fees

INDODAX current trade fee, withdrawal fee, API throttles, and market filters are intentionally not copied into defaults because they were not verified against a current official response during this audit. Fetch and record those terms in a read-only preflight before selecting them for the simulator.

## Adapter contracts added

- Market data: canonical symbol + quote identity + UTC timestamp + bid/ask validation. The new `IndodaxPublicMarketDataAdapter` uses the official public `/api/pairs` catalogue to load pair ID, price increment, quantity increment, minimum quote notional and minimum base quantity, then reads `/api/depth/{pair_id}` with bounded timeout, response-size limit, small bounded transient-error retries, and strict book validation. It makes no credentialed request. CI uses fixture transport only; this session did not call the real endpoint from the runner.
- Capital source: metadata contract, not private account data.
- Custody: balance lookup contract separated from the exchange/order API.
- Transfer route planner: validates that source/destination network names match, both ends are verified, and minimum/fee values are known. It never sends a transfer.
- Execution: only deterministic in-memory paper execution is implemented. Uses Decimal arithmetic, venue/ticker identity, quantity steps, price ticks, minimum notional, quote age, spread ceilings, and an idempotency key. A non-marketable limit returns NO_FILL rather than fabricating a fill.
- Ledger: idempotent paper receipt contract rejects conflicting replays.
- Live policy: always rejects LIVE mode in this project because a live executor is not implemented. Even when approval flags are supplied, the gate remains closed.

## Remaining acceptance gates

1. From a read-only INDODAX source, verify SOL/IDR market metadata, bid/ask/depth, server time, lot/tick/min-notional, and endpoint/rate-limit behavior.
2. Confirm on both Reku send preview and INDODAX receive screen that the exact asset/network is Solana; capture current minimum, fee, estimated net amount, and receiving minimum. Do not transfer during this phase.
3. Validate quote freshness and cost inputs under realistic spread/latency scenarios; measure the route's actual observable spread rather than borrowing Reku fees for another exchange.
4. Repair strategy evidence separately: zero trades on the last historical 60/20/20 evaluation means INSUFFICIENT_EVIDENCE. Do not loosen filters to manufacture trades.
5. Run a forward paper campaign only after the selected venue-specific quote adapter and its assumptions are proven; keep live execution disabled.
