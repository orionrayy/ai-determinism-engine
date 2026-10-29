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

## INDODAX mode-specific minimum and live fee policy

The official INDODAX help page states that the current all-in fee structure changed effective 1 October 2026; current exact fee components are displayed under the account's Profile/Trading Fees screen. The same help page states a Rp10,000 Lite minimum and Rp25,000 minimum for direct INDODAX Pro orders, with Pro-interface orders from Rp10,000 to Rp24,999 routed via Lite: https://help.indodax.com/hc/en-us/articles/4416646599705-Details-of-Transaction-Fees-on-INDODAX.

`IndodaxPublicMarketDataAdapter.get_market_spec()` reads the exchange's `trade_min_base_currency` market metadata. It accepts optional `mode_minimum_notional_quote` to enforce a separately verified interface-mode floor; e.g. pass `Decimal("25000")` for a research configuration that models direct Pro, or `Decimal("10000")` for Lite. Do not infer or hard-code a user-specific fee tier. The default remains the market metadata value if no mode has been selected. The paper simulation must label the selected mode and fee source explicitly.

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


## Verified native-history smoke test (2026-10-29 UTC)

GitHub Actions research run [#40948205968](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40948205968), at code/workflow commit `e7cc7ab2a810028af5a13d43c011674792a08a57`, made read-only public requests and completed native INDODAX history capture successfully:

- `data_status=OK`; 30 days; exactly 43,200 contiguous 1-minute candles.
- Range: 2026-09-29 20:22:00 UTC through 2026-10-29 20:21:59.999 UTC.
- Venue/pair/currency: INDODAX SOL/IDR; no FX conversion.
- Dataset SHA-256: `9dd63e357e3a8f4ba168bccd65edf3fe621d3758884a97549e8425c812ab2316`.
- Public market filters observed in the contemporaneous `/api/pairs` response: price tick `1` IDR, quantity increment `1e-8` SOL, per-market quote minimum IDR 10,000 and per-market minimum base quantity `0.00559952`. These are point-in-time public metadata; the selected interface mode may apply a higher minimum.
- The artifact also shows the Reku legacy native activity-history endpoint returned HTTP 404 in the same run, while the Reku public market snapshot returned `data_status=OK`. This is a data-history endpoint failure, not proof that Reku SOL withdrawals are unavailable.

The artifact belongs to the run above and has 7-day retention. This smoke test verifies the public endpoint from GitHub's runner, not from the user's Android device and not the correctness/availability of a recipient's deposit network. It was captured before the later native-IDR paper-engine denomination guard was merged into the branch; current code tests cover identity FX conversion and metadata-hash guards. The latest full strategy research result is not updated by this capture alone.

## Native-currency execution gating

The paper engine and chronological evaluator now resolve source venue/pair/quote currency from capture metadata, verify the CSV digest when the sidecar carries one, and normalize native INDODAX IDR candles with identity conversion (rate 1.0). The fee-sensitive backtest refuses native IDR unless `--fee-model indodax-custom` is supplied with explicit entry/exit fee, spread, slippage, minimum-notional and a source label. Reku cost presets reject native INDODAX data. This prevents a data-currency conversion bug and prevents a silent cross-venue cost-model substitution; it does not establish profitable native-market signals.

Example public-data-only capture:

```sh
python projects/sol-sentinel/capture_indodax_history.py --symbol SOL/IDR --days 30 \
  --output ~/.sol-sentinel/INDODAX_SOLIDR_1m.csv \
  --metadata-output ~/.sol-sentinel/INDODAX_SOLIDR_1m.csv.meta.json
```

Do not run a native-IDR backtest until the actual fee model and selected mode's minimum order value have been verified and explicitly supplied. Current trade fee components are displayed in INDODAX Profile/Trading Fees; the generic market metadata minimum (Rp10,000 in the captured snapshot) is not a substitute for the direct-Pro mode minimum (Rp25,000 per current official help, with Lite available at Rp10,000). Source: https://help.indodax.com/hc/en-us/articles/4416646599705-Details-of-Transaction-Fees-on-INDODAX.

