# SOL Sentinel — infrastructure decision and target economics
**Reviewed:** 2026-10-30 UTC  
**Status:** DECISION FOR RESEARCH ONLY; no live execution authorized  
**Target:** capital reference Rp300,000; aspirational equity target Rp3,500,000

## Decision summary

1. Keep Reku as the currently known candidate fiat on-ramp only, not as the assumed trading execution venue. The repository's existing specification records that the user previously completed Reku KYC; that historical context is not proof of current balance, available withdrawal routes, or current prices.
2. Use a dedicated self-custody Solana wallet only as the intended custody/signing boundary. A wallet by itself is not a market, price feed, or automated strategy.
3. Integrate Jupiter Swap V2 as a **read-only quote adapter first**. The initial implementation at `projects/sol-sentinel/jupiter_quote_probe.py` requests `/swap/v2/order` without a `taker`, does not sign or submit, and rejects unexpected transaction payloads. Never infer execution readiness from quote availability.
4. Do not yet conclude that a DEX route is cheaper than local CEX trading for Rp300,000. Current exact SOL withdrawal fee, withdrawal minimum, SOL-deposit compatibility on return, full Jupiter route cost, and any off-ramp spread are not locked in from one consistent quote set. Until those values are captured and measured, the economic venue decision is **BLOCKED / INSUFFICIENT_EVIDENCE**.
5. Do not add another local exchange adapter for trading without current official documentation proving API access, supported assets, account eligibility, rate limits, and terms. Pintu Pro is documented here as a candidate IDR on-ramp/venue for comparison, not as a verified automated-execution endpoint.
6. Mainnet signing/execution remains disabled. No credentials or wallet secrets belong in this repository, Actions logs, or chat.

## Confirmed source references and limitations

| Topic | Official source and observed evidence | Consequence |
|---|---|---|
| OJK framework / venue list | OJK's currently discovered list: https://ojk.go.id/id/Fungsi-Utama/ITSK/Perizinan-ITSK-Aset-Keuangan-Digital-Aset-Kripto/Pages/Daftar-Penyelenggara-Perdagangan-Aset-Keuangan-Digital-Posisi-22-September-2026.aspx ; PADK No. 3/2026: https://ojk.go.id/id/regulasi/Pages/PADK-Nomor-3-Tahun-2026-Penyelenggaraan-Perdagangan-Aset-Keuangan-Digital-termasuk-Aset-Kripto.aspx | Re-check the latest OJK publication and exact entity name immediately before opening/using an account. This document is not legal advice and does not attest any specific entity beyond the linked regulator record. |
| Reku local IDR on-ramp fees | https://reku.id/fees/crypto ; page lists Pro buy 0.1% trading + 0.00555% CFX and Pro sell 0.1% trading + 0.21% PPh + 0.00555% CFX. Lightning lists zero trading fee plus 0.00555% CFX on buy, and 0.21% PPh + 0.00555% CFX on sell. | Simplified fee components are 0.4211% Pro round trip and 0.2211% Lightning round trip, excluding spread/slippage and all crypto transfers. Check the mode and final order preview each time. |
| Reku external withdrawal | Same https://reku.id/fees/crypto indicates external wallet transfer is charged a variable network fee; the page extraction used for this audit did not verify a current SOL-specific amount or minimum. | Treat the SOL withdrawal cost/minimum as unknown until captured from the current official UI or written official support response. Do not invent a number. |
| Pintu Pro fees & limits | https://www.pintu.co.id/en/faq/spot-trading-at-pintu-pro (fee schedule starts 2026-10-01) lists maker 0.12%, taker 0.17%, 0% buy tax, 0.21% sell PPh 22, and 0.00555% CFX each side. SOL minimum shown as Rp10,000. External withdrawal fee remains asset/network specific and must be checked in the app; https://www.pintu.co.id/limit-and-fees?tab=fees | Simplified maker round-trip fee components are 0.4611%; taker round-trip 0.5611%, before spread/slippage/transfer. API trading support was not verified in official current docs in this audit; do not treat Pintu as an automated venue. |
| Jupiter Swap V2 | https://developers.jup.ag/docs/swap/order-and-execute ; API plans https://developers.jup.ag/pricing ; platform changelog https://developers.jup.ag/changelog/developer-platform | `/swap/v2/order` supports inputMint, outputMint, amount; omit `taker` for quote-only. The free developer plan is listed at $0 with 1 request/second; keyless access is available at lower rate limits (verify current docs). The order is a quote, not guaranteed fill. |
| Solana network fees | https://solana.com/docs/core/fees and https://solana.com/docs/core/fees/fee-structure | Current base fee is 5,000 lamports/signature; priority fee may apply. Fees are charged even when a transaction fails. This is a base-fee reference, not a complete swap-cost estimate. |
| Phantom app UI | https://help.phantom.com/articles/5985106844435 | Phantom help indicates an app swap fee of 0.85% on most swaps (actual route/app conditions may vary). Use as wallet interface comparison only; do not assume UI swaps are free or use its displayed fee as the direct Jupiter API route fee. |

## Cost model for Rp300,000 reference capital

The computations below use source-published fee components and the explicit simplification that buy and sell notionals are both Rp300,000. They are not complete round-trip estimates.

| Route/component | Simplified fee | Nominal cost on Rp300,000 | Excludes |
|---|---:|---:|---|
| Reku Pro buy + sell | 0.4211% | Rp1,263.30 | Spread, slippage, SOL transfer out/in, price impact, and taxes/fees that may vary by order mode/current rules |
| Reku Lightning buy + sell | 0.2211% | Rp663.30 | Spread, slippage, SOL transfer out/in, price impact |
| Pintu Pro maker buy + sell | 0.4611% | Rp1,383.30 | Spread, slippage, withdrawal/deposit, any extra route cost |
| Pintu Pro taker buy + sell | 0.5611% | Rp1,683.30 | Spread, slippage, withdrawal/deposit, any extra route cost |
| Solana base tx fee | 5,000 lamports/signature baseline | Not translated to IDR here | Priority fee, number of signatures, quote/swap platform fee, rent/ATA costs if relevant, failed tx fees, and exchange withdrawal charge |

Do not simply add CEX round-trip fee plus all on-chain swap fee as though every path has identical notional. Build a route ledger with an explicit sequence: IDR -> SOL on-ramp; SOL withdrawal; optional SOL/USDC swaps; network and any priority costs; transfer to a supported receiving exchange; asset sale; IDR withdrawal. Avoid repeated exchange-to-wallet transfers and small repeated swaps if fixed withdrawal costs dominate. Before a transfer out, verify the destination supports the exact asset and Solana network; before a return transfer, verify the receiving exchange supports the exact deposit network/address/memo requirements. Do not send a test transfer larger than necessary, and do not transfer until the user separately authorizes it.

At Rp300,000, the on-chain route cannot be declared cheaper until the current official SOL withdrawal fee/minimum and exit path are known. If the withdrawal fee is a material percentage of the capital, centralized exchange manual trading or simply remaining in research mode may be more economical than transferring to a self-custody wallet for frequent small trades. Conversely, if verified withdrawal/return costs are acceptable and algorithmic swap capability is the primary requirement, Jupiter plus a dedicated wallet is a feasible architecture to test, still not approved for live operation.

## Readiness and evaluation gates (pre-registered; not a return promise)

- **Gate 0 — research data integrity:** exact data-source URL/venue, fetch time, SHA-256, timezone, completed candles only, no gaps/duplicates, validation of OHLCV and temporal alignment. Any missing or stale required input fails closed.
- **Gate 1 — deterministic strategy baseline:** reproduce the frozen version; report all rejection counts and overlapping failure combinations, not just top counters. Add ablation tests one filter at a time in development data only. Never tune thresholds against the final OOS segment.
- **Gate 2 — net-cost strategy evidence:** hold parameters fixed across chronological 60/20/20 segments. OOS must have adequate trade count (pre-register at least 100 closed simulated trades before making a profitability claim), positive net expectancy after actual route costs with a 95% bootstrap lower confidence bound above zero, profit factor >= 1.2, and max drawdown within a pre-set risk budget. A smaller sample or a zero-trade segment means **INSUFFICIENT_EVIDENCE**, not a pass. These thresholds are engineering guardrails, not guarantees of future profit.
- **Gate 3 — quote quality:** collect synchronized on-chain quote snapshots during varied market regimes, label quote age, requested raw units, expected/minimum output, price impact, fees and quote errors. Confirm that the quote fields are consistent and capture actual network-fee/ATA requirements separately. Do not treat a read-only quote probe as a fill.
- **Gate 4 — forward paper:** run for >=30 calendar days with actual, timestamped quotes; record hypothetical missed/partial/failed fill effects conservatively and reconcile each paper trade. Require zero ledger-integrity errors, idempotent replay, and risk limits enforced. If fewer than 100 valid closed trades accrue, remain insufficient evidence and extend the sample rather than fabricate a pass.
- **Gate 5 — operational safety review:** separate wallet key management from GitHub Actions; least privilege; isolated wallet; no withdrawal permission on exchange API keys if an official exchange key is eventually used; signer access control, slippage/price-impact caps, daily loss lock, max exposure, transaction dedupe, confirmation/reconciliation, stale-quote timeout, rate-limit backoff and kill switch tested.
- **Gate 6 — separately authorized live pilot:** outside the current task; requires an explicit distinct user authorization after Gates 0–5 pass. This record does not authorize it.

Current result: **RESEARCH_ONLY / INSUFFICIENT_EVIDENCE**. Baseline backtest had zero trades in the 60/20/20 segments, so the current strategy has no measured positive expectancy, profit factor, or win rate. A green CI run is code-quality evidence only; it is not strategy or operational readiness.

## Rp300,000 -> Rp3,500,000 equity target

The aspirational ending equity target is 3,500,000 / 300,000 = 11.6667x starting capital. The required cumulative return is 1,066.67% (before external contributions, taxes/fees, and losses). This is a target arithmetic, not a forecast or time estimate.

Drawdown recovery arithmetic at the Rp300,000 starting capital:

| Drawdown | Loss at starting equity | Equity after loss | Gain needed from reduced equity to recover |
|---|---:|---:|---:|
| 10% | Rp30,000 | Rp270,000 | 11.11% |
| 20% | Rp60,000 | Rp240,000 | 25.00% |
| 30% | Rp90,000 | Rp210,000 | 42.86% |
| 50% | Rp150,000 | Rp150,000 | 100.00% |

Risk settings in the research simulator (e.g., 0.5% risk per trade, 50% max exposure, 3% daily lock) are simulated assumptions and do not guarantee those limits can be enforced under real execution. The Rp3.5m target must not be used to justify leverage, unbounded loss, loosening filters without validation, or risking money the user cannot afford to lose.

## Reproducible references and current code boundary

- Existing research audit: `projects/sol-sentinel/RESEARCH_AUDIT_2026-10-30.md`.
- Quote-only probe: `projects/sol-sentinel/jupiter_quote_probe.py`; its unit test: `projects/sol-sentinel/test_jupiter_quote_probe.py`.
- Termux runbook: `projects/sol-sentinel/RUNBOOK_TERMUX.md`.
- CI test workflow is offline unit testing only; it does not call Jupiter with real network data, does not sign transactions, and does not place orders.
- The exact Jupiter quote was not verified from a live API call during this audit. Verify API behavior and current quote fields with a read-only test before relying on the adapter.
- GitHub repository cannot be cloned in the current session environment due DNS resolution failure; reported CI status is from GitHub Actions run records, not local full-suite execution.


## Reku transfer-service status cross-check — 2026-10-30

Additional official availability check found that Reku announced on 2 October 2026 that external crypto send/receive was restored: https://reku.id/en/information/send-receive-crypto-sudah-dapat-dilakukan. Reku separately announced on 2 October 2026 that IDR withdrawals and deposits were reopened: https://reku.id/en/information/layanan-withdrawal-idr-telah-dibuka, and on 3 October reiterated that IDR deposit/withdrawal and crypto send/receive were active: https://reku.id/information/buy-sell-direct-deposit-usd-staking-unstaking-kini-sudah-tersedia.

This means the previously documented end-to-end route is **not blocked by a published blanket suspension** in the latest service updates found. It still does not verify that the user's current account can withdraw SOL, which Solana-network minimum/fee applies, what processing delay applies, or whether the exact destination token/network will be accepted on the return path. The official fees page's parsed table does not expose a current SOL-specific minimum/withdrawal fee in this research snapshot: https://reku.id/fees/crypto. Check the actual official app preview and destination-deposit screen before a separately authorized transfer; no transfer was made here.


## Adapter implementation follow-up — 2026-10-30

- Implementation now exists on PR #178 branch at `70b1734f260202fad70a60b31730c5097fb8bb95`: venue-neutral adapter contracts and mock paper execution plus a read-only INDODAX public pair/depth adapter. This improves portability but does not implement order placement, account authentication or wallet signing.
- Latest tests run #40942942848 passed 116 offline tests. Tests validate mock payloads only. The official INDODAX API PDF reviewed for schema/rate-limit documents `/api/pairs`, `/api/depth/$pair_id`, `/api/server_time`, pair increments/minimums and a public cap of 180 requests/minute. We could not retrieve current `/api/pairs` or depth JSON through the browser connector in this audit; direct public endpoint smoke test is still required before calling the adapter endpoint integration verified.
- Current official INDODAX SOL/IDR page displays about Rp1,822,124 last price and Rp5.9bn 24h IDR volume at the time of the crawl; do not hard-code these. Spread, fee tier, endpoint latency, API trade rules and this user's account/network deposit availability are not yet verified. https://indodax.com/trade/SOLIDR
- Reku remains acquisition/on-ramp only; verify the precise SOL network/withdrawal fee/minimum from the official app before any separately approved transfer. No funds were moved.
- Status remains `RESEARCH_ONLY` / `INSUFFICIENT_EVIDENCE`; research run #27 still shows zero trades in all 60/20/20 splits. `ExecutionPolicy(LIVE)` rejects unconditionally because no live executor exists.


## Latest integration state — native SOL/IDR history verified, fee gates tightened (2026-10-30)

### Source choice

INDODAX is now the leading *candidate* centralized execution venue for future SOL/IDR paper integration because the unauthenticated public pair catalogue and native OHLC-history endpoint returned a coherent, contiguous 30-day result in a GitHub Actions smoke test. The test used run [#40948530609](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40948530609) and observed:
- `data_status=OK`; 43,200 SOL/IDR one-minute candles.
- SHA-256 `4297c660ed1c3485bb5641babb5a286b8738290e88f8bb18c9f58bb903fa687b`.
- Input quote currency IDR; no FX conversion; first open 2026-09-29 20:24:00 UTC; last close 2026-10-29 20:23:59.999 UTC.
- Pair ID `solidr`; public history endpoint `https://indodax.com/tradingview/history_v2`.

This is a real public-data endpoint smoke test, not a test of order placement, account permissions, SOL deposit/withdrawal route, or Android/Termux. The artifact retention is short (7 days), so retain the dataset hash and the run URL in the audit rather than assuming the artifact remains downloadable indefinitely.

### Newly resolved integration bug class

The backtest/evaluator contracts now identify source venue, pair and quote currency. Binance SOLUSDT input remains an explicitly labeled USDT reference transformed with the dated USD/IDR proxy. Native INDODAX SOL/IDR input uses identity rate 1.0 and disables FX conversion; the report identifies the native IDR venue and fee model. CSV sidecar data status/hash are checked before source metadata is accepted. Reku fee sensitivity presets refuse a native IDR source. Native IDR backtests require explicitly supplied fee, spread, slippage and order minimum, plus a fee-source label; non-finite/negative fee assumptions and a zero minimum are rejected.

### Fee decision remains open pending account-specific verification

The snapshot's pair metadata reported IDR 10,000 quote minimum and SOL base minimum 0.00559952, but neither is a complete account mode/fee contract. Official INDODAX help says Lite minimum is IDR 10,000 and direct Pro minimum is IDR 25,000, with Pro interface orders IDR 10,000–24,999 routed via Lite. It also says current exact all-in fee components appear in the account Profile/Trading Fees screen, with the changed fee structure effective 1 October 2026. Before native IDR backtest:
1. Verify account mode and current fee components in the official UI.
2. Choose the applicable mode floor and supply that value explicitly.
3. Measure the live public bid/ask spread over a defined observation window; do not extrapolate one quote as historical spread.
4. Label all unmeasured slippage/latency values as assumptions and test sensitivity without tuning to OOS.

Official references:
- API endpoints/schema/rate: https://indodax.com/downloads/INDODAXCOM-API-DOCUMENTATION.pdf
- Fee and minimum policy: https://help.indodax.com/hc/en-us/articles/4416646599705-Details-of-Transaction-Fees-on-INDODAX
- Pro API permissions: https://help.indodax.com/hc/en-us/articles/5315217255833-How-to-Integrate-the-INDODAX-API

### Economic/strategy status

Latest complete 30-day Binance research run [#40948530609](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40948530609) still produced zero candidates/trades on the baseline and zero trades in each fixed 60/20/20 split. The theoretical Reku-Lightning fee-only floor produced 10 trades and net P&L around IDR -3,893.22 while assuming zero spread and slippage; it is not an executable estimate. Native INDODAX candles were captured in this run but were not backtested with INDODAX-specific fee inputs. No new positive edge claim is established.

### Release gate

The implementation remains `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`. Latest verified unit CI at the associated current feature checkpoint succeeded with 135 offline tests. The new fee-gate patch is triggering another full CI run; update the final audit only after exact-head status is confirmed. No transfers, orders, signatures, credential use, or modifications to `main` were performed. PR #178 and PR #180 remain draft/open/unmerged.


## Final research and input-cost gate reconciliation (2026-10-30 WIB)

The complete research run [#40949409224](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40949409224) completed successfully at engine code SHA `cb3ebab0b22a82f5ed1c9a6e748cfb5930d1439a`; strategy/backtest code is unchanged at current code head `39d093b6e44635431475ea635f018327310793fd`. Exact-head CI [#40949432502](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40949432502) passed 135 offline tests. The newest run #58 [#40949432867](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40949432867) was still in progress at the last check; do not claim it complete before final state/artifacts are available.

The full Binance SOLUSDT dataset had 43,200 1m bars, hash `cb685aa993bbde9e2abec8c9c350e98097ba97d6b7c50fa323de1921ed1a7a47`. Baseline and development/validation/OOS all produced zero candidates/trades. Rejection counts show multiple simultaneous filters (trend, breakout, volume, momentum/RSI) and only 195 estimated-net-edge rejections in the full window; because counts overlap, no single filter is proven to be the sole cause. No strategy threshold was changed to force trades.

The same workflow captured native INDODAX SOL/IDR 1m history: 43,200 contiguous bars, SHA `08f20992fed30c0921dee547e4cb4623c4dbf3b3c74ef732ce895ebc1fcf5633`, status OK, no FX conversion. This is evidence that the public history endpoint works from GitHub Actions. It is not evidence that INDODAX offers the user's account a specific fee tier, that an account can withdraw/deposit on a selected network, or that the strategy is tradable.

The new source-aware engine enforces:
- Binance SOLUSDT inputs are explicitly USDT reference and get a dated USD/IDR proxy.
- Native INDODAX SOL/IDR inputs remain IDR and must use FX identity 1.0.
- Native IDR data cannot use Reku Pro/Lightning sensitivity presets.
- Native IDR backtest requires `indodax-custom`, finite non-negative fee/spread/slippage, a strictly positive mode-specific IDR minimum, and a fee-source label.
- CSV sidecar status and source digest are checked before source metadata is trusted.

Run #58 was re-triggered because the test suite changed after adding the finite fee/minimum regression. It is expected to repeat the current strategy backtest using the same engine logic, not alter the previously observed edge. The decision remains **do not enable live trading** until strategy is validated after costs, account/network/fee facts are verified, Termux tests pass, and separate explicit approval/readiness gates exist.


## Live native-depth evidence — latest successful research run #60 (2026-10-30 WIB)

Run [#40950218320](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950218320) completed successfully on workflow head `89a8d7181954a1524379d3b03fa77e703eb9331d). The current feature branch’s later changes are documentation-only; exact-head test CI [#40950649418](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950649418) passed 137 tests, 0 failures.

The read-only native capture result is:
- 43,200 contiguous 1-minute SOL/IDR candles, 30-day range 2026-09-29 20:36:00 UTC through 2026-10-29 20:35:59.999 UTC.
- Dataset SHA-256 `c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30`.
- Public depth response `market_quote_status=OK`, no error, source INDODAX public `/api/depth/solidr`, request-start timestamp 2026-10-29T20:37:47.062073Z: bid Rp1,777,113; ask Rp1,779,923; mid Rp1,778,518; spread ~15.7997 bps. `orders_enabled=false`.
- Market metadata in that snapshot: price tick IDR 1, quantity increment 1e-8 SOL, market quote minimum IDR 10,000, minimum base quantity 0.00563136 SOL.

An earlier successful depth probe [#40950150440](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950150440) showed a wider spread of ~25.6669 bps two minutes earlier, reinforcing that the quoted spread is a time-stamped observation rather than a stable historical cost. This is live read-only endpoint evidence from GitHub's runner; it is not a fill, authenticated order result, user-account permission, or wallet deposit/withdrawal test.

### Decision

INDODAX is the initial centralized venue adapter candidate for source-aware paper experiments, not a production-approved venue. Native IDR history can now be captured and the public top-of-book endpoint has been smoke-tested. Before backtesting the native series, verify the current fee tier in the account UI, explicitly select the Lite/Pro minimum, and supply documented entry/exit fee plus measured spread and slippage. Keep Reku costs restricted to the Binance SOLUSDT reference scenario. The latest complete strategy result still has zero candidates/trades in every 60/20/20 segment and therefore does not establish positive expectancy.

The optional Reku native activity-history step continues to return HTTP 404 in the successful research run, with the collector marked non-fatal; the Reku public market snapshot itself reports OK. No real order, transfer, signing, private key, or credential was used. The live gate remains closed.


## Latest quote and data snapshot — 2026-10-30 WIB

Research run [#40950218320](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950218320) captured 30 days / 43,200 native INDODAX SOL/IDR candles with SHA-256 `c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30`. The same read-only run recorded bid Rp1,777,113, ask Rp1,779,923 and spread 15.7996714 bps at 2026-10-29 20:37:47 UTC. The prior point observation was ~25.6669 bps. These snapshots show observed spread variation, but are not a time series; do not use a single point as a fixed backtest slippage/spread model.

Native INDODAX prices were not backtested with unverified mode fee inputs. Obtain current fee components from the account's Profile/Trading Fees screen; explicitly select Lite/Pro minimum and provide dated fee and spread/slippage assumptions. The 30-day Binance SOLUSDT reference backtest still shows zero candidates/trades in all 60/20/20 segments. The optional Reku activity-history endpoint remains 404 while its public quote snapshot is available. Keep live execution blocked.
