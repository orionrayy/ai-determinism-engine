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
