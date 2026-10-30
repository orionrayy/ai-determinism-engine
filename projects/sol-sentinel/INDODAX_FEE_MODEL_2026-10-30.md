# INDODAX SOL/IDR — fee and venue constraints audit (2026-10-30)

## Scope

This file records the official public fee schedule used for the 30 October native SOL/IDR research replay. It is a Pro-mode sensitivity input, not a statement about the user's account. No login, account fee page, order preview, or private trading API was accessed.

## Official Pro all-in fees

Official INDODAX help material states the spot-fee change is effective 1 October 2026. The Pro IDR-market all-in fee table reports:

| Side | Maker | Taker |
|---|---:|---:|
| Buy | 0.10555% (10.555 bps) | 0.20555% (20.555 bps) |
| Sell | 0.31555% (31.555 bps) | 0.41555% (41.555 bps) |

The all-in table includes the CFX component (0.00555% per side on the IDR market) and the sell-side PPh component (0.21%). Use the all-in rates above as a whole; do not add the tax/CFX components again.

Official source: https://help.indodax.com/hc/id/articles/4416646599705-Rincian-Biaya-Transaksi-di-INDODAX

The later, date-specific INDODAX announcement is authoritative for the 1 October 2026 cutover: https://blog.indodax.com/penurunan-biaya-cfx-1-okt-2026/ . Published 29 September 2026, it states that the IDR-market CFX fee is 0.00555% per side from 1 October 2026 (replacing 0.0111%). The static screenshot embedded in the general help page is timestamped 28 September 2026 and still shows the pre-cutover 0.0111% IDR CFX value; do not use that image as the post-1-October fee schedule. The all-in fee values above use the post-cutover CFX component, published service-fee components, and the sell-side 0.21% tax. The account's own Profile → Settings → Trading Fees display remains the final account-specific check.

## Minimum order and pair filters

The same official help page states:
- Lite minimum order: Rp10,000.
- Direct Pro minimum order: Rp25,000.
- Pro-interface orders from Rp10,000 through Rp24,999 are routed through Lite.

The public pair metadata captured by workflow #40993527618 reported the following SOL/IDR filters at capture time:
- price_tick: Rp1
- quantity_step: 0.00000001 SOL
- api_min_notional_quote: Rp10,000
- minimum_base_quantity: 0.00576948 SOL

The prior fixed dataset artifact #12931446637 had the same tick/step and a historical minimum_base_quantity of 0.00563136 SOL. These are time-varying public pair metadata. The public API notional floor is not interchangeable with the selected app mode's effective minimum order. The replay explicitly selected the Pro direct minimum of Rp25,000.

Lite-specific all-in fee details were not confidently established from the public fee table used here. Do not reuse Pro rates for Lite or infer Lite fees from Reku. Before configuring a particular account, verify Profile → Trading Fees and the live order preview for the selected mode.

## Cost decomposition used in offline replay

The backtest computes the conservative round-trip friction gate as:

entry_fee_bps + exit_fee_bps + spread_bps_round_trip + 2 × slippage_bps_per_side

For the locked historical native dataset (SHA-256 c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30), workflow #40993527618 tested:

| Scenario | Entry / exit fee | Spread input | Slippage | Modeled round-trip cost |
|---|---:|---:|---:|---:|
| Pro taker, observed snapshot | 20.555 / 41.555 bps | 15.799671 bps | 5 bps/side | 87.909671 bps |
| Pro taker, stress | 20.555 / 41.555 bps | 25.6669 bps | 10 bps/side | 107.7769 bps |
| Pro maker, observed snapshot | 10.555 / 31.555 bps | 15.799671 bps | 5 bps/side | 67.909671 bps |

These are modeled sensitivities, not observed realized costs. The maker case does not model queue priority, post-only acceptance, partial fills, or missed fills. A limit order that crosses the book may execute as taker. Slippage values are declared assumptions and were not calibrated from fill data.

The source snapshot at the locked dataset was bid Rp1,777,113 / ask Rp1,779,923, spread 15.799671 bps at 2026-10-29T20:37:47Z. A prior point was about 25.6669 bps. The next current-run capture (workflow #40993527618) observed bid Rp1,732,501 / ask Rp1,732,825, spread 1.869954 bps at 2026-10-30T02:48:55Z. These sparse observations cannot establish a representative spread distribution, regime dependence, depth impact, or slippage. The latest spread must not be extrapolated across the historical period.

## Transfer and withdrawal evidence limits

INDODAX's public help instructions require the user to select the asset and network in the authenticated wallet flow; the interface displays the network-specific withdrawal fee after network selection. The amount, minimum withdrawal, and destination minimum are not certified by this repository's public market-data capture. Never hardcode a SOL withdrawal fee or assume that two exchanges accept identical network labels.

Before a future, separately authorized route assessment, both source and destination must independently confirm:
1. the exact asset/mint and Solana network;
2. withdrawal and deposit availability for the account/region;
3. minimum withdrawal, network fee, destination minimum, and arrival/confirmation rules;
4. destination custody address/network compatibility and supported trading route;
5. the full round-trip economic cost including off-ramp.

Official network-selection guidance:
- https://help.indodax.com/hc/id/
- https://help.indodax.com/hc/id/articles/4416646599705-Rincian-Biaya-Transaksi-di-INDODAX

No real purchase, deposit, withdrawal, transfer, wallet signing, or order was performed in this audit.

## Current decision

The official Pro fee schedule is adequate to run clearly labelled Pro-only offline sensitivity cases. It is not sufficient to certify user-account costs, Lite mode economics, executable spread/slippage, transfer routes, or order fills. Keep the mode at RESEARCH_ONLY / INSUFFICIENT_EVIDENCE.


## Follow-up same-run native replay — workflow #40994319663

A recurring workflow revision now evaluates the native INDODAX CSV captured during the same workflow run. It no longer downloads the short-retention artifact #12931446637 by hardcoded run ID. The current-run capture was verified with:

- Dataset SHA-256: 909b24977537110d2bd0ca114d5715d29646eb36b6ed9a268bf9470b64a05c59
- 43,200 contiguous 1m bars; 2026-09-30T02:54:00Z through 2026-10-30T02:53:59.999Z
- Venue/pair/currency: INDODAX / SOL/IDR / IDR
- FX conversion: false
- Fresh public depth snapshot at 2026-10-30T02:55:48.533456Z: bid Rp1,729,198 / ask Rp1,729,703; spread 2.920003 bps
- API pair filters: price tick Rp1, quantity step 1e-8 SOL, API notional Rp10,000, minimum base quantity 0.00578134 SOL

The same fixed strategy and chronology split were evaluated without retuning:

| Scenario | Fee total | Spread input | Slippage assumption | Total modeled friction |
|---|---:|---:|---:|---:|
| Pro taker, current-run snapshot | 62.110 bps | 2.920003 bps | 5 bps/side | 75.030003 bps |
| Pro taker, stress | 62.110 bps | 25.6669 bps | 10 bps/side | 107.7769 bps |
| Pro maker, current-run snapshot | 42.110 bps | 2.920003 bps | 5 bps/side | 55.030003 bps |

Results:
- Taker current-snapshot and taker stress: zero candidates, fills and closed trades in development, validation and out-of-sample.
- Maker sensitivity: one development-only candidate/fill/trade, zero in validation and zero in out-of-sample.
- The single development trade (2026-10-15 14:57:00Z to 15:05:59.999Z, 9 one-minute bars) reported net P&L +Rp466.818, all-in exchange fees Rp574.698, and execution friction Rp175.991. The segment had one winning trade, so its 100% win rate and infinite profit factor are not statistically informative.
- The maker fee sensitivity does not establish maker execution. The simulation has no queue priority, post-only rejection, fill probability, or maker-vs-taker classification from actual orders. It applies one point spread snapshot across historical bars; that is a sensitivity, not a historically measured cost curve.
- No scenario produced an out-of-sample trade. Expectancy, win rate and profit factor remain undefined for OOS.

This does not satisfy strategy acceptance criteria. Preserve RESEARCH_ONLY / INSUFFICIENT_EVIDENCE.

## Newest rolling-window replay — workflow #40994685989

- Research run #40994685989 completed successfully on 2026-10-30, artifact #12951295388. Native dataset SHA-256 d76434d4c9019249e924e72c5b05d0b200c0dfe3c4709c02a6d821fa0b22327c; 43,200 contiguous 1m SOL/IDR candles from 2026-09-30T03:01:00Z to 2026-10-30T03:00:59.999Z, no FX conversion.
- Public quote snapshot 2026-10-30T03:02:06.143046Z: bid Rp1,737,289; ask Rp1,739,906; mid Rp1,738,597.5; spread 15.0523626084 bps. Public metadata: price tick Rp1, quantity step 1e-8 SOL, API notional floor Rp10,000, minimum base quantity 0.00576894 SOL.
- Replayed friction totals using this capture: Pro taker current snapshot 87.162363 bps; Pro taker stress 107.7769 bps; Pro maker current snapshot 67.162363 bps. This is modeled round-trip cost, not observed realized cost.
- All three scenarios had zero signal candidates, order attempts, fills and closed trades across development, validation and out-of-sample. P&L zero is due to no positions. Expectancy, win rate and profit factor remain undefined.
- The prior run #40994319663 had a single development-only maker-sensitivity trade with +Rp466.818 net, but the newest rolling window did not reproduce it. This variability reinforces that no edge has been demonstrated.
- Current-window overlapping signal-rejection reason events on taker baseline: breakout 40,506; volume confirmation 35,873; 5m momentum 28,306; 15m trend 25,981; 5m RSI 23,415; 1m exhaustion/missing 3,135; cost gate 143. These counts are not distinct bars.
- Current decision: the Pro fee schedule remains an explicit research input only. Spread is represented by isolated point quotes; slippage remains assumed. Maker execution, queue priority, actual account-specific fee mode and OOS strategy edge remain unvalidated.