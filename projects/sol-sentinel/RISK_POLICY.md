# SOL Sentinel — Risk and Safety Policy

Status: Phase 0, read-only. This policy is binding for all later work.

## Capital and target

- Starting cash/fiat balance reported by user: IDR 300,000.
- Aspirational equity target: IDR 3,500,000.
- Target multiple: 3,500,000 / 300,000 = 11.6667x.
- Required profit: IDR 3,200,000, equal to approximately +1,066.7% net return from initial capital.
- The target is not a promised result, not a required daily quota, and not a reason to increase position size or leverage.

## Hard prohibitions in Phase 0

- No live order placement, cancellation, withdrawal, bridging, or wallet transaction.
- No API keys, tokens, passwords, seed phrases, or private keys in source control.
- No undocumented exchange endpoints or attempts to bypass platform restrictions/KYC.
- No martingale, loss-chasing, automatic leverage escalation, or averaging down.
- No blind replay of an uncertain side effect.
- No trading based only on an LLM's confidence or narrative.

## Initial paper-risk parameters

These are starting parameters for simulation, not proven optimal settings:
- Reference capital: IDR 300,000.
- Candidate initial exposure for later manual purchase: no more than IDR 150,000 until costs/withdrawal and signal quality are validated.
- Reserve: IDR 150,000 remains in fiat during initial evaluation.
- Paper risk budget per trade: 0.5% of equity (IDR 1,500 at IDR 300,000), including estimated fees and slippage.
- Paper daily loss lock: 3% of starting daily equity (IDR 9,000 at IDR 300,000).
- No leverage.
- One open paper position at a time until position accounting and fills are tested.

A risk budget is not a guaranteed maximum loss; real slippage and gaps can exceed model assumptions. Parameters must be configurable and recorded.

## Fail-closed conditions

Return NO_TRADE and preserve an audit record if:
- Candle history is insufficient, malformed, out of order, incomplete, or stale.
- Market sources disagree beyond a configured tolerance.
- Quote or order book data are stale, shallow, or unavailable for a future execution path.
- Round-trip estimated costs plus a safety buffer exceed the conservative gross-move estimate.
- Spread, slippage, volatility, or network latency exceed tested bounds.
- daily loss lock, exposure limit, cooldown, or kill switch is active.
- provider limits/terms or availability are unknown.
- transaction status is unknown; reconcile before any next side effect.
- wallet/network/pair/decimals are not independently verified.

## Cost model

Every future result must separate:
1. purchase fee;
2. sale fee;
3. round-trip bid/ask spread cost (modeled separately, as an assumption until quoted spread is measured);
4. slippage / price impact per side;
5. network fee and optional priority fee;
6. transfer/withdrawal fee, if relevant;
7. unrealized mark-to-market versus realized P&L.

Never claim net profitability using gross candle returns alone. The current published Reku Pro fee table is the source for default entry/exit fee components, but mode, applicable taxes, fees, and an actual order preview must be rechecked before use. Spread and slippage remain unmeasured assumptions. Binance candles are not executable Reku SOL/IDR quotes.

## Secret and approval controls

- Never commit private keys or wallet seed phrases.
- Never request that a user paste a private key into chat.
- Keep a separate, low-balance trading wallet if live Solana swaps are eventually approved.
- Use explicit, scoped approvals for live effects. Approval must bind to exact action, parameters, wallet, max value, and expiry.
- Use provider-side transaction/signature lookup and durable intent identity to reconcile timeout cases.
- If execution result is uncertain, block subsequent orders until reconciliation settles the state.
- The kill switch must be independently usable by the user and tested before live deployment.

## Stop conditions

Pause the system if:
- actual fees or slippage materially exceed estimates;
- paper results are negative after costs over the preregistered evaluation window;
- feed fails or source timestamps are stale;
- the process restarts without recovering the position/order ledger consistently;
- any unexplained balance or P&L discrepancy occurs;
- a risk-policy test fails.


## Current offline cost assumptions

- Published Reku Pro fee table checked 2026-10-29: entry/buy 10.555 bps and exit/sell 31.555 bps; approximate fee sum 42.11 bps for equal nominal values. Source: https://reku.id/fees/crypto. Recheck current mode and final order preview before live use.
- Scanner default: fee 42.11 bps round trip (fee reference from Reku Pro), spread 10 bps round trip (assumption), slippage 20 bps round trip (assumption).
- Backtest/evaluator default: entry fee 10.555 bps, exit fee 31.555 bps (published Reku Pro fee components), spread 10 bps round trip (assumption), slippage 10 bps per side (assumption). All are configurable.
- Spread cost is modeled by adverse half-spread adjustment on each side; slippage is a distinct adverse adjustment. Both are included in fill prices and modeled risk sizing. Exchange fees remain separate in the ledger.
- These defaults must be replaced with the actual venue/market measurements before interpreting net expectancy. A wrong cost assumption can flip a small trading edge negative.


## Quote-currency consistency

The history capture is SOLUSDT while the starting capital and nominal P&L are IDR. The paper engine must convert OHLC prices into IDR before position sizing, stop/target arithmetic, and P&L accounting. It reads a dated USD/IDR reference from the capture sidecar or requires an explicit rate override. The reference uses USD as a proxy for USDT and is not the executable Reku SOL/IDR quote. If the rate is absent, invalid, or older than five calendar days, the CLI fails closed rather than silently sizing in mixed currencies. Exact live position sizing must use a verified live IDR price and actual venue rules.
