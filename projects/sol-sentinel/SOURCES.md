# SOL Sentinel — Sources and Tool Inventory

Verification date: 2026-10-29 (Asia/Jakarta). Re-check before any live use; provider pricing, APIs, quotas, regulations, and terms can change.

## Market data

### Binance Spot public market data
- Docs: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
- Public klines endpoint: `GET /api/v3/klines`.
- Use: reference SOLUSDT candle feed at 1m/5m/15m without account credentials.
- Caveat: Binance spot data are not the executable Reku SOL/IDR price. Regional availability and endpoint uptime must be checked on the user's device. Respect documented rate limits.

### DEX Screener
- API reference: https://docs.dexscreener.com/api/reference
- Use: secondary pool/pair/volume/liquidity cross-check for Solana tokens.
- Current docs report 300 requests/min for pair lookup, pair search, and token-pair lookup; some profile/boost endpoints use 60 requests/min.
- Caveat: DEX prices are not Reku prices and pools differ in liquidity, routing, and execution impact.

### Reku
- Status/coin availability: https://reku.id/status
- Current published fee table: https://reku.id/fees/crypto (checked 2026-10-29).
- Current CFX fee adjustment notice (effective 2026-10-01): https://reku.id/information/informasi-penyesuaian-biaya-cfx
- Published Pro Mode fee components: buy 0.1% trading + 0% PPN + 0.00555% CFX = 0.10555% (10.555 bps); sell 0.1% trading + 0.21% PPh + 0.00555% CFX = 0.31555% (31.555 bps). The sum at equal nominal notional is approximately 0.42110% (42.11 bps) before spread/slippage and subject to the mode and current order preview.
- Published minimum order: IDR 30,000 for Pro Mode and IDR 5,000 for Lightning: https://reku.id/help/batas-minimal-rupiah-dalam-order-jual-beli
- Public API endpoint catalogue (historical; treat as unverified until live docs/terms confirm): https://api.reku.id/v2/openapi.json
- User context: Reku KYC is already complete; user reports IDR 300,000 deposited. User screenshot showed a limit-price UI at IDR 1,820,000 and an IDR 150,000 buy size, with estimated 0.08241759 SOL; open order count was 0 at screenshot time.
- Known gap: public market-data APIs have been referenced historically, but a currently supported authenticated order-placement API and permission to automate Reku trades were not verified in this research. Do not automate Reku orders using undocumented endpoints.

## Solana swap / chain components (future phase only)

### Jupiter Developer Platform
- Docs: https://developers.jup.ag/docs
- Swap guide: https://developers.jup.ag/docs/swap/index.md
- Use: quote/route and swap transaction orchestration for Solana self-custody wallet.
- Caveat: route, endpoint version, required API key, quote/execution rate limits, signing format, and API terms must be reverified before implementation. Jupiter swap path is separate from Reku; funds must already be on-chain and the relevant assets/pairs available. On-chain costs still apply.

### Helius
- Pricing: https://www.helius.dev/pricing
- Docs: https://www.helius.dev/docs
- Current advertised free tier: 1M credits/month, 10 requests/sec, 1 sendTransaction/sec, standard WebSocket (subject to plan terms at verification time).
- Use: optional read-only Solana RPC/monitoring fallback.
- Caveat: do not assume free tier has production SLA. The Phase 0 scanner does not require a Solana RPC.

### Solana SDK
- TypeScript SDK: https://github.com/anza-xyz/kit
- Python/Rust toolkit: https://github.com/kevinheavey/solders
- Use: only if a future on-chain adapter requires these dependencies. Phase 0 deliberately stays standard-library Python.

### LiteSVM
- Repository: https://github.com/LiteSVM/litesvm
- Use: optional local Solana program/testing VM in future transaction/program tests.
- Caveat: does not reproduce all real-world Jupiter route liquidity, fees, and network execution conditions.

## Backtesting and runtime

- Python standard library: https://docs.python.org/3/library/
- GitHub Actions docs: https://docs.github.com/actions
- Cloudflare Workers limits: https://developers.cloudflare.com/workers/platform/limits/
- Cloudflare Workers Free currently documents 100,000 requests/day and 10 ms CPU per invocation. Suitable for status/control endpoints, not continuous strategy loops.
- GitHub Actions is for tests and offline/bounded jobs, not a sub-minute trading loop.
- Termux is the intended initial local runtime, subject to Android process suspension, network instability, and battery limits.

## Repository/control plane

- Project repository: https://github.com/orionrayy/ai-determinism-engine
- Canonical orchestration context: https://github.com/orionrayy/ai-determinism-engine/blob/main/ORCHESTRATION_MEMORY.md
- Base commit observed before this work: `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5` (GitHub API tree response at branch setup). The ORCHESTRATION_MEMORY.md baseline may refer to an earlier observed head; always query current refs before modifying.
- Implementation branch: `feat/sol-sentinel-paper-trader-v1`

## Regulatory note

- OJK PADK No. 3 of 2026: https://www.ojk.go.id/id/regulasi/Pages/PADK-Nomor-3-Tahun-2026-Penyelenggaraan-Perdagangan-Aset-Keuangan-Digital-termasuk-Aset-Kripto.aspx
- Verify current Indonesian regulatory, tax, platform, and KYC obligations with authoritative sources. A self-custody wallet or DEX does not imply exemption from applicable law.

## Tools used for this implementation/checkpoint

- GitHub connector: inspected repository tree, README, canonical memory, and current main tree SHA; created the isolated feature branch.
- Web source verification: official Binance, DEX Screener, Jupiter, Helius, Cloudflare, and OJK source pages listed above.
- Runtime implementation: Python stdlib only; no model API, paid data plan, private credential, or financial side effect.


## Automated research workflow

- Workflow file: `.github/workflows/sol-sentinel-research.yml`.
- Trigger: manual `workflow_dispatch` with a bounded 1/3/7/14/30 day window and a daily scheduled run.
- Scheduled run time: 14:15 UTC (21:15 Asia/Jakarta). GitHub Actions requires workflow_dispatch and scheduled workflows to exist on the repository default branch, so merge this branch before the workflow can be manually dispatched or scheduled.
- Actions performed: offline tests, public history capture, offline backtest, fixed-parameter chronological split evaluation, and upload of research outputs for 7 days.
- No credentials or account APIs are used; it never sends orders. The output CSV contains public market data only, but reports should remain research artifacts and must not be interpreted as live account equity.


## FX unit conversion reference (research only)

- Official project/docs: https://frankfurter.dev/
- Endpoint: https://api.frankfurter.dev/v2/rate/USD/IDR
- Access: public, no API key; daily reference rates from central-bank/official sources, not an intraday trading quote. Documentation describes the API as free with no published quota beyond abuse-prevention rate limiting.
- Usage: the history-capture script records a USD/IDR reference rate and date in the local CSV sidecar. The offline simulator converts SOLUSDT prices to IDR before sizing and calculating nominal P&L, because the configured capital is in IDR.
- Important caveat: the system uses USD/IDR as a proxy for USDT/IDR and applies a single daily rate to the sampled window. It does not model the USDT premium/discount, FX changes within the historical window, or Reku SOL/IDR executable prices. This rate is for unit consistency and indicative reporting only, not a transaction rate.
- Attribution/docs: https://frankfurter.dev/ . Check the response date and fail closed on stale/missing rates. A verified manual `--quote-to-idr-rate` may be supplied with a descriptive `--quote-to-idr-source` override.


## Market data endpoint correction — 2026-10-29

The first GitHub Actions capture against `api.binance.com` returned HTTP 451 in the runner. Binance's official Market Data Only documentation lists `https://data-api.binance.vision` for public endpoints including `GET /api/v3/klines`. The collector and scanner now use that documented host. First end-to-end research validation succeeded with a 7-day capture and generated downloadable artifacts; see `EXECUTION_LOG.md` for run IDs and results.
