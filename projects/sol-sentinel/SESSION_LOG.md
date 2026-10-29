# SOL Sentinel — Session Log / Durable Context

Timezone: Asia/Jakarta. This log contains user-visible project decisions and verified execution checkpoints, not private chain-of-thought.

## 2026-10-29 — Project initiation

### User goal
Build a highly automated, efficient, carefully calibrated SOL scalping project targeting equity of IDR 3,500,000 from reported starting capital IDR 300,000. User requests free-first APIs/open-source projects and that sources, tools, decisions, and project history be stored in GitHub so the assistant can inspect the current implementation in future sessions.

### Decisions
- Keep SOL as the initial candidate because it fits the Solana/Jupiter execution design and has large-market liquidity; do not claim it will achieve the target.
- Target is an aspiration, not a guaranteed outcome. IDR 300,000 to IDR 3,500,000 implies 11.6667x equity and +1,066.7% net return.
- Primary fiat on-ramp is the user's existing verified Reku account; no KYC-bypass route is assumed.
- Do not assume Reku supports official authenticated automated order placement until current official API docs and terms verify it.
- First deliver a read-only market scanner using public market data, followed by strategy research and a cost-aware paper simulator.
- No live trade, withdrawal, wallet creation, or real order has been submitted by the assistant.
- Do not store credentials, private keys, seed phrases, account IDs, or personal screenshots in the public repository.

### Screenshot facts supplied by user
A Reku SOL Pro screen showed: buy side selected; order type Limit Order; limit price IDR 1,820,000; buy amount IDR 150,000; estimated size 0.08241759 SOL; available balance IDR 300,000; order count shown as zero. This is a point-in-time UI snapshot only. It does not prove a buy order was submitted or filled.

### Financial risk design (initial paper assumptions)
- Up to IDR 150,000 may be considered for a future manual initial exposure only after the user checks the live quote/fees; keep the other IDR 150,000 in IDR during early validation.
- Initial simulated risk budget: 0.5% equity per trade, IDR 1,500 when equity is IDR 300,000.
- Initial simulated daily stop: 3% equity, IDR 9,000 at IDR 300,000.
- No leverage, martingale, loss-chasing, automatic position escalation, or averaging down.
- Actual loss may exceed planned risk due to gaps, spread, slippage, platform/network issues, or failed/partial execution.

### Candidate free-first sources
- Binance public klines: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
- DEX Screener API: https://docs.dexscreener.com/api/reference
- Reku status: https://reku.id/status
- Reku historical API spec (must revalidate): https://api.reku.id/v2/openapi.json
- Jupiter docs: https://developers.jup.ag/docs
- Helius free tier: https://www.helius.dev/pricing
- Solana TypeScript SDK: https://github.com/anza-xyz/kit
- Solana Python toolkit: https://github.com/kevinheavey/solders
- LiteSVM: https://github.com/LiteSVM/litesvm
- Cloudflare Workers free limits: https://developers.cloudflare.com/workers/platform/limits/
- OJK PADK No. 3/2026: https://www.ojk.go.id/id/regulasi/Pages/PADK-Nomor-3-Tahun-2026-Penyelenggaraan-Perdagangan-Aset-Keuangan-Digital-termasuk-Aset-Kripto.aspx

### Repository checkpoint
- Repository: https://github.com/orionrayy/ai-determinism-engine
- Main tree SHA observed: eaa864a554c2c995ce73bd9f6b4d07206ecc56c5
- Branch created: feat/sol-sentinel-paper-trader-v1
- No changes to main branch in this checkpoint.
- Planned artifacts: project spec, risk policy, sources, this session log, Termux runbook, scanner, offline tests, CI test workflow.

## Current implementation state

This checkpoint's goal is the first read-only scanner release. Do not label it as profitable, calibrated, or production ready.

### Next work items
1. Verify branch and all files through GitHub API after creation.
2. Check CI workflow syntax and run unit tests; fix failures before reporting success.
3. Validate public API reachability from Termux; environment may behave differently from CI.
4. Add structured historical capture/replay and paper ledger with realistic Reku fee/spread assumptions.
5. Research and confirm the official market-data adapter for Reku, current fee schedule, withdrawal fee and minimum amount.
6. Before any live integration, require explicit user approval and a separate security review.
