# SOL Sentinel — Canonical Project Memory

Last updated: 2026-10-29 (Asia/Jakarta)
Repository: https://github.com/orionrayy/ai-determinism-engine
Project path: `projects/sol-sentinel/`
Implementation branch: `feat/sol-sentinel-paper-v1`

## Goal

Build a high-quality, free-first SOL scalping research and automation foundation, readable/auditable by the user's AI orchestration workflow. The user wants to pursue an aspirational portfolio value of Rp3,500,000 from Rp300,000 initial capital. No software can guarantee that outcome, especially over one night. The target is not an execution constraint: the bot must never increase risk, use leverage, or force trades to meet it.

## Known user-provided operating context

- Initial funds available: Rp300,000 deposited to Reku; user showed a SOL purchase screen with a limit price of Rp1,820,000 and a draft order value of Rp150,000 corresponding to 0.08241759 SOL.
- The screenshot showed no open orders; it did not prove that any order had been submitted or filled.
- The user has an existing KYC-verified Reku account and is interested in SOL, automated scalping, free-first infrastructure, and making project state accessible through GitHub.
- The intended runtime constraints are Android/Termux and minimal dependencies; public GitHub repository should not contain personal raw chat logs, account identifiers, secrets, wallet material, or financial private data.

## Decisions

1. Start with SOL/IDR on Reku because the user already has funds there. Do not change assets just to chase a higher return.
2. Use the Reku documented public market-data surface as the initial read-only adapter: `/v2/coins`, `/v2/price`, `/v2/bidask`, `/v2/orderbook`, and `/v2/chart`. Public docs are legacy/old, so current response shapes and availability must be verified in the user's live environment. Fail closed if unavailable or malformed.
3. Default mode is local paper trading. There is no live order submission, authenticated trading API, wallet signer, Jupiter execution, leverage, or automated movement of funds in this project version.
4. Use standard-library Python, SQLite local persistence, deterministic signal gates, explicit fee/slippage assumptions, freshness and spread checks, and a hard daily loss guard.
5. Reserve GitHub for project specification, code, test results, source records, and sanitized decisions. Do not put the complete raw conversation into a public repository.
6. AI/LLM is a supervisor/research assistant. It may review evidence and propose strategy changes; it must not override deterministic risk guards or directly control a private key.
7. The user asked for the project to be prepared until Rp3,500,000 is realized. The project can be prepared and evaluated, but the return cannot be guaranteed and no asset/trade outcome should be claimed until evidenced.

## Architecture

`Reku public market data -> validation/freshness guard -> candle features -> signal eligibility gate -> paper risk engine -> SQLite journal -> JSON stdout/status summaries -> supervised review through ai-determinism-engine`.

Cross-market data, if added later, must be labelled by venue and quote currency; Binance SOL/USDT must never be treated as a Reku SOL/IDR executable quote.

## Starting assumptions — verify, do not treat as fact

- Initial paper equity: Rp300,000.
- Maximum paper position notional: 50% of equity.
- Risk budget: 0.5% of equity per position, including an allowance for stop distance and modeled round-trip costs.
- Daily realized/unrealized loss stop: 3% of day-start equity.
- Reku fee estimates from research may become stale; current app costs override the sample configuration.
- Default slippage, spread, and signal thresholds are conservative test parameters, not calibrated production values and not a promise of performance.

## State and continuation protocol

At each work session:
1. Read this file, `SPEC.md`, `SOURCES.md`, `SECURITY.md`, and the current source/test files.
2. Inspect current Git branch/head and working tree before changing files.
3. Run offline tests first; network checks are separate and explicitly read-only.
4. Record new source checks, configuration changes, bugs, and test outcomes in this folder.
5. Never assert live trading, a filled order, profitability, or target realization without verifiable evidence.
6. Treat any unknown live order outcome as in-doubt; do not retry a financial side effect blindly.

## Sanitization boundary

This file is a project memory, not a transcript archive. It intentionally excludes raw conversation text, private account metadata, tokens, phone numbers, wallet addresses, screenshots, and unrelated user details. Complete private context cannot safely be placed into a public repository.