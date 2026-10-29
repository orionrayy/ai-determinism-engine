# SOL Sentinel

SOL Sentinel is a free-first, standard-library Python market scanner and paper-trading simulator for **SOL/IDR on Reku**. It is designed for Termux/Android and for supervised integration with `orionrayy/ai-determinism-engine`.

> **Safety boundary:** this release is read-only. It never places live orders, signs transactions, requests Reku credentials, connects to a wallet, or moves funds. The Rp3,500,000 target is an aspirational objective, not a promised or enforceable return. A system must not increase risk merely because the target has not been reached.

## What is included

- Legacy public Reku market-data adapter for coin metadata, prices, bid/ask, order book, and 5-minute candles.
- Data validation, age checks, spread checks, technical features (EMA 9/21, RSI 14, ATR 14, relative volume, recent breakout).
- Fee-aware paper simulator with position limits, risk budget, stop/target, daily loss guard, SQLite journal, and duplicate-entry guard.
- Standard-library-only unit tests; no API key or package installation required.
- Source register, architecture, security boundaries, Termux runbook, and sanitized project memory.

The Reku public API documentation currently discoverable at https://api.reku.id/ is dated/legacy documentation. The first live run must verify endpoint response shapes. If the API fails, the scanner fails closed and records an error; it does not silently fabricate a price or accept stale candles.

## Quick start (Termux)

```sh
pkg update
pkg install python
git clone https://github.com/orionrayy/ai-determinism-engine.git
cd ai-determinism-engine
git fetch origin feat/sol-sentinel-paper-v1
git checkout feat/sol-sentinel-paper-v1
cd projects/sol-sentinel
python -m unittest -v
python sol_sentinel.py --doctor --once
python sol_sentinel.py --once
python sol_sentinel.py --interval 30
```

Stop a continuous run with Ctrl+C. The default database is `data/sol_sentinel.sqlite3`; use `--db /path/to/file.sqlite3` to change it. Run the live data check before leaving the process unattended. Android battery optimization, network loss, API changes, and process termination are not handled by pretending uptime is guaranteed.

## Default operational posture

- Market data: public Reku endpoints; unauthenticated.
- Trading mode: paper only.
- Initial paper capital: Rp300,000.
- Maximum notional per paper position: 50% of paper equity.
- Risk budget per trade: 0.5% of paper equity before modelled execution costs are added.
- Daily paper loss guard: 3% of day-start equity.
- Leverage, martingale, averaging down, and live-order APIs: absent.
- All fee and slippage assumptions are illustrative configuration inputs and must be checked against the Reku app before interpreting paper P&L.

## Exit criteria before any live-trading project

Do not add a live executor merely because the scanner runs. First require a validated current API contract and explicitly permitted execution path; out-of-sample backtest and paper-trading evidence; measured spread, fees, slippage, and failed-fill behavior; independent review of secret handling; tested idempotency/reconciliation; and a separate explicit user approval. A future executor must be a separately reviewed module with independent risk limits.

## Project layout

- `sol_sentinel.py` — data adapter, indicator calculations, signal gate, paper simulator, and CLI.
- `tests/test_sol_sentinel.py` — deterministic offline tests.
- `config.example.json` — editable non-secret assumptions.
- `PROJECT_MEMORY.md` — sanitized durable record of goals and decisions.
- `SOURCES.md` — source register with confidence notes.
- `SECURITY.md` — safety and privacy boundaries.
- `RUNBOOK_TERMUX.md` — install, run, troubleshoot, and export instructions.
- `SPEC.md` — behavior contract and validation gates.

This folder is additive and does not alter the orchestrator's runtime behavior.