# SOL Sentinel — forward-paper operations v1.3

## Purpose and operating boundary

This is the operating guide for the persistent INDODAX SOL/IDR forward-paper pipeline. It starts research as soon as the engineering gates pass; there is no 30-day activation delay.

The worker uses only unauthenticated public market-data GET requests. It has no account login, exchange order endpoint, API-key handling, transaction signer, wallet, deposit, withdrawal, transfer, or live-order route. Every execution receipt is simulated against captured public order-book liquidity.

Keep three decisions separate:

- **Engineering readiness**: the data, strategy, risk, paper execution, persistence, recovery, and monitoring path work and pass tests.
- **Research evidence readiness**: enough independent observations and trades exist to estimate performance with useful uncertainty. This can remain insufficient while the paper process continues.
- **Live authorization**: a separate future review with explicit, scoped user approval. This code does not implement live execution.

## Integrated cycle

`forward_paper.py` combines the existing project modules:

1. `IndodaxPublicMarketDataAdapter` reads the live SOL/IDR pair specification and completed native 1-minute IDR candles.
2. `indodax_microstructure.DepthClient` collects public order-book depth; `DepthStore` records the observation, levels, latency, and request result.
3. The existing `paper_engine.signal_at`, `aggregate_bars`, and indicator routines evaluate the frozen 1m/5m/15m strategy using complete, chronologically aligned candles.
4. `RiskSupervisor` marks the IDR paper portfolio, checks daily loss, drawdown, exposure, open orders, spread, depth age, venue minimums, visible liquidity, and order count.
5. `depth_execution.SQLitePaperExecutionLedger` simulates Decimal multi-level fills, VWAP, fee accounting, partial fills, GTC residuals, IOC policy support, and durable order events.
6. Strategy position metadata, signal decisions, execution cycles, daily risk, and control state are stored in the same SQLite database.
7. Each completed cycle performs ledger/portfolio reconciliation and a WAL checkpoint. An inconsistent replay changes the operating state to manual review.

The cycle is bounded to one observation. A scheduler invokes one cycle at a time; it does not leave an unbounded Python process running inside GitHub Actions.

## Activation and scheduling

The implementation branch is `feat/sol-sentinel-paper-trader-v1`. After its test workflow passes, the push-triggered `forward-paper-activation` job attempts one bounded live-data paper cycle. The initial durable database is stored on the dedicated `sol-sentinel-paper-state` branch.

The separate `.github/workflows/sol-sentinel-forward-paper.yml` workflow is configured for one cycle every five minutes and a manual `run`, `halt`, or `resume` operation. GitHub only schedules workflows present on the repository's default branch. Therefore the recurring schedule becomes available after the reviewed implementation/workflow changes are merged to `main`; the current draft PR is not described as already scheduled. The initial push-triggered activation is independent of that recurring schedule.

Workflow concurrency is serialized under `sol-sentinel-forward-paper-state`, because both operations use the same SQLite checkpoint. Each invocation restores the last SQLite file, runs exactly one bounded cycle, verifies SQLite integrity, and commits the checkpoint back to the state branch. A failed network cycle still persists any collector-attempt diagnostics when a database exists. A successful activation record is written only after the cycle and portfolio reconciliation complete.

## Local execution

From the repository root with Python 3.10+ (stdlib only):

```sh
python -m py_compile projects/sol-sentinel/forward_paper.py
python -m unittest discover -s projects/sol-sentinel -p 'test_*.py' -v

python projects/sol-sentinel/forward_paper.py \
  --db-path ~/.sol-sentinel/forward-paper.sqlite3 \
  --cycles 1 --interval-seconds 0 --json

# Bounded multi-cycle observation; 100 cycles maximum per invocation
python projects/sol-sentinel/forward_paper.py \
  --db-path ~/.sol-sentinel/forward-paper.sqlite3 \
  --cycles 12 --interval-seconds 300 --json
```

The first run initializes the simulated portfolio with IDR 300,000 unless a pre-existing database is supplied. The database records a configuration hash. A later configuration change against the same database fails closed instead of resetting the portfolio. Back up the SQLite database before a deliberate new experiment; use a separate database path for an intentionally fresh simulation.

Persistent operator controls:

```sh
python projects/sol-sentinel/forward_paper.py \
  --db-path ~/.sol-sentinel/forward-paper.sqlite3 \
  --halt --halt-reason "operator requested paper halt" --json

python projects/sol-sentinel/forward_paper.py \
  --db-path ~/.sol-sentinel/forward-paper.sqlite3 \
  --resume --resume-reason "reviewed ledger and risk state" --json
```

A resume requires a consistent fill/portfolio ledger and an explicit reason. Manual-review state must be reconciled before any resumption. Never delete or recreate an existing state file to clear a halt.

## Default paper risk and cost assumptions

- Initial simulated cash: IDR 300,000; no initial SOL inventory.
- Entry risk budget: 0.5% of marked equity.
- Maximum marked SOL exposure: 50% of equity.
- Daily loss circuit breaker: 3% of UTC-day opening equity.
- Maximum peak-to-trough drawdown: 10%; warning threshold: 2%.
- Maximum open orders: 1; maximum daily entry intents: 10.
- Maximum entry spread: 50 bps; maximum depth quote age: 10 seconds; maximum latest candle age: 120 seconds.
- INDODAX Pro official fee components are encoded as 20.555 bps buy and 41.555 bps sell; the user's account-specific trading mode and exact fee display are not verified by CI.
- Slippage budget is 10 bps per side for conservative risk sizing. It is an uncalibrated assumption, not an added synthetic fill or measured slippage result. Simulated fill prices come from the captured book walk, including the price impact visible in those levels.
- Direct Pro minimum notional is modeled at IDR 25,000. Confirm current mode and order preview before interpreting a personal trading route.

All of these are paper assumptions. They are versioned through the configuration hash. They do not set or authorize real orders.

## Fail-closed and recovery behavior

A missing/ambiguous pair, invalid candle, timestamp gap, stale candle/depth, crossed book, invalid quantity, future timestamp, excessive entry spread, insufficient liquidity, minimum-notional failure, risk halt, or SQLite inconsistency blocks new entries. A protective simulated SELL residual may continue during a halt so a stop or portfolio close is not trapped behind the entry circuit breaker. BUY residuals are cancelled when the circuit breaker or spread entry gate fires.

Signal and order IDs are deterministic and durable. An order intent is written before it is submitted to the local paper ledger. On restart, the worker searches the ledger by its client order ID and either attaches the pre-existing order, recreates an unexpired missing order, or marks an expired unsubmitted intent as closed. The order ledger is the accounting authority; forward state is reconciled against immutable fill rows.

The first cycle may correctly report `NO_TRADE`. An active forward-paper system does not require a fabricated signal or fill. Research evidence remains insufficient until the actual trade and observation sample supports estimation; the worker does not wait 30 days before starting.

## Validation boundaries

GitHub Actions tests and the Termux Doctor run on an Ubuntu runner. They do not prove compatibility on the user's Android handset. Run the Termux Doctor directly on the device before claiming physical validation:

```sh
python projects/sol-sentinel/termux_doctor.py --json
```

The collector's spread/depth summary only describes snapshots actually captured after activation. It is not backfilled from OHLC candles. Queue position, maker fill probability, true slippage distribution, and exchange account fee tier remain uncalibrated/unverified.

The current program is forward-paper only. A green software test, successful activation record, or positive simulated P&L is not permission to deploy real money.
