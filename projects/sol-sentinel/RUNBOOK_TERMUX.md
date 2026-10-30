# SOL Sentinel — Termux Runbook

## Requirements

- Android + Termux.
- Python 3.10+ available as `python`.
- Network access to Binance public API.
- No pip package or API key required for Phase 0.

## Install and run

Clone or fetch the feature branch, then from repository root:

```sh
git fetch origin
git checkout feat/sol-sentinel-paper-trader-v1
python --version
python -m unittest discover -s projects/sol-sentinel -p 'test_*.py' -v
python projects/sol-sentinel/sol_sentinel.py --json
python projects/sol-sentinel/capture_history.py --symbol SOLUSDT --interval 1m --days 3 --output ~/.sol-sentinel/SOLUSDT_1m.csv
python projects/sol-sentinel/paper_engine.py --csv ~/.sol-sentinel/SOLUSDT_1m.csv --report-path ~/.sol-sentinel/backtest.json
python projects/sol-sentinel/evaluate_splits.py --csv ~/.sol-sentinel/SOLUSDT_1m.csv --output ~/.sol-sentinel/evaluation.json
python projects/sol-sentinel/cost_sensitivity.py --csv ~/.sol-sentinel/SOLUSDT_1m.csv --output ~/.sol-sentinel/cost_sensitivity.json
python projects/sol-sentinel/reku_market.py --symbol SOL --binance-csv ~/.sol-sentinel/SOLUSDT_1m.csv --output ~/.sol-sentinel/reku_sol_idr_market.json
```

To run a continuous scanner while the terminal stays active:

```sh
python projects/sol-sentinel/sol_sentinel.py --watch-seconds 30 --log-path sol_sentinel_runs.jsonl
```

Use Termux wake-lock if available and keep the device on a stable connection. Android may still kill processes; no always-on guarantee is implied. To stop watch mode press Ctrl+C.

## Parameters

- `--symbol SOLUSDT`: public Binance spot market reference.
- `--fee-bps-round-trip 42.11`: fee reference based on the current published Reku Pro buy/sell fee components (https://reku.id/fees/crypto; checked 2026-10-29). Actual fee treatment depends on mode/order and must be checked in the order preview; spread/slippage are separate assumptions.
- `--slippage-bps-round-trip 20`: initial slippage allowance, not a measured estimate.
- `--min-net-edge-bps 25`: minimum net edge buffer used for the candidate filter.
- `--log-path FILE`: append scan results as JSON Lines.
- `--json`: output structured JSON.
- `--watch-seconds 30`: repeat at a safe interval (minimum 15 seconds).

Do not set fees to zero to force a signal. If changing fees, store the reason and current source in the research log.

## Interpret the output

- `NO_TRADE`: a valid risk-managed result; inspect reasons.
- `PAPER_CANDIDATE`: a rule-based observation eligible for paper testing only; this is not a buy order or recommendation.
- `data_status=STALE/INSUFFICIENT/ERROR`: no signal may be acted on.
- Cost estimate is a sensitivity input; Binance data are not Reku executable prices.
- Every scan is a point-in-time observation, not an account balance or realized P&L.

## Troubleshooting

- HTTP errors/timeouts: retry later, do not loop aggressively; the worker uses bounded timeouts.
- Unsupported endpoint/region: scanner exits with a clear error; validate public API availability from this device and research a documented alternate source.
- Clock skew: enable Android automatic time.
- Android process stopped: restart manually and inspect the last JSONL record.
- Unexpected signal: inspect the JSON metrics and replay that exact data offline before changing any parameters.


## Historical capture and offline backtest

1. Capture up to 30 days of public Binance 1-minute candles (bounded to 60 pages and bounded retries):
   `python projects/sol-sentinel/capture_history.py --symbol SOLUSDT --interval 1m --days 3 --output ~/.sol-sentinel/SOLUSDT_1m.csv`
2. Review the generated `.meta.json` sidecar for source, timestamps, count, and SHA-256. Keep the CSV and sidecar local; never commit private datasets or account records.
3. Run the backtest:
   `python projects/sol-sentinel/paper_engine.py --csv ~/.sol-sentinel/SOLUSDT_1m.csv --report-path ~/.sol-sentinel/backtest.json`
4. The engine writes a local SQLite ledger by default at `~/.sol-sentinel/paper_ledger.sqlite3`. Use `--db-path ""` to disable persistence. A repeated identical run is idempotent.
5. The default entry/exit fee components match Reku's published Pro Mode table checked 2026-10-29; recheck https://reku.id/fees/crypto and the actual order preview before use. Spread and slippage remain assumptions.
6. Reject any backtest whose data has gaps, bad timestamps, or insufficient history. Do not tune parameters using the out-of-sample segment.
7. Expect the strategy to produce zero trades on some datasets; this is valid and preferable to forcing entries.

## Chronological evaluation

Use the fixed-parameter development/validation/out-of-sample evaluator after the basic backtest:

`python projects/sol-sentinel/evaluate_splits.py --csv ~/.sol-sentinel/SOLUSDT_1m.csv --output ~/.sol-sentinel/evaluation.json`

The evaluator splits the input by time in a 60/20/20 ratio. Each segment begins at the configured initial capital independently, and every result uses the same fixed strategy parameters. Do not tune parameters against the out-of-sample segment. The default minimum segment length is 600 complete 1-minute bars, so very short histories are rejected. This is an initial research check, not statistical proof of a profitable edge; require adequate trades and representative regimes.


## Cost assumptions

The scanner accepts `--fee-bps-round-trip`, `--spread-bps-round-trip`, and `--slippage-bps-round-trip`. The backtest/evaluator accept `--entry-fee-bps`, `--exit-fee-bps`, `--spread-bps-round-trip`, and `--slippage-bps-per-side`. These separate estimates to make assumptions visible. Do not set spread/slippage to zero just to obtain more signals. Published fee inputs are sourced from Reku's Pro Mode fee table; spread/slippage defaults are placeholders and should be replaced only by measured quotes, execution data, or a documented sensitivity test.


## Quote currency and IDR conversion

The CSV contains SOLUSDT prices, while the configured capital and reports use IDR. The history-capture script fetches a daily USD/IDR reference from Frankfurter and records the rate/date in the `.csv.meta.json` sidecar. The backtest and chronological evaluator read that sidecar by default. The reference is a USD proxy for USDT and applies one daily rate to the entire dataset; it is not Reku's live SOL/IDR price. If the reference is missing/stale, recapture or provide a verified manual value:

`python projects/sol-sentinel/paper_engine.py --csv ~/.sol-sentinel/SOLUSDT_1m.csv --quote-to-idr-rate YOUR_VERIFIED_RATE --quote-to-idr-source "source and date" --report-path ~/.sol-sentinel/backtest.json`

Do not paste a private or account-specific value into GitHub. This conversion corrects units for simulation; it does not make Binance prices executable on Reku.


## Cost sensitivity

Run the predeclared comparison with identical technical rules on the same historical dataset:

`python projects/sol-sentinel/cost_sensitivity.py --csv ~/.sol-sentinel/SOLUSDT_1m.csv --output ~/.sol-sentinel/cost_sensitivity.json`

The report compares: a Reku Pro fee baseline with assumed spread/slippage; a lower-friction hypothetical with the same fees; and two fee-only mathematical floors (Pro and Lightning) that explicitly assume zero spread/slippage and are not executable scenarios. All technical parameters remain fixed. Do not choose a scenario simply because its historical P&L is highest; use the report to understand whether costs erase the edge.


## Reku SOL/IDR public quote and order-book snapshot

After capturing Binance history (to provide the dated FX proxy), run:

`python projects/sol-sentinel/reku_market.py --symbol SOL --binance-csv ~/.sol-sentinel/SOLUSDT_1m.csv --output ~/.sol-sentinel/reku_sol_idr_market.json`

The adapter calls the public endpoints documented at https://api.reku.id/: `/v2/coins`, `/v2/price`, `/v2/bidask`, and `/v2/orderbook?id=...`. It records direct spread in basis points, order-book best levels and top-five notional depth, and optional cross-venue reference deviation. No login or credentials are used. The documentation is legacy, so use the report's `data_status` and `warnings`: `OK` means the fields parsed coherently, `PARTIAL` means quote/book consistency could not be fully validated, and `ERROR` means no reliable snapshot. Do not treat the USD/IDR proxy as a transaction rate or the adapter as an order API.


### Reku public quote caveat

A prior GitHub Actions snapshot returned last price Rp1,845,000, bid Rp1,840,000 and ask Rp1,845,000 (about 27.14 bps quoted spread), but the legacy `/v2/orderbook?id=51` route returned HTTP 400 and the output was labelled `PARTIAL`. The code now tries the documented `/v2/orderbookall?id=...` fallback. Use a fresh report's `data_status`, `orderbook_status`, endpoint used, and capture timestamp. This old snapshot must not be used as the current market quote.


## Jupiter V2 quote-only probe (read-only)

This step is optional and is for route/quote observation only. It does not require a wallet address, does not sign or submit a transaction, and explicitly refuses an unexpected transaction payload from the quote endpoint. The returned quote is not a guaranteed fill.

Use a raw amount in the input token's smallest unit; for wrapped SOL, 1 SOL = 1,000,000,000 lamports. Example to quote 0.01 SOL -> USDC:

```sh
python projects/sol-sentinel/jupiter_quote_probe.py --amount-units 10000000
```

By default, the script uses the keyless Jupiter Developer Platform path, subject to the documented keyless rate limit. For a free developer API key, set `JUPITER_API_KEY` in the local Termux environment only; never paste it into chat, commit it to Git, or include it in logs. The current free API plan is 1 request/second; a keyless request has a lower rate limit. Check current plan/limits before relying on the service:
- https://developers.jup.ag/pricing
- https://developers.jup.ag/changelog/developer-platform

Official Swap V2 order docs: https://developers.jup.ag/docs/swap/order-and-execute

The probe prints raw input/output units and any available estimated price impact/fee fields. Convert using verified token mint decimals; do not compare raw lamports and USDC units directly. A complete cost model also requires priority fee, possible account rent, transfer-to/from-exchange costs, and eventual IDR off-ramp costs. **The probe never executes a trade.** This code path is strictly read-only and not a paper-fill or production executor.


## Native INDODAX SOL/IDR history (public read-only)

The native source uses the INDODAX pair ID and TradingView history endpoint. It does not convert SOL/IDR to USDT or apply an FX rate.

Capture the latest completed 1-minute bars (up to 30 days):

```sh
python -m py_compile projects/sol-sentinel/execution_adapters.py projects/sol-sentinel/capture_indodax_history.py
python projects/sol-sentinel/capture_indodax_history.py --symbol SOL/IDR --days 30 --output ~/.sol-sentinel/INDODAX_SOLIDR_1m.csv --metadata-output ~/.sol-sentinel/INDODAX_SOLIDR_1m.csv.meta.json
```

Review the sidecar and CSV hash before any analysis. The collector fails closed if the source omits expected bars, timestamps do not align, OHLCV is invalid, or a duplicate candle conflicts. A successful mocked unit test proves parsing/validation logic only; an API workflow artifact is needed to confirm endpoint behavior from a live public network.

To run a native-IDR offline backtest, first verify the exact fee components shown in the current INDODAX Profile → Trading Fees page and the selected trading mode's order minimum. Set explicit fee/spread/slippage and minimum-order inputs; never reuse the Reku fee defaults for INDODAX. Example syntax (replace each placeholder with a measured/documented input before running):

```sh
python projects/sol-sentinel/paper_engine.py \
  --csv ~/.sol-sentinel/INDODAX_SOLIDR_1m.csv \
  --fee-model indodax-custom \
  --entry-fee-bps VERIFIED_ENTRY_BPS \
  --exit-fee-bps VERIFIED_EXIT_BPS \
  --spread-bps-round-trip MEASURED_ROUND_TRIP_SPREAD_BPS \
  --slippage-bps-per-side DOCUMENTED_SLIPPAGE_BPS \
  --minimum-order-notional-idr VERIFIED_MODE_MINIMUM_IDR \
  --fee-source "INDODAX Profile/Trading Fees checked YYYY-MM-DD; spread/slippage source identified" \
  --report-path ~/.sol-sentinel/indodax_backtest.json \
  --db-path ~/.sol-sentinel/indodax_paper_ledger.sqlite3
```

The flags are intentionally mandatory for this source. If the exact INDODAX fee schedule or selected mode minimum is not verified, do not run it with invented values. For chronological evaluation, use the same explicit source metadata and cost inputs:

```sh
python projects/sol-sentinel/evaluate_splits.py \
  --csv ~/.sol-sentinel/INDODAX_SOLIDR_1m.csv \
  --fee-model indodax-custom \
  --entry-fee-bps VERIFIED_ENTRY_BPS \
  --exit-fee-bps VERIFIED_EXIT_BPS \
  --spread-bps-round-trip MEASURED_ROUND_TRIP_SPREAD_BPS \
  --slippage-bps-per-side DOCUMENTED_SLIPPAGE_BPS \
  --minimum-order-notional-idr VERIFIED_MODE_MINIMUM_IDR \
  --fee-source "INDODAX fee display + dated quote measurement" \
  --output ~/.sol-sentinel/indodax_evaluation.json
```

The existing `cost_sensitivity.py` scenarios are explicitly Reku fee-reference scenarios on Binance SOLUSDT/USDT data. They reject native INDODAX IDR history rather than relabeling it or applying the wrong venue's fee schedule. Native IDR backtests skip FX conversion; reports record source venue, pair, currency and selected fee model.

## Venue-neutral adapter contracts (mock/paper only)

For the current development branch, the platform boundary and evidence requirements are documented in `PLATFORM_ADAPTER_DECISION_2026-10-30.md`. Contract code is `execution_adapters.py`; offline regression cases are `test_execution_adapters.py`.

Run the contract tests without networking or secrets:

```sh
python -m py_compile projects/sol-sentinel/execution_adapters.py projects/sol-sentinel/test_execution_adapters.py
python -m unittest discover -s projects/sol-sentinel -p 'test_*.py' -v
```

The included market-data, capital-source and custody adapters are fixtures only. `PlanningOnlyTransferAdapter` validates network names, verified network flags, known minimums/fees and fee-evidence age; it cannot send a transfer. `PaperExecutionAdapter` uses a mock bid/ask and Decimal quantities. `ExecutionPolicy(LIVE)` deliberately fails because no live adapter exists.

Do not supply a wallet seed/private key or API secret to these tests. Do not use a withdrawal-enabled API key for a trading worker. Exact Solana withdrawal fee/minimum, destination deposit minimum, market tick/lot/minimum, and current venue-specific quote/fee schedule must be verified from current official sources before the paper model is configured for a chosen platform. No live order, transfer, signature or signer integration is currently present.


## Lightweight Termux diagnostics — read-only

The standard-library-only diagnostic runner checks the Python/SQLite runtime, Termux detection, available memory and disk, core module compilation, SQLite close/reopen and idempotent replay, and the public INDODAX SOL/IDR pair catalogue. Its only network action is an unauthenticated, bounded HTTP GET to the public /api/pairs metadata endpoint. It never calls an order, wallet, signing, transfer, deposit, or withdrawal endpoint.

From the repository root:

```sh
python projects/sol-sentinel/termux_doctor.py
python projects/sol-sentinel/termux_doctor.py --json
```

To validate the native historical dataset after it has been downloaded to the device:

```sh
python projects/sol-sentinel/termux_doctor.py \
  --data ~/.sol-sentinel/INDODAX_SOLIDR_1m.csv \
  --expected-sha256 c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30
```

The doctor streams the CSV rather than loading the full 43,200-row file into memory. It verifies the hash, sidecar identity/status when present, no FX conversion, contiguous 1-minute timestamps, close-time boundaries, finite OHLCV, and OHLC ordering. The expected hash above applies only to artifact #12931446637; a newer dataset must use its own verified SHA-256.

To execute the offline suite on the phone:

```sh
python projects/sol-sentinel/termux_doctor.py --skip-network --run-tests --json
```

Interpretation:
- FAIL is a failed diagnostic invariant and exits with code 1.
- WARN is an environment limitation (for example, non-Termux host, unavailable /proc memory fields, or a transient public endpoint problem); inspect the detail.
- SKIP means a check was deliberately omitted.
- A successful CI run or a run on a desktop/GitHub runner is not proof of compatibility on the user's Android handset. Preserve the JSON output with timestamp and device/Python versions when the user runs it locally.
- The doctor does not execute the backtest automatically because native INDODAX evaluation requires a selected mode and explicit fee/spread/slippage assumptions. Use the audited Pro-only configuration in INDODAX_FEE_MODEL_2026-10-30.md; do not substitute Reku defaults.


## INDODAX depth microstructure collector — read-only

The feature branch includes `indodax_microstructure.py`, a standard-library collector for the documented public `GET /api/depth/solidr` endpoint. It parses multi-level bid/ask arrays, rejects invalid/unsorted/duplicate/crossed books, and records observation time, optional exchange timestamp, top-of-book prices and quantities, midpoint, quoted spread, all captured levels, collector version, and a book SHA-256 in a local SQLite archive.

Default collection is finite: 30 snapshots at 10-second intervals. The collector enforces at least one second between requests (60/minute at most), bounded response size, request timeout/retries, maximum levels/samples, and a consecutive-error stop. It records successful/error attempts and request duration. Summary queries report mean, median, nearest-rank P95 and population standard deviation for spread, plus bid/ask IDR notional depth within a midpoint band. The statistics describe only observations saved by the collector; no historical book depth is reconstructed from candles or old point quotes.

From the repository root:

```sh
python projects/sol-sentinel/indodax_microstructure.py --samples 30 --interval-seconds 10
python projects/sol-sentinel/indodax_microstructure.py --summary-only --database ~/.sol-sentinel/indodax_depth.sqlite3
```

The default archive is `~/.sol-sentinel/indodax_depth.sqlite3`. Keep it local; it contains public market observations and can grow over time. The collector is read-only and unauthenticated. It cannot access balances, place/cancel orders, sign, deposit, withdraw, or submit chain transactions. A CI fixture test proves parser/storage behavior only; run the first real collection on the target device/network and preserve its emitted summary before claiming empirical spread/depth distributions.

The current research dataset remains the previously recorded 43,200 native INDODAX 1m candles. No new order-book snapshots were captured as part of this code change.


## Depth-aware paper execution engine — local simulation only

The separate `depth_execution.py` module extends collected order-book evidence into a deterministic, SQLite-backed paper simulator. It does not change the older `MarketQuote` or `PaperExecutionAdapter` contracts and has no live execution interface.

The simulator:
- Walks the observed ask levels for buys and bid levels for sells in price order, recording per-level fills and Decimal VWAP.
- Enforces limit-price boundaries, quote freshness, symbol/venue identity, tick and quantity precision, available IDR cash including fees, and available simulated SOL inventory.
- Shares each snapshot's liquidity across all simulated orders. Observed depth is journaled once and each level tracks reserved/consumed/remaining quantity; the same snapshot cannot be reused to manufacture additional fills.
- Persists order state, fills, events, snapshot levels, liquidity consumption and the paper portfolio in one SQLite transaction per order/snapshot cycle. GTC residuals can continue only on a later eligible snapshot; IOC residuals are cancelled after the current attempt.
- Preserves past fills on cancellation/expiry, deduplicates order creation and order/snapshot cycles, supports reopen/recovery and reconciliation, and exposes failure-injection hooks used by the tests.

Run only the offline regression suite:

```sh
python -m py_compile projects/sol-sentinel/depth_execution.py projects/sol-sentinel/test_depth_execution.py
python -m unittest discover -s projects/sol-sentinel -p 'test_depth_execution.py' -v
```

The existing collector can be connected in memory with `DepthSnapshot.from_indodax(parsed_snapshot)`, where `parsed_snapshot` is returned by the microstructure parser. The execution engine does not yet read the collector's SQLite archive automatically and does not itself run an unattended market-data loop. Explicitly configure client order ID, order side/type, fee bps, tick/lot rules, maximum quote age, initial simulated IDR cash, and timestamps. The sample ledger starts with IDR 300,000 cash and zero SOL as a simulation seed; it is not a statement about a real account balance. Nonzero initial SOL is refused until an auditable inventory cost basis is supported.

Model limitations: no maker queue or post-only fill probability, no exchange matching-engine emulation, no real order-book capture in the implementation CI, and no integrated strategy/risk/daily-loss forward-paper cycle yet. Treat the depth model as an approximation. Fixture tests are not empirical fill calibration or profitability evidence.


### Conservative shared-liquidity rule

Depth consumption is pooled by the exact observed side-book fingerprint `venue + symbol + side + ordered price/quantity levels`, independently for bid and ask. A later timestamp with an unchanged ask side does not make that ask liquidity new; a bid-side change does not reset ask-side consumption. A changed side-book fingerprint creates a new observed pool. This is deliberately conservative because trade-flow data are not yet available to distinguish replenished size from an unchanged displayed queue. It is not an exchange matching-engine rule.

The ledger backfills pool-consumption mappings when it opens a database created by the first depth-engine schema, aggregating prior consumption across identical side books. If the aggregate exceeds observed depth, migration fails closed and requires manual audit.

 
## Persistent forward-paper on Android / Termux

The integrated worker uses Python standard library modules and the existing project files:

```sh
python -m py_compile projects/sol-sentinel/forward_paper.py
python -m unittest discover -s projects/sol-sentinel -p 'test_*.py' -v
python projects/sol-sentinel/termux_doctor.py --json
python projects/sol-sentinel/forward_paper.py --db-path ~/.sol-sentinel/forward-paper.sqlite3 --cycles 1 --json
```

Use a bounded run for repeated observations:

```sh
python projects/sol-sentinel/forward_paper.py --db-path ~/.sol-sentinel/forward-paper.sqlite3 --cycles 12 --interval-seconds 300 --json
```

The worker stores state in the selected file-backed SQLite database. Keep this file between restarts. It includes the simulated account, orders, fills, market-depth snapshots, risk lock, signals and execution cycles. Do not delete it to clear a halt or reset the portfolio.

Operator halt/resume commands and recovery limitations are documented in `FORWARD_PAPER_OPERATIONS.md`. This runbook does not claim that the user's handset has been tested; CI runs on Ubuntu. Forward-paper has no live exchange order, wallet-signing, transfer, deposit or withdrawal capability.
