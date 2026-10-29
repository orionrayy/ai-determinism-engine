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
