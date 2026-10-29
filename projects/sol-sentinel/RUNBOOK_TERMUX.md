# Termux Runbook

## 1. Install

```sh
pkg update
pkg install python git
git clone https://github.com/orionrayy/ai-determinism-engine.git
cd ai-determinism-engine/projects/sol-sentinel
python -m unittest -v
```

If the repository is already cloned, fetch the feature branch separately or use the branch that contains this project. No pip packages are required.

## 2. First data check

```sh
python sol_sentinel.py --doctor --once
```

This is read-only. It calls public Reku endpoints and prints JSON diagnostics. It must not need a Reku password, API key, OTP, or wallet key. A DNS/TLS/HTTP/schema error means the live feed is not verified; do not interpret the script's state as a trading signal.

## 3. One cycle then continuous paper mode

```sh
python sol_sentinel.py --once
python sol_sentinel.py --interval 30
```

Use Ctrl+C to stop. Do not turn on an Android battery restriction exemption without considering battery and background-process behavior. A stopped process cannot keep scanning.

## 4. Read status

```sh
python sol_sentinel.py --status
python sol_sentinel.py --status --db data/sol_sentinel.sqlite3
```

The script prints paper equity, cash, active position, daily lock and latest sample status. Default SQLite data is local and should not be committed.

## 5. Export a safe report

```sh
python sol_sentinel.py --export-report reports/latest-report.json
```

Inspect the export before sharing. It is intended for aggregate paper-trading results, not private account details. Do not publish a local database without reviewing its rows.

## 6. Errors

- `SOL not found`: current API response may have changed, or Reku uses another coin code. Inspect public `/v2/coins` response locally and update the parser with a test; do not guess the ID from an old screenshot.
- `chart unavailable / malformed`: the documented legacy chart endpoint may have changed. No entry signal should be trusted until corrected.
- `no fresh candles`: data has gone stale, network is unavailable, or candle timestamps have changed semantics.
- `spread guard`: bid/ask spread is too wide for the strategy's modeled edge.
- `no setup`: normal state; the scanner is not required to trade.
- `daily loss lock`: paper simulation pauses new entries until the next local trading date; do not disable the guard to chase a goal.

## 7. Backup

Stop the process before copying the SQLite database. Keep database files and any future credentials outside the Git repository. The source code in Git is the auditable memory; trading state and personal records remain local.
