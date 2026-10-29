# SOL Sentinel — Termux Runbook

## Requirements

- Android + Termux.
- Python 3.10+ available as \`python\`.
- Network access to Binance public API.
- No pip package or API key required for Phase 0.

## Install and run

Clone or fetch the feature branch, then from repository root:

\`\`\`sh
git fetch origin
git checkout feat/sol-sentinel-paper-trader-v1
python --version
python -m unittest discover -s projects/sol-sentinel -p 'test_*.py' -v
python projects/sol-sentinel/sol_sentinel.py --json
\`\`\`

To run a continuous scanner while the terminal stays active:

\`\`\`sh
python projects/sol-sentinel/sol_sentinel.py --watch-seconds 30 --log-path sol_sentinel_runs.jsonl
\`\`\`

Use Termux wake-lock if available and keep the device on a stable connection. Android may still kill processes; no always-on guarantee is implied. To stop watch mode press Ctrl+C.

## Parameters

- \`--symbol SOLUSDT\`: public Binance spot market reference.
- \`--fee-bps-round-trip 42.11\`: optional assumption equivalent to 0.4211% round trip. This is a placeholder based on a previously viewed Reku Pro fee example; verify current exact fees before relying on it.
- \`--slippage-bps-round-trip 20\`: initial slippage allowance, not a measured estimate.
- \`--min-net-edge-bps 25\`: minimum net edge buffer used for the candidate filter.
- \`--log-path FILE\`: append scan results as JSON Lines.
- \`--json\`: output structured JSON.
- \`--watch-seconds 30\`: repeat at a safe interval (minimum 15 seconds).

Do not set fees to zero to force a signal. If changing fees, store the reason and current source in the research log.

## Interpret the output

- \`NO_TRADE\`: a valid risk-managed result; inspect reasons.
- \`PAPER_CANDIDATE\`: a rule-based observation eligible for paper testing only; this is not a buy order or recommendation.
- \`data_status=STALE/INSUFFICIENT/ERROR\`: no signal may be acted on.
- Cost estimate is a sensitivity input; Binance data are not Reku executable prices.
- Every scan is a point-in-time observation, not an account balance or realized P&L.

## Troubleshooting

- HTTP errors/timeouts: retry later, do not loop aggressively; the worker uses bounded timeouts.
- Unsupported endpoint/region: scanner exits with a clear error; validate public API availability from this device and research a documented alternate source.
- Clock skew: enable Android automatic time.
- Android process stopped: restart manually and inspect the last JSONL record.
- Unexpected signal: inspect the JSON metrics and replay that exact data offline before changing any parameters.
