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

The implementation was integrated through PR #181. The general `SOL Sentinel Tests` workflow is a regression-only workflow: after PR #185 it no longer contains a `forward-paper-activation` job and must not run or write a paper cycle on push/PR events. CI tests use isolated fixtures. The durable paper database remains on the dedicated `sol-sentinel-paper-state` branch. The dedicated `.github/workflows/sol-sentinel-forward-paper.yml` workflow is the recurring state-mutating runner; it is guarded to `refs/heads/main`.

The dedicated `.github/workflows/sol-sentinel-forward-paper.yml` workflow declares one bounded cycle every five minutes (`2-57/5 * * * *`, UTC) and a manual `run`, `status`, `halt`, or `resume` operation. The forward-paper job is guarded to execute only when the ref is `main`; a dispatch against another ref cannot write the shared paper-state branch. After the SQLite WAL checkpoint/integrity check, the worker runs a read-only `--status --json` query against the same database and uploads `status.json` with `result.json`. The artifact is uploaded only after status capture. The workflow Step Summary reports event/run/ref/commit, latest cycle and completion time, last depth observation, signal reasons, circuit-breaker/daily-loss/drawdown state, cash/SOL, realized P&L, fees, position, open-order/residual quantities, ledger counts/reconciliation, recovery and last error. Failed operations, halted/manual-review states or inconsistent reconciliation are explicitly flagged in the Step Summary. The recurring schedule's presence in YAML does not establish that GitHub has enabled it or that it has executed: require a completed run with workflow name `SOL Sentinel Forward Paper`, event `schedule`, branch `main`, matching commit, result/status artifacts and a consistent state checkpoint before marking `SCHEDULED_OPERATION_VERIFIED`. Manual dispatch, push and PR tests are not cron evidence.

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
- INDODAX Pro IDR official **all-in** fees effective 2026-10-01: BUY maker 10.555 bps, BUY taker 20.555 bps, SELL maker 31.555 bps, SELL taker 41.555 bps. These are distinct from the CFX subcomponent (0.555 bps). The fee structure is published by INDODAX; the account-specific displayed mode/route must still be checked in Profile → Settings → Trading Fees. A limit order that immediately matches resting liquidity can be charged as taker, not maker.
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
REPORT="$HOME/sol-sentinel-termux-report.json"; python projects/sol-sentinel/termux_doctor.py --run-tests --json --timeout 5 > "$REPORT"; RC=$?; cat "$REPORT"; echo "exit_code=$RC"; exit "$RC"
```

The collector's spread/depth summary only describes snapshots actually captured after activation. It is not backfilled from OHLC candles. Queue position, maker fill probability, true slippage distribution, and exchange account fee tier remain uncalibrated/unverified.

The current program is forward-paper only. A green software test, successful activation record, or positive simulated P&L is not permission to deploy real money.


## v1.4 operational hardening checkpoint — 2026-10-30

- PR [#182](https://github.com/orionrayy/ai-determinism-engine/pull/182) merged to `main` as `8c8e0c0362161ec6644ba44fa538d55dcb9fe18c`.
- The recurring cron is now `2-57/5 * * * *` UTC (12 runs/hour, at minutes 02, 07, ..., 57) to avoid the start-of-hour contention window. GitHub documents that scheduled events can be delayed or dropped at high-load times and only execute from the default branch. No distinct `SOL Sentinel Forward Paper` event with `event=schedule` was found after merge when checked at 2026-10-30T05:48:56Z; keep `FIRST_CRON_PENDING` until the actual event and artifact are observed.
- The paper job now has a main-ref guard and a run Step Summary. The shared SQLite branch cannot be changed by a manual dispatch against a non-main ref. The `SOL Sentinel Tests` workflow has `contents: read` and no state-mutating activation job; recurring state changes belong only to the dedicated forward-paper workflow on `main`.
- SOL Sentinel Tests now run on pushes to `main`, and PR path filters include the forward-paper workflow. Post-merge run [#41010295827](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41010295827) on the exact main commit passed compile, **216 tests**, and Termux Doctor (Ubuntu: 6 PASS / 1 WARN / 0 FAIL); activation was skipped as designed.
- The PR test on branch `fix/sol-sentinel-schedule-ci-hardening`, run [#41010214471](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41010214471), also passed 216 tests; activation was skipped. This was a test event, not a paper cycle or schedule proof.
- Durable storage remains SQLite file `state/forward-paper.sqlite3` checkpointed to the dedicated Git branch `sol-sentinel-paper-state`, not a runner-local durable disk, object store, managed database or CI cache. The branch head at review was `c3f259242d450d64b4baeb01ae083c178fd4d8b0`; its linear history includes at least 12 checkpoint commits attributed to separate run IDs from 41005882837 through 41006890361. This proves repeated persisted checkpoints, while non-flat order/fill/residual reconstruction remains unexercised because the recorded account is flat.
- Latest recorded paper artifact remains run [#41006890361](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006890361), artifact [#12955034631](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006890361/artifacts/12955034631): NO_TRADE, risk NORMAL, equity/cash IDR 300,000, SOL 0, 0 orders/fills/fees, consistent ledger/portfolio, source commit `5bf6c2cc9598d49c4ecec4458373538b3afa18f5`. No post-merge scheduled paper result is claimed.
- Official INDODAX Pro IDR fee table (effective 2026-10-01) lists all-in BUY maker 10.555 bps, BUY taker 20.555 bps, SELL maker 31.555 bps, SELL taker 41.555 bps. The CFX component alone is 0.555 bps. The Pro direct order minimum is Rp25,000; orders Rp10,000–24,999 are routed through Lite. The user's own fee display remains unverified. A maker quote is not proof of a maker fill.
- Android/Termux must still be run on-device. The command above writes the full JSON report to `$HOME/sol-sentinel-termux-report.json` and executes the offline project tests plus a bounded public metadata GET.


### v1.4 monitoring addition — 2026-10-30

PR [#183](https://github.com/orionrayy/ai-determinism-engine/pull/183) merged as `0bcfe96c47c6d566051a48d1b031e993d96062d0`. Each scheduled or manual-main invocation now reads the durable database using `--status --json` after the integrity check and includes `status.json` in the run artifact. The operator summary now includes last cycle, observation freshness, strategy reasons, risk/circuit-breaker state, daily loss, drawdown, portfolio P&L/fees, position, open orders/residuals, ledger counts/reconciliation, recovery and last error. This status query is read-only and does not create a new market observation or cycle.


## v1.7 operational checkpoint — 2026-10-30 15:38 WIB

### Scheduler inspection result

The dedicated Actions workflow is **enabled** at this inspection time. An authenticated, read-only inspection of the workflow overflow menu showed the action `Disable workflow`, not `Enable workflow`; the page had no disabled/suspended banner and reported zero runs. Repository Actions settings allow all actions and reusable workflows, default GITHUB_TOKEN permissions are read-only for contents/packages, and no schedule-specific repository restriction or warning was visible. The file on the default branch declares `2-57/5 * * * *` UTC and has the main-ref guard. These facts rule out a manually disabled workflow and a wrong default branch, but do not explain why the scheduler has emitted no Sentinel event.

The cron expression was last changed in PR #182 (merge commit `8c8e0c0362161ec6644ba44fa538d55dcb9fe18c`), authored by `orionrayy`. This account is the repository owner/admin and is still active; other scheduled repository runs show actor `orionrayy`. No Sentinel run exists to expose its own actor metadata. Keep the exact cause of missing schedule dispatch unresolved rather than attributing it to a disabled workflow or suspended actor.

Actions history filtered by `event=schedule` and `branch=main` returned 153 runs across two pages, all from `AI Orchestrator`; none used this workflow path. The workflow page itself reports 0 runs. Status remains `FIRST_CRON_PENDING`. No manual dispatch, disable/enable toggle, rerun, or paper cycle was invoked.

### CI compatibility fix

PR #187 originally failed the Orchestrator Tests Actionlint step because its pinned actionlint v1.7.12 does not recognize GitHub's documented `concurrency.queue` property. The workflow's `queue: max` and `cancel-in-progress: false` are retained. A narrow ignore rule was added to `.github/actionlint.yaml`, scoped only to the Sentinel forward-paper workflow and this exact unknown-key diagnostic. The existing Sentinel contract tests continue to assert that the queue settings remain present. On PR head `f1de2a142a8530489677fe47d9cbb8e8c6810734`, Sentinel Tests #41027680475 and Orchestrator Tests #41027680469 both completed successfully. PR #187 is still open/unmerged; no automatic merge is authorized.

### State and execution boundary

State branch HEAD remains `c3f259242d450d64b4baeb01ae083c178fd4d8b0`. Its database Git blob remains `a9bbe1ae63aa1aa496fe000f23d30d4259de1c09` (1,323,008 bytes), identical to the last read-only SQL-audited database. The last raw SHA-256 and SQLite findings are recorded in canonical memory and the research audit; the database binary was not re-opened for a new SQL query in this checkpoint. Do not describe the prior audit as a new audit.

The last SQL audit reported integrity `ok`, zero FK violations, 17 forward cycles, 17 book observations, 5,100 levels, 17 attempts, no fills/orders/residuals, and a reconciled flat portfolio of Rp300,000 cash and 0 SOL. Synthetic non-flat recovery remains verified only in the isolated three-process temporary-DB test. Operational non-flat recovery remains unproven.

### Termux and research

The on-device diagnostic command is:

```sh
python projects/sol-sentinel/termux_doctor.py --json
```

It uses the standard library, performs a bounded unauthenticated GET to the public INDODAX pair catalogue, and exercises SQLite only on a temporary test database. For a no-network/offline check with tests:

```sh
python projects/sol-sentinel/termux_doctor.py --skip-network --run-tests --json
```

CI's prior Ubuntu result is 6 PASS / 1 WARN / 0 FAIL; physical Android/Termux remains `NOT_TESTED`. Research remains `NO_TRADE / INSUFFICIENT_EVIDENCE`; 17 depth observations are too short and lack exchange timestamps, prints and queue depletion for slippage calibration. The 10 bps per-side assumption remains unvalidated. Live trading remains `DISABLED`.

### Next step

The current evidence does not justify a workflow enablement toggle. A same-cadence schedule-registration refresh may be prepared as a separate reviewable change, but it must not be applied until explicitly approved. A genuine scheduled run on `main`, both result/status artifacts and a matching durable state checkpoint are still required for `SCHEDULED_OPERATION_VERIFIED`.


## v1.8 scheduler registration refresh — 2026-10-30 16:04 WIB

### Workflow identity and activation

GitHub REST identifies the workflow as ID `401397705`, name `SOL Sentinel Forward Paper`, path `.github/workflows/sol-sentinel-forward-paper.yml`, and state `active`. The authenticated Actions page also offers `Disable workflow` (not `Enable workflow`), displays no disabled/suspended banner, and reports zero workflow runs. The repository default branch is `main`, and the workflow exists on that branch.

### Schedule registration refresh

PR [#188](https://github.com/orionrayy/ai-determinism-engine/pull/188) was merged to `main` as `1ec2a6a11dbe8bcf84d910fc3017c9d0173ad01a`. It changed only the cron spelling from:

```cron
2-57/5 * * * *
```

to:

```cron
2,7,12,17,22,27,32,37,42,47,52,57 * * * *
```

Both expressions select the same twelve UTC minutes per hour: 02, 07, 12, 17, 22, 27, 32, 37, 42, 47, 52 and 57. A contract test pins this exact cadence. No state logic, concurrency, job permissions, main-ref condition, executor, artifact handling or strategy configuration changed.

GitHub's troubleshooting documentation says a scheduled workflow can stop while still enabled if its associated actor account is suspended/deleted, and that a commit changing cron syntax can reactivate a deactivated schedule and establish a new actor. The account associated with the prior cron change is `orionrayy`, an active repository owner/admin; therefore actor suspension is not confirmed. The schedule refresh is a supported, conservative re-registration attempt, not proof of the previous root cause.

### Current scheduler result

An exact GitHub REST query against workflow ID `401397705`, with `event=schedule` and `branch=main`, returned `total_count=0` and no pagination link at the checkpoint time. No event, run ID, artifact, cycle ID or state checkpoint can be reported. Keep `FIRST_CRON_PENDING` and `WORKFLOW_ENABLED_FIRST_RUN_PENDING` separate. Do not count AI Orchestrator schedule events or PR/push tests.

Do not invoke manual `run` to simulate cron. A future first scheduled run must be verified using the exact workflow ID and event, branch, head SHA, timestamps, run conclusion, `result.json`, `status.json`, cycle ID, state branch advancement, database integrity and ledger reconciliation.

### State and risk boundary

State branch remains `sol-sentinel-paper-state` at `c3f259242d450d64b4baeb01ae083c178fd4d8b0`. The database Git blob remains `a9bbe1ae63aa1aa496fe000f23d30d4259de1c09`, size 1,323,008 bytes; its previously audited raw SHA-256 is `149ea9184abb9977cdba032820e7bc57fa33526349530633906008d49d33427c`. The binary was not reopened for a new SQL pass in this checkpoint. Last SQL audit still reports flat cash Rp300,000, 0 SOL, zero fills/orders/residuals, integrity `ok`, and 0 foreign-key violations. Do not represent this as a new integrity query.

Synthetic three-process recovery remains verified only on an isolated temporary database. Active operational non-flat recovery is not yet observed. Strategy remains `NO_TRADE / INSUFFICIENT_EVIDENCE`; physical Android/Termux remains `NOT_TESTED`; live trading remains `DISABLED`.


## v1.9 operational checkpoint — 2026-10-30

### Scheduler evidence
- Isolated schedule-only diagnostic workflow `.github/workflows/sol-sentinel-schedule-canary.yml` was added through PR #190. It has workflow ID `401581054`, API state `active`, and cron `0,30 * * * *`. It only reports run/workflow provenance, uses `actions:read`, a two-minute timeout, and does not check out or call the trading engine or paper-state branch.
- Forward Paper remains workflow ID `401397705`, API state `active`, cron `2,7,12,17,22,27,32,37,42,47,52,57 * * * *`.
- Read-only queries filtered to `event=schedule` and `branch=main` returned 0 runs for both workflows after the 10:00 UTC canary slot and again at about 10:06 UTC. No schedule run IDs, artifacts, or execution results exist yet. One missed canary slot is not enough to establish a repository-wide dispatch fault. Keep `FIRST_CRON_PENDING`; root cause unresolved.
- The isolated Research workflow was run manually once for public data capture, workflow run #41037517595. This is not schedule proof and is not a paper cycle.

### Active state — fresh SQL audit
The explicit opt-in audit job was run once via workflow_dispatch audit input; see run #41035585382. It checked out `sol-sentinel-paper-state` without persisted write credentials and opened SQLite read-only/query-only. Branch HEAD `c3f259242d450d64b4baeb01ae083c178fd4d8b0`, DB blob `a9bbe1ae63aa1aa496fe000f23d30d4259de1c09`, raw SHA-256 `149ea9184abb9977cdba032820e7bc57fa33526349530633906008d49d33427c`, file size1,323,008 bytes. Integrity `ok`, FK errors0, WAL, schema1. 17 cycles, 10 signals, 17 depth observations, 5,100 levels, 17 successful attempts; 0 orders/fills/events/execution cycles/positions. Cash Rp300,000, SOL0, fees/P&L0; cash and SOL reconcile to fills. This is flat-state persistence, not non-flat recovery.

### Operating decision
Keep the state branch unchanged. Do not dispatch the forward-paper workflow to simulate cron. Do not promote or activate a candidate until the research gate passes. No live route, credentials, wallet signing, orders, transfers, deposits or withdrawals are enabled. Physical Android/Termux validation still requires the user's on-device Doctor output; Ubuntu CI is not a substitute.
