# Roadmap and acceptance criteria

## P0 — reconcile code and checkpoint
- Reconcile v1.0.0-reported timestamp validator changes with exact tracked source; mixed aware/naive timestamps must raise a controlled typed validation error, not TypeError.
- Test unsorted timestamps, missing/duplicate source indices, invalid OHLC, non-finite values and negative volume.
- Quality report should include time monotonicity, source-index continuity, duplicate count, interval median/p95, session gaps, irregular intervals, zero volume, flat candles, price jumps, null/invalid rows, provenance and data hash.
- Regenerate manifest/test/benchmark outputs from the exact release code and preserve failures as evidence.

Acceptance: fail-closed typed errors, deterministic benchmark artifacts, tests reproducible from clean extraction.

## P1 — widen data and regime coverage
- Acquire licensed multi-year XAUUSD data with documented origin, timezone/DST, broker/instrument, volume semantics, timeframe and Bid/Ask/tick availability.
- Sample multiple years, sessions, trending/ranging and volatility regimes plus point-in-time news windows.
- Keep source index and timestamps audited; no concatenation of disconnected remote offset ranges.
- Use chronological train/model-selection/calibration/final-holdout partitions and matured-label purging where labels overlap.

Acceptance: provenance manifest complete, multi-regime data passes structural and temporal gates, untouched final holdout maintained.

## P2 — execution-aware strategy evaluation
- Replay with broker-relevant Bid/Ask or ticks; include dynamic spread, commission, swap, slippage, gaps and timeout rules.
- Report net expectancy, R distribution, drawdown, exposure, tail loss, trade count, profit factor, Brier/log loss, calibration/reliability and confidence intervals.
- Compare against no-trade, naive/base-rate and fixed-rule baselines; control multiple testing and parameter search.

Acceptance: any proposed edge survives measured costs, regime slices, uncertainty and untouched holdout.

## P3 — model governance and risk gate
- Separate direction and TP-before-SL targets; calibration only on held-out forecasts.
- Require sufficient mature labels; otherwise return `insufficient_history` and no probability.
- Log data/features/model hashes and every decision.
- Risk gate should require validated model/data, acceptable spread/session conditions, risk limits, confidence threshold and stale-feed checks.

Acceptance: OOS calibrated forecasts beat base-rate comparators consistently; no in-sample score turns on live output.

## P4 — point-in-time event and cross-asset evidence
- Add macro/news release timestamps, revisions/vintages, source latency and event windows.
- Test DXY, yields/real rates, volatility and other features without look-ahead; report ablations against price-only baseline.
- Track stale/conflicting provider states and abstain on data-quality failure.

Acceptance: chronological event-study protocol and ablation tests pass without causal overclaim.

## P5 — integration, security and operations
- Test Telegram/OANDA sandbox integration with authorized credentials; account connectors remain separate, consent-gated modules.
- MT5/cTrader account connection is not implemented; OANDA adapter currently covers market data only.
- Implement least privilege, token rotation/revocation, idempotency, webhook-secret checks, audit logs, rate-limits, health probes and rollback.
- Paper-trade before any live order routing; keep order execution disabled pending dedicated safety review.

Acceptance: end-to-end sandbox/paper evidence, reversible deployment, no secrets in source/logs, signed-off security and risk plan.

## v1.1.0 follow-up — online dataset evidence gate

- [x] Reconcile and build the source from the v1.0.0 archive; regenerate release manifest and SHA-256; re-extract and rerun tests.
- [x] Add a bounded, fixed-host public dataset fetcher with 16 MiB limit, bounded retries, strict schema/finite/OHLC/time-order checks and source checksum.
- [x] Add the 50 timestamp-spread × 5 baseline protocol, 1,200 counterfactual path contracts over five anchors, and a 240-case software robustness suite.
- [x] Add the CI workflow, test module, and JSON/CSV/Markdown evidence artifacts without storing the raw online dataset in Git.
- [ ] Verify first successful hosted workflow log and uploaded artifact; update DS51 from `reference_only` only after the upstream download, current data quality, and intended-use licensing have been reviewed.
- [ ] Acquire licensed broker Bid/Ask/tick data and document timezone/DST, symbol contract, volume semantics, spread/commission and fill information.
- [ ] Run chronological multi-fold/out-of-sample evaluation on broader independent time regimes; keep all trade outcomes matured and preserve an untouched final holdout.


## Final v1.1.0 rebuild — verified 2026-10-16

After adding the source-register entry and the portable online benchmark suite, the release was rebuilt from a clean verification path. Final artifact SHA-256 is 259103a2bf406498048e31c8ea97970c9f1ecd6bbf7636bd3bed07d7ca0ecb6e (245,370 bytes; 98 files in ZIP; manifest lists 51 sources). Final verification: pytest 169 passed; unittest discovery 102 passed; compileall PASS; source dataset audit has zero hard failures and one expected unknown-timezone warning; baseline and online-smoke benchmark entry points PASS; HTML IDs/inline JS checks PASS; extraction test reran pytest and passed. The source remains a research prototype, and remote acquisition of DS51 is pending the hosted GitHub Actions run.
