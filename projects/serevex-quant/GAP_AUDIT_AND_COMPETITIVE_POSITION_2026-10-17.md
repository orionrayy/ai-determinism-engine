# Serevex Quant Research — Risk-Weighted Gap Audit and Competitive Positioning

**Date:** 2026-10-17 (Asia/Jakarta)  
**Status:** Assessment and roadmap artifact on the existing draft PR branch. This document does not change application behavior, enable live signals, authorize order execution, or assert production readiness.

## 1. Executive assessment

The user-supplied inventory of 100 weaknesses and 100 comparisons is useful as a broad hypothesis register, but it is not yet a verified 200-point audit. It mixes confirmed implementation gaps, historical checkpoints, inferred competitor features, platform-category differences, and advanced capabilities that may not create measurable user value. Several items repeat the same underlying problem (for example, spread/commission/slippage/gap handling; data quality and chronology; authentication/session/RBAC controls; and broad asset coverage).

Serevex's best next step is not to maximize the number of features or models. It is to build a **verifiable XAUUSD decision-support pipeline** where every output is traceable to its data, timestamp, decision rule/model version, net-of-cost assumptions, calibration evidence and abstention reason. Its initial defensible differentiation should be evidence quality, honest uncertainty, safe defaults and frictionless Telegram UX—not an unsubstantiated claim of global predictive superiority.

## 2. Verified current baseline and important distinctions

The current project checkpoint is v1.3.1 on draft PR #164, `feature/serevex-online-dataset-benchmark`. It is not merged into `main` and is not a production release. The PR records 185 pytest tests, 109 unittest tests, compilation, Mini App syntax/HTML checks and clean-archive retesting as passing at the recorded checkpoint. These are source/package and contract-test results, not evidence that a strategy is profitable.

There are two distinct data realities that must not be conflated:

- **Bundled candidate application data:** 60 XAUUSD M5 OHLCV bars. The TP-before-SL model reports `insufficient_history` and zero OOS forecasts on this bundled sample.
- **Hosted research benchmark input:** 35,954 rows of historical community-sourced XAUUSD M5 OHLCV, checksum `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`, with represented timestamps from 2026-03-26 through 2026-09-25. This is historical—not live—and its intended-use rights, independent origin, intended timezone, broker/quote-side semantics and actual fills remain unverified.

The full-history replay records 35,893 sequential opportunities per fixed rule over five chronological folds. Four active fixed-rule baselines were net negative over full history and the final tail under an assumed 0.50 price-unit round-trip cost. Prequential direction-only baselines did not demonstrate meaningful Brier skill over the constant-0.5 reference. The reported 1,200 synthetic market-path contracts and 240 software robustness contracts test simulated scenarios and software behavior; they do **not** turn synthetic paths into real market observations or prove predictive skill.

Current guardrails therefore remain mandatory: `NO_TRADE`, `live_signal_enabled=false`, `deployment_calibrated=false`, and `validated=false`.

## 3. Corrections required before treating the 200 items as an audit

1. **Separate bundled data from hosted benchmark data.** “Only 60 bars exist” is true of the bundled model sample, but not of the separate online replay artifact. Neither fact removes provenance, licensing, timezone or broker-execution uncertainty.
2. **Separate software tests from trading evidence.** A passing test, correct parser, valid ZIP, scenario contract or deterministic replay proves only the property tested. It does not prove model accuracy, calibration, alpha, live fills or profitability.
3. **Do not equate platform categories.** Bloomberg Terminal and LSEG Workspace are primarily data/news/analytics/workflow offerings; QuantConnect/LEAN is an algorithm research and trading engine; TradingView emphasizes charting, scripting and alerts; MetaTrader 5 and TradeStation have trading-platform and brokerage ecosystems. One comparison matrix should measure shared capabilities only and use separate category-specific scorecards.
4. **Remove universal competitor claims without evidence.** Assertions such as “every competitor has a 99.9% SLA”, “all competitors are externally audited”, “all offer 24/7 support”, or “all natively support reinforcement learning” need product-, plan-, jurisdiction- and contract-specific evidence. Do not copy these statements into positioning.
5. **Do not use programming language or model complexity as a proxy for quality.** Python is not intrinsically unsuitable. A more complex model is justified only when it beats simple baselines on untouched chronological data after realistic costs and uncertainty analysis.
6. **Distinguish missing features from blockers.** Lack of SSO, social trading, options analytics, HFT, visual programming or multi-asset support is not a launch blocker for a Telegram-first XAUUSD research assistant. Unknown data provenance, leakage, unvalidated probabilities, cost-blind backtests and unsafe order execution are blockers.
7. **Treat zero-dollar as an operating constraint, not a guarantee about data rights or scale.** Open-source code, local execution, public datasets and free service tiers can minimize infrastructure costs. They do not guarantee rights to commercialize data, continuous uptime, unlimited quota, low-latency feeds or institutional execution.
8. **Make all competitor benchmarks reproducible.** Capture product name, edition/plan, date checked, official source URL, test script, test input, observed result, cost/limit and confidence. “Not found in documentation” must not be reported as “not supported.”

## 4. Risk-weighted remediation register

### P0 — Evidence integrity and research validity (block predictive claims)

**P0.1 Data rights, provenance and point-in-time semantics**
- Keep every source in a registry with source owner, canonical URL, license/terms checked, allowed storage/derivative/commercial uses, instrument, quote side (bid/ask/mid/last), bar/tick resolution, timezone and DST semantics, source refresh timestamp, checksum, acquisition method and known limitations.
- Keep DS51/reference-only until intended-use rights and source semantics receive a documented review.
- Fail closed for unknown/ambiguous timestamp semantics, duplicate timestamps, non-monotonic source indices, impossible OHLC values and suspicious gaps. Never concatenate arbitrary offsets or sort away evidence of source-order corruption.
- Do not store or redistribute raw remote data until its rights and intended-use terms permit it.

**Acceptance evidence:** a source audit report for every promoted dataset; deterministic hash; repeatable parser tests; tests for shuffled rows, duplicate/missing indices, mixed aware/naive timestamps, DST/timezone ambiguity, malformed numbers, impossible OHLC, gap intervals and source-index continuity.

**P0.2 Chronological evaluation and leakage control**
- Keep direction, next-candle return and TP-before-SL as distinct targets.
- Use time-ordered training, calibration and evaluation windows; matured-label purging/embargo where target intervals overlap; preserve a final untouched holdout.
- Compare against simple, no-trade and constant-probability baselines. Preserve every attempted strategy/configuration in an experiment ledger to make multiple-testing risk visible.
- Evaluate across non-overlapping sessions/regimes and additional independent data sources before inferring generality. Synthetic paths remain stress tests only.

**Acceptance evidence:** each fold's date range, observation/label counts, maturity cut-off, feature timestamp contract, model/config hash, baseline results and failure causes; a mutation test showing that changing future/holdout labels cannot alter earlier predictions or fitted preprocessing.

**P0.3 Net economic utility, not direction accuracy alone**
- Add an explicit execution/cost model once broker evidence is available: bid/ask entry and exit, time-dependent spread, commission, slippage, financing/swap where applicable, gaps, partial fills and order rejection.
- Until measured data exists, label each cost value as an assumption and publish sensitivity ranges. Never present assumed costs as observed broker costs.
- Report net PnL, drawdown, exposure, turnover, profit factor, trade count, tail losses and uncertainty intervals. Report direction metrics separately.

**Acceptance evidence:** reproducible gross-versus-net report; sensitivity to plausible cost levels; trade/event-level ledger; holdout results; confidence intervals that account for serial dependence where appropriate. If the edge does not survive cost and uncertainty analysis, keep `NO_TRADE`.

**P0.4 Probability calibration and abstention**
- Continue separating calibration demonstration code from financial validation.
- For the exact deployed target, evaluate Brier score, log loss, reliability bins and ECE/MCE with sample counts and uncertainty intervals; include base-rate and uncalibrated comparators.
- Fit calibration only from eligible past OOS forecasts whose outcome labels have matured before the target prediction time. Monitor drift and unsupported regimes.
- Calibrated probabilities are not automatically profitable; decision thresholds must also account for net costs, tail risk, sample uncertainty and applicability.

**Acceptance evidence:** matured-label anti-leakage tests, chronological OOS scoring, adequate observations across relevant regimes, uncertainty intervals, baseline comparisons and a documented model card. Small synthetic calibration gains must never be promoted as real-market evidence.

### P1 — Operational safety and dependable read-only service

**P1.1 Read-only production path before order routing**
- First validate the complete Telegram → application → provider/data validation → analysis → response path in an authorized staging environment.
- Add request timeouts, bounded retries, rate limiting, idempotent inbound-update handling, clear provider failure states and correlation IDs that exclude secrets/PII.
- Keep provider adapters explicitly marked as adapters until authenticated requests, response semantics and errors have been tested against an authorized sandbox/account.
- Do not create live order-routing capability as a side effect of research or connector work.

**Acceptance evidence:** recorded end-to-end tests with authorized credentials stored only in secrets; duplicate Telegram update suppression; timeout/rate-limit/provider-failure tests; no-secret logs; safe fallback to data-quality warning or `NO_TRADE`.

**P1.2 Minimum security and recovery**
- Use least-privilege tokens, secret rotation, private-chat/user allowlists, consent boundaries, input validation and bounded logs. Keep raw credentials and sensitive identifiers out of commits and report artifacts.
- Establish health checks, alerting on failed jobs, artifact retention, backup/restore tests for required state and a documented recovery procedure. SLO targets should be chosen after observing service behavior—not fabricated.
- Perform dependency and secret scanning in CI. Add auth/session hardening only to surfaces that actually need it; do not add enterprise identity features without an identified user/workflow requirement.

**Acceptance evidence:** threat model, CI security checks, staging smoke test, restore drill result, negative authorization tests and a short incident/recovery runbook.

### P1 — Benchmarking and research reproducibility

**P1.3 Real workload and failure benchmark harness**
- Maintain both a fast deterministic unit/contract suite and a separate online research suite. Label results by evidence class: real historical, broker-observed, synthetic, or software-contract.
- Pull public/community datasets only from allow-listed endpoints, respect terms and rate limits, pin checksums, bound download sizes/retries, and preserve the failed-fetch result rather than quietly swapping the source.
- Include input-quality archetypes derived from observed source failures (gaps, timezone shifts, duplicate bars, spread spikes, stale quotes) alongside synthetic adversarial cases. Do not call these historical cases unless they came from historical observations.
- Compare multiple simple strategies, costs and regime splits before spending effort on deep learning, reinforcement learning or genetic parameter search.

**Acceptance evidence:** one command/CI job reproduces data acquisition (when allowed), validation, chronological replay and score report; exact input/code/config hashes; raw JSON plus human-readable Markdown; stable tests on offline fixtures and clear classification of online versus offline results.

### P2 — Product usefulness and low-cost differentiation

**P2.1 Evidence-carrying decision output**
Each analysis should expose, as applicable: instrument and timeframe, source timestamp/timezone confidence, data-quality status, rule/model version, forecast target definition, calibrated/uncalibrated status, sample/holdout evidence, cost assumptions, regime support, reasons for abstention and “what would invalidate this view”. Never turn an internal probability into an unqualified certainty or promise of profit.

**P2.2 Telegram-first UX**
- Keep simple mode readable; offer advanced diagnostics progressively without concealing material uncertainty or risk.
- Add an explainer for `NO_TRADE`, source/data quality, methodology and limitations; exportable analysis/journal data should be introduced only after privacy and schema contracts exist.
- Prioritize reliability, clarity, accessibility, response latency and user-configurable but bounded notifications above social feeds or decorative AI features.

**P2.3 Free-first architecture**
- Use stdlib/open-source packages where their maintenance and license cost is acceptable; run local tests and use existing free CI/tier quotas carefully.
- Cache deterministic public metadata; bound provider calls; make optional expensive/external data features opt-in and display their limitations.
- Do not promise perpetual $0, guaranteed free live data, unlimited usage or institutional-grade uptime. Define quotas and degraded modes explicitly.

**Acceptance evidence:** task-based usability tests, request-cost accounting, dependency/licensing review, measured latency/error rates and a clear privacy/data-retention contract.

### P3 — Expansion after the evidence gates pass

Possible later work includes MT5/cTrader/TradingView workflow connectors, more timeframes, additional assets, richer charting, journal analytics, portfolio/risk utilities and selected ensemble/ML methods. Each increment must have an identified user job, an independently testable module, a data-rights review, maintenance-cost analysis and rollback plan. Reinforcement learning, HFT, smart order routing, social/copy trading, institution-grade compliance certifications and many asset classes are not first-wave priorities for a $0 Telegram-first product.

## 5. Competitor benchmark map

| Comparator | Correct benchmark category | Measure against Serevex | Do not infer |
|---|---|---|---|
| Bloomberg Terminal | Financial data, news, analytics and professional workflows | Data lineage, source authority, cross-asset context, search, collaboration and explainability concepts | That Serevex can reproduce proprietary/global content at $0 |
| LSEG Workspace / Refinitiv lineage | Financial data, news, market analysis and trading workflows | Provenance, point-in-time context, data coverage disclosure and workflow integration | That every paid dataset or feature is available without contract or license |
| QuantConnect / LEAN | Algorithm research and trading engine | Event/time frontier, chronological replay, broker/cost models, reproducible strategy experiments, paper/live boundary | That Serevex must rebuild an entire execution platform before a research assistant is useful |
| TradingView | Charts, scripting, alerts and trader-facing UX | Chart usability, alert reliability, script/notification workflow and clear data timestamps | That a charting/alert feature proves signal quality or grants broker-account access |
| MetaTrader 5 / TradeStation | Trading software and brokerage/execution ecosystem | Indicator/script interface, realistic tester assumptions, order lifecycle, broker compatibility and user workflow | That all brokerage features/markets/fees are available identically in every region or account |

Use official product documentation and record the actual date/plan consulted. Relevant starting references:
- Bloomberg Terminal: https://bloomberg.com/professional/solution/bloomberg-terminal/
- LSEG Workspace data and content: https://www.lseg.com/en/data-analytics/products/workspace/data-and-content
- QuantConnect LEAN overview: https://www.quantconnect.com/docs/v2/lean-engine/getting-started
- QuantConnect engine/event-flow concepts: https://www.quantconnect.com/docs/v2/writing-algorithms/key-concepts/algorithm-engine
- QuantConnect paper trading: https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/brokerages/quantconnect-paper-trading
- TradingView alerts: https://www.tradingview.com/pine-script-docs/concepts/alerts/

## 6. Evidence-gated delivery sequence

**Gate 0 — Auditability:** source/data/license register, deterministic hashes, CI result ledger, release manifest and precise separation between packaged sample and online benchmark.  
**Gate 1 — Data integrity:** provenance review, timestamp/chronology contract, robust failure tests and an approved dataset for its intended use.  
**Gate 2 — Research validity:** causal feature construction, matured labels, purged chronological folds, simple baselines, experiment tracking and untouched holdout.  
**Gate 3 — Economic validity:** broker-grounded or transparently assumed costs, net results, cost sensitivity and uncertainty intervals.  
**Gate 4 — Probability validity:** target-specific OOS calibration, calibration uncertainty, regime coverage and explicit abstention policy.  
**Gate 5 — Staging operations:** secure, authorized Telegram/provider end-to-end tests, operational monitoring, recovery and paper/shadow mode.  
**Gate 6 — Limited external beta:** explicit opt-in, no live orders by default, feedback/incident handling and predeclared evaluation criteria.  
**Gate 7 — Expansion:** only after previous gates pass, expand datasets, assets, integrations or model complexity.

Any gate failure returns to the earliest failed gate; later phases do not proceed around data, calibration or risk failures. Pass thresholds should be preregistered against an appropriate baseline before examining the final holdout, not chosen post hoc to make the result look good.

## 7. Definition of progress

Serevex moves forward when uncertainty is reduced with reproducible evidence—not merely when code volume, model complexity, feature count or marketing language grows. The first meaningful milestone is not “beat Bloomberg” or “be #1”; it is a trustworthy, affordable XAUUSD decision-support workflow whose data rights, evaluation, calibration limits, operational behavior and abstention decisions can be independently checked.

**Current decision remains:** `NO_TRADE`. No live order execution, validated trading edge, or global superiority claim is authorized by this audit.
