> **Identity migration note (2026-10-17):** The current project identity is **Serevex Quant Research**, with canonical path `projects/serevex-quant/`. This file preserves historical experiment descriptions, source claims, CI run IDs, checksums, and provider-issued artifact names where changing them would damage auditability. Any prior branding in historical entries is archived metadata, not active branding.

# Serevex Quant Research — historical v1.3.0 implementation checkpoint

**Checkpoint:** 2026-10-16, Asia/Jakarta. Status: research/UX prototype, not production trading software. Decision gate remains NO_TRADE.

## Implemented and locally verified

- Full-history replay evaluates all eligible sequential next-bar opportunities after 60 warm-up bars; the 50 timestamps × 5 rules output remains a separate selected-case diagnostic.
- Five contiguous, disjoint chronological folds. Folds 1–4 partition development; fold 5 is exactly the final 20% chronological tail.
- Causal prequential rate baselines for next-bar candle body up/not-up: constant 0.5, expanding Laplace, rolling 288-bar Laplace.
- Brier score, log loss, Brier skill, reliability bins, descriptive ECE/MCE and deterministic moving-block bootstrap intervals.
- Optimized fixed-rule calculations; regression tests cover rule parity, full-history coverage, disjoint folds, causality/leakage invariance, metrics and insufficient-history behavior.

Clean source release checks: 181 pytest, 109 application unittest, 11 online benchmark tests; compileall PASS; Mini App JS syntax PASS; 48 unique HTML IDs, no duplicates; ZIP integrity and extracted-archive pytest PASS.

Downloadable source artifact: aurum-intelligence-v1.3.0.zip, 278,402 bytes, SHA-256 53dabb5e86c8ce74691790e4c865d2624667cf531e572818bc571527bd4c2b67, 107 ZIP entries. ZIP is not tracked as a binary in this repository.

## Hosted online evidence

Run [#38888666520](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38888666520) passed on source commit d790435b9f09d623a795aa6ab7ee4d1732515aa5. Public dataset has 35,954 XAUUSD M5 OHLCV rows, SHA-256 acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db, date range 2026-03-26 through 2026-09-25.

- Full replay: 35,893 sequential opportunities per rule.
- Five disjoint folds: 7,175 / 7,176 / 7,176 / 7,176 / 7,190; fold 5 equals final 20% tail.
- 1,200/1,200 synthetic market-path and 240/240 software robustness contracts passed.
- All four active fixed-rule baselines lose in complete replay and tail under assumed total costs of 0.50 price units/trade.
- Expanding base-rate Brier 0.250019 vs 0.250000 for constant 0.5; the approximate improvement interval crosses zero. Rolling-288 Brier 0.250749, worse than constant. These are direction diagnostics, not TP-before-SL probabilities.

## Remaining gates

Rights/provenance, timestamp interpretation, broker quote side, Bid/Ask/tick inputs, real execution costs/fills, independent regimes, external holdout and paper trading remain unvalidated. No world-leading accuracy or profitability claim is made; live order execution remains disabled.
