# Aurum Intelligence — persistent project memory

This directory is the durable working notebook for the XAUUSD/gold analysis and Telegram-assistant project. It records architecture, progress, benchmarking, source provenance, current blockers, and a prioritized roadmap.

Start with **PROJECT_MEMORY.md**, then **PROGRESS_LOG.md**, **BENCHMARKS_AND_EVIDENCE.md**, **RESEARCH_SUMMARY.md**, **ROADMAP.md**, and **SOURCE_SNAPSHOT.md**.

## Current state (2026-10-16; latest checkpoint v1.1.0)
- The reconciled complete application source is built as `aurum-intelligence-v1.1.0.zip` (245,370 bytes; SHA-256 `259103a2bf406498048e31c8ea97970c9f1ecd6bbf7636bd3bed07d7ca0ecb6e`). It is a downloadable conversation artifact, not a binary committed to this folder.
- Final release builder evidence: pytest 169 passed, unittest 102 passed, compileall and all release checks passed; the 98-file archive was extracted and pytest rerun.
- Primary bundled dataset remains 60 community XAUUSD M5 bars with unknown timezone and broker/execution semantics. Source register now has 51 entries; DS51 is a separate GetData public OHLCV candidate and remains reference-only until live CI acquisition, quality review, and intended-use licensing checks pass.
- The new online CI benchmark is in `.github/workflows/aurum-online-dataset-benchmark.yml`; it requires 50 chronological timestamps, 250 fixed-rule comparisons, 200+ market scenarios, 200+ software stress cases, and uploads JSON/CSV/Markdown evidence without committing the raw feed.
- Verified offline smoke: 30 timestamp cases / 150 baseline comparisons, 480 market paths, and 240 software input cases passed their expected validation contracts. The live hosted dataset result remains pending until the workflow runs.
- These tests do not show predictive edge. Model forecasts remain unvalidated and production decision output stays `NO_TRADE`.

Never add credentials, tokens, .env files, or personal account information to this folder.