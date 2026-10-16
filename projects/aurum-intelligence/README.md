# Aurum Intelligence — persistent project memory

This directory is the durable working notebook for the XAUUSD/gold analysis and Telegram-assistant project. It records architecture, progress, benchmarking, source provenance, current blockers, and a prioritized roadmap.

Start with **PROJECT_MEMORY.md**, then **PROGRESS_LOG.md**, **BENCHMARKS_AND_EVIDENCE.md**, **RESEARCH_SUMMARY.md**, **ROADMAP.md**, and **SOURCE_SNAPSHOT.md**.

## Current state (2026-10-16; latest checkpoint v1.1.0)
- The reconciled complete application source is built as `aurum-intelligence-v1.1.0.zip` (245,370 bytes; SHA-256 `259103a2bf406498048e31c8ea97970c9f1ecd6bbf7636bd3bed07d7ca0ecb6e`). It is a downloadable conversation artifact, not a binary committed to this folder.
- Final release builder evidence: pytest 169 passed, unittest 102 passed, compileall and all release checks passed; the 98-file archive was extracted and pytest rerun.
- Primary bundled dataset remains 60 community XAUUSD M5 bars with unknown timezone and broker/execution semantics. Source register now has 51 entries; DS51 is a separate GetData public OHLCV candidate and remains reference-only until live CI acquisition, quality review, and intended-use licensing checks pass.
- The new online CI benchmark is in `.github/workflows/aurum-online-dataset-benchmark.yml`; it requires 50 chronological timestamps, 250 fixed-rule comparisons, 200+ market scenarios, 200+ software stress cases, and uploads JSON/CSV/Markdown evidence without committing the raw feed.
- Verified offline smoke: 30 timestamp cases / 150 baseline comparisons, 480 market paths, and 240 software input cases passed their expected validation contracts. The hosted live dataset benchmark now passes: 35,954 rows, 50 cases/250 baseline evaluations, 1,200/1,200 market paths, and 240/240 software cases. See `benchmarks/live_dataset_benchmark_2026-10-16.md`; source terms, intended timezone and broker semantics remain under review, so DS51 remains reference-only.
- These tests do not show predictive edge. Model forecasts remain unvalidated and production decision output stays `NO_TRADE`.

Never add credentials, tokens, .env files, or personal account information to this folder.

## Final hosted verification — 2026-10-16, run 38851436990

The GitHub Actions run is **green**, including unit tests, live acquisition, dataset/count assertions, job summary and artifact upload. Run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990
Artifact: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990/artifacts/12025522071
Artifact SHA-256: `7c661df1afafc372fd2816f2074ceb05aa7af0f8c7a388e64f07aa6b4bc43ef9` (9,208 bytes; retention expires 2026-10-30 04:35:53Z).

Fresh online sample: 35,954 rows, 3,070,226 bytes, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`; timestamp range `2026-03-26T02:30:00+00:00`–`2026-09-25T20:40:00+00:00`; median interval 300 seconds; 136 intervals longer than 600 seconds. The remote report ran 50 timestamp cases / 250 rule comparisons, 1,200/1,200 market-path contracts and 240/240 software robustness contracts. Runtime 458.588 ms. All counts/assertions passed; decision output remains `NO_TRADE`.

The +00:00 offset is the representation seen in the file, not independent proof of the provider's intended timezone. Broker, quote-side, license scope and fill semantics remain unverified. DS51 remains `reference_only`. The positive PnL of the breakout baseline comes from only seven trades and cannot establish alpha. See `benchmarks/live_dataset_benchmark_2026-10-16.md` and its JSON companion.
