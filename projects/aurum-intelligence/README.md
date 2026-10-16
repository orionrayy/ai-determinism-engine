# Aurum Intelligence — persistent project memory

This directory is the durable working notebook for the XAUUSD/gold analysis and Telegram-assistant project. It records architecture, progress, benchmarking, source provenance, current blockers, and a prioritized roadmap.

Start with **PROJECT_MEMORY.md**, then **PROGRESS_LOG.md**, **BENCHMARKS_AND_EVIDENCE.md**, **RESEARCH_SUMMARY.md**, **ROADMAP.md**, and **SOURCE_SNAPSHOT.md**.

## Current state (2026-10-16)
- Latest complete source artifact available for audit: v0.9.0; it contains 85 archive entries with app code, tests, data sample, benchmark results, scripts, provider adapters, and Mini App.
- A v1.0.0 progress checkpoint reports timestamp-validator hardening and richer dataset-quality diagnostics, but a complete v1.0.0 source archive was not present in the working filesystem. Those changes must be reconciled into the source and re-tested before claiming a v1.0.0 code release.
- Current evidence is a 60-bar XAUUSD M5 community excerpt; 50 sources are registered, only one sampled.
- 250 case-based next-candle rows, 90 TP/SL barrier rows, and two separate 240-case matrices (software robustness and synthetic market paths) are diagnostics, not proof of predictive edge.
- ML candidates return `insufficient_history` and zero OOS forecasts on the bundled sample. No probability is calibrated; production signal gate remains `NO_TRADE`.

Never add credentials, tokens, .env files, or personal account information to this folder.