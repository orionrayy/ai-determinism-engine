# Aurum Intelligence — persistent project memory

This directory is the durable working notebook for the XAUUSD/gold analysis and Telegram-assistant project. It records architecture, progress, benchmarking, source provenance, current blockers, and a prioritized roadmap.

Start with **PROJECT_MEMORY.md**, then **PROGRESS_LOG.md**, **BENCHMARKS_AND_EVIDENCE.md**, **RESEARCH_SUMMARY.md**, **ROADMAP.md**, and **SOURCE_SNAPSHOT.md**.

## Current state (2026-10-16; latest checkpoint v1.2.0)
- Latest source artifact: aurum-intelligence-v1.2.0.zip (260,781 bytes; SHA-256 19912b690e74af802a7170316703d40048a67731a1eb26d806c9929f048d2db3). It remains a downloadable artifact, not a binary tracked in this folder.
- Clean release validation: 176 pytest passed; 109 unittest passed; compileall, inline JS syntax and HTML-ID audit pass; clean ZIP extraction reruns 176 tests.
- UX increment: persistent-accessible Sederhana/Tingkat lanjut mode in the Mini App; Telegram mode preference backed by SQLite with backward-compatible migration. Mini App mode is browser-local, not server-synchronized. Both modes preserve the NO_TRADE decision gate.
- Local reproducible dataset remains 60 public community XAUUSD M5 bars, with unknown timezone/broker provenance. Release smoke has 50 observed cases / 250 comparisons, 1,200/1,200 market-path contracts and 240/240 software robustness cases.
- Hosted online CI run [#38851629049](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851629049) passed on the benchmark branch using 35,954 public CSV rows (SHA-256 acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db), 50 timestamp cases / 250 evaluations, 1,200/1,200 market paths and 240/240 software checks. The sampled data ends 2026-09-25; it is not a live feed.
- The source repository declares an MIT license; market-data rights scope/provenance, quote side, broker semantics and actual fills still need review. The raw 3 MB CSV is not redistributed. Forecast model has zero out-of-sample predictions on the bundled sample and is not calibrated; model superiority is not claimed.
- Live Telegram/OANDA credentials, MT5/cTrader account integration, real browser/mobile acceptance testing, paper trading and deployment are not validated. Keep trading disabled.

Never add credentials, tokens, .env files, or personal account information to this folder.

## Final hosted verification — 2026-10-16, run 38851436990

The GitHub Actions run is **green**, including unit tests, live acquisition, dataset/count assertions, job summary and artifact upload. Run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990
Artifact: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990/artifacts/12025522071
Artifact SHA-256: `7c661df1afafc372fd2816f2074ceb05aa7af0f8c7a388e64f07aa6b4bc43ef9` (9,208 bytes; retention expires 2026-10-30 04:35:53Z).

Fresh online sample: 35,954 rows, 3,070,226 bytes, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`; timestamp range `2026-03-26T02:30:00+00:00`–`2026-09-25T20:40:00+00:00`; median interval 300 seconds; 136 intervals longer than 600 seconds. The remote report ran 50 timestamp cases / 250 rule comparisons, 1,200/1,200 market-path contracts and 240/240 software robustness contracts. Runtime 458.588 ms. All counts/assertions passed; decision output remains `NO_TRADE`.

The +00:00 offset is the representation seen in the file, not independent proof of the provider's intended timezone. Broker, quote-side, license scope and fill semantics remain unverified. DS51 remains `reference_only`. The positive PnL of the breakout baseline comes from only seven trades and cannot establish alpha. See `benchmarks/live_dataset_benchmark_2026-10-16.md` and its JSON companion.


## Latest checkpoint — v1.3.0 (local release verified; hosted full-history benchmark pending)

- Source artifact: aurum-intelligence-v1.3.0.zip, SHA-256 51a82f710033eee0cb7723c8940721d8c3bcd64565159650ba90e6235945ff0b, 267,854 bytes.
- Local tests: 181 pytest, 109 unittest, 11 dedicated online-runner tests; compiler, UI static checks and clean ZIP extraction passed.
- Online benchmark now replays all eligible bars, reports five chronological folds, and evaluates prequential probability baselines with Brier/log loss, reliability/ECE/MCE and a deterministic moving-block bootstrap. The successful live dataset result from the previous code was only 50 sampled timestamps; the new full-history metrics remain pending a successful workflow run on the v1.3.0 source.
- Decision gate remains NO_TRADE. Synthetic 35,954-row throughput is a contract test, not real-market predictive evidence.
