# Artifact index — 2026-10-16 checkpoint

This page separates what is committed to GitHub from the portable artifacts attached to the ChatGPT project checkpoint.

## Committed in this repository

Directory: `projects/aurum-intelligence/`

- `README.md` — entry point and current caveats.
- `PROJECT_MEMORY.md` — architecture, source modules, provider boundaries, data status and safety rules.
- `PROGRESS_LOG.md` — milestones and resume protocol.
- `BENCHMARKS_AND_EVIDENCE.md` — exact dataset hash, test counts, strategy metrics, scenario scope and limitations.
- `RESEARCH_SUMMARY.md` — feasibility, competitive positioning, UX and operating architecture conclusions.
- `ROADMAP.md` — prioritized implementation with acceptance criteria.
- `SOURCE_SNAPSHOT.md` — release/hash and explicit boundary between source code and newer progress notes.
- `IMPLEMENTATION_STATUS.json` — machine-readable status and next actions.

## Portable files attached to the current project conversation

1. `aurum-intelligence-v0.9.0.zip`
   - Exact source snapshot; 85 entries including source, tests, data, benchmarks, scripts, docs and Mini App.
   - SHA-256: `ce35237ae32f0edafdc8288ae14edfab9d5c3df09f61d73a458e61ce782bda3f`
2. `aurum-intelligence-project-memory-v1.0.0.zip`
   - Expanded snapshot with v0.9.0 source tree, four long-form research files and project memory/checkpoint docs.
   - SHA-256: `3e57249d26d69fc5739edc5cb5a45cc18f5a22c3162a688bdae4ad01753673d9`

**Storage boundary:** the two ZIPs above are attached downloadable artifacts in the current conversation, not binary files stored in this GitHub directory. The memory documents are durable in the repository; the actual ZIP should be uploaded/committed separately before claiming that the repository itself contains the full source bundle.

## Required version reconciliation

The latest complete code snapshot is v0.9.0. Notes about v1.0.0 timestamp-validator and data-quality improvements are recorded as progress but must be reconciled into the tracked source and retested before a v1.0.0 source release is claimed.


## Latest verified artifact — v1.1.0

- `aurum-intelligence-v1.1.0.zip` — 223,353 bytes, SHA-256 `4f3aafa682d1777a1dc6d220ad65b4b931b76760dafa8700818f8d1c0c18e623`; built by the release script from reconciled v1.0.0 source and verified through fresh ZIP extraction.
- Base release verification: pytest 153 passed, unittest 92 passed, compileall PASS, release checks PASS, extracted archive pytest PASS. Later extra portable-runner tests are documented in the progress log.
- Portable hosted benchmark source: `online-benchmark/ci_benchmark.py` and `test_ci_benchmark.py`, committed under `projects/aurum-intelligence/online-benchmark/` on branch `feature/aurum-online-dataset-benchmark`.
- CI: `.github/workflows/aurum-online-dataset-benchmark.yml`; it acquires the allow-listed public CSV at runtime, does not push or persist raw data in Git, emits the SHA-256 and report, and uploads result artifacts for 14 days.
- Live upstream data: pending the first hosted workflow result. An offline smoke result is not a substitute for that verification.
- The ZIP is available as a conversation artifact and is not a committed binary file under this repository.


## Final v1.1.0 rebuild — verified 2026-10-16

After adding the source-register entry and the portable online benchmark suite, the release was rebuilt from a clean verification path. Final artifact SHA-256 is 259103a2bf406498048e31c8ea97970c9f1ecd6bbf7636bd3bed07d7ca0ecb6e (245,370 bytes; 98 files in ZIP; manifest lists 51 sources). Final verification: pytest 169 passed; unittest discovery 102 passed; compileall PASS; source dataset audit has zero hard failures and one expected unknown-timezone warning; baseline and online-smoke benchmark entry points PASS; HTML IDs/inline JS checks PASS; extraction test reran pytest and passed. The source remains a research prototype, and remote acquisition of DS51 is pending the hosted GitHub Actions run.


## Final hosted verification — 2026-10-16, run 38851436990

The GitHub Actions run is **green**, including unit tests, live acquisition, dataset/count assertions, job summary and artifact upload. Run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990
Artifact: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990/artifacts/12025522071
Artifact SHA-256: `7c661df1afafc372fd2816f2074ceb05aa7af0f8c7a388e64f07aa6b4bc43ef9` (9,208 bytes; retention expires 2026-10-30 04:35:53Z).

Fresh online sample: 35,954 rows, 3,070,226 bytes, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`; timestamp range `2026-03-26T02:30:00+00:00`–`2026-09-25T20:40:00+00:00`; median interval 300 seconds; 136 intervals longer than 600 seconds. The remote report ran 50 timestamp cases / 250 rule comparisons, 1,200/1,200 market-path contracts and 240/240 software robustness contracts. Runtime 458.588 ms. All counts/assertions passed; decision output remains `NO_TRADE`.

The +00:00 offset is the representation seen in the file, not independent proof of the provider's intended timezone. Broker, quote-side, license scope and fill semantics remain unverified. DS51 remains `reference_only`. The positive PnL of the breakout baseline comes from only seven trades and cannot establish alpha. See `benchmarks/live_dataset_benchmark_2026-10-16.md` and its JSON companion.

## Latest verified artifact — v1.2.0

- File: aurum-intelligence-v1.2.0.zip, 260,781 bytes, SHA-256 19912b690e74af802a7170316703d40048a67731a1eb26d806c9929f048d2db3; 105 ZIP entries.
- Verification: 176 pytest, 109 unittest, compileall, Mini App JS syntax, HTML uniqueness, ZIP integrity and clean-extraction test all pass.
- New v1.2 changes: Mini App Sederhana/Tingkat lanjut switch; Telegram equivalent; SQLite migration; four new regression tests for mode UI plus database mode selection/migration. Benchmark version labels read VERSION.
- Hosted online benchmark run #38851629049 and outputs are preserved in the source artifact and documented in V1_2_0_IMPLEMENTATION_CHECKPOINT.md. Result: 35,954 rows / 250 baseline evaluations / 1,200 market-path scenarios / 240 software stress cases, all structural/scenario checks passing. The sample ends 2026-09-25; it is not live data.
- **Storage boundary:** the ZIP has not been committed as a binary to the GitHub folder. It is available as a downloadable source artifact. Project memory and release metadata are committed on this branch; this PR is still draft and must not be treated as merged.

## v1.3.0 persisted full-history evidence

- Current local release: aurum-intelligence-v1.3.0.zip, 278,402 bytes, SHA-256 53dabb5e86c8ce74691790e4c865d2624667cf531e572818bc571527bd4c2b67; 107 ZIP entries. Tests: 181 pytest, 109 unittest, 11 online benchmark tests, compileall, browser syntax/HTML checks and clean extraction all pass.
- Hosted run [#38888666520](https://github.com/orionrayy/ai-determinism-engine/actions/runs/38888666520): 35,954 public community rows, 35,893 all-bar decision opportunities per fixed rule, five disjoint chronological folds, 1,200 market-path and 240 robustness checks pass.
- Permanent summaries: projects/aurum-intelligence/benchmarks/full_history_replay_2026-10-16.md and .json. No raw 35,954-row CSV is redistributed.
- All four active fixed-rule baselines are net-negative under assumed costs in full history and final tail. No calibrated trading model exists; NO_TRADE remains active.

## v1.3.1 hosted CI artifact

- Run #38897894594 passed on commit `6017eec3141fb803907583abeca006a230e90be4`.
- Artifact #12044756186, 14,966 bytes; SHA-256 `a2561de0161e79dbaac794988b5545cd8f32fe2cf55d38c875e8aed91e99d9b3`.
- Includes full-history diagnostics, direct-case outcomes, 1,200 market scenarios, 240 software scenarios and the 240/240 temporal-calibration contract report.
- The app release ZIP is downloadable, not committed as a binary. No raw public CSV is redistributed.


## Target-specific TP-before-SL OOS diagnostic candidate — 2026-10-17

- `online-benchmark/barrier_oos_benchmark.py` — fixed-protocol long/short TP-before-SL diagnostic with causal features, matured-label purging, calibration-period Platt fit, untouched chronological holdout and proper-score/reliability metrics.
- `online-benchmark/test_barrier_oos_benchmark.py` — feature-causality, barrier-collision, gap-open, maturation-boundary and no-live-execution regression tests.
- `ONLINE_MULTI_SOURCE_BENCHMARK.md` — protocol and caveats. New hosted CI is pending; no result may be described as validated until its run/artifact is inspected.
- The benchmark consumes the already-downloaded public-source rows in CI and does not commit raw market CSVs. It is not yet a rebuilt/released Aurum application ZIP and it does not alter the Telegram/Mini App production path.


## Competitive gap audit — 2026-10-17

- `GAP_AUDIT_AND_COMPETITIVE_POSITION_2026-10-17.md` — evidence-led gap register, risk-weighted implementation phases, competitor category distinctions and free-first guardrails.
