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
