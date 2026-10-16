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
