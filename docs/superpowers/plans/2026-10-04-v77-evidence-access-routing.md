# v77 Evidence Access and Source Routing Implementation Plan

## Task 1 — RED
Modify orchestrator/test_research_providers.py and add/extend evidence record tests for:
- Crossref public payload normalization.
- Crossref included in free-only default order.
- Crossref search function does not require an API key.
- access metadata classification for metadata-only, abstract, and full-text records.
- authority score is bounded 0..1 and explicitly marked as heuristic.

Run focused tests and confirm RED.

## Task 2 — GREEN
Modify:
- orchestrator/research_budget.py
- orchestrator/research_providers.py
- orchestrator/evidence_records.py
- relevant tests.

Implement public Crossref search with optional CROSSREF_MAILTO; use the existing cache and retry wrapper.
Implement deterministic access/authority metadata derivation.

## Task 3 — Verification
Run compileall, full unittest suite, actionlint, worker syntax, and ORCHESTRATOR_FREE_ONLY=true evaluation.

## Non-goals
- no paid provider;
- no full-text downloader;
- no external evidence database;
- no semantic entailment model;
- no claim that source authority score equals factual correctness.
