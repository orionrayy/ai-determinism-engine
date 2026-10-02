# ORCHESTRATION MEMORY — Canonical Control-Plane Context

## Purpose

This file is the durable, version-controlled memory of the AI orchestration control plane. Before modifying the orchestrator, treat this document and the current source tree as the source of truth. Do not assume older conversation state is still accurate without checking the repository.

## Current baseline

Repository: `orionrayy/ai-determinism-engine`
Primary branch: `main`
Current main baseline: orchestration hardening v55 durable connector output redaction + v54 connector output contracts/response bounds + v53 terminal workflow state lifecycle/compaction + v52 identity-first ingress deduplication + v51 repository-event target routing + v50 goal-ingress state-load repair + v49 targeted workflow hydration + v48 sharded workflow persistence + v47 continuation/ingress/persistence race closure + v46 discovery snapshot provenance + v45 SSRF-safe artifact verification + v44 free-only reconciliation cost closure + v43 connector upstream cost gate + v42 free Gemini model gate + v41 private structured input boundary + v40 recovery routing/exact Actions run-attempt binding + v39 federation fairness/backpressure + earlier durable control-plane generations.
Execution model: GitHub Actions + stdlib Python
Cost policy: free-first; `ORCHESTRATOR_FREE_ONLY=true` in the production workflow
Current execution-fabric branch: `main`


## Orchestration hardening v42 — free Gemini model gate
- `orchestrator/tools.json` is now authoritative for Gemini model cost policy: it declares the default model plus an explicit `free_models` allowlist.