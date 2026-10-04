# v77 Evidence Access and Source Routing

## Goal

Make free-first research materially more robust by adding a public metadata fallback, explicit access-level classification, and clearer provenance/authority signals without claiming metadata indexes are primary evidence.

## Design

- Add Crossref to the free provider pool. Crossref REST is publicly accessible without signup; use optional MAILTO identification when configured, cache results, and retain bounded retry/backoff. Crossref metadata is an access/identity route, not proof of claim truth.
- Add deterministic evidence fields:
  - access_level: L0 locator, L1 metadata, L2 abstract/structured text, L3 full-text URL, L4 primary API/dataset.
  - access_route: provider/search/oa/full_text metadata route.
  - authority_tier: metadata_index, scholarly_index, biomedical_index, preprint_repository, peer_review_signal.
  - authority_score: routing heuristic only, never truth probability.
- Preserve work-level independence as an explicit proxy; do not count provider copies as independent works.
- Free-only mode must reject paid/key-required CORE and OpenAI paths before network access.
- Existing cache remains the primary quota-control mechanism.

## Acceptance

1. Free provider order is semantic_scholar, europe_pmc, crossref.
2. Crossref works without an API key.
3. Cache and bounded retry behavior remains intact.
4. Normalized records expose access_level/access_route/authority_tier/authority_score.
5. Full-text URL raises access level above abstract/metadata, but never changes authority tier by itself.
6. Evidence independence count remains work-based.
7. Full test suite and free-only evaluation remain green.
