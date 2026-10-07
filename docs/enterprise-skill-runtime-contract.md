# Skill Runtime and Research Capability Contract

Status: design specification only; no dynamic code execution enabled by this document.
Date: 2026-10-05

## 1. Core rule

Future exploration and multi-skill execution must follow:

```text
discover -> validate -> authorize -> budget -> schedule -> execute -> verify -> learn
```

Never allow discover -> execute.

## 2. Skill identity

A skill is identified by skill_id, semantic version and manifest digest. Mutable repository branches, URLs or repository HEADs are not immutable execution identities.

## 3. Skill manifest

```yaml
skill_id: research/search
version: 1.0.0
manifest_digest: sha256:...
capabilities: [research.search]
input_schema: ...
output_schema: ...
runtime:
  kind: python
  entrypoint: package.module:function
  protocol_version: 1
permissions:
  network: restricted
  filesystem: workspace
  secrets: []
  side_effects: false
policy:
  risk_class: read_only
  free_tier: bounded
  max_requests: 20
  max_runtime_s: 120
quality:
  validation_level: evidence_bound
  replayable: true
provenance:
  source_uri: ...
  source_digest: sha256:...
compatibility:
  orchestrator_protocol_min: 1
  orchestrator_protocol_max: 1
```

Eligibility is deterministic:

```text
manifest.valid
AND policy.allows
AND runtime.compatible
AND capability.matches
AND budget.available
```

Scoring can rank eligible candidates but can never override those filters.

## 4. Registry trust

Recommended precedence:

1. built-in trusted registry;
2. repository-local registry;
3. organization-signed registry;
4. optional external discovery index.

External discovery can propose a skill. It cannot silently replace a higher-trust implementation.

## 5. Lifecycle and health

```text
DISCOVERED -> VALIDATED -> ENABLED -> HEALTHY
                              |
                              v
                         DEGRADED -> QUARANTINED -> DISABLED
```

Track success rate, validation failure rate, timeout rate, retry rate, contradiction rate, p95 latency, provider throttling, billing violations and manifest freshness.

Health may lower priority or quarantine a skill. It must never broaden permissions.

## 6. Dynamic skill security boundary

Discovered code is untrusted until validated. The controller must never import or execute arbitrary discovered code.

Execution requirements:

- validate manifest;
- verify implementation digest;
- pin package/runtime version;
- run in an isolated worker or sandbox;
- restrict network egress;
- restrict filesystem access;
- provide only declared secrets;
- validate output against the contract;
- record provenance and execution identity.

This allows future multi-skill expansion without turning the scheduler into a code-execution trust boundary.

## 7. Research skill graph

Make research composable:

```text
research.query_decompose
research.search
research.retrieve
research.deduplicate
research.verify_access
research.extract_passage
research.detect_retraction
research.detect_conflict
research.build_claim_graph
research.synthesize
research.archive
```

Each capability can be independently routed, rate-limited and quality-scored.

## 8. Research source contract

Every normalized Evidence Record should preserve:

```text
canonical_id
provider
provider_id
title
venue
year
doi/pmid/arxiv/pmcid
landing_url
full_text_url
access_level
access_route
access_verification
publication_status
retraction_signal
independence_key
source_digest
corpus_digest
retrieved_at
claim_refs
```

An evidence record should be immutable after verification except through explicit versioned corrections.

## 9. Passage-level evidence

Material direct-supported claims must satisfy:

```text
claim
 -> evidence_refs
 -> validated evidence record
 -> exact normalized passage/span
 -> corpus match
```

Model confidence must not substitute for evidence authority.

The repository's current v79 passage validation, retraction awareness, challenge-scoped references and conservative truth-lock should be extracted as reusable research primitives rather than remaining coupled to one workflow path.

## 10. Exploration protocol

Autonomous exploration is a bounded workload producer. Every run gets:

```text
max_depth
max_queries
max_sources
max_providers
max_llm_calls
max_runtime
max_bytes
max_retries
allowed_skills
allowed_domains
stop_conditions
```

Budget changes are explicit policy transitions. The LLM cannot self-authorize more budget.

## 11. Staged research strategy

Stage A — discovery: optimize recall cheaply.
Stage B — verification: confirm source identity and access.
Stage C — extraction: collect bounded passages from validated corpus.
Stage D — conflict analysis: compare independent evidence and preserve uncertainty.
Stage E — synthesis: write only from the adjudicated evidence graph.

This separates discovery from epistemic authority.

## 12. Multi-skill composition

A workflow should exchange immutable references between skills:

```text
research.search
 -> research.retrieve
 -> research.verify_access
 -> research.extract_passage
 -> analysis.compare
 -> synthesis.write
 -> artifact.verify
```

Large outputs become artifact references rather than shared mutable state.

## 13. Free-first policy

Use explicit billing classes:

```text
free_public
free_allowance
credentialed_optional
paid
unknown
```

With hard-free enabled:

- free_public is allowed;
- free_allowance is allowed only while its local allowance remains;
- credentialed_optional, paid and unknown are rejected.

The billing class is part of policy, not a description supplied by the worker.

## 14. Research memory

Persist research as an immutable-reference graph:

```text
Query
 -> Search Run
 -> Source
 -> Passage
 -> Claim
 -> Verification
 -> Synthesis
 -> Artifact
```

Adapters may project this graph into SQLite, PostgreSQL, Git or a search/vector index. Canonical identifiers and digests must remain storage-independent.

## 15. Learning loop

The future quality ledger may learn from validation success, validation failure, rework count, contradiction count, latency, resource use, provider throttling and human override.

Learning is ranking only. A learned score cannot override capability compatibility, security policy, evidence requirements, cost policy or tenant isolation.

## 16. Enterprise implication

The skill/runtime layer is effectively an operating system boundary for AI work:

```text
registry      = package/capability discovery
scheduler     = policy kernel
task broker   = process delivery
workers       = execution environments
evidence graph= epistemic data plane
artifacts     = durable outputs
policy/budget = resource control
telemetry     = observability
Git           = portable audit surface
```

This gives the current deterministic engine a path to future exploration, dynamic research, and multi-skill composition without making the GitHub repository itself the runtime scheduler.