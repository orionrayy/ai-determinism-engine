# General Blueprint Orchestration

The deterministic blueprint compiler is a side-effect-free workload adapter. It normalizes structured specifications and converts them into bounded, traceable execution units. It never executes tools and never bypasses the supervisor's DAG, federation, checkpoint, retry/replan, idempotency, risk, or approval policy.

Structured blueprints support requirements, dependencies, workstreams, constraints, risk, agent-role hints, capabilities, acceptance criteria, artifacts, and source references. Markdown/text ingestion only extracts headings and provenance metadata; semantic requirement extraction remains an explicit research/analysis step.

Hard resource bounds: 256 KiB source/structured input, 512 requirements, 64 execution units, 8 requirements per unit, 24 KiB per unit/packet, and 480 KiB per compilation manifest. File ingestion is confined to ORCHESTRATOR_WORKLOAD_ROOT.

Recommended lifecycle: ingest -> normalize -> reconcile -> compile bounded units -> validate unit DAG -> execute unit -> checkpoint/evidence -> validate -> retry/replan -> continue.

The capability is credential-free and free-first. Downstream providers remain subject to the existing registry and free-only policy.
