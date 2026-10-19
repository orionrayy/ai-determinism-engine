# Portfolio Panel → Selection Guard Adapter v1.0.0

## Purpose

Connect a validated synchronized return panel to the existing White Reality Check implementation without silently dropping periods, omitting strategy trials, or promoting diagnostic panels.

## Contract

The adapter accepts only a mapping from build_synchronized_return_panel and checks:
- time-alignment status and evidence-eligible status;
- panel row count, strategy count, clock cadence, timestamp/session arrays, and benchmark-vector length;
- exact equality between panel strategy IDs and the independently registered trial IDs;
- the per-strategy ledger hash family and panel SHA-256;
- distinct SHA-256 digests for the full trial registry and frozen source dataset;
- exactly five evidence gates and references: complete trial registry, unseen locked holdout, qualified source, verified costs, and verified portfolio replay.

After structural gating, it calls selection_bias_guard_v1 with every panel return series on the common clock. It does not change values or remove rows. The underlying guard still validates timestamps, cadence, session continuity, finite returns, block-length sensitivity and evidence gates.

## Interpretation

The adapter can return BLOCKED before statistical evaluation. When it returns PASS, that means the adapter successfully validated its declared input contract and ran the statistical guard. A synthetic contract test is always labelled SYNTHETIC_ONLY_NO_MARKET_PROMOTION. A real-market evaluation lacking any independent evidence gate is blocked. Even the strongest allowed result is eligible for independent replication only; it is not trading authorization.

Gate booleans, URLs and hashes are assertions/lineage, not proof that the source is authentic or the cost model matches the broker. Evidence references must be independently reviewed. Keep NO_TRADE and both live execution flags false.
