# Aurum bid/ask tick probe — verified hosted run

**CI result:** PASS for the bounded read-only sample, not a strategy or execution-cost validation.

- Workflow [#39021204557](https://github.com/orionrayy/ai-determinism-engine/actions/runs/39021204557) passed all 13 steps on commit `f2706642ed8da7c7727b53881ec7e045473ae06c`.
- Artifact #12096921783: 55,004 bytes, SHA-256 `cca5e4e8c735b24261bfafc336a749a5422d97c3cd7472196dd4bc8dcd22ab27`.
- Dataset [CarlosSilva1/xauusd-ticks](https://huggingface.co/datasets/CarlosSilva1/xauusd-ticks) reports 286,823,542 rows and claims CC-BY-4.0. The probe reads only 3,000 quotes via 30 API pages (614,982 response bytes); raw rows are not committed or retained.

## Observed spread distribution

Spread = ask − bid in feed price units; bps are computed relative to quote midpoint. Timezone is normalized to UTC only under the dataset-card claim and is not independently verified.

| Window | Row offsets | UTC timestamps assumed | Median spread | P95 | P99 | Maximum | Median bps | P95 bps | P99 bps | Crossed/locked | Sample SHA-256 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Head | 0–999 | 2021-05-24 00:00:00.256 to 00:08:50.766 | 0.417 | 0.447 | 0.457 | 0.491 | 2.215 | 2.374 | 2.428 | 0 / 0 | `43ec1d9b4441db7984e04aceb1391c83f6897289b51e6d7c7d2bca3c58daf0a3` |
| Middle | 143,411,700–143,412,699 | 2024-07-10 13:23:26.657 to 13:26:34.937 | 0.377 | 0.414 | 0.447 | 0.514 | 1.585 | 1.741 | 1.879 | 0 / 0 | `6708dfd3c96e12f5935d07370411931c47b4313ba9569dc1e8c88651d49f664a` |
| Tail | 286,822,500–286,823,499 | 2026-05-22 20:50:01.172 to 20:59:37.043 | 1.040 | 1.504 | 1.677 | 2.104 | 2.307 | 3.335 | 3.719 | 0 / 0 | `e26223d5cb9efa272abac02915855f16c9fb5abcf110545ec7a029ecd79770f1` |

All 3,000 sampled rows had positive spreads, with no crossed/locked quotes or timestamp-order violations in these windows. The tail median spread is about 2.48× the head median; its P95 is about 3.36×. This is only a three-window sample, not a continuous spread regime model.

The probe shows why OHLCV-only replay cannot supply observed Bid/Ask distributions. It does **not** establish actual fills, slippage, independently verified broker lineage, full-dataset quality, commercial data rights, or executable cost calibration. Upstream data rights and provenance remain unresolved.

Production cost calibration and commercial model training remain blocked. Decision gate: `NO_TRADE`; live signals/orders disabled.
