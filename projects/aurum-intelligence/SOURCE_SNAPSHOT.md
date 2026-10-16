# Source snapshot and version boundary

The latest complete source artifact available during checkpoint creation was **Aurum Intelligence v0.9.0**. Its original ZIP contains 85 archive entries: application, tests, data sample, benchmark outputs, source register, scripts, Docker configuration, Mini App and documentation.

- Original source ZIP SHA-256: `ce35237ae32f0edafdc8288ae14edfab9d5c3df09f61d73a458e61ce782bda3f`
- Canonical dataset SHA-256: `dae535a16e13cf49f123231b1ee72e19787259e7a5f943b017a013346f36a93b`
- Validation recorded in the v0.9.0 package: 126 pytest passed, 92 unittest passed, compileall and Mini App JavaScript syntax check passed.

**Version boundary:** Separate v1.0.0 notes report hardening of mixed timezone-aware/naive timestamp validation and richer data-quality auditing. A complete v1.0.0 source ZIP/tree was not present in the workspace at checkpoint time. Treat these as reported progress pending reconciliation in the source and a rerun of all tests; do not claim that the changes are contained in v0.9.0.

**Repository-storage boundary:** this GitHub directory currently persists the project memory, status, research summary, roadmap and benchmark evidence. The binary ZIP itself has not yet been committed to GitHub. The ZIP is available as a downloadable artifact in the current ChatGPT conversation; do not treat the local/sandbox artifact as a durable GitHub file until it is committed separately.

Key source paths in the v0.9.0 archive: `app/main.py`, `app/config.py`, `app/engine/`, `app/providers/`, `app/infra/database.py`, `app/telegram/api.py`, `webapp/index.html`, `scripts/`, `tests/`, `data/` and `benchmarks/`.

No credentials are included. Never commit Telegram tokens, broker API keys, OAuth refresh tokens, personal account identifiers, or private logs.


## Current source artifact supersedes the preceding snapshot notes

The earlier v0.9.0 section records the source archive that was available before exact v1.0.0 reconciliation. On 2026-10-16, the v1.0.0 archive was retrieved and tested, then rebuilt as v1.1.0.

- Current downloadable artifact: `aurum-intelligence-v1.1.0.zip`; SHA-256 `4f3aafa682d1777a1dc6d220ad65b4b931b76760dafa8700818f8d1c0c18e623`; size 223,353 bytes.
- Base v1.1.0 clean-build evidence: 153 pytest pass, 92 unittest pass, compileall PASS, release checks PASS, and extracted ZIP re-tests PASS.
- The source package now includes the bounded external dataset benchmarking pipeline, generated smoke artifacts, ISO-8601 timestamp regression and source register DS51 (reference-only). The user project memory also has a portable online runner plus GitHub Actions workflow so the dataset is fetched from a hosted network-enabled runner.
- The ZIP is a conversation-download artifact. The memory repository does not contain the full application ZIP as a tracked binary; the source code for the CI checker is committed under this folder once this branch is reviewed/merged.
- Offline smoke results must not be substituted for live-upstream evidence. The live runner's observed row count and SHA are still pending.


## Final v1.1.0 rebuild — verified 2026-10-16

After adding the source-register entry and the portable online benchmark suite, the release was rebuilt from a clean verification path. Final artifact SHA-256 is 259103a2bf406498048e31c8ea97970c9f1ecd6bbf7636bd3bed07d7ca0ecb6e (245,370 bytes; 98 files in ZIP; manifest lists 51 sources). Final verification: pytest 169 passed; unittest discovery 102 passed; compileall PASS; source dataset audit has zero hard failures and one expected unknown-timezone warning; baseline and online-smoke benchmark entry points PASS; HTML IDs/inline JS checks PASS; extraction test reran pytest and passed. The source remains a research prototype, and remote acquisition of DS51 is pending the hosted GitHub Actions run.


## Final hosted verification — 2026-10-16, run 38851436990

The GitHub Actions run is **green**, including unit tests, live acquisition, dataset/count assertions, job summary and artifact upload. Run: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990
Artifact: https://github.com/orionrayy/ai-determinism-engine/actions/runs/38851436990/artifacts/12025522071
Artifact SHA-256: `7c661df1afafc372fd2816f2074ceb05aa7af0f8c7a388e64f07aa6b4bc43ef9` (9,208 bytes; retention expires 2026-10-30 04:35:53Z).

Fresh online sample: 35,954 rows, 3,070,226 bytes, SHA-256 `acb0d2f86b030a3ead93d92038ea27c21db6f5171acfec7a7c316faf90a9e3db`; timestamp range `2026-03-26T02:30:00+00:00`–`2026-09-25T20:40:00+00:00`; median interval 300 seconds; 136 intervals longer than 600 seconds. The remote report ran 50 timestamp cases / 250 rule comparisons, 1,200/1,200 market-path contracts and 240/240 software robustness contracts. Runtime 458.588 ms. All counts/assertions passed; decision output remains `NO_TRADE`.

The +00:00 offset is the representation seen in the file, not independent proof of the provider's intended timezone. Broker, quote-side, license scope and fill semantics remain unverified. DS51 remains `reference_only`. The positive PnL of the breakout baseline comes from only seven trades and cannot establish alpha. See `benchmarks/live_dataset_benchmark_2026-10-16.md` and its JSON companion.
