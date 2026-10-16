# Source snapshot and version boundary

The latest complete source artifact available during checkpoint creation was **Aurum Intelligence v0.9.0**. Its original ZIP contains 85 archive entries: application, tests, data sample, benchmark outputs, source register, scripts, Docker configuration, Mini App and documentation.

- Original source ZIP SHA-256: `ce35237ae32f0edafdc8288ae14edfab9d5c3df09f61d73a458e61ce782bda3f`
- Canonical dataset SHA-256: `dae535a16e13cf49f123231b1ee72e19787259e7a5f943b017a013346f36a93b`
- Validation recorded in the v0.9.0 package: 126 pytest passed, 92 unittest passed, compileall and Mini App JavaScript syntax check passed.

**Version boundary:** Separate v1.0.0 notes report hardening of mixed timezone-aware/naive timestamp validation and richer data-quality auditing. A complete v1.0.0 source ZIP/tree was not present in the workspace at checkpoint time. Treat these as reported progress pending reconciliation in the source and a rerun of all tests; do not claim that the changes are contained in v0.9.0.

**Repository-storage boundary:** this GitHub directory currently persists the project memory, status, research summary, roadmap and benchmark evidence. The binary ZIP itself has not yet been committed to GitHub. The ZIP is available as a downloadable artifact in the current ChatGPT conversation; do not treat the local/sandbox artifact as a durable GitHub file until it is committed separately.

Key source paths in the v0.9.0 archive: `app/main.py`, `app/config.py`, `app/engine/`, `app/providers/`, `app/infra/database.py`, `app/telegram/api.py`, `webapp/index.html`, `scripts/`, `tests/`, `data/` and `benchmarks/`.

No credentials are included. Never commit Telegram tokens, broker API keys, OAuth refresh tokens, personal account identifiers, or private logs.
