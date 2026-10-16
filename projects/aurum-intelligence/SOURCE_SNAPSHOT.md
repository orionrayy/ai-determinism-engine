# Source snapshot and version boundary

The latest complete code artifact available during checkpoint creation was **Aurum Intelligence v0.9.0**, whose source ZIP contains 85 archive entries (application, tests, data, benchmark outputs, scripts, Docker configuration, docs and Mini App). Its validation report records 126 pytest passes, 92 unittest passes, compileall success and a JavaScript syntax check.

Source ZIP SHA-256: `$(sha256)` — see the v0.9.0 artifact checksum in the saved conversation artifacts. Canonical dataset SHA-256: `dae535a16e13cf49f123231b1ee72e19787259e7a5f943b017a013346f36a93b`.

**Version boundary:** separate v1.0.0 notes report a mixed-timezone timestamp validation fix and richer data-quality auditing, but the complete v1.0.0 source ZIP/tree was not materialized in the workspace at checkpoint time. This repository memory therefore documents that work as pending reconciliation; it does not claim that it is included in v0.9.0 code. Do not tag a new source release until changes are present in the tracked tree and tests are rerun.

Important source paths: `app/main.py`, `app/config.py`, `app/engine/{analysis,market_data,backtest,research_benchmark,logistic_research,barrier_benchmark,barrier_probability,market_scenario_lab,scenario_lab,cross_asset,chart,decision,real_data_cases}.py`, `app/providers/{oanda,news}.py`, `app/infra/database.py`, `app/telegram/api.py`, `webapp/index.html`, `scripts/`, `tests/`, `data/`, `benchmarks/`.

Secrets are deliberately excluded. No live bot token, broker token, OAuth refresh token, MT5/cTrader session, or personal account ID belongs in an archive.