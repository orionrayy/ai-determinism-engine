# Evidence Gate v1 — strict temporal semantics and selection-bias control

**Status:** research-only control introduced on an isolated branch; not merged, not deployed, and not market-validated.

## Why this increment exists

The previous review found two distinct problems:

1. Cross-source matching silently interpreted timezone-naive timestamps as UTC. This can be useful as a documented diagnostic assumption, but it is not verified temporal semantics. The default is now to reject naive timestamps; an explicit assume-UTC-for-diagnostics-only override permits comparison while preserving timestamp_semantics_verified=false and decision_output=NO_TRADE.
2. The current replay generates strategy events, but no verified regular-cadence, capital-aware net-portfolio return panel with a complete trial registry exists. Event-level overlapping trade rows cannot be treated as an aligned return stream for family-wise testing.

## Selection-adjusted inference contract

selection_bias_guard_v1.py implements a non-studentized White Reality Check max-mean statistic with a synchronized, segmented moving-block bootstrap:

- One column per materially tested strategy/configuration; columns remain aligned by timestamp.
- Explicit benchmark vector; include no-trade periods as zero returns rather than dropping them.
- Exact cadence within documented session segments; sessions cannot silently be stitched together.
- Shared row indices across strategies preserve cross-strategy dependence; blocks preserve short-horizon time dependence without crossing session boundaries.
- Bootstrap centers each strategy's return differential under the null and tests the maximum mean across the candidate family.
- A preregistered set of block lengths is evaluated. The candidate family does not pass this gate unless every configured block-length sensitivity rejects the null at the specified alpha and the observed best mean excess is positive.
- The full trial registry, dataset hash, cost evidence, source qualification, locked holdout and verified portfolio replay must all have stable references before an actual-market result can be eligible for independent replication.
- Synthetic fixtures are tagged synthetic_contract_test and can never yield market-promotion status, regardless of p-values.

This is one inferential gate, not a comprehensive solution to selection bias. It is not White's Superior Predictive Ability test and does not implement Deflated Sharpe Ratio. It does not handle unregistered analyst discretion, strategy families omitted from the registry, or every form of regime shift. Its p-value is diagnostic under the specified block-bootstrap design.

## Explicit non-goals

This change does not change the Telegram decision path, invent new alpha, change existing TP/SL outcomes, correct source prices, infer missing timestamps, enable live signals, authorize orders or declare any strategy profitable.

## Acceptance checks

- Unit/regression tests reject timestamp assumptions without explicit policy.
- Unit/regression tests validate alignment, finite values, timestamp cadence, session segmentation, deterministic seeds and evidence gates.
- Synthetic positive-effect fixtures test algorithm behavior only and are never represented as market evidence.
- Existing hosted benchmark and quote-side replay must pass unchanged. The new guard is not wired to historical sampled anchors because that sample is not a regular-cadence portfolio return panel.
- Existing NO_TRADE and live-execution-disabled boundaries remain in force.

## Research references

- White, H. (2000), “A Reality Check for Data Snooping,” Econometrica, DOI: 10.1111/1468-0262.00152.
- Sullivan, R., Timmermann, A. and White, H. (1999), “Data-Snooping, Technical Trading Rule Performance, and the Bootstrap,” Journal of Finance 54(5), 1647–1691, DOI: 10.1111/0022-1082.00163.
- Bailey, D. H. and López de Prado, M. (2014), “The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality,” Journal of Portfolio Management 40(5), 94–107, DOI: 10.3905/jpm.2014.40.5.094.
- CFTC official COT release schedule: https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm.
- FRED vintage dates / ALFRED historical values: https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html and https://alfred.stlouisfed.org/help/downloaddata.

The cited papers motivate controls but do not validate this project's signals. Reference selection and protocol choices need independent review before any production claim.
