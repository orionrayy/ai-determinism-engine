# Serevex Quant Research — identity and evidence policy

## Canonical identity

The active project directory is `projects/serevex-quant/`. The active hosted benchmark workflow is `.github/workflows/serevex-research-benchmark.yml`. Current package labels, executable user agents, benchmark output prefixes, and project status use the Serevex identity.

## Evidence-preservation rule

Historical CI run IDs, immutable digests, and externally issued artifact filenames are evidence identifiers. They must not be rewritten where that would imply an artifact was originally emitted under a different filename. Current outputs use the new identity; historical identifiers remain unchanged where needed for reproducibility.

## Release boundary

The project remains a quantitative research prototype. A passing CI pipeline verifies the procedures exercised by that workflow; it does not establish predictive skill, calibration quality, net profitability, source-license eligibility, or readiness to route orders. The research-readiness gate therefore keeps `NO_TRADE`, live signals disabled, and live order execution disabled.

## Renaming acceptance

- The canonical project directory and active benchmark workflow use the Serevex identity.
- The active README, status, roadmap, benchmark entry points, and release gate use the new identity.
- A contract test rejects stale names in the active code and workflow surface.
- Historical experiment evidence remains traceable by immutable run IDs, commit hashes, and artifact digests.
