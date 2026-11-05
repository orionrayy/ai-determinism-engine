# Free-First Opportunity Radar (experimental)

A standard-library-only Python tool for discovering leads for paid work. It is not an income generator and cannot guarantee acceptance or payment.

## Strategy

US$500/month is unlikely to come from low-end mobile PoW mining, bandwidth sharing, or a hands-off bot with no capital. This prototype searches for work that rewards completed tasks without requiring the user to sell a subscription or product:

1. AI evaluation / Indonesian language work (direct application; screening and task availability apply).
2. Public open-source issues with bounty/reward signals.
3. Contributor and security-bounty programs (security activity only within explicit written authorization and scope).

At the advertised maximum of US$7.50/hour on the Indonesian AI voice-evaluator listing, US$500 gross requires 66.7 paid hours/month (~15.4 hours/week). Actual rates, hours, screening, fees and taxes may differ. This is a target calculation, not an income prediction.

## Features

- Works with Android Termux and a standard Python installation.
- Uses Python standard library only; no paid API or dependency.
- Searches public GitHub Issues for bounty/reward language.
- Extracts explicit money mentions and scores skill fit / recency transparently.
- Includes direct leads for Outlier Indonesian AI evaluation, RWS TrainAI Indonesian speech-data work, TELUS Digital Indonesian QA work, and contributor/security-bounty platforms; all leads are caveated and must be re-verified.
- Writes Markdown and JSON reports.
- Has an offline mode for report generation.
- Does not auto-apply, claim bounties, test security targets, submit pull requests, or perform financial actions.

## Run in Termux

    pkg install python
    python income_radar/scout.py --out-dir reports/opportunity-radar

Offline report generation:

    python income_radar/scout.py --offline --out-dir reports/opportunity-radar

Tests:

    python -m unittest discover -s income_radar -p 'test_*.py' -v

An optional GITHUB_TOKEN environment variable can increase API rate limits. Never hardcode a token or include secrets in reports.

## Actions

A workflow is included for manual execution and daily scanning on the repository's default branch. It uploads a short-lived report artifact and does not commit findings to the repository. On this feature branch, the scheduled workflow is not active until the workflow is merged into the default branch.

## Safety and economic guardrails

- Treat every result as an unverified lead until funding, active status, claim procedure, country eligibility, acceptance criteria and withdrawal route are checked.
- Never pay an application fee, buy a course, rent compute, share seed phrases/private keys, or run unknown binaries to qualify.
- The tool performs no security scans or probes.
- Do not count speculative tokens, airdrops, unclaimed bounties or non-withdrawable balances as income.
- Applications, identity verification, legal attestations and external submissions remain user-controlled.
