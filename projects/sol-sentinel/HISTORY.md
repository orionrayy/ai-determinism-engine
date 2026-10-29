# SOL Sentinel — Change and Conversation History

This file records a sanitized, durable history of decisions and implementation milestones. It is not a verbatim conversation export and intentionally excludes personal/account secrets.

## 2026-10-29 — Project initiation

1. User requested aggressive automated SOL scalping with starting capital originally discussed as Rp500,000 and an aspirational same-night target of Rp3,000,000.
2. The discussion identified that a 6x portfolio target in one night is extremely speculative and not an outcome a bot can guarantee.
3. User clarified no exchange account was initially available and requested a free-first project that can be read/controlled through the existing AI orchestration repository, with a preference for avoiding additional KYC.
4. A DEX/self-custody architecture was considered (Solana wallet + Jupiter) but the on-ramp from physical IDR cash to SOL remained the unresolved dependency.
5. User then clarified that Reku is already KYC-verified and IDR funds could be deposited there. This made Reku SOL/IDR the preferred initial data/market context rather than introducing a new asset venue.
6. User updated actual available capital to Rp300,000 and the aspirational target to Rp3,500,000. This implies a required total-value multiple of about 11.67x and net increase of Rp3,200,000; it is not a forecast or an acceptable reason to increase risk.
7. A Reku screen supplied by the user showed a SOL/IDR limit-order draft at Rp1,820,000 for Rp150,000 (displayed quantity 0.08241759 SOL), available IDR Rp300,000, and zero open orders. It does not prove that an order was submitted or filled. The screenshot itself is intentionally not stored in this public repository.
8. The design was changed to prioritize market intelligence and paper trading before committing cash, because a small account is particularly sensitive to spread, fees, network charges, and minimum withdrawal rules.
9. Research discovered legacy public Reku API docs for /v2/coins, /v2/price, /v2/bidask, /v2/orderbook, and /v2/chart. The docs appear old, so current availability/schema must be verified via the read-only doctor command. There is no claim of authenticated order-placement API verification.
10. An initial project scaffold and implementation was committed to branch feat/sol-sentinel-paper-v1, with CI tests. No real transaction was placed and no balance was changed.

## Explicitly rejected behaviors

- Do not claim that Rp3,500,000 is realized until independent account/portfolio evidence proves it.
- Do not promise a return or trade to satisfy a numeric target.
- Do not use leverage, martingale, averaging down, thinly traded tokens, or unbounded position sizes to chase the target.
- Do not place private chat transcripts, Reku credentials, transaction details, wallet seed phrases, private keys, or account identifiers in this public repository.
- Do not implement a live executor against undocumented or merely inferred endpoints.
- Do not blindly retry a transaction/order when the result is unknown.
- Do not treat USDT prices from another exchange as executable Reku IDR quotes.

## Active backlog

- [ ] Verify Reku public API responses on the user's device using python sol_sentinel.py --doctor --once.
- [ ] Adjust the schema parser only against observed, sanitized public response samples and add fixtures.
- [ ] Validate latest Reku fee schedule, order minimum, and marketable/limit-order rules in the app.
- [ ] Collect uninterrupted market-data samples and paper trades across varied market regimes.
- [ ] Report net P&L after modeled buy/sell fees, spread, slippage, drawdown, profit factor, expectancy, signal count, and missing-data intervals.
- [ ] Validate a current, permitted live execution path before separately designing one. No such executor exists in this release.
- [ ] Continue keeping local portfolio/journal data private and use only sanitized aggregate reports in public Git.

## Verification record

- GitHub branch: feat/sol-sentinel-paper-v1.
- Project source, test harness, workflow, specs, source register, security policy and runbook are tracked under projects/sol-sentinel/.
- Build environment could not resolve api.reku.id during this task; a live network smoke test was therefore not possible here.
- CI is the authoritative test result for the committed code; do not infer a passing test state before inspecting the workflow run.