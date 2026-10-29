# Security and Privacy Boundary

## Current release: read-only and paper-only

This release contains no authenticated trading API, live-order endpoint, wallet key loading, wallet signing, withdrawal capability, or automatic transfer of assets. Never add such a capability casually to the market-data adapter.

## Secrets and private context

- Never commit Reku passwords, API tokens, OTPs, recovery codes, wallet seed phrases/private keys, screenshots of account pages, full account identifiers, or private transaction reports.
- Never send a seed phrase or private key to the assistant, GitHub Actions, an LLM, issue body, commit, or logs.
- The repository is public. Project memory is sanitized; raw private chat archives are intentionally excluded.
- Local SQLite records market observations and paper trades only. Keep it locally unless you explicitly review whether its content is safe to publish.
- If a real credential is accidentally committed, revoke/rotate it immediately; deleting the file later does not remove Git history.

## API safety

- Only fixed HTTPS public market-data URLs are used.
- Responses are size-bounded and have timeouts.
- No arbitrary URL from CLI/config is fetched.
- API errors, schema drift, missing SOL, stale candles, and implausible bid/ask values fail closed.
- Never treat a cross-market signal as an executable quote. This version does not use Binance data.

## Risk safety

- This is research software, not investment advice.
- No target return is guaranteed. The Rp3,500,000 goal must never trigger a larger order, leverage, martingale, or bypass of a risk guard.
- Paper results cannot prove future live profitability.
- A future live adapter requires a separate threat model, official API documentation verification, idempotent order identity, durable pre-side-effect records, reconciliation for unknown outcomes, and explicit approval.
- A stop-loss is not guaranteed to fill at its threshold in a gap/illiquid market.
