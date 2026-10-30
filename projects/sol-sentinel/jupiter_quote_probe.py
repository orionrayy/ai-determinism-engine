#!/usr/bin/env python3
"""Read-only Jupiter Swap V2 quote probe. Never signs or submits transactions."""
from __future__ import annotations
import argparse, json, math, os, sys, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone
from typing import Any

ORDER_ENDPOINT = "https://api.jup.ag/swap/v2/order"
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USER_AGENT = "SOL-Sentinel-ReadOnly-QuoteProbe/1.0"

def build_order_url(input_mint: str, output_mint: str, amount_units: int) -> str:
    """Build a quote-only URL; deliberately omits taker and transaction fields."""
    for label, mint in (("input_mint", input_mint), ("output_mint", output_mint)):
        if not isinstance(mint, str) or not mint or mint.strip() != mint or any(ch.isspace() for ch in mint):
            raise ValueError(f"{label} must be a non-empty mint address without whitespace")
        if len(mint) > 64:
            raise ValueError(f"{label} is too long")
    if input_mint == output_mint:
        raise ValueError("input_mint and output_mint must differ")
    if isinstance(amount_units, bool) or not isinstance(amount_units, int) or amount_units <= 0:
        raise ValueError("amount_units must be a positive integer in the input token's smallest unit")
    query = urllib.parse.urlencode({"inputMint": input_mint, "outputMint": output_mint, "amount": str(amount_units)})
    return f"{ORDER_ENDPOINT}?{query}"

def normalize_quote(payload: Any, *, checked_at: str | None = None) -> dict[str, Any]:
    """Validate and whitelist quote fields; never persist or emit an assembled transaction."""
    if not isinstance(payload, dict):
        raise ValueError("Jupiter response must be a JSON object")
    transaction = payload.get("transaction")
    if isinstance(transaction, str) and transaction:
        raise ValueError("unexpected assembled transaction in quote-only response; refusing to retain it")
    if transaction not in (None, ""):
        raise ValueError("unexpected transaction field type in quote-only response")
    def positive_int_string(field: str) -> str:
        value = payload.get(field)
        if isinstance(value, bool) or value is None:
            raise ValueError(f"Jupiter response missing valid {field}")
        text_value = str(value)
        if not text_value.isdigit() or int(text_value) <= 0:
            raise ValueError(f"Jupiter response has invalid {field}")
        return text_value
    in_amount = positive_int_string("inAmount")
    out_amount = positive_int_string("outAmount")
    threshold = payload.get("otherAmountThreshold")
    if threshold is not None:
        threshold = str(threshold)
        if not threshold.isdigit() or int(threshold) < 0:
            raise ValueError("Jupiter response has invalid otherAmountThreshold")
    impact = payload.get("priceImpactPct")
    if impact is not None:
        try: impact = float(impact)
        except (TypeError, ValueError, OverflowError): raise ValueError("Jupiter response has invalid priceImpactPct") from None
        if not math.isfinite(impact): raise ValueError("Jupiter response has non-finite priceImpactPct")
    fee_bps = payload.get("feeBps")
    if fee_bps is not None:
        try: fee_bps = float(fee_bps)
        except (TypeError, ValueError, OverflowError): raise ValueError("Jupiter response has invalid feeBps") from None
        if not math.isfinite(fee_bps) or fee_bps < 0: raise ValueError("Jupiter response has invalid feeBps")
    return {
        "schema_version": "sol-sentinel-jupiter-quote-v1",
        "mode": "READ_ONLY_QUOTE_ONLY", "execution_enabled": False,
        "transaction_signed": False, "transaction_submitted": False,
        "checked_at": checked_at or datetime.now(timezone.utc).isoformat(),
        "input_mint": str(payload.get("inputMint", "")),
        "output_mint": str(payload.get("outputMint", "")),
        "input_amount_units": in_amount, "expected_output_amount_units": out_amount,
        "minimum_output_amount_units": threshold, "price_impact_pct": impact,
        "fee_bps": fee_bps, "fee_mint": payload.get("feeMint"),
        "router": payload.get("router"), "route_mode": payload.get("mode"),
        "request_id": payload.get("requestId"),
        "transaction_payload_discarded": transaction in (None, ""),
        "limitations": [
            "Quote only: no taker address was sent; no transaction was signed or submitted.",
            "Quoted output is not a guaranteed fill; price, liquidity and route may change.",
            "Not a complete cost model: include route fees, priority fee, account rent if any, transfer costs and eventual fiat off-ramp.",
            "Amounts are raw token units; verify mint addresses and decimals before comparing with IDR."
        ]
    }

def fetch_quote(input_mint: str, output_mint: str, amount_units: int, *, api_key: str | None = None, timeout: float = 8.0) -> dict[str, Any]:
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 30:
        raise ValueError("timeout must be finite and between 0 and 30 seconds")
    url = build_order_url(input_mint, output_mint, amount_units)
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if api_key: headers["x-api-key"] = api_key
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200: raise RuntimeError(f"Jupiter quote endpoint returned HTTP {response.status}")
            raw = response.read(1_000_001)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Jupiter quote endpoint returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Jupiter quote request failed: {type(exc).__name__}") from None
    if len(raw) > 1_000_000: raise RuntimeError("Jupiter quote response exceeded 1 MB safety limit")
    try: payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError): raise RuntimeError("Jupiter quote endpoint returned invalid JSON") from None
    quote = normalize_quote(payload)
    quote["requested_input_mint"] = input_mint
    quote["requested_output_mint"] = output_mint
    quote["requested_input_amount_units"] = str(amount_units)
    quote["api_auth_mode"] = "free_api_key" if api_key else "keyless"
    if quote["input_amount_units"] != str(amount_units): raise RuntimeError("Jupiter quote input amount does not match request")
    if quote["input_mint"] and quote["input_mint"] != input_mint: raise RuntimeError("Jupiter quote input mint does not match request")
    if quote["output_mint"] and quote["output_mint"] != output_mint: raise RuntimeError("Jupiter quote output mint does not match request")
    return quote

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mint", default=SOL_MINT, help="input SPL token mint (default: wrapped SOL)")
    parser.add_argument("--output-mint", default=USDC_MINT, help="output SPL token mint (default: USDC)")
    parser.add_argument("--amount-units", type=int, required=True, help="positive input token units; SOL uses lamports (1 SOL = 1e9 lamports)")
    parser.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout in seconds (max 30)")
    args = parser.parse_args(argv)
    try:
        quote = fetch_quote(args.input_mint, args.output_mint, args.amount_units, api_key=os.environ.get("JUPITER_API_KEY") or None, timeout=args.timeout)
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"mode": "READ_ONLY_QUOTE_ONLY", "execution_enabled": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(quote, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
