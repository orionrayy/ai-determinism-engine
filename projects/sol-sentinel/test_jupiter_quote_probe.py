from __future__ import annotations
import json, unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse
import jupiter_quote_probe as probe

class FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self.payload, self.status = payload, status
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, limit: int = -1) -> bytes:
        value = json.dumps(self.payload).encode()
        return value if limit < 0 else value[:limit]

class JupiterQuoteProbeTests(unittest.TestCase):
    def sample(self):
        return {
            "inputMint": probe.SOL_MINT, "outputMint": probe.USDC_MINT,
            "inAmount": "10000000", "outAmount": "1234567",
            "otherAmountThreshold": "1220000", "priceImpactPct": "0.001",
            "feeBps": 0, "feeMint": probe.USDC_MINT,
            "router": "metis", "mode": "ultra", "requestId": "opaque-request-id",
            "transaction": None,
        }
    def test_quote_url_omits_taker_and_contains_only_required_quote_fields(self):
        url = probe.build_order_url(probe.SOL_MINT, probe.USDC_MINT, 10_000_000)
        query = parse_qs(urlparse(url).query)
        self.assertEqual(query, {"inputMint": [probe.SOL_MINT], "outputMint": [probe.USDC_MINT], "amount": ["10000000"]})
        self.assertNotIn("taker", query)
    def test_invalid_amount_fails_closed(self):
        for amount in (0, -1, True, 1.5):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                probe.build_order_url(probe.SOL_MINT, probe.USDC_MINT, amount)
    def test_invalid_mints_fail_closed(self):
        with self.assertRaises(ValueError): probe.build_order_url(" ", probe.USDC_MINT, 1)
        with self.assertRaises(ValueError): probe.build_order_url(probe.SOL_MINT, probe.SOL_MINT, 1)
    def test_normalized_report_is_read_only_and_whitelists_fields(self):
        q = probe.normalize_quote(self.sample(), checked_at="2026-10-30T00:00:00+00:00")
        self.assertFalse(q["execution_enabled"])
        self.assertFalse(q["transaction_signed"])
        self.assertFalse(q["transaction_submitted"])
        self.assertEqual(q["expected_output_amount_units"], "1234567")
        self.assertNotIn("transaction", q)
        self.assertEqual(q["fee_bps"], 0.0)
    def test_unexpected_transaction_payload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexpected assembled transaction"):
            probe.normalize_quote({**self.sample(), "transaction": "BASE64_TX_BYTES"})
    def test_missing_or_invalid_output_is_rejected(self):
        bad = self.sample(); bad.pop("outAmount")
        with self.assertRaises(ValueError): probe.normalize_quote(bad)
        with self.assertRaises(ValueError): probe.normalize_quote({**self.sample(), "priceImpactPct": "NaN"})
    def test_fetch_quote_sends_no_taker_and_never_emits_api_key(self):
        fake = FakeResponse(self.sample())
        with patch.object(probe.urllib.request, "urlopen", return_value=fake) as mocked:
            result = probe.fetch_quote(probe.SOL_MINT, probe.USDC_MINT, 10_000_000, api_key="test-secret-key")
        request = mocked.call_args.args[0]
        query = parse_qs(urlparse(request.full_url).query)
        self.assertNotIn("taker", query)
        self.assertNotIn("test-secret-key", request.full_url)
        self.assertEqual(request.get_header("X-api-key"), "test-secret-key")
        self.assertNotIn("test-secret-key", json.dumps(result))
        self.assertEqual(result["api_auth_mode"], "free_api_key")
    def test_fetch_quote_rejects_response_for_different_input_amount(self):
        altered = self.sample(); altered["inAmount"] = "100"
        with patch.object(probe.urllib.request, "urlopen", return_value=FakeResponse(altered)):
            with self.assertRaisesRegex(RuntimeError, "does not match request"):
                probe.fetch_quote(probe.SOL_MINT, probe.USDC_MINT, 10_000_000)
    def test_timeout_is_bounded(self):
        with self.assertRaises(ValueError):
            probe.fetch_quote(probe.SOL_MINT, probe.USDC_MINT, 1, timeout=31)

if __name__ == "__main__":
    unittest.main()
