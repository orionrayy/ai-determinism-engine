import hashlib
import hmac
import json
import os
import unittest
from unittest.mock import patch

import private_input as pi


class PrivateInputTests(unittest.TestCase):
    def test_ref_is_deterministic_and_secret_bound(self):
        payload = {"title": "Hello"}
        digest = pi.input_digest(payload)
        a = pi.derive_input_ref("e" * 64, digest, "secret-a")
        b = pi.derive_input_ref("e" * 64, digest, "secret-a")
        c = pi.derive_input_ref("e" * 64, digest, "secret-b")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 64)

    def test_store_posts_private_payload_with_opaque_ref(self):
        captured = {}

        def fake_urlopen(request, timeout=30):
            captured["request"] = request
            class Response:
                status = 201
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def read(self, *args): return json.dumps({
                    "ok": True,
                    "input_ref": pi.derive_input_ref(
                        "e" * 64,
                        pi.input_digest({"title": "Hello"}),
                        "secret",
                    ),
                }).encode()
            return Response()

        with patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example.test",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True), patch.object(pi.urllib.request, "urlopen", side_effect=fake_urlopen):
            ref = pi.store_private_input(
                execution_id="e" * 64,
                intent_fingerprint="f" * 64,
                payload={"title": "Hello"},
            )

        body = captured["request"].data.decode()
        self.assertEqual(ref, pi.derive_input_ref(
            "e" * 64, pi.input_digest({"title": "Hello"}), "secret"
        ))
        self.assertIn('"payload":{"title":"Hello"}', body)
        self.assertEqual(captured["request"].get_header("Idempotency-key"), ref)
        self.assertTrue(captured["request"].get_header("X-orchestrator-signature").startswith("sha256="))

    def test_fetch_verifies_digest_and_execution_binding(self):
        payload = {"title": "Hello"}
        digest = pi.input_digest(payload)
        ref = pi.derive_input_ref("e" * 64, digest, "secret")

        def fake_urlopen(request, timeout=30):
            class Response:
                status = 200
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def read(self, *args):
                    return json.dumps({
                        "ok": True,
                        "input_ref": ref,
                        "execution_id": "e" * 64,
                        "input_digest": digest,
                        "intent_fingerprint": "f" * 64,
                        "payload": payload,
                    }).encode()
            return Response()

        with patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example.test",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True), patch.object(pi.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = pi.fetch_private_input(
                input_ref=ref,
                execution_id="e" * 64,
                expected_digest=digest,
                expected_intent_fingerprint="f" * 64,
            )
        self.assertEqual(result, payload)

    def test_fetch_rejects_ref_mismatch_before_network(self):
        payload = {"title": "Hello"}
        digest = pi.input_digest(payload)
        with patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example.test",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True), patch.object(pi.urllib.request, "urlopen") as call:
            with self.assertRaisesRegex(pi.PrivateInputError, "does not match execution identity"):
                pi.fetch_private_input(
                    input_ref="0" * 64,
                    execution_id="e" * 64,
                    expected_digest=digest,
                )
        call.assert_not_called()

    def test_invalid_transport_is_rejected(self):
        with patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "http://private.example.test",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True):
            with self.assertRaisesRegex(pi.PrivateInputError, "HTTPS"):
                pi.private_input_config()


if __name__ == "__main__":
    unittest.main()
