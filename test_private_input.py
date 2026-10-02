import hashlib
import hmac
import json
import os
import time
import unittest
from unittest.mock import patch

import private_input as pi


class PrivateInputTests(unittest.TestCase):
    def test_request_signature_binds_method_and_path(self):
        secret = "secret"
        body = b"{}"
        timestamp = 1700000000
        get_one = pi.request_signature(
            method="GET",
            path="/v1/inputs/" + "a" * 64,
            timestamp=timestamp,
            body=body,
            secret=secret,
        )
        get_two = pi.request_signature(
            method="GET",
            path="/v1/inputs/" + "b" * 64,
            timestamp=timestamp,
            body=body,
            secret=secret,
        )
        post_one = pi.request_signature(
            method="POST",
            path="/v1/inputs",
            timestamp=timestamp,
            body=body,
            secret=secret,
        )
        self.assertNotEqual(get_one, get_two)
        self.assertNotEqual(get_one, post_one)
        self.assertEqual(
            get_one,
            pi.request_signature(
                method="GET",
                path="/v1/inputs/" + "a" * 64,
                timestamp=timestamp,
                body=body,
                secret=secret,
            ),
        )

    def test_ref_rejects_malformed_execution_identity(self):
        with self.assertRaisesRegex(pi.PrivateInputError, "execution identity"):
            pi.derive_input_ref(
                "not-an-execution-id",
                "a" * 64,
                "secret",
            )

    def test_ref_is_deterministic_and_secret_bound(self):
        payload = {"title": "Hello"}
        digest = pi.input_digest(payload)
        a = pi.derive_input_ref("e" * 64, digest, "secret-a")
        b = pi.derive_input_ref("e" * 64, digest, "secret-a")
        c = pi.derive_input_ref("e" * 64, digest, "secret-b")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 64)

    def test_signature_binds_protocol(self):
        secret = "secret"
        body = b"{}"
        timestamp = 1700000000
        current = pi.request_signature(
            method="GET",
            path="/v1/inputs/" + "a" * 64,
            timestamp=timestamp,
            body=body,
            secret=secret,
        )
        with patch.object(pi, "PROTOCOL", "different-protocol/v9"):
            self.assertNotEqual(
                current,
                pi.request_signature(
                    method="GET",
                    path="/v1/inputs/" + "a" * 64,
                    timestamp=timestamp,
                    body=body,
                    secret=secret,
                ),
            )

    def test_store_rejects_invalid_intent_fingerprint(self):
        with patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example.test",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True):
            with self.assertRaisesRegex(pi.PrivateInputError, "intent fingerprint"):
                pi.store_private_input(
                    execution_id="e" * 64,
                    intent_fingerprint="bad",
                    payload={"title": "Hello"},
                )

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
        actual_signature = captured["request"].get_header("X-orchestrator-signature")
        timestamp = int(captured["request"].get_header("X-orchestrator-timestamp"))
        expected_signature = pi.request_signature(
            method="POST",
            path="/v1/inputs",
            timestamp=timestamp,
            body=captured["request"].data,
            secret="secret",
        )
        self.assertEqual(actual_signature, "sha256=" + expected_signature)
        self.assertEqual(
            captured["request"].get_header("X-Orchestrator-protocol"),
            pi.PROTOCOL,
        )

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

    def test_fetch_rejects_expired_envelope(self):
        payload = {"title": "Expired"}
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
                        "expires_at": int(time.time()) - 1,
                        "payload": payload,
                    }).encode()
            return Response()

        with patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example.test",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True), patch.object(pi.urllib.request, "urlopen", side_effect=fake_urlopen):
            with self.assertRaisesRegex(pi.PrivateInputError, "expired"):
                pi.fetch_private_input(
                    input_ref=ref,
                    execution_id="e" * 64,
                    expected_digest=digest,
                    expected_intent_fingerprint="f" * 64,
                )

    def test_fetch_rejects_intent_mismatch(self):
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
                        "expires_at": int(time.time()) + 3600,
                        "payload": payload,
                    }).encode()
            return Response()

        with patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example.test",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True), patch.object(pi.urllib.request, "urlopen", side_effect=fake_urlopen):
            with self.assertRaisesRegex(pi.PrivateInputError, "intent fingerprint mismatch"):
                pi.fetch_private_input(
                    input_ref=ref,
                    execution_id="e" * 64,
                    expected_digest=digest,
                    expected_intent_fingerprint="0" * 64,
                )

    def test_delete_private_input_signs_method_and_path(self):
        captured = {}
        def fake_urlopen(request, timeout=15):
            captured["request"] = request
            class Response:
                status = 204
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def read(self, *args): return b""
            return Response()

        with patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example.test",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True), patch.object(pi.urllib.request, "urlopen", side_effect=fake_urlopen):
            self.assertTrue(pi.delete_private_input("a" * 64))
        self.assertEqual(captured["request"].method, "DELETE")
        timestamp = int(captured["request"].get_header("X-Orchestrator-timestamp"))
        expected = pi.request_signature(
            method="DELETE",
            path="/v1/inputs/" + "a" * 64,
            timestamp=timestamp,
            body=b"",
            secret="secret",
        )
        self.assertEqual(
            captured["request"].get_header("X-Orchestrator-signature"),
            "sha256=" + expected,
        )

    def test_invalid_transport_is_rejected(self):
        with patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "http://private.example.test",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True):
            with self.assertRaisesRegex(pi.PrivateInputError, "HTTPS"):
                pi.private_input_config()


if __name__ == "__main__":
    unittest.main()
