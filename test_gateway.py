import hashlib
import hmac
import json
import os
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gateway


class GatewayTests(unittest.TestCase):
    def test_health_configuration_is_safe(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(gateway.authorized({}))

    def test_authorization_uses_bearer_secret(self):
        with patch.dict(os.environ, {"GATEWAY_SHARED_SECRET": "test-secret"}, clear=False):
            self.assertTrue(gateway.authorized({"Authorization": "Bearer test-secret"}))
            self.assertFalse(gateway.authorized({"Authorization": "Bearer wrong"}))

    def test_authorization_accepts_hmac_signature(self):
        body = b'{"goal":"hello"}'
        timestamp = str(int(time.time()))
        signed = timestamp.encode() + b"\n" + body
        digest = hmac.new(b"test-secret", signed, hashlib.sha256).hexdigest()
        headers = {
            "X-Orchestrator-Timestamp": timestamp,
            "X-Orchestrator-Signature": "sha256=" + digest,
        }
        with patch.dict(os.environ, {"GATEWAY_SHARED_SECRET": "test-secret"}, clear=False):
            self.assertTrue(gateway.authorized(headers, body))

    def test_github_dispatch_propagates_event_id(self):
        captured = {}

        def fake_urlopen(request, timeout=30):
            captured["body"] = request.data
            class Response:
                status = 204
                def __enter__(self): return self
                def __exit__(self, *args): return None
            return Response()

        with patch.dict(os.environ, {
            "GITHUB_GATEWAY_TOKEN": "token",
            "GITHUB_REPOSITORY": "owner/repo",
        }, clear=True):
            with patch.object(gateway.urllib.request, "urlopen", side_effect=fake_urlopen):
                gateway.github_dispatch("hello", {"source": "test"}, event_id="evt-123")

        payload = json.loads(captured["body"].decode())
        self.assertEqual(payload["client_payload"]["event_id"], "evt-123")

    def test_structured_execution_event_sanitizes_private_input(self):
        payload = {
            "event_id": "evt-1",
            "execution_id": "a" * 64,
            "workflow_id": "wf-1",
            "domain": "wattpad-romance-publisher",
            "operation": "publication.schedule",
            "payload": {"secret": "do-not-forward"},
            "source": "test",
            "attempt": 2,
        }
        goal, metadata, event_id = gateway.build_execution_event(payload)
        self.assertEqual(
            goal,
            "Execute orchestration operation wattpad-romance-publisher.publication.schedule",
        )
        self.assertEqual(event_id, "evt-1")
        self.assertEqual(metadata["execution_id"], "a" * 64)
        self.assertNotIn("secret", json.dumps(metadata))
        self.assertEqual(len(metadata["input_digest"]), 64)

    def test_structured_execution_event_rejects_intent_tampering(self):
        with self.assertRaisesRegex(ValueError, "intent_fingerprint_mismatch"):
            gateway.build_execution_event({
                "event_id": "evt-2",
                "domain": "wattpad-romance-publisher",
                "operation": "publication.schedule",
                "payload": {"part_no": 1},
                "intent_fingerprint": "0" * 64,
            })

    def test_authorization_rejects_old_hmac_signature(self):
        body = b'{"goal":"hello"}'
        timestamp = str(int(time.time()) - 301)
        signed = timestamp.encode() + b"\n" + body
        digest = hmac.new(b"test-secret", signed, hashlib.sha256).hexdigest()
        headers = {
            "X-Orchestrator-Timestamp": timestamp,
            "X-Orchestrator-Signature": "sha256=" + digest,
        }
        with patch.dict(os.environ, {"GATEWAY_SHARED_SECRET": "test-secret"}, clear=False):
            self.assertFalse(gateway.authorized(headers, body))


if __name__ == "__main__":
    unittest.main()
