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
        signature = gateway.hmac_signature(
            timestamp, "POST", "/event", "", body, "test-secret"
        )
        headers = {
            "X-Orchestrator-Timestamp": timestamp,
            "X-Orchestrator-Signature": signature,
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
        self.assertEqual(payload["client_payload"]["workflow_id"], "evt-123")

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
            "requested_mode": "dry-run",
        }
        goal, metadata, event_id = gateway.build_execution_event(payload)
        self.assertEqual(
            goal,
            "Execute orchestration operation wattpad-romance-publisher.publication.schedule",
        )
        self.assertEqual(event_id, "evt-1")
        self.assertEqual(metadata["execution_id"], "a" * 64)
        self.assertEqual(metadata["requested_mode"], "dry-run")
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

    def test_structured_execution_defaults_to_dry_run(self):
        goal, metadata, event_id = gateway.build_execution_event({
            "event_id": "evt-default",
            "domain": "wattpad-romance-publisher",
            "operation": "chapter.produce",
            "payload": {"story_id": "s1"},
        })
        self.assertEqual(metadata["requested_mode"], "dry-run")
        self.assertEqual(event_id, "evt-default")
        self.assertIn("wattpad-romance-publisher.chapter.produce", goal)

    def test_structured_live_execution_fails_closed_without_private_channel(self):
        with self.assertRaisesRegex(ValueError, "live_structured_requires_private_input_channel"):
            gateway.build_execution_event({
                "event_id": "evt-live",
                "domain": "wattpad-romance-publisher",
                "operation": "publication.schedule",
                "payload": {"story_id": "s1"},
                "requested_mode": "live",
            })

    def test_hmac_binds_method_and_path(self):
        body = b'{"goal":"hello"}'
        timestamp = str(int(time.time()))
        signature = gateway.hmac_signature(
            timestamp, "POST", "/event", "", body, "test-secret"
        )
        headers = {
            "X-Orchestrator-Timestamp": timestamp,
            "X-Orchestrator-Signature": signature,
        }
        with patch.dict(os.environ, {"GATEWAY_SHARED_SECRET": "test-secret"}, clear=False):
            self.assertTrue(gateway.authorized(headers, body, method="POST", path="/event"))
            self.assertFalse(gateway.authorized(headers, body, method="GET", path="/event"))
            self.assertFalse(gateway.authorized(headers, body, method="POST", path="/other"))

    def test_hmac_binds_idempotency_key(self):
        body = b'{"goal":"hello"}'
        timestamp = str(int(time.time()))
        signature = gateway.hmac_signature(
            timestamp, "POST", "/event", "key-1", body, "test-secret"
        )
        headers = {
            "X-Orchestrator-Timestamp": timestamp,
            "X-Orchestrator-Signature": signature,
            "Idempotency-Key": "key-1",
        }
        with patch.dict(os.environ, {"GATEWAY_SHARED_SECRET": "test-secret"}, clear=False):
            self.assertTrue(gateway.authorized(headers, body))
            altered = dict(headers)
            altered["Idempotency-Key"] = "key-2"
            self.assertFalse(gateway.authorized(altered, body))

    def test_unstructured_event_id_is_deterministic(self):
        first = gateway.derive_unstructured_event_id("do work", {"source": "test"})
        second = gateway.derive_unstructured_event_id("do work", {"source": "test"})
        other = gateway.derive_unstructured_event_id("do work", {"source": "other"})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, other)

    def test_hmac_header_names_are_case_insensitive(self):
        body = b'{"goal":"hello"}'
        timestamp = str(int(time.time()))
        signature = gateway.hmac_signature(
            timestamp, "POST", "/event", "key-1", body, "test-secret"
        )
        headers = {
            "x-orchestrator-timestamp": timestamp,
            "x-orchestrator-signature": signature,
            "idempotency-key": "key-1",
        }
        with patch.dict(os.environ, {"GATEWAY_SHARED_SECRET": "test-secret"}, clear=False):
            self.assertTrue(gateway.authorized(headers, body))

    def test_authorization_rejects_old_hmac_signature(self):
        body = b'{"goal":"hello"}'
        timestamp = str(int(time.time()) - 301)
        signature = gateway.hmac_signature(
            timestamp, "POST", "/event", "", body, "test-secret"
        )
        headers = {
            "X-Orchestrator-Timestamp": timestamp,
            "X-Orchestrator-Signature": signature,
        }
        with patch.dict(os.environ, {"GATEWAY_SHARED_SECRET": "test-secret"}, clear=False):
            self.assertFalse(gateway.authorized(headers, body))


if __name__ == "__main__":
    unittest.main()
