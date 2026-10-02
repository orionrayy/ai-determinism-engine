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

    def test_authorization_is_case_insensitive_for_http_headers(self):
        body = b'{"goal":"hello"}'
        timestamp = str(int(time.time()))
        digest = hmac.new(
            b"test-secret",
            timestamp.encode() + b"\n" + body,
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "authorization": "Bearer test-secret",
        }
        with patch.dict(os.environ, {"GATEWAY_SHARED_SECRET": "test-secret"}, clear=False):
            self.assertTrue(gateway.authorized(headers))
        headers = {
            "x-orchestrator-timestamp": timestamp,
            "x-orchestrator-signature": "sha256=" + digest,
        }
        with patch.dict(os.environ, {"GATEWAY_SHARED_SECRET": "test-secret"}, clear=False):
            self.assertTrue(gateway.authorized(headers, body))

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
        self.assertEqual(metadata["idempotency_key"], "evt-1")
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

    def test_structured_execution_event_rejects_digest_tampering(self):
        with self.assertRaisesRegex(ValueError, "input_digest_mismatch"):
            gateway.build_execution_event({
                "event_id": "evt-digest",
                "domain": "wattpad-romance-publisher",
                "operation": "chapter.produce",
                "payload": {"part_no": 1},
                "input_digest": "0" * 64,
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
        with self.assertRaisesRegex(ValueError, "private_input_unavailable"):
            gateway.build_execution_event({
                "event_id": "evt-live",
                "domain": "wattpad-romance-publisher",
                "operation": "publication.schedule",
                "payload": {"story_id": "s1"},
                "requested_mode": "live",
            })

    def test_structured_execution_rejects_supplied_execution_id_mismatch(self):
        payload = {
            "event_id": "evt-mismatch",
            "execution_id": "f" * 64,
            "domain": "notion",
            "operation": "create_page",
            "payload": {"title": "x"},
            "requested_mode": "dry-run",
        }
        with self.assertRaisesRegex(ValueError, "execution_id_mismatch"):
            gateway.build_execution_event(payload)

    def test_structured_live_dispatch_failure_cleans_private_input(self):
        metadata = {"private_input_ref": "b" * 64}
        with patch.object(
            gateway, "github_dispatch", side_effect=RuntimeError("dispatch down")
        ), patch.object(
            gateway, "delete_private_input", return_value=True
        ) as cleanup:
            with self.assertRaises(RuntimeError):
                gateway.dispatch_execution("goal", metadata, event_id="evt-cleanup")
        cleanup.assert_called_once_with("b" * 64)

    def test_dispatch_cleanup_does_not_mask_original_error(self):
        metadata = {"private_input_ref": "b" * 64}
        with patch.object(
            gateway, "github_dispatch", side_effect=RuntimeError("dispatch down")
        ), patch.object(
            gateway, "delete_private_input", side_effect=RuntimeError("cleanup down")
        ):
            with self.assertRaisesRegex(RuntimeError, "dispatch down"):
                gateway.dispatch_execution("goal", metadata, event_id="evt-cleanup")

    def test_structured_live_execution_stores_private_payload(self):
        payload = {
            "event_id": "evt-live",
            "execution_id": "a" * 64,
            "workflow_id": "wf-live",
            "domain": "notion",
            "operation": "create_page",
            "payload": {"title": "Secret title"},
            "requested_mode": "live",
        }
        with patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example.test",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True), patch.object(
            gateway,
            "store_private_input",
            return_value="b" * 64,
        ) as store:
            goal, metadata, event_id = gateway.build_execution_event(payload)
        store.assert_called_once()
        self.assertEqual(event_id, "evt-live")
        self.assertEqual(metadata["private_input_ref"], "b" * 64)
        self.assertNotIn("Secret title", json.dumps(metadata))
        self.assertIn("notion.create_page", goal)

    def test_gateway_rejects_non_positive_content_length(self):
        with self.assertRaisesRegex(ValueError, "payload size"):
            gateway.parse_content_length("-1")
        with self.assertRaisesRegex(ValueError, "payload size"):
            gateway.parse_content_length("0")

    def test_gateway_rejects_invalid_content_length(self):
        with self.assertRaisesRegex(ValueError, "content_length_invalid"):
            gateway.parse_content_length("not-a-number")

    def test_structured_execution_rejects_oversized_input_object(self):
        with self.assertRaisesRegex(ValueError, "input_has_too_many_properties"):
            gateway.build_execution_event({
                "event_id": "evt-many",
                "domain": "publisher",
                "operation": "chapter.produce",
                "payload": {str(i): i for i in range(257)},
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
