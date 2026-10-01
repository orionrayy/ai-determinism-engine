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


    def test_native_result_payload_requires_protocol_and_exact_identifiers(self):
        payload = {
            "protocol": "wrong",
            "workflow_id": "wf",
            "node_id": "n1",
            "execution_id": "0" * 64,
            "token": "secret",
            "result": {"ok": True},
        }
        with self.assertRaises(ValueError):
            gateway.validate_native_result_payload(payload)

    def test_native_result_dispatches_exact_payload(self):
        payload = {
            "protocol": "ai-orchestrator.native-worker/v1",
            "workflow_id": "wf",
            "node_id": "n1",
            "execution_id": "0" * 64,
            "token": "secret-token",
            "result": {"ok": True},
        }
        with patch.object(gateway, "github_repository_dispatch", return_value={"github_status": 204}) as dispatch:
            result = gateway.handle_native_result(payload)
        self.assertEqual(result["github_status"], 204)
        dispatch.assert_called_once_with("orchestrator.native_result", payload)

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
