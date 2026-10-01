import unittest
from types import SimpleNamespace
from unittest.mock import patch

import connector_bridge as cb


class ConnectorBridgeTests(unittest.TestCase):
    def node(self, payload=None):
        return SimpleNamespace(
            id="n01-connector",
            input={
                "workflow_id": "wf_bridge",
                "connector": "notion",
                "action": "create_page",
                "payload": payload or {"title": "Hello"},
            },
        )

    def test_build_request_is_deterministic(self):
        a = cb.build_request(self.node(), "build a page")
        b = cb.build_request(self.node(), "build a page")
        self.assertEqual(a.request_id, b.request_id)
        self.assertEqual(len(a.request_id), 64)
        self.assertEqual(a.protocol, "ai-orchestrator.connector/v1")

    def test_signature_matches_hmac_sha256(self):
        body = b'{"ok":true}'
        sig = cb.sign(123, body, "secret")
        expected = __import__("hmac").new(
            b"secret", b"123\n" + body, __import__("hashlib").sha256
        ).hexdigest()
        self.assertEqual(sig, "sha256=" + expected)

    def test_dry_run_never_requires_bridge_secret(self):
        with patch.dict(cb.os.environ, {}, clear=True):
            result = cb.execute_connector_bridge(self.node(), "bridge it", dry_run=True)
        self.assertTrue(result["simulated"])
        self.assertEqual(result["request_id"], cb.execution_id("wf_bridge", "n01-connector"))

    def test_live_requires_https_and_secret(self):
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "http://example.test/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True):
            with self.assertRaises(cb.ConnectorBridgeError):
                cb.execute_connector_bridge(self.node(), "bridge it", dry_run=False)

    def test_live_sends_idempotency_key_and_signature(self):
        captured = {}

        def fake_urlopen(request, timeout=60):
            captured["headers"] = dict(request.header_items())
            captured["body"] = request.data
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def read(self): return b'{"ok": true, "bridge_job_id": "job-1"}'
            return Response()

        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/invoke",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True):
            with patch.object(cb.urllib.request, "urlopen", side_effect=fake_urlopen):
                result = cb.execute_connector_bridge(self.node(), "bridge it", dry_run=False)
        self.assertEqual(result["response"]["bridge_job_id"], "job-1")
        self.assertEqual(
            captured["headers"]["Idempotency-key"],
            cb.execution_id("wf_bridge", "n01-connector"),
        )
        self.assertTrue(captured["headers"]["X-orchestrator-signature"].startswith("sha256="))


if __name__ == "__main__":
    unittest.main()
