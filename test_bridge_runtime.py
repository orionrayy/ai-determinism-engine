import hashlib
import os
import hmac
import json
import time
import unittest
from unittest.mock import patch

import bridge_runtime as br


class BridgeRuntimeTests(unittest.TestCase):
    def payload(self, request_id=None):
        request_id = request_id or hashlib.sha256(b"wf:n1").hexdigest()
        return {
            "protocol": br.PROTOCOL,
            "request_id": request_id,
            "workflow_id": "wf",
            "node_id": "n1",
            "connector": "notion",
            "action": "create_page",
            "goal": "create",
            "input": {"title": "Hello"},
            "sent_at": int(time.time()),
        }

    def test_signature_is_verified_with_replay_window(self):
        body = br.canonical_json(self.payload())
        ts = int(time.time())
        headers = {
            "x-orchestrator-timestamp": str(ts),
            "x-orchestrator-signature": br.sign(ts, body, "secret"),
        }
        self.assertTrue(br.verify_signature(headers, body, "secret", now=ts))
        self.assertFalse(br.verify_signature(headers, body, "wrong", now=ts))
        self.assertFalse(br.verify_signature(headers, body, "secret", now=ts + 301))

    def test_route_allowlist_rejects_unknown_action(self):
        routes = {"notion": {"actions": ["read_page"]}}
        with self.assertRaises(br.BridgeRuntimeError):
            br.validate_envelope(self.payload(), routes)

    def test_capability_discovery_is_sanitized_and_sorted(self):
        routes = {
            "clickup": {
                "url": "https://upstream.example/clickup",
                "secret_env": "CLICKUP_SECRET",
                "actions": ["update_task", "create_task"],
                "capabilities": ["execute", "publish"],
                "risk": "high",
                "free_tier": False,
            },
            "notion": {
                "url": "https://upstream.example/notion",
                "secret_env": "NOTION_SECRET",
                "actions": ["append_block", "create_page"],
                "capabilities": ["publish"],
                "risk": "high",
                "free_tier": True,
            },
        }
        with patch.dict(os.environ, {}, clear=True):
            discovered = br.describe_routes(routes)
        self.assertEqual(list(discovered), ["clickup", "notion"])
        self.assertEqual(discovered["clickup"]["actions"], ["create_task", "update_task"])
        self.assertEqual(discovered["notion"]["capabilities"], ["publish"])
        self.assertFalse(discovered["notion"]["configured"])
        self.assertNotIn("NOTION_SECRET", json.dumps(discovered))
        self.assertEqual(
            discovered["clickup"]["action_specs"]["create_task"],
            {"required": [], "types": {}, "idempotent": False},
        )

    def test_capability_discovery_reports_configured_secret_without_exposing_it(self):
        routes = {
            "notion": {
                "url": "https://upstream.example/notion",
                "secret_env": "NOTION_SECRET",
                "actions": ["create_page"],
            }
        }
        with patch.dict(os.environ, {"NOTION_SECRET": "do-not-leak"}, clear=True):
            discovered = br.describe_routes(routes)
        self.assertTrue(discovered["notion"]["configured"])
        self.assertNotIn("do-not-leak", json.dumps(discovered))

    def test_action_specs_are_sanitized_and_enforced(self):
        routes = {
            "notion": {
                "actions": ["create_page"],
                "url": "https://upstream.example.test/invoke",
                "action_specs": {
                    "create_page": {
                        "required": ["title", "properties.name"],
                        "types": {"title": "string", "properties.name": "string", "ignored": "secret"},
                        "idempotent": True,
                        "secret": "must-not-leak",
                    }
                },
            }
        }
        discovered = br.describe_routes(routes)
        self.assertEqual(
            discovered["notion"]["action_specs"]["create_page"],
            {
                "required": ["properties.name", "title"],
                "types": {"properties.name": "string", "title": "string"},
                "idempotent": True,
            },
        )
        with patch.object(br, "load_routes", return_value=routes), patch.object(
            br, "dispatch_upstream", return_value={"status_code": 200, "data": {"id": "p1"}}
        ):
            payload = self.payload(request_id=hashlib.sha256(b"wf:schema").hexdigest())
            payload["input"]["properties"] = {"name": "N"}
            result = br.handle_request(payload, "secret")
            self.assertTrue(result["ok"])
            bad = self.payload(request_id=hashlib.sha256(b"wf:schema-bad").hexdigest())
            bad["input"] = {"title": 42, "properties": {"name": "N"}}
            with self.assertRaises(br.BridgeRuntimeError):
                br.handle_request(bad, "secret")

    def test_idempotent_response_is_replayed(self):
        payload = self.payload()
        routes = {"notion": {
            "actions": ["create_page"],
            "url": "https://upstream.example.test/invoke",
        }}
        first = {"status_code": 200, "data": {"id": "p1"}}
        with patch.object(br, "load_routes", return_value=routes),              patch.object(br, "dispatch_upstream", return_value=first):
            a = br.handle_request(payload, "secret")
            b = br.handle_request(payload, "secret")
        self.assertFalse(a.get("idempotent_replay", False))
        self.assertTrue(b["idempotent_replay"])

    def test_upstream_requires_https(self):
        routes = {"notion": {"actions": ["create_page"], "url": "http://bad.example.test"}}
        with patch.object(br, "load_routes", return_value=routes):
            with self.assertRaises(br.BridgeRuntimeError):
                br.handle_request(self.payload(), "secret")

    def test_payload_size_is_bounded(self):
        payload = self.payload()
        payload["input"] = {"data": "x" * (70 * 1024)}
        routes = {"notion": {"actions": ["create_page"], "url": "https://upstream.example.test"}}
        with patch.object(br, "load_routes", return_value=routes):
            with self.assertRaises(br.BridgeRuntimeError):
                br.handle_request(payload, "secret")


if __name__ == "__main__":
    unittest.main()
