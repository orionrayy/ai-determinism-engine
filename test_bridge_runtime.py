import hashlib
import os
import hmac
import json
import time
import threading
import unittest
from unittest.mock import patch

import bridge_runtime as br


class BridgeRuntimeTests(unittest.TestCase):
    def payload(self, request_id=None, input_value=None):
        input_value = {"title": "Hello"} if input_value is None else input_value
        payload = {
            "protocol": br.PROTOCOL,
            "workflow_id": "wf",
            "node_id": "n1",
            "connector": "notion",
            "action": "create_page",
            "goal": "create",
            "input": input_value,
            "sent_at": int(time.time()),
        }
        payload["request_id"] = (
            request_id
            or br.connector_request_id(
                payload["workflow_id"],
                payload["node_id"],
                payload["connector"],
                payload["action"],
                payload["input"],
            )
        )
        return payload

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

    def test_request_id_must_match_request_intent(self):
        payload = self.payload()
        payload["request_id"] = hashlib.sha256(b"wrong-intent-key").hexdigest()
        routes = {
            "notion": {
                "actions": ["create_page"],
            }
        }
        with self.assertRaises(br.BridgeRuntimeError):
            br.validate_envelope(payload, routes)

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
            {"required": [], "types": {}, "idempotent": False, "response_fields": []},
        )

    def test_route_target_fingerprint_is_exposed_without_url(self):
        url = "https://upstream.example.test/invoke"
        routes = {
            "notion": {
                "url": url,
                "actions": ["create_page"],
            }
        }
        discovered = br.describe_routes(routes)
        self.assertEqual(
            discovered["notion"]["target_fingerprint"],
            br.route_target_fingerprint(url),
        )
        self.assertNotIn(url, json.dumps(discovered))

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
            payload = self.payload(
                input_value={"title": "Hello", "properties": {"name": "N"}}
            )
            result = br.handle_request(payload, "secret")
            self.assertTrue(result["ok"])
            bad = self.payload(
                input_value={"title": 42, "properties": {"name": "N"}}
            )
            with self.assertRaises(br.BridgeRuntimeError):
                br.handle_request(bad, "secret")


    def test_response_fields_are_sanitized_and_bounded(self):
        routes = {
            "notion": {
                "url": "https://upstream.example/invoke",
                "actions": ["create_page"],
                "action_specs": {
                    "create_page": {
                        "response_fields": ["upstream.data.id"],
                    }
                },
            }
        }
        discovered = br.describe_routes(routes)
        self.assertEqual(
            discovered["notion"]["action_specs"]["create_page"]["response_fields"],
            ["upstream.data.id"],
        )

    def test_reconciliation_capability_is_discovered(self):
        routes = {
            "notion": {
                "url": "https://upstream.example/notion",
                "reconciliation_url": "https://upstream.example/notion/reconcile",
                "actions": ["create_page"],
            }
        }
        discovered = br.describe_routes(routes)
        self.assertTrue(discovered["notion"]["reconciliation"])

    def test_reconciliation_returns_explicit_state(self):
        routes = {
            "notion": {
                "actions": ["create_page"],
                "reconciliation_url": "https://upstream.example.test/reconcile",
            }
        }
        payload = self.payload()
        captured = {}

        def fake_urlopen(request, timeout=45):
            captured["url"] = request.full_url
            class Response:
                status = 200
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def read(self): return b'{"state":"applied","external_id":"p1"}'
            return Response()

        with patch.object(br, "load_routes", return_value=routes),              patch.object(br.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = br.handle_reconciliation(payload)
        self.assertTrue(result["ok"])
        self.assertEqual(result["state"], "applied")
        self.assertIn("request_id=", captured["url"])

    def test_reconciliation_rejects_endpoint_drift(self):
        routes = {
            "notion": {
                "actions": ["create_page"],
                "url": "https://upstream.example.test/invoke",
                "reconciliation_url": "https://reconcile-current.example.test/status",
            }
        }
        payload = self.payload()
        payload["reconciliation_target_fingerprint"] = br.route_target_fingerprint(
            "https://reconcile-original.example.test/status"
        )
        with patch.object(br, "load_routes", return_value=routes),              patch.object(br, "dispatch_reconciliation") as reconcile:
            with self.assertRaises(br.BridgeRuntimeError):
                br.handle_reconciliation(payload)
        reconcile.assert_not_called()

    def test_reconciliation_rejects_target_drift(self):
        routes = {
            "notion": {
                "actions": ["create_page"],
                "url": "https://upstream-current.example.test/invoke",
                "reconciliation_url": "https://upstream-current.example.test/reconcile",
            }
        }
        payload = self.payload()
        payload["target_fingerprint"] = br.route_target_fingerprint(
            "https://upstream-original.example.test/invoke"
        )
        with patch.object(br, "load_routes", return_value=routes),              patch.object(br, "dispatch_reconciliation") as reconcile:
            with self.assertRaises(br.BridgeRuntimeError):
                br.handle_reconciliation(payload)
        reconcile.assert_not_called()

    def test_reconciliation_rejects_invalid_state(self):
        routes = {
            "notion": {
                "actions": ["create_page"],
                "reconciliation_url": "https://upstream.example.test/reconcile",
            }
        }
        payload = self.payload()
        with patch.object(br, "load_routes", return_value=routes),              patch.object(br.urllib.request, "urlopen") as urlopen:
            class Response:
                status = 200
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def read(self): return b'{"state":"maybe"}'
            urlopen.return_value = Response()
            with self.assertRaises(br.BridgeRuntimeError):
                br.handle_reconciliation(payload)
    def test_idempotent_response_is_replayed(self):
        br._COMPLETED.clear()
        br._INFLIGHT.clear()
        payload = self.payload()
        routes = {"notion": {
            "actions": ["create_page"],
            "url": "https://upstream.example.test/invoke",
        }}
        first = {"status_code": 200, "data": {"id": "p1"}}
        with patch.object(br, "load_routes", return_value=routes),              patch.object(br, "dispatch_upstream", return_value=first) as dispatch:
            a = br.handle_request(payload, "secret")
            b = br.handle_request(payload, "secret")
        self.assertEqual(dispatch.call_count, 1)
        self.assertFalse(a.get("idempotent_replay", False))
        self.assertTrue(b["idempotent_replay"])
        br._COMPLETED.clear()
        br._INFLIGHT.clear()

    def test_identical_concurrent_requests_are_single_flight(self):
        br._COMPLETED.clear()
        br._INFLIGHT.clear()
        payload = self.payload()
        routes = {"notion": {
            "actions": ["create_page"],
            "url": "https://upstream.example.test/invoke",
        }}
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def fake_dispatch(route, request_payload):
            calls.append(request_payload["request_id"])
            entered.set()
            self.assertTrue(release.wait(2))
            return {"status_code": 200, "data": {"id": "p1"}}

        results = []

        def invoke():
            results.append(br.handle_request(payload, "secret"))

        with patch.object(br, "load_routes", return_value=routes),              patch.object(br, "dispatch_upstream", side_effect=fake_dispatch):
            first_thread = threading.Thread(target=invoke)
            second_thread = threading.Thread(target=invoke)
            first_thread.start()
            self.assertTrue(entered.wait(1))
            second_thread.start()
            time.sleep(0.02)
            self.assertEqual(calls, [payload["request_id"]])
            release.set()
            first_thread.join(2)
            second_thread.join(2)

        self.assertEqual(len(results), 2)
        self.assertEqual(calls, [payload["request_id"]])
        self.assertEqual(
            sum(1 for result in results if result.get("idempotent_replay", False)),
            1,
        )
        br._COMPLETED.clear()
        br._INFLIGHT.clear()

    def test_same_request_id_different_target_fails_closed(self):
        br._COMPLETED.clear()
        br._INFLIGHT.clear()
        payload = self.payload()
        routes_a = {
            "notion": {
                "actions": ["create_page"],
                "url": "https://upstream-a.example.test/invoke",
            }
        }
        routes_b = {
            "notion": {
                "actions": ["create_page"],
                "url": "https://upstream-b.example.test/invoke",
            }
        }
        with patch.object(br, "load_routes", side_effect=[routes_a, routes_b]),              patch.object(br, "dispatch_upstream", return_value={"status_code": 200}) as dispatch:
            first = br.handle_request(payload, "secret")
            self.assertTrue(first["ok"])
            with self.assertRaises(br.BridgeRuntimeError):
                br.handle_request(payload, "secret")
        dispatch.assert_called_once()
        br._COMPLETED.clear()
        br._INFLIGHT.clear()

    def test_upstream_http_failure_is_not_cached(self):
        br._COMPLETED.clear()
        br._INFLIGHT.clear()
        payload = self.payload()
        routes = {
            "notion": {
                "actions": ["create_page"],
                "url": "https://upstream.example.test/invoke",
            }
        }
        failure = br.UpstreamConnectorError(
            "upstream connector returned HTTP 500",
            status_code=500,
            uncertain=True,
        )
        with patch.object(
            br, "load_routes", return_value=routes
        ), patch.object(
            br,
            "dispatch_upstream",
            side_effect=[failure, {"status_code": 200, "data": {"id": "p1"}}],
        ) as dispatch:
            with self.assertRaises(br.UpstreamConnectorError):
                br.handle_request(payload, "secret")
            result = br.handle_request(payload, "secret")
        self.assertTrue(result["ok"])
        self.assertEqual(dispatch.call_count, 2)
        self.assertTrue(result["upstream"]["status_code"] == 200)
        br._COMPLETED.clear()
        br._INFLIGHT.clear()

    def test_identical_concurrent_failure_is_single_flight(self):
        br._COMPLETED.clear()
        br._INFLIGHT.clear()
        payload = self.payload(
            request_id=hashlib.sha256(b"concurrent-failure").hexdigest()
        )
        routes = {
            "notion": {
                "actions": ["create_page"],
                "url": "https://upstream.example.test/invoke",
            }
        }
        entered = threading.Event()
        release = threading.Event()
        calls = []
        results = []

        def fake_dispatch(route, request_payload):
            calls.append(request_payload["request_id"])
            entered.set()
            self.assertTrue(release.wait(2))
            raise br.UpstreamConnectorError(
                "upstream connector returned HTTP 500",
                status_code=500,
                uncertain=True,
            )

        def invoke():
            try:
                results.append(br.handle_request(payload, "secret"))
            except Exception as exc:
                results.append(exc)

        with patch.object(br, "load_routes", return_value=routes),              patch.object(br, "dispatch_upstream", side_effect=fake_dispatch) as dispatch:
            first_thread = threading.Thread(target=invoke)
            second_thread = threading.Thread(target=invoke)
            first_thread.start()
            self.assertTrue(entered.wait(1))
            second_thread.start()
            time.sleep(0.02)
            self.assertEqual(calls, [payload["request_id"]])
            release.set()
            first_thread.join(2)
            second_thread.join(2)

            self.assertEqual(dispatch.call_count, 1)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(isinstance(item, Exception) for item in results))

            recovery = br.handle_request(payload, "secret")

        self.assertTrue(recovery["ok"])
        self.assertEqual(dispatch.call_count, 2)
        br._COMPLETED.clear()
        br._INFLIGHT.clear()

    def test_same_idempotency_key_cannot_change_request_intent(self):
        request_id = hashlib.sha256(b"stable-intent-key").hexdigest()
        first_payload = self.payload()
        request_id = first_payload["request_id"]
        changed_payload = self.payload(input_value={"title": "Different"})
        changed_payload["request_id"] = request_id
        changed_payload["input"] = {"title": "Different"}

        routes = {
            "notion": {
                "actions": ["create_page"],
                "url": "https://upstream.example.test/invoke",
            }
        }
        first = {"status_code": 200, "data": {"id": "p1"}}
        br._COMPLETED.clear()
        with patch.object(br, "load_routes", return_value=routes),              patch.object(br, "dispatch_upstream", return_value=first) as dispatch:
            result = br.handle_request(first_payload, "secret")
            self.assertTrue(result["ok"])
            with self.assertRaises(br.BridgeRuntimeError):
                br.handle_request(changed_payload, "secret")
        dispatch.assert_called_once()
        br._COMPLETED.clear()

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
