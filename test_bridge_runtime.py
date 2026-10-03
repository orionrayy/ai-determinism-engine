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
    def setUp(self):
        br._COMPLETED.clear()
        br._INFLIGHT.clear()
        self._free_only_env = patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "false"}, clear=False)
        self._free_only_env.start()
        self.addCleanup(self._free_only_env.stop)

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
            {"required": [], "types": {}, "idempotent": False, "free_tier": False},
        )

    def test_action_free_tier_inherits_and_can_be_overridden(self):
        routes = {
            "notion": {
                "url": "https://upstream.example/notion",
                "actions": ["read_page", "write_page"],
                "free_tier": True,
                "action_specs": {
                    "write_page": {"free_tier": False},
                },
            }
        }
        discovered = br.describe_routes(routes)
        self.assertTrue(discovered["notion"]["action_specs"]["read_page"]["free_tier"])
        self.assertFalse(discovered["notion"]["action_specs"]["write_page"]["free_tier"])

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
                "free_tier": False,
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

    def test_free_only_blocks_uncertified_reconciliation(self):
        routes = {
            "notion": {
                "url": "https://upstream.example.test/notion",
                "actions": ["create_page"],
                "free_tier": True,
                "reconciliation_url": "https://upstream.example.test/reconcile",
                "action_specs": {"create_page": {"free_tier": False}},
            }
        }
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=True), patch.object(
            br, "load_routes", return_value=routes
        ), patch.object(br, "dispatch_reconciliation") as dispatch:
            with self.assertRaisesRegex(br.BridgeRuntimeError, "not certified for free-only execution"):
                br.handle_reconciliation(self.payload())
        dispatch.assert_not_called()

    def test_reconciliation_returns_explicit_state(self):
        routes = {
            "notion": {
                "actions": ["create_page"],
                "free_tier": True,
                "reconciliation_url": "https://upstream.example.test/reconcile",
            }
        }
        payload = self.payload()
        captured = {}

        def fake_urlopen(request, timeout=45):
            captured["url"] = request.full_url
            class Response:
                status = 200
                def __init__(self): self._done = False
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def read(self, limit=None):
                    if self._done:
                        return b""
                    self._done = True
                    return b'{"state":"applied","external_id":"p1"}'
            return Response()

        with patch.object(br, "load_routes", return_value=routes),              patch.object(br.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = br.handle_reconciliation(payload)
        self.assertTrue(result["ok"])
        self.assertEqual(result["state"], "applied")
        self.assertIn("request_id=", captured["url"])

    def test_reconciliation_rejects_invalid_state(self):
        routes = {
            "notion": {
                "actions": ["create_page"],
                "free_tier": True,
                "reconciliation_url": "https://upstream.example.test/reconcile",
            }
        }
        payload = self.payload(request_id=hashlib.sha256(b"wf:invalid-state").hexdigest())
        with patch.object(br, "load_routes", return_value=routes),              patch.object(br.urllib.request, "urlopen") as urlopen:
            class Response:
                status = 200
                def __init__(self): self._done = False
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def read(self, limit=None):
                    if self._done:
                        return b""
                    self._done = True
                    return b'{"state":"maybe"}'
            urlopen.return_value = Response()
            with self.assertRaises(br.BridgeRuntimeError):
                br.handle_reconciliation(payload)
    def test_idempotency_key_conflict_fails_closed(self):
        payload = self.payload()
        routes = {"notion": {
            "actions": ["create_page"],
            "url": "https://upstream.example.test/invoke",
        }}
        with patch.object(br, "load_routes", return_value=routes),              patch.object(br, "dispatch_upstream", return_value={"status_code": 200, "data": {"id": "p1"}}) as dispatch:
            br.handle_request(payload, "secret")
            changed = self.payload()
            changed["input"] = {"title": "Different"}
            with self.assertRaisesRegex(br.BridgeRuntimeError, "idempotency key conflicts"):
                br.handle_request(changed, "secret")
            dispatch.assert_called_once()

    def test_current_free_only_policy_is_rechecked_before_cached_replay(self):
        payload = self.payload()
        routes = {"notion": {
            "actions": ["create_page"],
            "url": "https://upstream.example.test/invoke",
            "free_tier": True,
        }}
        with patch.object(br, "load_routes", return_value=routes),              patch.object(br, "dispatch_upstream", return_value={"status_code": 200, "data": {"id": "p1"}}):
            br.handle_request(payload, "secret")
            routes["notion"]["free_tier"] = False
            with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=True),                  self.assertRaisesRegex(br.BridgeRuntimeError, "not certified for free-only execution"):
                br.handle_request(payload, "secret")

    def test_upstream_http_error_is_not_cached(self):
        payload = self.payload(request_id=hashlib.sha256(b"wf:http-error").hexdigest())
        routes = {"notion": {
            "actions": ["create_page"],
            "url": "https://upstream.example.test/invoke",
            "free_tier": True,
        }}
        with patch.object(br, "load_routes", return_value=routes),              patch.object(br, "dispatch_upstream", side_effect=br.BridgeUpstreamError("upstream down")) as dispatch:
            with self.assertRaises(br.BridgeUpstreamError):
                br.handle_request(payload, "secret")
            dispatch.assert_called_once()

    def test_dispatch_upstream_rejects_non_2xx(self):
        payload = self.payload(request_id=hashlib.sha256(b"wf:non2xx").hexdigest())
        route = {"url": "https://upstream.example.test/invoke"}
        class Response:
            status = 503
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, limit=None): return b'{"error":"down"}'
        with patch.object(br.urllib.request, "urlopen", return_value=Response()):
            with self.assertRaisesRegex(br.BridgeUpstreamError, "HTTP 503") as ctx:
                br.dispatch_upstream(route, payload)
            self.assertTrue(ctx.exception.uncertain)
            self.assertEqual(ctx.exception.status_code, 502)

    def test_dispatch_upstream_bounds_response(self):
        payload = self.payload(request_id=hashlib.sha256(b"wf:oversized-upstream").hexdigest())
        route = {"url": "https://upstream.example.test/invoke"}
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, limit=None): return b"x" * (br.MAX_UPSTREAM_RESPONSE_BYTES + 1)
        with patch.object(br.urllib.request, "urlopen", return_value=Response()):
            with self.assertRaisesRegex(br.BridgeUpstreamError, "exceeds 128 KiB"):
                br.dispatch_upstream(route, payload)

    def test_concurrent_duplicate_requests_single_flight_upstream(self):
        payload = self.payload(
            request_id=hashlib.sha256(b"wf:concurrent-single-flight").hexdigest()
        )
        routes = {
            "notion": {
                "actions": ["create_page"],
                "url": "https://upstream.example.test/invoke",
                "free_tier": True,
            }
        }
        started = threading.Event()
        release = threading.Event()
        calls = {"count": 0}
        calls_lock = threading.Lock()
        results = []
        errors = []

        def fake_dispatch(route, value):
            with calls_lock:
                calls["count"] += 1
            started.set()
            if not release.wait(2):
                raise br.BridgeUpstreamError(
                    "test upstream release timeout",
                    status_code=503,
                    uncertain=True,
                )
            return {"status_code": 200, "data": {"id": "p1"}}

        def invoke():
            try:
                results.append(br.handle_request(payload, "secret"))
            except Exception as exc:
                errors.append(exc)

        with patch.object(br, "load_routes", return_value=routes),              patch.object(br, "dispatch_upstream", side_effect=fake_dispatch):
            first = threading.Thread(target=invoke)
            second = threading.Thread(target=invoke)
            first.start()
            self.assertTrue(started.wait(1))
            second.start()
            release.set()
            first.join(3)
            second.join(3)

        self.assertFalse(errors, errors)
        self.assertEqual(calls["count"], 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(
            sorted(bool(result.get("idempotent_replay")) for result in results),
            [False, True],
        )

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

    def test_free_only_blocks_uncertified_invocation(self):
        routes = {
            "notion": {
                "actions": ["create_page"],
                "url": "https://upstream.example.test/notion",
                "free_tier": True,
                "action_specs": {"create_page": {"free_tier": False}},
            }
        }
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=True), patch.object(
            br, "load_routes", return_value=routes
        ), patch.object(br, "dispatch_upstream") as dispatch:
            with self.assertRaisesRegex(br.BridgeRuntimeError, "not certified for free-only execution"):
                br.handle_request(self.payload(), "secret")
        dispatch.assert_not_called()

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

    def test_cleanup_idempotency_handles_expired_three_tuple_entry(self):