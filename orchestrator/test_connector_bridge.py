import hashlib
import json
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

    def test_discovery_url_respects_bridge_path(self):
        self.assertEqual(
            cb.discovery_url("https://bridge.example.test/api/bridge"),
            "https://bridge.example.test/api/bridge/capabilities",
        )
        self.assertEqual(
            cb.discovery_url("https://bridge.example.test/bridge"),
            "https://bridge.example.test/bridge/capabilities",
        )

    def test_validate_discovered_result_accepts_declared_output_contract(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "action_specs": {
                    "create_page": {
                        "result_required": ["bridge_job_id"],
                        "result_types": {"bridge_job_id": "string"},
                    }
                },
            }
        }
        cb.validate_discovered_result(
            "notion", "create_page", {"bridge_job_id": "job-1"}, inventory
        )

    def test_validate_discovered_result_rejects_missing_required_field(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "action_specs": {
                    "create_page": {"result_required": ["bridge_job_id"]}
                },
            }
        }
        with self.assertRaisesRegex(cb.ConnectorBridgeError, "response missing required field"):
            cb.validate_discovered_result("notion", "create_page", {}, inventory)

    def test_validate_discovered_result_rejects_wrong_result_type(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "action_specs": {
                    "create_page": {"result_types": {"bridge_job_id": "string"}}
                },
            }
        }
        with self.assertRaisesRegex(cb.ConnectorBridgeError, "response field bridge_job_id must be string"):
            cb.validate_discovered_result("notion", "create_page", {"bridge_job_id": 1}, inventory)

    def test_validate_discovered_action_accepts_configured_action(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "capabilities": ["publish"],
                "configured": True,
                "risk": "high",
                "free_tier": True,
            }
        }
        cb.validate_discovered_action("notion", "create_page", inventory)

    def test_validate_discovered_action_rejects_unadvertised_action(self):
        inventory = {"notion": {"actions": ["read_page"], "configured": True}}
        with self.assertRaises(cb.ConnectorBridgeError):
            cb.validate_discovered_action("notion", "create_page", inventory)

    def test_live_requires_https_and_secret(self):
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "http://example.test/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True):
            with self.assertRaises(cb.ConnectorBridgeError):
                cb.execute_connector_bridge(self.node(), "bridge it", dry_run=False)



    def test_discovery_snapshot_has_stable_digest(self):
        inventory = {
            "notion": {
                "actions": ["read_page"],
                "configured": True,
                "free_tier": True,
                "action_specs": {"read_page": {"free_tier": True}},
            }
        }
        first = cb.build_discovery_snapshot(inventory)
        second = cb.build_discovery_snapshot(inventory)
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertRegex(first["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(first["count"], 1)
        self.assertEqual(
            first["sha256"],
            hashlib.sha256(
                json.dumps(
                    {
                        "protocol": cb.PROTOCOL,
                        "connectors": inventory,
                        "count": 1,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )

    def test_discovery_rejects_oversized_response(self):
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self, limit=None):
                return b'{' + b'x' * (cb.MAX_DISCOVERY_BYTES + 10)
        with patch.object(cb.urllib.request, "urlopen", return_value=Response()):
            with self.assertRaises(cb.ConnectorBridgeError):
                cb.discover_capabilities(
                    "https://bridge.example.test/api/bridge",
                    force_refresh=True,
                )

    def test_reconciliation_url_respects_bridge_path(self):
        self.assertEqual(
            cb.reconciliation_url("https://bridge.example.test/api/bridge"),
            "https://bridge.example.test/api/bridge/reconcile",
        )
        self.assertEqual(
            cb.reconciliation_url("https://bridge.example.test/bridge"),
            "https://bridge.example.test/bridge/reconcile",
        )

    def test_reconciliation_requires_advertised_support(self):
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True):
            inventory = {
                "notion": {
                    "actions": ["create_page"],
                    "configured": True,
                    "reconciliation": False,
                }
            }
            with patch.object(cb, "discover_capabilities", return_value=inventory):
                with self.assertRaises(cb.ConnectorReconciliationError):
                    cb.reconcile_connector_execution(
                        self.node(),
                        "reconcile it",
                        dry_run=False,
                    )

    def test_reconciliation_returns_explicit_state(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "free_tier": True,
                "reconciliation": True,
            }
        }
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True), patch.object(
            cb, "discover_capabilities", return_value=inventory
        ), patch.object(
            cb, "post_reconciliation",
            return_value={"ok": True, "state": "not_applied"},
        ) as reconcile:
            result = cb.reconcile_connector_execution(
                self.node(),
                "reconcile it",
                dry_run=False,
            )
        self.assertEqual(result["state"], "not_applied")
        reconcile.assert_called_once()
    def test_live_preflights_discovered_connector_before_post(self):
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True):
            with patch.object(
                cb, "discover_capabilities",
                return_value={"notion": {"actions": ["create_page"], "configured": True, "free_tier": True}},
            ) as discovery, patch.object(
                cb, "post_request",
                return_value={"ok": True, "bridge_job_id": "job-1"},
            ) as post:
                result = cb.execute_connector_bridge(self.node(), "bridge it", dry_run=False)
        self.assertEqual(result["response"]["bridge_job_id"], "job-1")
        discovery.assert_called_once_with("https://bridge.example.test/api/bridge", force_refresh=True)
        post.assert_called_once()

    def test_uncertain_request_uses_declared_idempotency(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "free_tier": True,
                "action_specs": {"create_page": {"idempotent": True}},
            }
        }
        with patch.object(
            cb, "discover_capabilities", return_value=inventory
        ), patch.object(
            cb, "post_request",
            side_effect=cb.ConnectorRequestError("timeout", uncertain=True),
        ):
            with patch.dict(cb.os.environ, {
                "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
                "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
            }, clear=True):
                with self.assertRaises(cb.ConnectorRequestError) as ctx:
                    cb.execute_connector_bridge(self.node(), "bridge it", dry_run=False)
        self.assertTrue(ctx.exception.uncertain)
        self.assertTrue(ctx.exception.idempotent)

    def test_non_idempotent_uncertain_request_is_not_retryable(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "free_tier": True,
                "action_specs": {"create_page": {"idempotent": False}},
            }
        }
        with patch.object(
            cb, "discover_capabilities", return_value=inventory
        ), patch.object(
            cb, "post_request",
            side_effect=cb.ConnectorRequestError("timeout", uncertain=True),
        ):
            with patch.dict(cb.os.environ, {
                "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
                "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
            }, clear=True):
                with self.assertRaises(cb.ConnectorRequestError) as ctx:
                    cb.execute_connector_bridge(self.node(), "bridge it", dry_run=False)
        self.assertTrue(ctx.exception.uncertain)
        self.assertFalse(ctx.exception.idempotent)


    def test_reconciliation_result_does_not_persist_upstream_payload(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "free_tier": True,
                "reconciliation": True,
            }
        }
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True), patch.object(
            cb, "discover_capabilities", return_value=inventory
        ), patch.object(
            cb, "post_reconciliation",
            return_value={
                "ok": True,
                "state": "applied",
                "upstream": {"private_token": "should-not-persist"},
            },
        ):
            result = cb.reconcile_connector_execution(
                self.node(),
                "reconcile it",
                dry_run=False,
            )
        encoded = cb.canonical_json(result).decode("utf-8")
        self.assertEqual(result["state"], "applied")
        self.assertNotIn("private_token", encoded)
        self.assertNotIn("should-not-persist", encoded)

    def test_free_only_rejects_uncertified_reconciliation(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "free_tier": True,
                "reconciliation": True,
                "action_specs": {"create_page": {"free_tier": False}},
            }
        }
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True), patch.object(
            cb, "discover_capabilities", return_value=inventory
        ), patch.object(cb, "post_reconciliation") as reconcile:
            with self.assertRaisesRegex(cb.ConnectorReconciliationError, "not certified for free-only execution"):
                cb.reconcile_connector_execution(self.node(), "reconcile it", dry_run=False)
        reconcile.assert_not_called()

    def test_live_rejects_invalid_payload_before_post(self):
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True):
            inventory = {
                "notion": {
                    "actions": ["create_page"],
                    "configured": True,
                    "free_tier": True,
                    "action_specs": {
                        "create_page": {
                            "required": ["title", "properties.name"],
                            "types": {"title": "string", "properties.name": "string"},
                            "idempotent": True,
                        }
                    },
                }
            }
            with patch.object(cb, "discover_capabilities", return_value=inventory), patch.object(cb, "post_request") as post:
                invalid = self.node({"title": "Hello", "properties": {}})
                with self.assertRaises(cb.ConnectorBridgeError):
                    cb.execute_connector_bridge(invalid, "bridge it", dry_run=False)
                post.assert_not_called()

    def test_live_rejects_connector_not_in_live_inventory(self):
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True):
            with patch.object(
                cb, "discover_capabilities",
                return_value={"clickup": {"actions": ["create_task"], "configured": True}},
            ), patch.object(cb, "post_request") as post:
                with self.assertRaises(cb.ConnectorBridgeError):
                    cb.execute_connector_bridge(self.node(), "bridge it", dry_run=False)
                post.assert_not_called()

    def test_free_only_rejects_uncertified_upstream_action(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "free_tier": True,
                "action_specs": {"create_page": {"free_tier": False}},
            }
        }
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True), patch.object(
            cb, "discover_capabilities", return_value=inventory
        ), patch.object(cb, "post_request") as post:
            with self.assertRaisesRegex(cb.ConnectorRequestError, "not certified for free-only execution"):
                cb.execute_connector_bridge(self.node(), "bridge it", dry_run=False)
        post.assert_not_called()

    def test_live_rejects_oversized_connector_response(self):
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self, limit=None):
                return b'x' * (cb.MAX_RESPONSE_BYTES + 1)

        inventory = {"notion": {"actions": ["create_page"], "configured": True, "free_tier": True}}
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True), patch.object(cb, "discover_capabilities", return_value=inventory), \
             patch.object(cb.urllib.request, "urlopen", return_value=Response()):
            with self.assertRaisesRegex(cb.ConnectorRequestError, "128 KiB"):
                cb.execute_connector_bridge(self.node(), "bridge it", dry_run=False)

    def test_live_rejects_response_contract_violation(self):
        captured = {}
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self, limit=None): return b'{"bridge_job_id":123}'

        inventory = {"notion": {"actions": ["create_page"], "configured": True, "free_tier": True,
                                  "action_specs": {"create_page": {"result_types": {"bridge_job_id": "string"}}}}
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True), patch.object(cb, "discover_capabilities", return_value=inventory), \
             patch.object(cb.urllib.request, "urlopen", side_effect=lambda req, timeout=60: Response()):
            with self.assertRaisesRegex(cb.ConnectorBridgeError, "response field bridge_job_id must be string"):
                cb.execute_connector_bridge(self.node(), "bridge it", dry_run=False)

    def test_live_sends_idempotency_key_and_signature(self):
        captured = {}

        def fake_urlopen(request, timeout=60):
            captured["headers"] = dict(request.header_items())
            captured["body"] = request.data
            class Response:
                status = 200
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def read(self): return b'{"ok": true, "bridge_job_id": "job-1"}'
            return Response()

        inventory = {"notion": {"actions": ["create_page"], "configured": True, "free_tier": True}}
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True):
            with patch.object(cb, "discover_capabilities", return_value=inventory),                  patch.object(cb.urllib.request, "urlopen", side_effect=fake_urlopen):
                result = cb.execute_connector_bridge(self.node(), "bridge it", dry_run=False)
        self.assertEqual(result["response"]["bridge_job_id"], "job-1")
        self.assertEqual(
            captured["headers"]["Idempotency-key"],
            cb.execution_id("wf_bridge", "n01-connector"),
        )
        self.assertTrue(captured["headers"]["X-orchestrator-signature"].startswith("sha256="))
        self.assertEqual(result["discovery"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
