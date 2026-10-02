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
        expected = cb.connector_request_id(
            "wf_bridge",
            "n01-connector",
            "notion",
            "create_page",
            {"title": "Hello"},
        )
        self.assertEqual(result["request_id"], expected)

    def test_discovery_url_respects_bridge_path(self):
        self.assertEqual(
            cb.discovery_url("https://bridge.example.test/api/bridge"),
            "https://bridge.example.test/api/bridge/capabilities",
        )
        self.assertEqual(
            cb.discovery_url("https://bridge.example.test/bridge"),
            "https://bridge.example.test/bridge/capabilities",
        )

    def test_validate_discovered_action_accepts_configured_action(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "capabilities": ["publish"],
                "configured": True,
                "risk": "high",
                "free_tier": False,
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

    def test_reconciliation_request_carries_target_bindings(self):
        node = self.node({"title": "Hello"})
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "reconciliation": True,
                "target_fingerprint": "target-a",
                "reconciliation_target_fingerprint": "reconcile-a",
                "action_specs": {"create_page": {"idempotent": True}},
            }
        }
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True), patch.object(
            cb, "discover_capabilities", return_value=inventory
        ), patch.object(
            cb, "post_reconciliation",
            return_value={"ok": True, "state": "applied"},
        ) as reconcile:
            result = cb.reconcile_connector_execution(node, "reconcile it", dry_run=False)
        self.assertEqual(result["state"], "applied")
        request = reconcile.call_args.args[2]
        self.assertEqual(request.target_fingerprint, "target-a")
        self.assertEqual(
            request.reconciliation_target_fingerprint,
            "reconcile-a",
        )

    def test_reconciliation_returns_explicit_state(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
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
                return_value={"notion": {"actions": ["create_page"], "configured": True, "action_specs": {"create_page": {"response_fields": ["bridge_job_id"]}}}},
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
        self.assertTrue(ctx.exception.retry_allowed)

    def test_non_idempotent_uncertain_request_is_not_retryable(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
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
        self.assertFalse(ctx.exception.retry_allowed)


    def test_reconciliation_blocks_target_contract_drift(self):
        node = self.node({"title": "Hello"})
        node.error = {
            "connector_action_contract_fingerprint": cb.action_contract_fingerprint(
                {"required": ["title"], "types": {"title": "string"}, "idempotent": True},
                target_fingerprint="target-a",
            )
        }
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "reconciliation": True,
                "target_fingerprint": "target-b",
                "action_specs": {
                    "create_page": {
                        "required": ["title"],
                        "types": {"title": "string"},
                        "idempotent": True,
                    }
                },
            }
        }
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True), patch.object(
            cb, "discover_capabilities", return_value=inventory
        ), patch.object(cb, "post_reconciliation") as reconcile:
            with self.assertRaises(cb.ConnectorReconciliationError):
                cb.reconcile_connector_execution(node, "reconcile it", dry_run=False)
        reconcile.assert_not_called()

    def test_reconciliation_blocks_endpoint_drift(self):
        node = self.node({"title": "Hello"})
        node.error = {
            "connector_reconciliation_target_fingerprint": "reconcile-a"
        }
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "reconciliation": True,
                "target_fingerprint": "target-a",
                "reconciliation_target_fingerprint": "reconcile-b",
                "action_specs": {
                    "create_page": {
                        "required": ["title"],
                        "types": {"title": "string"},
                        "idempotent": True,
                    }
                },
            }
        }
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True), patch.object(
            cb, "discover_capabilities", return_value=inventory
        ), patch.object(cb, "post_reconciliation") as reconcile:
            with self.assertRaises(cb.ConnectorReconciliationError):
                cb.reconcile_connector_execution(node, "reconcile it", dry_run=False)
        reconcile.assert_not_called()

    def test_reconciliation_result_does_not_persist_upstream_payload(self):
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
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

    def test_live_rejects_invalid_payload_before_post(self):
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True):
            inventory = {
                "notion": {
                    "actions": ["create_page"],
                    "configured": True,
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

    def test_request_id_changes_when_connector_intent_changes(self):
        base = self.node({"title": "Hello"})
        changed_action = SimpleNamespace(
            id=base.id,
            input={**base.input, "action": "update_page"},
        )
        a = cb.build_request(base, "goal")
        b = cb.build_request(changed_action, "goal")
        self.assertNotEqual(a.request_id, b.request_id)

        changed_payload = self.node({"title": "Different"})
        c = cb.build_request(changed_payload, "goal")
        self.assertNotEqual(a.request_id, c.request_id)

    def test_action_contract_fingerprint_is_stable(self):
        spec = {"required": ["title"], "types": {"title": "string"}, "idempotent": True}
        self.assertEqual(
            cb.action_contract_fingerprint(spec),
            cb.action_contract_fingerprint(dict(spec)),
        )
        self.assertEqual(
            cb.action_contract_fingerprint(spec, "target-a"),
            cb.action_contract_fingerprint(dict(spec), "target-a"),
        )

    def test_action_contract_fingerprint_changes_when_target_changes(self):
        spec = {"required": ["title"], "types": {"title": "string"}, "idempotent": True}
        self.assertNotEqual(
            cb.action_contract_fingerprint(spec, "target-a"),
            cb.action_contract_fingerprint(spec, "target-b"),
        )

    def test_contract_drift_blocks_second_connector_attempt(self):
        node = self.node({"title": "Hello"})
        node.error = {}
        inventories = [
            {"notion": {
                "actions": ["create_page"],
                "configured": True,
                "target_fingerprint": "target-a",
                "action_specs": {
                    "create_page": {
                        "required": ["title"],
                        "types": {"title": "string"},
                        "idempotent": True,
                    }
                },
            }},
            {"notion": {
                "actions": ["create_page"],
                "configured": True,
                "target_fingerprint": "target-b",
                "action_specs": {
                    "create_page": {
                        "required": ["title"],
                        "types": {"title": "string"},
                        "idempotent": True,
                    }
                },
            }},
        ]
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True), patch.object(cb, "discover_capabilities", side_effect=inventories):
            with patch.object(
                cb,
                "post_request",
                side_effect=cb.ConnectorRequestError("timeout", uncertain=True),
            ) as post:
                with self.assertRaises(cb.ConnectorRequestError):
                    cb.execute_connector_bridge(node, "bridge it", dry_run=False)
                post.assert_called_once()
            with self.assertRaises(cb.ConnectorBridgeError):
                cb.execute_connector_bridge(node, "bridge it", dry_run=False)
            self.assertEqual(
                node.error["connector_action_contract_fingerprint"],
                cb.action_contract_fingerprint(
                    inventories[0]["notion"]["action_specs"]["create_page"],
                    target_fingerprint="target-a",
                ),
            )

    def test_bridge_post_rejects_nested_upstream_http_failure(self):
        node = self.node({"title": "Hello"})
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True):
            inventory = {"notion": {"actions": ["create_page"], "configured": True}}
            class Response:
                status = 200
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def read(self):
                    return (
                        b'{"ok": true, "upstream": '
                        b'{"status_code": 500, "data": {"error": "upstream"}}}'
                    )
            with patch.object(cb, "discover_capabilities", return_value=inventory),                  patch.object(cb.urllib.request, "urlopen", return_value=Response()):
                with self.assertRaises(cb.ConnectorRequestError) as ctx:
                    cb.execute_connector_bridge(node, "bridge it", dry_run=False)
        self.assertTrue(ctx.exception.uncertain)
        self.assertFalse(ctx.exception.retry_allowed)

    def test_connector_response_persists_only_allowlisted_fields(self):
        node = self.node({"title": "Hello"})
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "target_fingerprint": "target-a",
                "reconciliation_target_fingerprint": "reconcile-a",
                "action_specs": {
                    "create_page": {
                        "idempotent": True,
                        "response_fields": ["upstream.data.id"],
                    }
                },
            }
        }
        raw_response = {
            "ok": True,
            "bridge_job_id": "job-1",
            "upstream": {
                "status_code": 200,
                "data": {
                    "id": "p1",
                    "private_token": "do-not-persist",
                },
            },
        }
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True), patch.object(
            cb, "discover_capabilities", return_value=inventory
        ), patch.object(cb, "post_request", return_value=raw_response):
            result = cb.execute_connector_bridge(node, "bridge it", dry_run=False)
        encoded = cb.canonical_json(result).decode("utf-8")
        self.assertEqual(result["response"]["fields"]["upstream.data.id"], "p1")
        self.assertEqual(result["response"]["status_code"], 200)
        self.assertNotIn("private_token", encoded)
        self.assertNotIn("do-not-persist", encoded)

    def test_live_result_does_not_persist_raw_bridge_url(self):
        node = self.node({"title": "Hello"})
        bridge_url = "https://private-bridge.example.test/api/bridge"
        inventory = {
            "notion": {
                "actions": ["create_page"],
                "configured": True,
                "target_fingerprint": "target-a",
                "reconciliation_target_fingerprint": "reconcile-a",
                "action_specs": {"create_page": {"idempotent": True}},
            }
        }
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": bridge_url,
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True), patch.object(
            cb, "discover_capabilities", return_value=inventory
        ), patch.object(
            cb, "post_request",
            return_value={"ok": True, "bridge_job_id": "job-1"},
        ):
            result = cb.execute_connector_bridge(node, "bridge it", dry_run=False)
        encoded = cb.canonical_json(result).decode("utf-8")
        self.assertNotIn(bridge_url, encoded)
        self.assertEqual(result["bridge_target_fingerprint"], "target-a")

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

        inventory = {"notion": {"actions": ["create_page"], "configured": True, "action_specs": {"create_page": {"response_fields": ["bridge_job_id"]}}}}
        with patch.dict(cb.os.environ, {
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example.test/api/bridge",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET": "secret",
        }, clear=True):
            with patch.object(cb, "discover_capabilities", return_value=inventory),                  patch.object(cb.urllib.request, "urlopen", side_effect=fake_urlopen):
                result = cb.execute_connector_bridge(self.node(), "bridge it", dry_run=False)
        self.assertEqual(result["response"]["bridge_job_id"], "job-1")
        self.assertEqual(
            captured["headers"]["Idempotency-key"],
            cb.connector_request_id(
                "wf_bridge",
                "n01-connector",
                "notion",
                "create_page",
                {"title": "Hello"},
            ),
        )
        self.assertTrue(captured["headers"]["X-orchestrator-signature"].startswith("sha256="))
        self.assertEqual(result["discovery"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
