import json
import os
import unittest
from unittest.mock import patch

import execution_lease as el


class ExecutionLeaseTests(unittest.TestCase):
    def workflow(self, used=0, maximum=16):
        return {
            "id": "wf_test",
            "execution_id": "a" * 64,
            "execution_budget": {
                "used_steps": used,
                "max_steps": maximum,
            },
        }

    def test_lease_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(el.lease_enabled())

    def test_owner_is_deterministic_for_github_run(self):
        with patch.dict(os.environ, {
            "ORCHESTRATOR_GITHUB_RUN_ID": "123",
            "ORCHESTRATOR_GITHUB_RUN_ATTEMPT": "2",
        }, clear=True):
            self.assertEqual(el.lease_owner(), "github:123:2")

    def test_acquire_seeds_remote_ledger_from_local_budget(self):
        captured = {}
        with patch.object(el, "_post", side_effect=lambda path, body: captured.update(path=path, body=body) or {
            "ok": True,
            "attempts": 7,
            "lease_until": int(__import__("time").time()) + 600,
        }), patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True):
            result = el.acquire_execution_lease(self.workflow(used=7, maximum=16))
        self.assertTrue(result["ok"])
        self.assertEqual(captured["path"], "/v1/leases/acquire")
        self.assertEqual(captured["body"]["initial_attempts"], 7)
        self.assertEqual(captured["body"]["max_attempts"], 16)

    def test_acquire_rejects_invalid_budget_before_network(self):
        with patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True), patch.object(el, "_post") as post:
            with self.assertRaises(el.ExecutionLeaseError):
                el.acquire_execution_lease(self.workflow(used=17, maximum=16))
        post.assert_not_called()

    def test_reserve_sends_renewal_ttl(self):
        captured = {}
        with patch.object(el, "_post", side_effect=lambda path, body: captured.update(path=path, body=body) or {
            "ok": True,
            "start_attempt": 8,
            "used_attempts": 8,
            "lease_until": int(__import__("time").time()) + 600,
        }), patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True):
            result = el.reserve_remote_execution_attempt(
                self.workflow(used=7, maximum=16),
                count=1,
                max_attempts=16,
                ttl_seconds=900,
            )
        self.assertTrue(result["ok"])
        self.assertEqual(captured["path"], "/v1/leases/reserve-attempt")
        self.assertEqual(captured["body"]["ttl_seconds"], 900)

    def test_acquire_rejects_malformed_backend_response(self):
        with patch.object(el, "_post", return_value={"ok": True, "attempts": 9999, "lease_until": int(__import__("time").time()) + 600}), patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True):
            with self.assertRaisesRegex(el.ExecutionLeaseError, "attempts"):
                el.acquire_execution_lease(self.workflow(used=0, maximum=16))

    def test_reserve_rejects_inconsistent_backend_response(self):
        with patch.object(el, "_post", return_value={
            "ok": True,
            "start_attempt": 5,
            "used_attempts": 9,
            "lease_until": int(__import__("time").time()) + 60,
        }), patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True):
            with self.assertRaisesRegex(el.ExecutionLeaseError, "range"):
                el.reserve_remote_execution_attempt(
                    self.workflow(used=4, maximum=16),
                    count=1,
                    max_attempts=16,
                )

    def test_release_is_best_effort(self):
        with patch.object(el, "_post", side_effect=el.ExecutionLeaseError("down")), patch.dict(os.environ, {
            "ORCHESTRATOR_PRIVATE_INPUT_URL": "https://private.example",
            "ORCHESTRATOR_PRIVATE_INPUT_SECRET": "secret",
        }, clear=True):
            self.assertFalse(el.release_execution_lease(self.workflow()))


if __name__ == "__main__":
    unittest.main()
