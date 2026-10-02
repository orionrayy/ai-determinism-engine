import os
import unittest
from unittest.mock import patch

from policy_integrity import build_policy_snapshot, fingerprint_policy


class PolicyIntegrityTests(unittest.TestCase):
    def test_snapshot_binds_route_candidates_and_tool_policy(self):
        registry = {
            "capability:publish": {
                "default_tool": "webhook",
                "fallback_tools": ["connector_bridge"],
            },
            "webhook": {
                "required_env": "ORCHESTRATOR_WEBHOOK_URL",
                "side_effects": ["external_request"],
                "free_tier": False,
                "risk": "high",
                "allowed_actions": ["invoke"],
            },
            "connector_bridge": {
                "required_env": "BRIDGE_URL",
                "secret_env": "BRIDGE_SECRET",
                "side_effects": ["external_request"],
                "free_tier": True,
                "risk": "high",
                "allowed_actions": ["invoke"],
            },
        }
        node = type("NodeLike", (), {
            "id": "n01-publish",
            "capability": "publish",
            "tool": "connector_bridge",
            "risk": "high",
            "input": {"action": "invoke"},
        })()
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            snapshot = build_policy_snapshot(registry, [node], live=True)
        self.assertEqual(snapshot["routes"][0]["candidates"], ["webhook", "connector_bridge"])
        self.assertEqual(snapshot["routes"][0]["tool"], "connector_bridge")
        self.assertEqual(snapshot["tool_specs"]["connector_bridge"]["free_tier"], True)
        self.assertEqual(len(fingerprint_policy(snapshot)), 64)

    def test_policy_fingerprint_changes_when_side_effect_policy_changes(self):
        base = {
            "capability:execute": {
                "default_tool": "noop",
                "fallback_tools": [],
            },
            "noop": {
                "required_env": None,
                "side_effects": [],
                "free_tier": True,
                "risk": "low",
                "allowed_actions": [],
            },
        }
        node = type("NodeLike", (), {
            "id": "n01-execute",
            "capability": "execute",
            "tool": "noop",
            "risk": "low",
            "input": {},
        })()
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            first = build_policy_snapshot(base, [node], live=False)
        changed = {
            **base,
            "noop": {
                **base["noop"],
                "side_effects": ["external_request"],
                "risk": "high",
            },
        }
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            second = build_policy_snapshot(changed, [node], live=False)
        self.assertNotEqual(fingerprint_policy(first), fingerprint_policy(second))


if __name__ == "__main__":
    unittest.main()
