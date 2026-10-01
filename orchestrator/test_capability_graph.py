import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import capability_graph as cg


class CapabilityGraphTests(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "capability:analyze": {
                "default_tool": "paid",
                "fallback_tools": ["free_keyed", "free"],
            },
            "paid": {
                "required_env": "PAID_KEY",
                "free_tier": False,
                "risk": "low",
            },
            "free_keyed": {
                "required_env": "FREE_KEY",
                "free_tier": True,
                "risk": "low",
            },
            "free": {
                "required_env": None,
                "free_tier": True,
                "risk": "low",
            },
        }

    def test_free_no_credential_wins_deterministically(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=True):
            self.assertEqual(
                cg.route_capability("analyze", self.registry, live=False),
                "free",
            )

    def test_live_prefers_available_credential_free_tool(self):
        with patch.dict(os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "FREE_KEY": "x",
        }, clear=True):
            self.assertEqual(
                cg.route_capability("analyze", self.registry, live=True),
                "free_keyed",
            )

    def test_quarantined_tool_is_skipped_during_cooldown(self):
        health = {
            "free": {
                "status": cg.QUARANTINED,
                "failure_streak": 2,
                "cooldown_until": 500,
            }
        }
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=True):
            self.assertEqual(
                cg.route_capability("analyze", self.registry, health, live=False, now=100),
                "free_keyed",
            )

    def test_expired_quarantine_enters_probe(self):
        health = {
            "free": {
                "status": cg.QUARANTINED,
                "failure_streak": 2,
                "cooldown_until": 100,
            }
        }
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=True):
            self.assertEqual(cg.effective_health(health, "free", now=100), cg.PROBING)

    def test_read_only_failures_degrade_then_quarantine(self):
        health = {}
        cg.record_tool_result(health, "free", success=False, now=10)
        self.assertEqual(health["free"]["status"], cg.DEGRADED)
        cg.record_tool_result(health, "free", success=False, now=20)
        self.assertEqual(health["free"]["status"], cg.QUARANTINED)
        cg.record_tool_result(health, "free", success=True, now=30)
        self.assertEqual(health["free"]["status"], cg.HEALTHY)
        self.assertEqual(health["free"]["failure_streak"], 0)

    def test_side_effect_failure_quarantines_immediately(self):
        health = {}
        cg.record_tool_result(health, "webhook", success=False, side_effecting=True, now=10)
        self.assertEqual(health["webhook"]["status"], cg.QUARANTINED)
        self.assertEqual(health["webhook"]["cooldown_until"], 910)

    def test_health_round_trip(self):
        health = {"free": {"status": cg.HEALTHY, "failure_streak": 0}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tool_health.json"
            cg.save_health(path, health)
            self.assertEqual(cg.load_health(path), health)


if __name__ == "__main__":
    unittest.main()
