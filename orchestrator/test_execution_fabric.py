import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import orchestrator as o


class ExecutionFabricTests(unittest.TestCase):
    def test_policy_reroutes_to_free_tool(self):
        nodes = [o.Node("n01", "analyze", "paid")]
        registry = {
            "capability:analyze": {
                "default_tool": "paid",
                "fallback_tools": ["free"],
            },
            "paid": {"required_env": "PAID_KEY", "free_tier": False, "risk": "low"},
            "free": {"required_env": None, "free_tier": True, "risk": "low"},
        }
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=True):
            o.enforce_node_policy(nodes, registry, live=False)
        self.assertEqual(nodes[0].tool, "free")

    def test_completed_node_persists_tool_health(self):
        workflow = {
            "id": "wf_health",
            "goal": "health",
            "live": False,
            "nodes": [o.asdict(o.Node("n01", "execute", "noop", []))],
        }
        registry = {
            "capability:execute": {
                "default_tool": "noop",
                "fallback_tools": [],
            },
            "noop": {"required_env": None, "free_tier": True, "side_effects": [], "risk": "low"},
        }
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with patch.object(o, "STATE_DIR", root),                      patch.object(o, "EVENT_FILE", root / "events.jsonl"),                      patch.object(o, "CHECKPOINT_DIR", root / "checkpoints"),                      patch.object(o, "load_registry", return_value=registry):
                    result = o.run_one_step(workflow)
                    self.assertEqual(result, "completed")
                    health = json.loads(
                        o.tool_health_path("noop").read_text(encoding="utf-8")
                    )
                    self.assertEqual(health["noop"]["status"], "healthy")
                    self.assertEqual(health["noop"]["failure_streak"], 0)


if __name__ == "__main__":
    unittest.main()
