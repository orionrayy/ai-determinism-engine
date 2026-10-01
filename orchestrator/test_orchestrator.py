import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import orchestrator as o


class OrchestratorTests(unittest.TestCase):
    def test_plan_is_acyclic(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "REGISTRY_FILE", Path(tmp) / "missing.json"):
                nodes = o.deterministic_plan("build and deploy a web app", {})
        o.validate_dag(nodes)
        self.assertEqual(len(nodes), 7)
        self.assertEqual(nodes[-1].capability, "notify")

    def test_cycle_is_rejected(self):
        a = o.Node("a", "x", "noop", ["b"])
        b = o.Node("b", "x", "noop", ["a"])
        with self.assertRaises(ValueError):
            o.validate_dag([a, b])

    def test_high_risk_requires_approval_in_live_mode(self):
        nodes = [
            o.Node("deploy", "deploy", "noop", [], risk="high"),
        ]
        workflow = {
            "id": "wf_test",
            "goal": "deploy",
            "live": True,
            "nodes": [o.asdict(nodes[0])],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)), \
                 patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                 patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"), \
                 patch.object(o, "load_registry", return_value={}):
                o.run_workflow(workflow, approve_high_risk=False)
        self.assertEqual(workflow["status"], "waiting_approval")
        self.assertEqual(workflow["nodes"][0]["status"], "waiting_approval")


if __name__ == "__main__":
    unittest.main()
