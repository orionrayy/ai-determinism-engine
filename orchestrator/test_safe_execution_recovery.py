import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("_orchestrator_safe_recovery", ROOT / "orchestrator.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load orchestrator module")
o = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = o
SPEC.loader.exec_module(o)


class SafeExecutionRecoveryTests(unittest.TestCase):
    def test_interrupted_safe_node_is_rearmed(self):
        node = o.Node(
            "n01",
            "research",
            "wikipedia",
            [],
            status="running",
            retry_count=1,
            output={"stale": True},
            error={"stale": True},
        )
        workflow = {"id": "wf-safe-recovery", "status": "running"}
        registry = {"wikipedia": {"side_effects": [], "free_tier": True}}
        with patch.object(o, "append_event"):
            recovered = o.recover_inflight_safe_nodes(workflow, [node], registry)
        self.assertTrue(recovered)
        self.assertEqual(node.status, "ready")
        self.assertEqual(node.retry_count, 1)
        self.assertEqual(node.output, {})
        self.assertEqual(node.error, {})
        self.assertEqual(workflow["status"], "running")

    def test_interrupted_side_effecting_node_is_not_rearmed_by_safe_recovery(self):
        node = o.Node(
            "n01",
            "publish",
            "connector_bridge",
            [],
            risk="high",
            status="running",
        )
        workflow = {"id": "wf-side-effect", "status": "running"}
        registry = {
            "connector_bridge": {"side_effects": ["external_request"], "free_tier": True},
        }
        with patch.object(o, "append_event") as events:
            recovered = o.recover_inflight_safe_nodes(workflow, [node], registry)
        self.assertFalse(recovered)
        self.assertEqual(node.status, "running")
        events.assert_not_called()


if __name__ == "__main__":
    unittest.main()
