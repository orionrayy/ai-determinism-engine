from datetime import datetime, timedelta, timezone
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("_orchestrator_federation_recovery", ROOT / "orchestrator.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load orchestrator module")
o = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = o
SPEC.loader.exec_module(o)


class StaleFederationRecoveryTests(unittest.TestCase):
    def test_stale_safe_federation_rearms_nodes_and_refunds_reservations(self):
        created = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        nodes = [
            o.Node("n01", "research", "wikipedia", [], status="delegated"),
            o.Node("n02", "analyze", "gemini", [], status="delegated"),
        ]
        workflow = {
            "id": "wf-stale-fed",
            "status": "waiting_agents",
            "attempts_used": 5,
            "max_attempts": 64,
            "federation_batches_used": 1,
            "federation_tasks_used": 2,
            "federation": {
                "id": "fed-stale",
                "status": "dispatched",
                "created_at": created,
                "tasks": [
                    {"task_id": "n01"},
                    {"task_id": "n02"},
                ],
            },
        }
        registry = {
            "wikipedia": {"side_effects": [], "free_tier": True},
            "gemini": {"side_effects": [], "free_tier": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)), patch.object(o, "persist_workflow"), patch.object(o, "append_event"):
                result = o.rearm_stale_federation(
                    workflow,
                    nodes,
                    registry,
                    now=datetime.now(timezone.utc),
                )
        self.assertEqual(result, "rearmed")
        self.assertEqual(workflow["status"], "ready")
        self.assertEqual(workflow["attempts_used"], 3)
        self.assertEqual(workflow["federation_tasks_used"], 0)
        self.assertEqual(workflow["federation_batches_used"], 0)
        self.assertEqual(workflow["federation"]["status"], "abandoned")
        self.assertTrue(all(node.status == "ready" for node in nodes))

    def test_stale_recovery_refuses_side_effecting_tasks(self):
        created = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        node = o.Node("n01", "publish", "connector_bridge", [], risk="high", status="delegated")
        workflow = {
            "id": "wf-stale-unsafe",
            "status": "waiting_agents",
            "attempts_used": 1,
            "max_attempts": 64,
            "federation": {
                "id": "fed-unsafe",
                "status": "dispatched",
                "created_at": created,
                "tasks": [{"task_id": "n01"}],
            },
        }
        registry = {
            "connector_bridge": {"side_effects": ["external_request"], "free_tier": True},
        }
        with patch.object(o, "persist_workflow"), patch.object(o, "append_event"):
            result = o.rearm_stale_federation(workflow, [node], registry)
        self.assertEqual(result, "failed")
        self.assertEqual(workflow["status"], "failed")
        self.assertEqual(node.status, "delegated")


if __name__ == "__main__":
    unittest.main()
