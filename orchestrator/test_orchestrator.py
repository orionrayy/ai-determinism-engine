import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import orchestrator as o


class OrchestratorTests(unittest.TestCase):
    def test_credential_free_research_prefers_wikipedia(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = {
                "capability:research": {
                    "default_tool": "firecrawl",
                    "fallback_tools": ["wikipedia"],
                },
                "firecrawl": {"required_env": "FIRECRAWL_API_KEY"},
                "wikipedia": {"required_env": None},
            }
            with patch.dict(o.os.environ, {}, clear=True):
                nodes = o.deterministic_plan("research AI safety", registry)
        self.assertEqual(nodes[0].tool, "wikipedia")

    def test_workflow_creation_is_persistable(self):
        workflow = o.create_workflow("build a small website", live=False)
        self.assertTrue(workflow["id"].startswith("wf_"))
        self.assertEqual(workflow["status"], "planning")
        self.assertEqual(workflow["execution_mode"], "dry-run")
        self.assertGreaterEqual(len(workflow["nodes"]), 4)

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

    def test_http_requires_https(self):
        with self.assertRaises(RuntimeError):
            o.http_json("http://example.com")

    def test_live_state_does_not_require_process_env_on_resume(self):
        nodes = [o.Node("safe", "execute", "noop", [], risk="low")]
        workflow = {
            "id": "wf_resume",
            "goal": "resume",
            "live": True,
            "nodes": [o.asdict(nodes[0])],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)), \
                 patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                 patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"), \
                 patch.object(o, "load_registry", return_value={}):
                o.run_workflow(workflow, approve_high_risk=False)
        self.assertEqual(workflow["execution_mode"], "live")
        self.assertEqual(workflow["status"], "completed")

    def test_dry_run_end_to_end_completes_without_credentials(self):
        workflow = o.create_workflow("research an offline technical topic", live=False)
        self.assertEqual(workflow["status"], "planning")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)), \
                 patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                 patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"), \
                 patch.object(o, "load_registry", return_value={}):
                o.run_workflow(workflow, approve_high_risk=False)
        self.assertEqual(workflow["status"], "completed")
        self.assertTrue(all(node["status"] == "completed" for node in workflow["nodes"]))

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
                 patch.object(o, "load_registry", return_value={}), \
                 patch.dict(o.os.environ, {"ORCHESTRATOR_LIVE": "true"}, clear=False):
                o.run_workflow(workflow, approve_high_risk=False)
        self.assertEqual(workflow["status"], "waiting_approval")
        self.assertEqual(workflow["nodes"][0]["status"], "waiting_approval")


if __name__ == "__main__":
    unittest.main()
