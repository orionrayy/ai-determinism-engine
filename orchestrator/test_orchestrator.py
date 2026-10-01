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
                "firecrawl": {"required_env": "FIRECRAWL_API_KEY", "free_tier": False},
                "wikipedia": {"required_env": None, "free_tier": True},
            }
            with patch.dict(o.os.environ, {}, clear=True):
                nodes = o.deterministic_plan("research AI safety", registry)
        self.assertEqual(nodes[0].tool, "wikipedia")

    def test_free_only_blocks_external_paid_adapters(self):
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            self.assertFalse(o.tool_available("openai", {"openai": {"free_tier": False}}))
            self.assertFalse(o.tool_available("firecrawl", {"firecrawl": {"free_tier": False}}))
            self.assertFalse(o.tool_available("webhook", {"webhook": {"free_tier": False}}))
            self.assertTrue(o.tool_available("wikipedia", {"wikipedia": {"free_tier": True}}))
    def test_one_step_does_not_require_github_token(self):
        workflow = {
            "id": "wf_no_token",
            "goal": "research without network",
            "live": False,
            "nodes": [o.asdict(o.Node("n01", "execute", "noop"))],
        }
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            with patch.dict(o.os.environ, {"GITHUB_TOKEN": ""}, clear=False):
                with tempfile.TemporaryDirectory() as tmp:
                    with patch.object(o, "STATE_DIR", Path(tmp)), \
                         patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                         patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"), \
                         patch.object(o, "load_registry", return_value={}):
                        result = o.run_one_step(workflow, approve_high_risk=False)
        self.assertEqual(result, "completed")
    def test_free_only_is_registry_driven(self):
        registry = {
            "future_paid": {"free_tier": False},
            "future_free": {"free_tier": True},
        }
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            self.assertFalse(o.tool_available("future_paid", registry))
            self.assertTrue(o.tool_available("future_free", registry))

    def test_policy_raises_risk_for_deploy_even_if_planner_says_low(self):
        node = o.Node("n01-deploy", "deploy", "noop", [], risk="low")
        o.enforce_node_policy([node], {"noop": {"side_effects": []}})
        self.assertEqual(node.risk, "high")

    def test_policy_raises_risk_for_github_issue_write(self):
        node = o.Node(
            "n01-build", "build", "github", [], risk="medium",
            input={"action": "create_issue"},
        )
        o.enforce_node_policy([node], {"github": {"side_effects": ["issue_write"]}})
        self.assertEqual(node.risk, "high")

    def test_github_read_file_decodes_base64(self):
        node = o.Node(
            "n01-read", "execute", "github", [],
            input={"action": "read_file", "path": "README.md", "branch": "main"},
        )
        def fake_http(url, method="GET", body=None, headers=None, timeout=60):
            import base64
            return {"status_code": 200, "data": {
                "path": "README.md",
                "sha": "abc",
                "content": base64.b64encode(b"hello").decode(),
            }}
        with patch.object(o, "github_headers", return_value={"Authorization": "Bearer x"}),              patch.object(o, "github_repository", return_value="owner/repo"),              patch.object(o, "http_json", side_effect=fake_http):
            result = o.execute_github(node)
        self.assertEqual(result["data"]["decoded_content"], "hello")

    def test_github_update_file_sends_existing_sha(self):
        node = o.Node(
            "n01-write", "build", "github", [],
            input={"action": "create_or_update_file", "path": "src/app.py", "content": "print(1)", "branch": "main"},
        )
        calls = []
        def fake_http(url, method="GET", body=None, headers=None, timeout=60):
            calls.append((url, method, body))
            if method == "GET":
                return {"status_code": 200, "data": {"sha": "existing-sha"}}
            return {"status_code": 201, "data": {"commit": {"sha": "new-sha"}}}
        with patch.object(o, "github_headers", return_value={"Authorization": "Bearer x"}),              patch.object(o, "github_repository", return_value="owner/repo"),              patch.object(o, "http_json", side_effect=fake_http):
            result = o.execute_github(node)
        self.assertEqual(result["status_code"], 201)
        self.assertEqual(calls[-1][1], "PUT")
        self.assertEqual(calls[-1][2]["sha"], "existing-sha")

    def test_github_path_rejects_traversal(self):
        node = o.Node(
            "n01-write", "build", "github", [],
            input={"action": "create_or_update_file", "path": "../secret.txt", "content": "x"},
        )
        with patch.object(o, "github_headers", return_value={"Authorization": "Bearer x"}),              patch.object(o, "github_repository", return_value="owner/repo"):
            with self.assertRaises(RuntimeError):
                o.execute_github(node)

    def test_policy_rejects_unknown_tool(self):
        node = o.Node("n01", "execute", "unknown_tool")
        with self.assertRaises(ValueError):
            o.enforce_node_policy([node], {"noop": {"side_effects": []}})

    def test_one_step_retry_reenters_running_state(self):
        workflow = {
            "id": "wf_retry",
            "goal": "retry",
            "live": False,
            "nodes": [o.asdict(o.Node("n01", "execute", "noop", [], max_retries=2))],
        }
        calls = {"n": 0}

        def flaky_execute(node, goal, dry_run):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return {"ok": True}

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={}),                  patch.object(o, "execute_node", side_effect=flaky_execute):
                result = o.run_one_step(workflow)
        self.assertEqual(result, "completed")
        self.assertEqual(calls["n"], 2)
        self.assertEqual(workflow["nodes"][0]["retry_count"], 1)

    def test_preapproved_high_risk_node_can_resume(self):
        node = o.Node(
            "n01-deploy", "deploy", "noop", [], risk="high",
            input={"approval_granted": True},
        )
        workflow = {
            "id": "wf_approval", "goal": "deploy", "live": True,
            "nodes": [o.asdict(node)],
        }
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(o, "STATE_DIR", Path(tmp)), \
                     patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                     patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"), \
                     patch.object(o, "load_registry", return_value={}):
                    result = o.run_one_step(workflow, approve_high_risk=False)
        self.assertEqual(result, "completed")
        self.assertEqual(workflow["status"], "completed")
    def test_resume_scheduler_prioritizes_oldest_updated_workflow(self):
        older = {
            "id": "wf_old", "goal": "old", "live": False,
            "status": "running", "updated_at": "2026-10-01T00:00:00+00:00",
            "nodes": [o.asdict(o.Node("n01", "execute", "noop"))],
        }
        newer = {
            "id": "wf_new", "goal": "new", "live": False,
            "status": "running", "updated_at": "2026-10-01T01:00:00+00:00",
            "nodes": [o.asdict(o.Node("n01", "execute", "noop"))],
        }
        state = {"version": 2, "workflows": {"wf_new": newer, "wf_old": older}, "last_workflow_id": "wf_new"}
        with patch.object(o, "run_one_step", side_effect=lambda wf, approve_high_risk=False: (wf.update({"status":"completed"}) or "completed")):
            with patch.object(o, "save_state"):
                count = o.resume_pending_workflows(state, step=True)
        self.assertEqual(count, 1)
        self.assertEqual(state["last_workflow_id"], "wf_old")
    def test_one_step_advances_dag_incrementally(self):
        workflow = o.create_workflow('build a website', live=False)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, 'STATE_DIR', Path(tmp)), \
                 patch.object(o, 'EVENT_FILE', Path(tmp) / 'events.jsonl'), \
                 patch.object(o, 'CHECKPOINT_DIR', Path(tmp) / 'checkpoints'), \
                 patch.object(o, 'load_registry', return_value={}):
                statuses = []
                for _ in range(len(workflow['nodes'])):
                    statuses.append(o.run_one_step(workflow, approve_high_risk=False))
                self.assertEqual(statuses[-1], 'completed')
                self.assertEqual(workflow['status'], 'completed')
                self.assertTrue(all(node['status'] == 'completed' for node in workflow['nodes']))
    def test_free_only_rejects_paid_node_at_execution_time(self):
        node = o.Node("n01", "analyze", "openai")
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            with self.assertRaises(RuntimeError):
                o.execute_node(node, "test", dry_run=False)

    def test_workflow_creation_falls_back_without_gemini_key(self):
        with patch.dict(o.os.environ, {}, clear=True):
            workflow = o.create_workflow("build a small website", live=False)
        self.assertGreaterEqual(len(workflow["nodes"]), 4)
        self.assertEqual(workflow["status"], "planning")

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
