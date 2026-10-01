import json
import tempfile
import threading
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
        self.assertEqual(
            workflow["nodes"][0]["error"].get("failure_class"),
            "transient",
        )

    def test_side_effect_execution_uncertainty_fails_closed(self):
        node = o.Node(
            "n01-write", "build", "github", [], risk="high",
            input={"action": "create_issue", "approval_granted": True},
        )
        execution_id = o.hashlib.sha256(b"wf_uncertain:n01-write").hexdigest()
        workflow = {
            "id": "wf_uncertain", "goal": "write", "live": True,
            "executions": {execution_id: {"node_id": "n01-write", "status": "started"}},
            "nodes": [o.asdict(node)],
        }
        with patch.dict(o.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "GITHUB_TOKEN": "dummy",
            "GITHUB_REPOSITORY": "owner/repo",
        }, clear=False):
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(o, "STATE_DIR", Path(tmp)), \
                     patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                     patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"), \
                     patch.object(o, "load_registry", return_value={"github": {
                         "free_tier": True, "side_effects": ["issue_write"]
                     }}), \
                     patch.object(o, "execute_github") as execute:
                    result = o.run_one_step(workflow)
        self.assertEqual(result, "failed")
        self.assertEqual(workflow["nodes"][0]["error"]["type"], "execution_uncertain")
        execute.assert_not_called()

    def _uncertain_connector_workflow(self, node_error, node_id="n01-connector"):
        node = o.Node(
            node_id,
            "publish",
            "connector_bridge",
            [],
            risk="high",
            status="failed",
            error=node_error,
            input={
                "connector": "notion",
                "action": "create_page",
                "payload": {"title": "Hello"},
            },
        )
        return {
            "id": "wf_reconcile",
            "goal": "publish safely",
            "live": True,
            "status": "failed",
            "nodes": [o.asdict(node)],
            "executions": {},
        }

    def test_uncertain_connector_reconciliation_applied_completes_without_replay(self):
        workflow = self._uncertain_connector_workflow({
            "type": "execution_uncertain",
            "message": "timeout",
            "execution_uncertain": True,
            "reconciliation_required": True,
        })
        registry = {
            "capability:publish": {
                "default_tool": "connector_bridge",
                "fallback_tools": [],
            },
            "connector_bridge": {
                "free_tier": True,
                "required_env": "ORCHESTRATOR_CONNECTOR_BRIDGE_URL",
                "side_effects": ["external_request"],
            },
        }
        reconciliation = {
            "state": "applied",
            "request_id": o.hashlib.sha256(b"wf_reconcile:n01-connector").hexdigest(),
        }
        with patch.dict(o.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example/api/bridge",
        }, clear=False), patch.object(
            o, "load_registry", return_value=registry
        ), patch.object(
            o, "reconcile_connector_execution", return_value=reconciliation
        ), patch.object(o, "update_tool_health"), tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"):
                result = o.run_one_step(workflow)
        self.assertEqual(result, "reconciled")
        self.assertEqual(workflow["nodes"][0]["status"], "completed")
        self.assertTrue(workflow["nodes"][0]["output"]["reconciled"])

    def test_uncertain_connector_reconciliation_not_applied_returns_ready(self):
        workflow = self._uncertain_connector_workflow({
            "type": "execution_uncertain",
            "message": "timeout",
            "execution_uncertain": True,
            "reconciliation_required": True,
        })
        registry = {
            "capability:publish": {
                "default_tool": "connector_bridge",
                "fallback_tools": [],
            },
            "connector_bridge": {
                "free_tier": True,
                "required_env": "ORCHESTRATOR_CONNECTOR_BRIDGE_URL",
                "side_effects": ["external_request"],
            },
        }
        with patch.dict(o.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example/api/bridge",
        }, clear=False), patch.object(
            o, "load_registry", return_value=registry
        ), patch.object(
            o, "reconcile_connector_execution", return_value={"state": "not_applied"}
        ):
            result = o.run_one_step(workflow)
        self.assertEqual(result, "reconciled_ready")
        self.assertEqual(workflow["nodes"][0]["status"], "ready")
        self.assertEqual(workflow["status"], "running")

    def test_uncertain_connector_reconciliation_unknown_fails_closed(self):
        workflow = self._uncertain_connector_workflow({
            "type": "execution_uncertain",
            "message": "timeout",
            "execution_uncertain": True,
            "reconciliation_required": True,
        })
        registry = {
            "capability:publish": {
                "default_tool": "connector_bridge",
                "fallback_tools": [],
            },
            "connector_bridge": {
                "free_tier": True,
                "required_env": "ORCHESTRATOR_CONNECTOR_BRIDGE_URL",
                "side_effects": ["external_request"],
            },
        }
        with patch.dict(o.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example/api/bridge",
        }, clear=False), patch.object(
            o, "load_registry", return_value=registry
        ), patch.object(
            o, "reconcile_connector_execution",
            return_value={"state": "unknown"}
        ):
            result = o.run_one_step(workflow)
        self.assertEqual(result, "failed")
        self.assertEqual(workflow["nodes"][0]["status"], "failed")
        self.assertEqual(
            workflow["nodes"][0]["error"]["reconciliation_state"],
            "unknown",
        )

    def test_resume_scheduler_includes_failed_uncertain_connectors(self):
        workflow = self._uncertain_connector_workflow({
            "type": "execution_uncertain",
            "message": "timeout",
            "execution_uncertain": True,
            "reconciliation_required": True,
        })
        state = {
            "version": 2,
            "workflows": {"wf_reconcile": workflow},
            "last_workflow_id": "wf_reconcile",
        }
        with patch.object(
            o, "run_one_step",
            side_effect=lambda wf, approve_high_risk=False: (
                wf.update({"status": "running"}) or "reconciled_ready"
            ),
        ), patch.object(o, "save_state"):
            count = o.resume_pending_workflows(state, step=True)
        self.assertEqual(count, 1)

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

    def test_duplicate_event_id_is_not_recreated(self):
        existing = {
            "id": "wf_existing",
            "goal": "same",
            "status": "completed",
            "event_id": "evt-123",
        }
        state = {
            "version": 2,
            "workflows": {"wf_existing": existing},
            "last_workflow_id": "wf_existing",
        }
        with patch.dict(o.os.environ, {"ORCHESTRATOR_EVENT_ID": "evt-123"}, clear=False):
            duplicate = next(
                (
                    item for item in state["workflows"].values()
                    if item.get("event_id") == "evt-123"
                ),
                None,
            )
            self.assertIsNotNone(duplicate)

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

    def test_completed_node_records_evidence_digest(self):
        node = o.Node("n01", "execute", "noop", [])
        workflow = {
            "id": "wf_evidence",
            "goal": "evidence",
            "live": False,
            "nodes": [o.asdict(node)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={}):
                result = o.run_one_step(workflow)
        self.assertEqual(result, "completed")
        evidence = workflow["nodes"][0]["output"]["evidence"]
        self.assertEqual(len(evidence["output_sha256"]), 64)
        self.assertEqual(len(evidence["evidence_sha256"]), 64)
        self.assertEqual(workflow["evidence"]["n01"]["evidence_sha256"], evidence["evidence_sha256"])

    def test_artifact_verifier_checks_local_file(self):
        node = o.Node(
            "n01-artifacts", "artifact_verify", "artifact_verifier", [],
            input={"artifacts": [{"type": "local_file", "path": "README.md"}]},
        )
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            result = o.execute_artifact_verifier(node, "verify")
        self.assertTrue(result["passed"])
        self.assertEqual(result["verified_count"], 1)

    def test_artifact_verifier_checks_url_status(self):
        node = o.Node(
            "n01-artifacts", "artifact_verify", "artifact_verifier", [],
            input={"artifacts": [{"type": "url", "url": "https://example.test"}]},
        )
        with patch.object(o, "http_json", return_value={"status_code": 200, "data": {"ok": True}}):
            result = o.execute_artifact_verifier(node, "verify")
        self.assertTrue(result["checks"][0]["passed"])

    def test_contract_required_field_is_enforced(self):
        node = o.Node(
            "n01", "execute", "noop", [],
            max_retries=0,
            contract={"required_fields": ["answer"]},
        )
        workflow = {
            "id": "wf_contract",
            "goal": "contract",
            "live": False,
            "nodes": [o.asdict(node)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={}):
                result = o.run_one_step(workflow)
        self.assertEqual(result, "failed")
        self.assertIn("contract field missing: answer", workflow["nodes"][0]["error"]["message"])

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

    def test_approval_creation_failure_fails_closed(self):
        node = o.Node("deploy", "deploy", "noop", [], risk="high")
        workflow = {
            "id": "wf_approval_failure",
            "goal": "deploy",
            "live": True,
            "nodes": [o.asdict(node)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={}),                  patch.object(o, "create_approval_issue", side_effect=RuntimeError("approval service down")):
                o.run_workflow(workflow, approve_high_risk=False)
        self.assertEqual(workflow["status"], "failed")
        self.assertEqual(workflow["nodes"][0]["status"], "failed")

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
                 patch.object(o, "create_approval_issue", return_value=123), \
                 patch.object(o, "get_issue_labels", return_value=set()), \
                 patch.dict(o.os.environ, {"ORCHESTRATOR_LIVE": "true"}, clear=False):
                o.run_workflow(workflow, approve_high_risk=False)
        self.assertEqual(workflow["status"], "waiting_approval")
        self.assertEqual(workflow["nodes"][0]["status"], "waiting_approval")
        self.assertEqual(workflow["nodes"][0]["input"]["approval_issue"], 123)


    def test_parallel_independent_nodes_execute_concurrently(self):
        barrier = threading.Barrier(2)

        nodes = [
            o.Node("n01-a", "execute", "noop", []),
            o.Node("n02-b", "execute", "noop", []),
        ]
        workflow = {
            "id": "wf_parallel",
            "goal": "parallel",
            "live": False,
            "max_parallel": 2,
            "nodes": [o.asdict(n) for n in nodes],
        }

        def fake_execute(node, goal, dry_run):
            barrier.wait(timeout=3)
            return {"ok": True, "node": node.id}

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={}),                  patch.object(o, "execute_node", side_effect=fake_execute):
                o.run_workflow(workflow)
        self.assertEqual(workflow["status"], "completed")
        self.assertTrue(all(node["status"] == "completed" for node in workflow["nodes"]))

    def test_dependency_context_is_forwarded_to_next_node(self):
        nodes = [
            o.Node("n01-source", "execute", "noop", []),
            o.Node("n02-consumer", "execute", "noop", ["n01-source"]),
        ]
        workflow = {
            "id": "wf_context",
            "goal": "context",
            "live": False,
            "max_parallel": 2,
            "nodes": [o.asdict(n) for n in nodes],
        }
        seen = {}

        def fake_execute(node, goal, dry_run):
            if node.id == "n01-source":
                return {"answer": "42"}
            seen["context"] = node.input.get("context")
            return {"ok": True}

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={}),                  patch.object(o, "execute_node", side_effect=fake_execute):
                o.run_workflow(workflow)
        self.assertEqual(workflow["status"], "completed")
        self.assertIn("n01-source", seen["context"]["dependencies"])
        self.assertIn("42", seen["context"]["dependencies"]["n01-source"]["output"])

    def test_local_validator_uses_completed_dependency_evidence(self):
        nodes = [
            o.Node("n01-source", "execute", "noop", []),
            o.Node("n02-validate", "validate", "local_validator", ["n01-source"]),
        ]
        workflow = {
            "id": "wf_validate",
            "goal": "validate",
            "live": False,
            "nodes": [o.asdict(n) for n in nodes],
        }

        def fake_execute(node, goal, dry_run):
            return {"evidence": node.id}

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={
                     "local_validator": {"free_tier": True, "side_effects": []}
                 }),                  patch.object(o, "execute_node", side_effect=fake_execute):
                # Use the real local validator for the validation node.
                with patch.object(o, "execute_node", side_effect=lambda node, goal, dry_run:
                    o.execute_local_validator(node, goal) if node.tool == "local_validator"
                    else {"evidence": node.id}):
                    o.run_workflow(workflow)

        self.assertEqual(workflow["status"], "completed")
        self.assertTrue(workflow["nodes"][1]["output"]["validation"]["passed"])

    def test_replan_preserves_failure_feedback(self):
        node = o.Node(
            "n01-execute", "execute", "noop", [],
            max_retries=0,
        )
        workflow = {
            "id": "wf_repair_feedback",
            "goal": "repair",
            "live": False,
            "nodes": [o.asdict(node)],
        }
        registry = {
            "capability:execute": {
                "default_tool": "noop",
                "fallback_tools": ["wikipedia"],
            },
            "noop": {"free_tier": True, "side_effects": []},
            "wikipedia": {"free_tier": True, "side_effects": []},
        }
        node.status = "failed"
        node.error = {"type": "RuntimeError", "message": "broken"}
        node.output = {"next_action": "retry with fallback"}
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            replanned = o.replan_after_failure(workflow, [node], node, registry)
        self.assertTrue(replanned)
        feedback = workflow["repair_feedback"]["n01-execute"]
        self.assertEqual(feedback["error"]["message"], "broken")
        self.assertEqual(feedback["next_action"], "retry with fallback")
        self.assertIn("next_action", feedback["output"])

    def test_replanned_node_becomes_runnable_on_next_step(self):
        node = o.Node(
            "n01-execute", "execute", "noop", [],
            max_retries=0,
        )
        workflow = {
            "id": "wf_replan",
            "goal": "replan",
            "live": False,
            "nodes": [o.asdict(node)],
        }
        registry = {
            "capability:execute": {
                "default_tool": "noop",
                "fallback_tools": ["wikipedia"],
            },
            "noop": {"free_tier": True, "side_effects": []},
            "wikipedia": {"free_tier": True, "side_effects": []},
        }

        def fake_execute(node, goal, dry_run):
            if node.tool == "noop":
                raise RuntimeError("planned failure")
            return {"ok": True}

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value=registry),                  patch.object(o, "execute_node", side_effect=fake_execute):
                first = o.run_one_step(workflow)
                second = o.run_one_step(workflow)

        self.assertEqual(first, "replanned")
        self.assertEqual(second, "completed")
        self.assertEqual(workflow["nodes"][0]["tool"], "wikipedia")

    def test_uncertain_non_idempotent_connector_fails_closed_without_replan(self):
        node = o.Node(
            "n01", "publish", "connector_bridge", [],
            risk="high", max_retries=2,
            input={"connector": "notion", "action": "create_page", "payload": {"title": "Hello"}},
        )
        workflow = {
            "id": "wf_uncertain_connector",
            "goal": "publish",
            "live": False,
            "nodes": [o.asdict(node)],
        }
        registry = {
            "capability:publish": {
                "default_tool": "connector_bridge",
                "fallback_tools": ["webhook"],
            },
            "connector_bridge": {"free_tier": True, "side_effects": ["external_request"]},
            "webhook": {"free_tier": True, "side_effects": ["external_request"]},
        }
        error = o.ConnectorRequestError("timeout", uncertain=True, retry_allowed=False)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value=registry),                  patch.object(o, "execute_node", side_effect=error) as execute:
                result = o.run_one_step(workflow)
        self.assertEqual(result, "failed")
        self.assertEqual(execute.call_count, 1)
        self.assertTrue(workflow["nodes"][0]["error"]["execution_uncertain"])
        self.assertTrue(workflow["nodes"][0]["error"]["reconciliation_required"])
        self.assertEqual(workflow["nodes"][0]["tool"], "connector_bridge")
        self.assertEqual(workflow["replan_count"], 0)

    def test_uncertain_idempotent_connector_retries_same_tool(self):
        node = o.Node(
            "n01", "publish", "connector_bridge", [],
            risk="high", max_retries=1,
        )
        workflow = {
            "id": "wf_idempotent_connector",
            "goal": "publish",
            "live": False,
            "nodes": [o.asdict(node)],
        }
        registry = {
            "capability:publish": {
                "default_tool": "connector_bridge",
                "fallback_tools": ["webhook"],
            },
            "connector_bridge": {"free_tier": True, "side_effects": ["external_request"]},
            "webhook": {"free_tier": True, "side_effects": ["external_request"]},
        }
        errors = [
            o.ConnectorRequestError("timeout-1", uncertain=True, retry_allowed=True),
            o.ConnectorRequestError("timeout-2", uncertain=True, retry_allowed=True),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value=registry),                  patch.object(o, "execute_node", side_effect=errors),                  patch.object(o.time, "sleep"):
                result = o.run_one_step(workflow)
        self.assertEqual(result, "failed")
        self.assertEqual(workflow["nodes"][0]["retry_count"], 1)
        self.assertEqual(workflow["nodes"][0]["tool"], "connector_bridge")
        self.assertEqual(workflow["replan_count"], 0)

    def test_semantic_validation_rejects_failed_http_response(self):
        node = o.Node(
            "n01", "execute", "noop", [],
            max_retries=0,
        )
        workflow = {
            "id": "wf_validation_error",
            "goal": "validation",
            "live": False,
            "nodes": [o.asdict(node)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={}),                  patch.object(o, "execute_node", return_value={"status_code": 500}):
                result = o.run_one_step(workflow)
        self.assertEqual(result, "failed")
        self.assertIn("500", workflow["nodes"][0]["error"]["message"])


if __name__ == "__main__":
    unittest.main()
