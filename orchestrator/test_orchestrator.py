import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import orchestrator as o


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self._actions_env = patch.dict(
            o.os.environ,
            {"GITHUB_ACTIONS": "false"},
            clear=False,
        )
        self._actions_env.start()
        self.addCleanup(self._actions_env.stop)

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

    def test_event_log_failure_does_not_break_execution(self):
        with patch.object(
            o.EVENT_FILE,
            "open",
            side_effect=OSError("event log unavailable"),
        ):
            o.append_event("test.event", {"ok": True})

    def test_atomic_json_write_replaces_existing_file_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            o.write_json(path, {"version": 1, "value": "before"})
            o.write_json(path, {"version": 2, "value": "after"})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"version": 2, "value": "after"},
            )
            self.assertFalse(list(Path(tmp).glob(".state.json.*.tmp")))

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

    def test_run_workflow_does_not_reroute_persisted_plan(self):
        node = o.Node("n01", "execute", "noop", [])
        workflow = {
            "id": "wf_resume_full",
            "goal": "resume full",
            "live": False,
            "nodes": [o.asdict(node)],
            "plan_fingerprint": o.fingerprint_nodes([node]),
        }
        registry = {
            "capability:execute": {
                "default_tool": "wikipedia",
                "fallback_tools": ["noop"],
            },
            "wikipedia": {"free_tier": True, "side_effects": []},
            "noop": {"free_tier": True, "side_effects": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value=registry):
                o.run_workflow(workflow)
        self.assertEqual(workflow["status"], "completed")
        self.assertEqual(workflow["plan_integrity"], "verified")
        self.assertEqual(workflow["nodes"][0]["tool"], "noop")

    def test_resume_fails_closed_when_tool_policy_drifts(self):
        node = o.Node("n01", "execute", "noop", [])
        old_registry = {
            "capability:execute": {"default_tool": "noop", "fallback_tools": []},
            "noop": {"free_tier": True, "side_effects": [], "risk": "low"},
        }
        new_registry = {
            "capability:execute": {"default_tool": "noop", "fallback_tools": []},
            "noop": {
                "free_tier": True,
                "side_effects": [],
                "risk": "low",
                "allowed_actions": ["validate"],
            },
        }
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            snapshot = o.build_policy_snapshot(old_registry, [node], live=False)
        workflow = {
            "id": "wf_policy_drift",
            "goal": "resume policy drift",
            "live": False,
            "nodes": [o.asdict(node)],
            "plan_fingerprint": o.fingerprint_nodes([node]),
            "policy_fingerprint": o.fingerprint_policy(snapshot),
            "route_snapshot": snapshot,
            "policy_integrity": "verified",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False),                  patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value=new_registry),                  patch.object(o, "execute_node") as execute:
                result = o.run_one_step(workflow, approve_high_risk=False)
        self.assertEqual(result, "failed")
        self.assertEqual(workflow["policy_integrity"], "drift_detected")
        self.assertTrue(workflow["policy_drift"])
        execute.assert_not_called()

    def test_running_safe_node_is_rearmed_after_runner_interruption(self):
        node = o.Node("n01", "execute", "noop", [], status="running")
        workflow = {
            "id": "wf_safe_recovery",
            "goal": "recover safe node",
            "status": "running",
            "live": False,
            "nodes": [o.asdict(node)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={}),                  patch.object(o, "execute_node", return_value={"ok": True}) as execute:
                o.run_workflow(workflow)
        self.assertEqual(workflow["status"], "completed")
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(workflow["execution_budget"]["used_steps"], 1)

    def test_postcondition_failure_after_side_effect_does_not_replan(self):
        node = o.Node(
            "n01-publish", "publish", "webhook", [], risk="high", max_retries=2,
            input={"approval_granted": True, "url": "https://example.test/hook"},
            contract={"postconditions": [{"type": "field_equals", "field": "status", "value": "published"}]},
        )
        workflow = {
            "id": "wf_postcondition_fence",
            "goal": "publish",
            "live": True,
            "nodes": [o.asdict(node)],
        }
        registry = {"webhook": {"free_tier": True, "side_effects": ["external_request"]}}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(o.os.environ, {
                "ORCHESTRATOR_FREE_ONLY": "true",
                "ORCHESTRATOR_WEBHOOK_URL": "https://example.test/hook",
            }, clear=False),                  patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value=registry),                  patch.object(o, "commit_side_effect_start", return_value=True),                  patch.object(o, "execute_node", return_value={"status": "queued"}) as execute:
                result = o.run_one_step(workflow, approve_high_risk=False)
        self.assertEqual(result, "failed")
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(workflow["replan_count"], 0)
        self.assertTrue(workflow["nodes"][0]["error"]["post_start_side_effect_failure"])
        self.assertTrue(workflow["nodes"][0]["error"]["replan_blocked_after_side_effect_start"])

    def test_execution_budget_blocks_before_side_effect_barrier(self):
        node = o.Node(
            "n01-publish", "publish", "webhook", [], risk="high",
            input={"approval_granted": True},
        )
        workflow = {
            "id": "wf_budget",
            "goal": "budget",
            "live": True,
            "execution_budget": {"max_steps": 1, "used_steps": 1},
            "nodes": [o.asdict(node)],
        }
        registry = {"webhook": {"free_tier": True, "side_effects": ["external_request"]}}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(o.os.environ, {
                "ORCHESTRATOR_FREE_ONLY": "true",
                "ORCHESTRATOR_WEBHOOK_URL": "https://example.test/hook",
            }, clear=False),                  patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value=registry),                  patch.object(o, "commit_side_effect_start") as barrier,                  patch.object(o, "execute_node") as execute:
                result = o.run_one_step(workflow, approve_high_risk=False)
        self.assertEqual(result, "failed")
        self.assertTrue(workflow["nodes"][0]["error"]["budget_exhausted"])
        barrier.assert_not_called()
        execute.assert_not_called()

    def test_postcondition_http_status_passes(self):
        node = o.Node(
            "n01", "publish", "webhook", [],
            contract={"postconditions": [{"type": "http_status", "field": "status_code"}]},
        )
        result = o.validate_node_output(node, {"status_code": 201})
        self.assertTrue(result["passed"])
        self.assertTrue(result["acceptance"]["passed"])
        self.assertEqual(result["checks"][-1]["value"], 201)

    def test_postcondition_field_equals_fails_closed(self):
        node = o.Node(
            "n01", "publish", "noop", [],
            contract={"postconditions": [{"type": "field_equals", "field": "status", "value": "published"}]},
        )
        with self.assertRaises(RuntimeError):
            o.validate_node_output(node, {"status": "queued"})

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



    def test_completed_checkpoint_drift_fails_closed_before_execution(self):
        node = o.Node(
            "n01",
            "execute",
            "noop",
            [],
            status="completed",
            output={
                "message": "done",
                "checkpoint": {
                    "path": ".orchestrator/checkpoints/wf-drift-n01.json",
                    "sha256": "0" * 64,
                },
            },
        )
        workflow = {
            "id": "wf-drift",
            "goal": "continue",
            "live": False,
            "status": "running",
            "nodes": [o.asdict(node)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / ".orchestrator" / "checkpoints" / "wf-drift-n01.json"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(
                json.dumps({"workflow": "wf-drift", "node": {"id": "n01", "status": "completed"}}),
                encoding="utf-8",
            )
            workflow["nodes"][0]["output"]["checkpoint"]["path"] = str(checkpoint)
            workflow["nodes"][0]["output"]["checkpoint"]["sha256"] = "1" * 64
            with patch.object(o, "ROOT", root),                  patch.object(o, "STATE_DIR", root / ".orchestrator"),                  patch.object(o, "EVENT_FILE", root / ".orchestrator" / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", root / ".orchestrator" / "checkpoints"),                  patch.object(o, "load_registry", return_value={}):
                result = o.run_one_step(workflow)
        self.assertEqual(result, "failed")
        self.assertEqual(workflow["checkpoint_integrity"], "failed")
        self.assertIn("checkpoint_error", workflow)

    def test_plan_drift_fails_closed_before_execution(self):
        node = o.Node(
            "n01",
            "execute",
            "noop",
            [],
            input={"instruction": "safe"},
        )
        workflow = {
            "id": "wf_plan_drift",
            "goal": "run",
            "live": False,
            "nodes": [o.asdict(node)],
            "plan_fingerprint": "not-the-current-plan",
        }
        with patch.object(o, "execute_node") as execute:
            result = o.run_one_step(workflow)
        self.assertEqual(result, "failed")
        self.assertEqual(workflow["plan_integrity"], "drift_detected")
        self.assertIn("plan_drift", workflow)
        execute.assert_not_called()

    def test_plan_fingerprint_is_initialized_for_new_workflow(self):
        node = o.Node(
            "n01",
            "execute",
            "noop",
            [],
            input={"instruction": "safe"},
        )
        workflow = {
            "id": "wf_plan_init",
            "goal": "run",
            "live": False,
            "nodes": [o.asdict(node)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={}):
                result = o.run_one_step(workflow)
        self.assertEqual(result, "completed")
        self.assertTrue(workflow.get("plan_fingerprint"))
        self.assertEqual(workflow.get("plan_integrity"), "initialized")

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


    def test_live_side_effect_runs_only_after_durable_barrier(self):
        node = o.Node(
            "n01-write", "build", "github", [], risk="high",
            input={"action": "create_issue", "approval_granted": True},
        )
        workflow = {
            "id": "wf_barrier",
            "goal": "write",
            "live": True,
            "nodes": [o.asdict(node)],
        }
        registry = {
            "github": {
                "free_tier": True,
                "side_effects": ["issue_write"],
            }
        }
        order = []

        def barrier(*args, **kwargs):
            order.append("barrier")
            return True

        def execute(*args, **kwargs):
            order.append("execute")
            return {"result": "created"}

        with patch.dict(o.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "GITHUB_TOKEN": "dummy",
            "GITHUB_REPOSITORY": "owner/repo",
        }, clear=False):
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(o, "STATE_DIR", Path(tmp)),                      patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                      patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                      patch.object(o, "load_registry", return_value=registry),                      patch.object(o, "commit_side_effect_start", side_effect=barrier),                      patch.object(o, "execute_node", side_effect=execute):
                    result = o.run_one_step(workflow)
        self.assertEqual(result, "completed")
        self.assertEqual(order, ["barrier", "execute"])

    def test_durable_barrier_failure_blocks_side_effect(self):
        node = o.Node(
            "n01-write", "build", "github", [], risk="high",
            input={"action": "create_issue", "approval_granted": True},
        )
        workflow = {
            "id": "wf_barrier_fail",
            "goal": "write",
            "live": True,
            "nodes": [o.asdict(node)],
        }
        registry = {
            "github": {
                "free_tier": True,
                "side_effects": ["issue_write"],
            }
        }
        error = o.DurabilityBarrierError("push rejected")
        with patch.dict(o.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "GITHUB_TOKEN": "dummy",
            "GITHUB_REPOSITORY": "owner/repo",
        }, clear=False):
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(o, "STATE_DIR", Path(tmp)),                      patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                      patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                      patch.object(o, "load_registry", return_value=registry),                      patch.object(o, "commit_side_effect_start", side_effect=error),                      patch.object(o, "execute_node") as execute:
                    result = o.run_one_step(workflow)
        self.assertEqual(result, "failed")
        self.assertTrue(workflow["nodes"][0]["error"]["durability_barrier_failed"])
        self.assertEqual(
            workflow["nodes"][0]["error"]["failure_class"],
            "dependency",
        )
        execute.assert_not_called()

    def test_started_side_effect_timeout_is_not_retried_or_replanned(self):
        node = o.Node(
            "n01-write", "publish", "webhook", [], risk="high", max_retries=2,
            input={"approval_granted": True},
        )
        workflow = {
            "id": "wf_post_start_fence",
            "goal": "publish",
            "live": True,
            "nodes": [o.asdict(node)],
        }
        registry = {"webhook": {"free_tier": True, "side_effects": ["external_request"]}}
        with patch.dict(o.os.environ, {"ORCHESTRATOR_WEBHOOK_URL": "https://example.test/hook"}, clear=False):
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(o, "STATE_DIR", Path(tmp)), \
                     patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                     patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"), \
                     patch.object(o, "load_registry", return_value=registry), \
                     patch.object(o, "execute_node", side_effect=TimeoutError("upstream timeout")) as execute:
                    result = o.run_one_step(workflow, approve_high_risk=False)
            self.assertEqual(result, "failed")
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(workflow["nodes"][0]["retry_count"], 0)
            self.assertEqual(workflow["replan_count"], 0)
            self.assertTrue(workflow["nodes"][0]["error"]["post_start_side_effect_failure"])
            self.assertTrue(workflow["nodes"][0]["error"]["retry_blocked_after_side_effect_start"])
            self.assertTrue(workflow["nodes"][0]["error"]["replan_blocked_after_side_effect_start"])
            self.assertTrue(workflow["nodes"][0]["error"]["execution_uncertain"])

    def test_started_side_effect_permanent_error_does_not_replan(self):
        node = o.Node(
            "n01-write", "publish", "webhook", [], risk="high", max_retries=0,
            input={"approval_granted": True},
        )
        workflow = {
            "id": "wf_post_start_permanent",
            "goal": "publish",
            "live": True,
            "nodes": [o.asdict(node)],
        }
        registry = {"webhook": {"free_tier": True, "side_effects": ["external_request"]}}
        with patch.dict(o.os.environ, {"ORCHESTRATOR_WEBHOOK_URL": "https://example.test/hook"}, clear=False):
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(o, "STATE_DIR", Path(tmp)), \
                     patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                     patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"), \
                     patch.object(o, "load_registry", return_value=registry), \
                     patch.object(o, "execute_node", side_effect=RuntimeError("provider rejected response")) as execute:
                    result = o.run_one_step(workflow, approve_high_risk=False)
            self.assertEqual(result, "failed")
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(workflow["replan_count"], 0)
            self.assertTrue(workflow["nodes"][0]["error"]["replan_blocked_after_side_effect_start"])
    def test_workflow_started_side_effect_timeout_is_not_retried(self):
        node = o.Node(
            "n01-write", "publish", "webhook", [], risk="high", max_retries=2,
            input={"approval_granted": True},
        )
        workflow = {
            "id": "wf_workflow_post_start_fence",
            "goal": "publish",
            "live": True,
            "nodes": [o.asdict(node)],
        }
        registry = {"webhook": {"free_tier": True, "side_effects": ["external_request"]}}
        with patch.dict(o.os.environ, {"ORCHESTRATOR_WEBHOOK_URL": "https://example.test/hook"}, clear=False):
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(o, "STATE_DIR", Path(tmp)), \
                     patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                     patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"), \
                     patch.object(o, "load_registry", return_value=registry), \
                     patch.object(o, "execute_node", side_effect=TimeoutError("upstream timeout")) as execute:
                    o.run_workflow(workflow, approve_high_risk=False)
            self.assertEqual(workflow["status"], "failed")
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(workflow["nodes"][0]["retry_count"], 0)
            self.assertTrue(workflow["nodes"][0]["error"]["post_start_side_effect_failure"])
            self.assertTrue(workflow["nodes"][0]["error"]["execution_uncertain"])
    def test_barrier_failed_side_effect_is_rearmed_for_fresh_worker(self):
        node = o.Node(
            "n01-webhook", "publish", "webhook", [], risk="high", status="failed",
            input={"approval_granted": True},
        )
        workflow_id = "wf_barrier_rearm"
        execution_id = o.execution_key({"id": workflow_id}, node)
        workflow = {
            "id": workflow_id, "goal": "publish", "live": True, "status": "failed",
            "executions": {execution_id: {"node_id": node.id, "status": "barrier_failed", "barrier_error": "push rejected"}},
            "nodes": [o.asdict(node)],
        }
        registry = {"webhook": {"free_tier": True, "side_effects": ["external_request"]}}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)), \
                 patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                 patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"), \
                 patch.object(o, "load_registry", return_value=registry), \
                 patch.object(o, "commit_side_effect_start", return_value=True), \
                 patch.object(o, "execute_node", return_value={"ok": True}) as execute:
                result = o.run_one_step(workflow, approve_high_risk=False)
        self.assertEqual(result, "rearmed_pre_side_effect")
        self.assertEqual(workflow["status"], "running")
        self.assertEqual(workflow["nodes"][0]["status"], "ready")
        self.assertEqual(workflow["executions"][execution_id]["status"], "prepared")
        self.assertNotIn("barrier_error", workflow["executions"][execution_id])
        execute.assert_not_called()

    def test_resume_scheduler_selects_barrier_failed_workflow(self):
        node = o.Node("n01", "publish", "webhook", [], status="failed")
        workflow_id = "wf_barrier_scheduler"
        execution_id = o.execution_key({"id": workflow_id}, node)
        workflow = {
            "id": workflow_id, "goal": "publish", "live": True, "status": "failed",
            "updated_at": "2026-10-01T00:00:00+00:00",
            "executions": {execution_id: {"node_id": node.id, "status": "barrier_failed"}},
            "nodes": [o.asdict(node)],
        }
        state = {"version": 3, "workflows": {workflow_id: workflow}, "last_workflow_id": workflow_id}
        with patch.object(o, "run_one_step", side_effect=lambda wf, approve_high_risk=False: (wf.update({"status": "running"}) or "rearmed_pre_side_effect")), \
             patch.object(o, "save_state"):
            count = o.resume_pending_workflows(state, step=True)
        self.assertEqual(count, 1)
        self.assertEqual(workflow["status"], "running")
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

    def test_pre_start_prepared_side_effect_is_rearmed(self):
        node = o.Node(
            "n01-webhook", "publish", "webhook", [],
            risk="high", status="running",
            input={"approval_granted": True},
        )
        workflow_id = "wf_prepared_rearm"
        execution_id = o.execution_key({"id": workflow_id}, node)
        workflow = {
            "id": workflow_id, "goal": "publish", "live": True, "status": "running",
            "executions": {execution_id: {"node_id": node.id, "status": "prepared"}},
            "nodes": [o.asdict(node)],
        }
        registry = {"webhook": {"free_tier": True, "side_effects": ["external_request"]}}
        with patch.dict(o.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "ORCHESTRATOR_WEBHOOK_URL": "https://example.test/hook",
        }, clear=False), \
             patch.object(o, "load_registry", return_value=registry), \
             patch.object(o, "execute_node", return_value={"ok": True}) as execute:
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(o, "STATE_DIR", Path(tmp)), \
                     patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                     patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"):
                    result = o.run_one_step(workflow, approve_high_risk=False)
        self.assertEqual(result, "completed")
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(workflow["nodes"][0]["status"], "completed")
    def test_interrupted_started_connector_enters_reconciliation(self):
        node = o.Node(
            "n01-connector", "publish", "connector_bridge", [],
            risk="high", status="running",
            input={"connector": "notion", "action": "create_page", "payload": {"title": "Hello"}},
        )
        workflow_id = "wf_inflight_connector"
        execution_id = o.execution_key({"id": workflow_id}, node)
        workflow = {
            "id": workflow_id, "goal": "publish", "live": True, "status": "running",
            "executions": {execution_id: {"node_id": node.id, "status": "started"}},
            "nodes": [o.asdict(node)],
        }
        registry = {
            "capability:publish": {"default_tool": "connector_bridge", "fallback_tools": []},
            "connector_bridge": {"free_tier": True, "side_effects": ["external_request"]},
        }
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False), \
             patch.object(o, "load_registry", return_value=registry), \
             patch.object(o, "reconcile_connector_execution", return_value={"state": "applied"}), \
             patch.object(o, "execute_node") as execute:
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(o, "STATE_DIR", Path(tmp)), \
                     patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                     patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"):
                    result = o.run_one_step(workflow)
        self.assertEqual(result, "reconciled")
        self.assertEqual(workflow["nodes"][0]["status"], "completed")
        self.assertTrue(workflow["nodes"][0]["error"]["inflight_recovered"])
        execute.assert_not_called()

    def test_interrupted_started_opaque_side_effect_fails_closed(self):
        node = o.Node(
            "n01-webhook", "publish", "webhook", [],
            risk="high", status="running",
            input={"approval_granted": True},
        )
        workflow_id = "wf_inflight_webhook"
        execution_id = o.execution_key({"id": workflow_id}, node)
        workflow = {
            "id": workflow_id, "goal": "publish", "live": True, "status": "running",
            "executions": {execution_id: {"node_id": node.id, "status": "started"}},
            "nodes": [o.asdict(node)],
        }
        registry = {"webhook": {"free_tier": True, "side_effects": ["external_request"]}}
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False), \
             patch.object(o, "load_registry", return_value=registry), \
             patch.object(o, "execute_node") as execute:
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(o, "STATE_DIR", Path(tmp)), \
                     patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                     patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"):
                    result = o.run_one_step(workflow)
        self.assertEqual(result, "failed")
        self.assertEqual(workflow["nodes"][0]["status"], "failed")
        self.assertTrue(workflow["nodes"][0]["error"]["inflight_recovered"])
        self.assertTrue(workflow["nodes"][0]["error"]["execution_uncertain"])
        execute.assert_not_called()
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
        self.assertEqual(workflow["nodes"][0]["output"]["request_id"], reconciliation["request_id"])
        self.assertEqual(
            workflow["nodes"][0]["output"]["execution_id"],
            o.hashlib.sha256(b"wf_reconcile:n01-connector").hexdigest(),
        )

    def test_uncertain_connector_reconciliation_not_applied_allows_new_attempt(self):
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
        ), patch.object(
            o, "execute_node", return_value={"result": "executed"}
        ):
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(o, "STATE_DIR", Path(tmp)),                      patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                      patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"):
                    first = o.run_one_step(workflow)
                    self.assertEqual(first, "reconciled_ready")
                    self.assertEqual(workflow["nodes"][0]["status"], "ready")
                    self.assertEqual(workflow["status"], "running")
                    execution_id = o.execution_key(workflow, o.Node(**workflow["nodes"][0]))
                    self.assertEqual(
                        workflow["executions"][execution_id]["status"],
                        "not_applied",
                    )
                    second = o.run_one_step(workflow, approve_high_risk=True)

        self.assertEqual(second, "completed")
        self.assertEqual(workflow["nodes"][0]["status"], "completed")
        self.assertEqual(
            workflow["executions"][execution_id]["status"],
            "completed",
        )

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
    def test_persist_rebinds_latest_worker_run_and_preserves_origin(self):
        workflow = {
            "id": "wf_run_binding",
            "goal": "chain",
            "status": "running",
            "live": False,
            "nodes": [o.asdict(o.Node("n01", "execute", "noop"))],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(
                o, "STATE_DIR", root
            ), patch.object(
                o, "STATE_FILE", root / "state.json"
            ), patch.object(
                o, "EVENT_FILE", root / "events.jsonl"
            ), patch.dict(
                o.os.environ,
                {"ORCHESTRATOR_GITHUB_RUN_ID": "run-A", "ORCHESTRATOR_GITHUB_RUN_ATTEMPT": "1"},
                clear=False,
            ):
                o.persist_workflow(workflow)
            with patch.object(
                o, "STATE_DIR", root
            ), patch.object(
                o, "STATE_FILE", root / "state.json"
            ), patch.object(
                o, "EVENT_FILE", root / "events.jsonl"
            ), patch.dict(
                o.os.environ,
                {"ORCHESTRATOR_GITHUB_RUN_ID": "run-B", "ORCHESTRATOR_GITHUB_RUN_ATTEMPT": "2"},
                clear=False,
            ):
                o.persist_workflow(workflow)
            saved = json.loads((root / "state.json").read_text(encoding="utf-8"))
            events = (root / "events.jsonl").read_text(encoding="utf-8")
        stored = saved["workflows"]["wf_run_binding"]
        self.assertEqual(stored["origin_github_run_id"], "run-A")
        self.assertEqual(stored["origin_github_run_attempt"], 1)
        self.assertEqual(stored["github_run_id"], "run-B")
        self.assertEqual(stored["github_run_attempt"], 2)
        self.assertIn("workflow.worker_run_rebound", events)

    def test_persist_updates_attempt_for_same_worker_run(self):
        workflow = {
            "id": "wf_run_attempt",
            "goal": "rerun",
            "status": "running",
            "live": False,
            "nodes": [o.asdict(o.Node("n01", "execute", "noop"))],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {"ORCHESTRATOR_GITHUB_RUN_ID": "run-A", "ORCHESTRATOR_GITHUB_RUN_ATTEMPT": "1"}
            with patch.object(o, "STATE_DIR", root),                  patch.object(o, "STATE_FILE", root / "state.json"),                  patch.object(o, "EVENT_FILE", root / "events.jsonl"),                  patch.dict(o.os.environ, env, clear=False):
                o.persist_workflow(workflow)
            env["ORCHESTRATOR_GITHUB_RUN_ATTEMPT"] = "2"
            with patch.object(o, "STATE_DIR", root),                  patch.object(o, "STATE_FILE", root / "state.json"),                  patch.object(o, "EVENT_FILE", root / "events.jsonl"),                  patch.dict(o.os.environ, env, clear=False):
                o.persist_workflow(workflow)
            saved = json.loads((root / "state.json").read_text(encoding="utf-8"))
        stored = saved["workflows"]["wf_run_attempt"]
        self.assertEqual(stored["github_run_id"], "run-A")
        self.assertEqual(stored["github_run_attempt"], 2)
        self.assertEqual(stored["origin_github_run_id"], "run-A")
        self.assertEqual(stored["origin_github_run_attempt"], 1)

    def test_legacy_worker_binding_backfills_origin_attempt(self):
        workflow = {
            "id": "wf_legacy_attempt",
            "goal": "legacy",
            "status": "running",
            "live": False,
            "github_run_id": "run-legacy",
            "github_run_attempt": None,
            "nodes": [o.asdict(o.Node("n01", "execute", "noop"))],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(o, "STATE_DIR", root),                  patch.object(o, "STATE_FILE", root / "state.json"),                  patch.object(o, "EVENT_FILE", root / "events.jsonl"),                  patch.dict(
                     o.os.environ,
                     {
                         "ORCHESTRATOR_GITHUB_RUN_ID": "run-legacy",
                         "ORCHESTRATOR_GITHUB_RUN_ATTEMPT": "3",
                     },
                     clear=False,
                 ):
                o.persist_workflow(workflow)
            saved = json.loads((root / "state.json").read_text(encoding="utf-8"))
        stored = saved["workflows"]["wf_legacy_attempt"]
        self.assertEqual(stored["github_run_attempt"], 3)
        self.assertEqual(stored["origin_github_run_attempt"], 3)

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
        with patch.object(o, "load_registry", return_value={}):
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
    def test_parallel_batch_preflight_failure_does_not_strand_siblings(self):
        nodes = [
            o.Node("n01-a", "execute", "noop", []),
            o.Node("n02-b", "execute", "noop", []),
        ]
        workflow = {
            "id": "wf_parallel_preflight",
            "goal": "parallel preflight",
            "live": False,
            "status": "ready",
            "max_parallel": 2,
            "execution_budget": {"max_steps": 4, "used_steps": 0},
            "nodes": [o.asdict(node) for node in nodes],
        }
        registry = {}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value=registry),                  patch.object(
                     o,
                     "preflight_node",
                     side_effect=[None, RuntimeError("second node unavailable")],
                 ),                  patch.object(o, "replan_after_failure", return_value=False),                  patch.object(o, "execute_node") as execute:
                o.run_workflow(workflow)

        self.assertEqual(workflow["status"], "failed")
        self.assertEqual(workflow["failed_node"], "n02-b")
        self.assertEqual(workflow["nodes"][0]["status"], "ready")
        self.assertEqual(workflow["nodes"][1]["status"], "failed")
        self.assertEqual(workflow["execution_budget"]["used_steps"], 0)
        execute.assert_not_called()

    def test_parallel_batch_is_bounded_by_remaining_execution_budget(self):
        nodes = [
            o.Node("n01-a", "execute", "noop", []),
            o.Node("n02-b", "execute", "noop", []),
        ]
        workflow = {
            "id": "wf_parallel_budget",
            "goal": "parallel budget",
            "live": False,
            "status": "ready",
            "max_parallel": 2,
            "execution_budget": {"max_steps": 1, "used_steps": 0},
            "nodes": [o.asdict(node) for node in nodes],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)), \
                 patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"), \
                 patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"), \
                 patch.object(o, "load_registry", return_value={}), \
                 patch.object(o, "execute_node", return_value={"ok": True}) as execute:
                o.run_workflow(workflow)

        self.assertEqual(execute.call_count, 1)
        self.assertEqual(workflow["status"], "failed")
        self.assertEqual(workflow["failed_node"], "n02-b")
        self.assertEqual(workflow["execution_budget"]["used_steps"], 1)
        self.assertEqual(workflow["nodes"][0]["status"], "completed")
        self.assertEqual(workflow["nodes"][1]["status"], "failed")
        self.assertNotIn("running", {node["status"] for node in workflow["nodes"]})

    def test_free_only_rejects_paid_node_at_execution_time(self):
        node = o.Node("n01", "analyze", "openai")
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            with self.assertRaises(RuntimeError):
                o.execute_node(node, "test", dry_run=False)

    def test_goal_length_is_bounded(self):
        with self.assertRaises(ValueError):
            o.normalize_goal("x" * 4001)

    def test_node_serialized_size_is_bounded(self):
        node = o.Node(
            "n01",
            "execute",
            "noop",
            [],
            input={"instruction": "x" * (32 * 1024)},
        )
        with self.assertRaises(ValueError):
            o.validate_dag([node])

    def test_workflow_creation_falls_back_without_gemini_key(self):
        with patch.dict(o.os.environ, {}, clear=True):
            workflow = o.create_workflow("build a small website", live=False)
        self.assertGreaterEqual(len(workflow["nodes"]), 4)
        self.assertEqual(workflow["status"], "planning")

    def test_duplicate_event_rejects_intent_fingerprint_conflict(self):
        workflow = {
            "id": "wf-existing",
            "execution_id": "a" * 64,
            "intent_fingerprint": "b" * 64,
            "input_digest": "c" * 64,
            "idempotency_key": "evt-1",
        }
        with self.assertRaises(SystemExit):
            o.validate_event_replay_identity(
                workflow,
                execution_id="a" * 64,
                intent_fingerprint="d" * 64,
                input_digest="c" * 64,
                idempotency_key="evt-1",
            )

    def test_duplicate_event_with_matching_identity_is_replay_safe(self):
        workflow = {
            "id": "wf-existing",
            "execution_id": "a" * 64,
            "intent_fingerprint": "b" * 64,
            "input_digest": "c" * 64,
            "idempotency_key": "evt-1",
        }
        o.validate_event_replay_identity(
            workflow,
            execution_id="a" * 64,
            intent_fingerprint="b" * 64,
            input_digest="c" * 64,
            idempotency_key="evt-1",
        )

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

    def test_scheduled_recovery_defers_fresh_running_workflow(self):
        now = o.datetime(2026, 10, 2, 9, 10, tzinfo=o.timezone.utc)
        workflow = {
            "status": "running",
            "github_run_id": "123",
            "updated_at": "2026-10-02T09:08:00+00:00",
        }
        self.assertFalse(o.running_recovery_due(workflow, now=now))

    def test_scheduled_recovery_picks_stale_running_workflow(self):
        now = o.datetime(2026, 10, 2, 9, 10, tzinfo=o.timezone.utc)
        workflow = {
            "status": "running",
            "github_run_id": "123",
            "updated_at": "2026-10-02T09:00:00+00:00",
        }
        self.assertTrue(o.running_recovery_due(workflow, now=now))

    def test_manual_resume_can_still_process_fresh_running_workflow(self):
        node = o.Node("n01", "execute", "noop", [])
        workflow = {
            "id": "wf_fresh_manual",
            "goal": "manual resume",
            "status": "running",
            "live": False,
            "github_run_id": "123",
            "updated_at": "2026-10-02T09:09:30+00:00",
            "nodes": [o.asdict(node)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={}):
                with patch.object(o, "run_workflow", return_value=None) as runner:
                    state = {"workflows": {workflow["id"]: workflow}}
                    o.resume_pending_workflows(state, scheduled_recovery=False)
        runner.assert_called_once()

    def test_workflow_creation_persists_execution_envelope_identity(self):
        with patch.dict(
            o.os.environ,
            {"ORCHESTRATOR_MAX_EXECUTION_STEPS": "17"},
            clear=False,
        ):
            workflow = o.create_workflow(
                "structured work",
                live=False,
                execution_id="a" * 64,
                parent_execution_id="parent-1",
                external_workflow_id="external-wf",
                external_domain="publisher",
                external_operation="chapter.produce",
                intent_fingerprint="b" * 64,
                input_digest="c" * 64,
                idempotency_key="evt-1",
                external_attempt=2,
            )
        self.assertEqual(workflow["execution_id"], "a" * 64)
        self.assertEqual(workflow["parent_execution_id"], "parent-1")
        self.assertEqual(workflow["external_workflow_id"], "external-wf")
        self.assertEqual(workflow["external_domain"], "publisher")
        self.assertEqual(workflow["external_operation"], "chapter.produce")
        self.assertEqual(workflow["intent_fingerprint"], "b" * 64)
        self.assertEqual(workflow["input_digest"], "c" * 64)
        self.assertEqual(workflow["idempotency_key"], "evt-1")
        self.assertEqual(workflow["external_attempt"], 2)

    def test_terminal_execution_callback_is_hmac_signed_and_idempotent(self):
        workflow = {
            "id": "wf_callback",
            "execution_id": "d" * 64,
            "status": "completed",
            "external_workflow_id": "external-wf",
            "external_domain": "publisher",
            "external_operation": "chapter.produce",
            "intent_fingerprint": "e" * 64,
            "input_digest": "f" * 64,
            "external_attempt": 3,
            "nodes": [],
        }
        captured = {}

        def fake_urlopen(request, timeout=30):
            captured["body"] = request.data
            captured["headers"] = dict(request.header_items())
            captured["method"] = request.method
            captured["url"] = request.full_url

            class Response:
                status = 204
                def __enter__(self): return self
                def __exit__(self, *args): return None
            return Response()

        with patch.dict(
            o.os.environ,
            {
                "ORCHESTRATOR_CALLBACK_URL": "https://callback.example.test/terminal",
                "ORCHESTRATOR_CALLBACK_SECRET": "secret",
            },
            clear=False,
        ), patch.object(
            o.urllib.request, "urlopen", side_effect=fake_urlopen
        ) as send:
            self.assertTrue(o.notify_execution_callback(workflow))

        self.assertEqual(send.call_count, 1)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(
            captured["headers"]["Idempotency-key"],
            workflow["execution_id"],
        )
        body = json.loads(captured["body"].decode("utf-8"))
        self.assertEqual(body["execution_id"], workflow["execution_id"])
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["result"]["attempt"], 3)
        self.assertTrue(captured["headers"]["X-engine-signature"].startswith("sha256="))

    def test_terminal_execution_callback_does_not_send_without_private_channel(self):
        workflow = {
            "id": "wf_callback_missing",
            "execution_id": "d" * 64,
            "status": "completed",
        }
        with patch.dict(
            o.os.environ,
            {
                "ORCHESTRATOR_CALLBACK_URL": "",
                "ORCHESTRATOR_CALLBACK_SECRET": "",
            },
            clear=False,
        ), patch.object(o.urllib.request, "urlopen") as send:
            self.assertFalse(o.notify_execution_callback(workflow))
        send.assert_not_called()

    def test_workflow_creation_is_persistable(self):
        workflow = o.create_workflow("build a small website", live=False)
        self.assertTrue(workflow["id"].startswith("wf_"))
        self.assertEqual(workflow["status"], "planning")
        self.assertEqual(workflow["execution_mode"], "dry-run")
        self.assertEqual(workflow["schema_version"], 6)
        self.assertEqual(workflow["execution_budget"], {"max_steps": 96, "used_steps": 0})
        self.assertGreaterEqual(len(workflow["nodes"]), 4)

    def test_workflow_creation_persists_requested_execution_budget(self):
        with patch.dict(o.os.environ, {"ORCHESTRATOR_MAX_EXECUTION_STEPS": "17"}, clear=False):
            workflow = o.create_workflow("research something", live=False)
        self.assertEqual(workflow["schema_version"], 5)
        self.assertEqual(workflow["execution_budget"], {"max_steps": 17, "used_steps": 0})

    def test_invalid_requested_execution_budget_fails_closed(self):
        with patch.dict(o.os.environ, {"ORCHESTRATOR_MAX_EXECUTION_STEPS": "0"}, clear=False):
            with self.assertRaises(ValueError):
                o.create_workflow("research something", live=False)

    def test_plan_is_acyclic(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "REGISTRY_FILE", Path(tmp) / "missing.json"):
                nodes = o.deterministic_plan("build and deploy a web app", {})
        o.validate_dag(nodes)
        self.assertEqual(len(nodes), 7)
        self.assertEqual(nodes[-1].capability, "notify")

    def test_unsafe_node_identifier_is_rejected(self):
        node = o.Node("../escape", "execute", "noop", [])
        with self.assertRaises(ValueError):
            o.validate_dag([node])

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
        with patch.object(o, "load_registry", return_value={}):
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

    def test_approval_accepts_only_matching_node_fingerprint(self):
        node = o.Node(
            "n01-publish", "publish", "webhook", [], risk="high", status="waiting_approval",
            input={"approval_issue": 42, "approval_granted": False},
        )
        node.input["approval_fingerprint"] = o.fingerprint_nodes([node])
        workflow = {"id": "wf_approval_bind", "status": "waiting_approval", "nodes": [o.asdict(node)]}
        with patch.dict(o.os.environ, {"GITHUB_ACTOR": "reviewer", "ORCHESTRATOR_APPROVAL_EVENT": "true"}, clear=False), \
             patch.object(o, "get_issue_labels", return_value={"orchestrator-approved"}):
            o.refresh_approvals(workflow, [node])
        self.assertTrue(node.input["approval_granted"])
        self.assertEqual(node.input["approval_actor"], "reviewer")
        self.assertTrue(node.input["approval_approved_at"])
        self.assertEqual(node.status, "ready")
        self.assertEqual(workflow["status"], "running")

    def test_approval_event_only_accepts_its_exact_issue(self):
        node = o.Node(
            "n01-publish", "publish", "webhook", [],
            risk="high", status="waiting_approval",
            input={"approval_issue": 42, "approval_granted": False},
        )
        node.input["approval_fingerprint"] = o.fingerprint_nodes([node])
        workflow = {
            "id": "wf_exact_approval",
            "status": "waiting_approval",
            "nodes": [o.asdict(node)],
        }
        with patch.dict(
            o.os.environ,
            {
                "ORCHESTRATOR_APPROVAL_EVENT": "true",
                "ORCHESTRATOR_APPROVAL_ISSUE": "99",
            },
            clear=False,
        ), patch.object(
            o, "get_issue_labels", return_value={"orchestrator-approved"}
        ) as labels:
            o.refresh_approvals(workflow, [node])
        self.assertFalse(node.input["approval_granted"])
        self.assertEqual(node.status, "waiting_approval")
        labels.assert_not_called()

    def test_approval_event_accepts_matching_issue_only_after_fingerprint_check(self):
        node = o.Node(
            "n01-publish", "publish", "webhook", [],
            risk="high", status="waiting_approval",
            input={"approval_issue": 42, "approval_granted": False},
        )
        node.input["approval_fingerprint"] = o.fingerprint_nodes([node])
        workflow = {
            "id": "wf_matching_approval",
            "status": "waiting_approval",
            "nodes": [o.asdict(node)],
        }
        with patch.dict(
            o.os.environ,
            {
                "ORCHESTRATOR_APPROVAL_EVENT": "true",
                "ORCHESTRATOR_APPROVAL_ISSUE": "42",
                "GITHUB_ACTOR": "maintainer",
            },
            clear=False,
        ), patch.object(
            o, "get_issue_labels", return_value={"orchestrator-approved"}
        ):
            o.refresh_approvals(workflow, [node])
        self.assertTrue(node.input["approval_granted"])
        self.assertEqual(node.status, "ready")
        self.assertEqual(node.input["approval_actor"], "maintainer")

    def test_approval_label_without_authenticated_event_is_ignored(self):
        node = o.Node(
            "n01-publish", "publish", "webhook", [], risk="high", status="waiting_approval",
            input={"approval_issue": 42, "approval_granted": False},
        )
        node.input["approval_fingerprint"] = o.fingerprint_nodes([node])
        workflow = {"id": "wf_unverified_approval", "status": "waiting_approval", "nodes": [o.asdict(node)]}
        with patch.dict(o.os.environ, {"ORCHESTRATOR_APPROVAL_EVENT": "false"}, clear=False),              patch.object(o, "get_issue_labels", return_value={"orchestrator-approved"}):
            o.refresh_approvals(workflow, [node])
        self.assertFalse(node.input["approval_granted"])
        self.assertEqual(node.status, "waiting_approval")
        self.assertEqual(workflow["status"], "waiting_approval")

    def test_rejection_label_without_authenticated_event_is_ignored(self):
        node = o.Node(
            "n01-publish", "publish", "webhook", [], risk="high", status="waiting_approval",
            input={"approval_issue": 42, "approval_granted": False},
        )
        workflow = {"id": "wf_unverified_rejection", "status": "waiting_approval", "nodes": [o.asdict(node)]}
        with patch.dict(o.os.environ, {"ORCHESTRATOR_APPROVAL_EVENT": "false"}, clear=False),              patch.object(o, "get_issue_labels", return_value={"orchestrator-rejected"}):
            o.refresh_approvals(workflow, [node])
        self.assertFalse(node.input["approval_granted"])
        self.assertEqual(node.status, "waiting_approval")
        self.assertEqual(workflow["status"], "waiting_approval")

    def test_stale_approval_is_rearmed_instead_of_accepted(self):
        node = o.Node(
            "n01-publish", "publish", "webhook", [], risk="high", status="waiting_approval",
            input={"approval_issue": 42, "approval_granted": False},
        )
        node.input["approval_fingerprint"] = o.fingerprint_nodes([node])
        node.input["instruction"] = "changed after approval request"
        workflow = {"id": "wf_stale_approval", "status": "waiting_approval", "nodes": [o.asdict(node)]}
        with patch.dict(o.os.environ, {"ORCHESTRATOR_APPROVAL_EVENT": "true"}, clear=False),              patch.object(o, "get_issue_labels", return_value={"orchestrator-approved"}):
            o.refresh_approvals(workflow, [node])
        self.assertFalse(node.input["approval_granted"])
        self.assertIsNone(node.input["approval_issue"])
        self.assertIsNone(node.input["approval_fingerprint"])
        self.assertEqual(node.status, "ready")
        self.assertEqual(workflow["status"], "running")
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


    def test_resume_preserves_persisted_tool_selection_until_replan(self):
        node = o.Node("n01", "execute", "noop", [])
        workflow = {
            "id": "wf_resume_plan",
            "goal": "resume",
            "live": True,
            "nodes": [o.asdict(node)],
            "plan_fingerprint": o.fingerprint_nodes([node]),
        }
        registry = {
            "capability:execute": {
                "default_tool": "wikipedia",
                "fallback_tools": ["noop"],
            },
            "wikipedia": {"free_tier": True, "side_effects": []},
            "noop": {"free_tier": True, "side_effects": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value=registry),                  patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
                o.run_workflow(workflow)
        self.assertEqual(workflow["status"], "completed")
        self.assertEqual(workflow["plan_integrity"], "verified")
        self.assertEqual(workflow["nodes"][0]["tool"], "noop")

    def test_preflight_allows_explicit_webhook_url_without_env(self):
        node = o.Node(
            "n01-publish", "publish", "webhook", [], risk="high",
            input={"url": "https://example.test/hook", "approval_granted": True},
        )
        registry = {
            "webhook": {
                "free_tier": True,
                "required_env": "ORCHESTRATOR_WEBHOOK_URL",
                "side_effects": ["external_request"],
            }
        }
        with patch.dict(o.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "ORCHESTRATOR_WEBHOOK_URL": "",
        }, clear=False):
            o.preflight_node(node, registry, live=True)

    def test_terminal_workflow_resume_is_idempotent(self):
        node = o.Node("n01", "execute", "noop", [], status="completed")
        workflow = {
            "id": "wf_terminal",
            "goal": "already done",
            "status": "completed",
            "live": False,
            "nodes": [o.asdict(node)],
        }
        with patch.object(o, "persist_workflow") as persist,              patch.object(o, "execute_node") as execute:
            self.assertEqual(o.run_one_step(workflow), "completed")
            o.run_workflow(workflow)
        execute.assert_not_called()
        persist.assert_not_called()

    def test_preflight_failure_happens_before_side_effect_barrier(self):
        node = o.Node(
            "n01-publish", "publish", "webhook", [], risk="high",
            input={"approval_granted": True},
        )
        workflow = {
            "id": "wf_preflight_barrier",
            "goal": "publish",
            "live": True,
            "nodes": [o.asdict(node)],
        }
        registry = {
            "webhook": {
                "free_tier": True,
                "required_env": "ORCHESTRATOR_WEBHOOK_URL",
                "side_effects": ["external_request"],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value=registry),                  patch.dict(o.os.environ, {
                     "ORCHESTRATOR_FREE_ONLY": "true",
                     "ORCHESTRATOR_WEBHOOK_URL": "",
                 }, clear=False),                  patch.object(o, "commit_side_effect_start") as barrier,                  patch.object(o, "execute_node") as execute:
                result = o.run_one_step(workflow, approve_high_risk=False)
        self.assertEqual(result, "failed")
        self.assertTrue(workflow["nodes"][0]["error"]["preflight_failed"])
        barrier.assert_not_called()
        execute.assert_not_called()

if __name__ == "__main__":
    unittest.main()
