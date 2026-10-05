import os
import json
import tempfile
import sys
import threading
import unittest
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from state_schema import CURRENT_STATE_VERSION, CURRENT_WORKFLOW_SCHEMA_VERSION

import importlib.util

_ORCHESTRATOR_DIR = Path(__file__).resolve().parent
if str(_ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR_DIR))

_CONTROL_PLANE_SPEC = importlib.util.spec_from_file_location(
    "_control_plane_orchestrator",
    _ORCHESTRATOR_DIR / "orchestrator.py",
)
if _CONTROL_PLANE_SPEC is None or _CONTROL_PLANE_SPEC.loader is None:
    raise RuntimeError("cannot load control-plane module for tests")
o = importlib.util.module_from_spec(_CONTROL_PLANE_SPEC)
sys.modules[_CONTROL_PLANE_SPEC.name] = o
_CONTROL_PLANE_SPEC.loader.exec_module(o)


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self._actions_env = patch.dict(
            o.os.environ,
            {"GITHUB_ACTIONS": "false"},
            clear=False,
        )
        self._actions_env.start()
        self.addCleanup(self._actions_env.stop)

    def test_private_connector_payload_is_fetched_just_in_time_and_scrubbed(self):
        node = o.Node(
            id="n01-private",
            capability="execute",
            tool="connector_bridge",
            risk="high",
            input={
                "workflow_id": "wf-private",
                "connector": "notion",
                "action": "create_page",
                "private_input_ref": "b" * 64,
                "private_input_digest": "d" * 64,
                "private_input_intent_fingerprint": "f" * 64,
                "private_input_execution_id": "e" * 64,
            },
        )
        seen = {}

        def fake_execute(runtime_node, goal, dry_run):
            seen["payload"] = dict(runtime_node.input["payload"])
            return {"ok": True}

        with patch.object(
            o, "fetch_private_input", return_value={"title": "Secret"}
        ) as fetch, patch.object(
            o, "execute_connector_bridge", side_effect=fake_execute
        ):
            result = o.execute_node(node, "private goal", dry_run=False)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(seen["payload"], {"title": "Secret"})
        self.assertNotIn("payload", node.input)
        fetch.assert_called_once()

    def test_private_structured_live_plan_avoids_llm(self):
        registry = {
            "connector_bridge": {
                "required_env": "ORCHESTRATOR_CONNECTOR_BRIDGE_URL",
                "secret_env": "ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET",
                "free_tier": True,
                "side_effects": ["external_request"],
            },
        }
        import sys
        from types import SimpleNamespace

        planner_called = {"value": False}

        def forbidden_plan(*args, **kwargs):
            planner_called["value"] = True
            raise AssertionError("private structured live input must not reach the LLM planner")

        fake_planner = SimpleNamespace(plan_goal=forbidden_plan)
        with patch.dict(
            o.os.environ,
            {
                "ORCHESTRATOR_LLM_PLANNER": "true",
                "GEMINI_API_KEY": "test-key",
            },
            clear=False,
        ), patch.object(o, "load_registry", return_value=registry), patch.dict(
            sys.modules, {"llm_planner": fake_planner}
        ):
            workflow = o.create_workflow(
                "Execute private notion.create_page",
                live=True,
                execution_id="e" * 64,
                external_domain="notion",
                external_operation="create_page",
                intent_fingerprint="f" * 64,
                input_digest="d" * 64,
                private_input_ref="b" * 64,
            )
        self.assertEqual(len(workflow["nodes"]), 1)
        node = workflow["nodes"][0]
        self.assertEqual(node["tool"], "connector_bridge")
        self.assertEqual(node["input"]["private_input_ref"], "b" * 64)
        self.assertNotIn("payload", node["input"])
        self.assertFalse(planner_called["value"])

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

    def test_free_only_rejects_expired_gemini_free_model(self):
        registry = {
            "gemini": {
                "free_tier": True,
                "default_model": "gemini-3.8-flash",
                "free_models": ["gemini-3.8-flash"],
                "free_until": {
                    "gemini-3.8-flash": "2020-01-01T00:00:00Z",
                },
            }
        }
        with patch.dict(
            os.environ,
            {"ORCHESTRATOR_FREE_ONLY": "true"},
            clear=False,
        ):
            self.assertFalse(
                o.gemini_model_allowed(
                    registry,
                    model="gemini-3.8-flash",
                )
            )

    def test_free_only_rejects_unlisted_gemini_model(self):
        registry = {
            "gemini": {
                "free_tier": True,
                "default_model": "gemini-3.8-flash",
                "free_models": ["gemini-3.8-flash"],
                "required_env": "GEMINI_API_KEY",
            }
        }
        with patch.dict(o.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "GEMINI_MODEL": "gemini-3.8-pro",
            "GEMINI_API_KEY": "key",
        }, clear=False):
            self.assertFalse(o.tool_available("gemini", registry))
            self.assertFalse(o.tool_available("gemini", registry, model="gemini-3.8-pro"))
            self.assertTrue(o.tool_available("gemini", registry, model="gemini-3.8-flash"))

    def test_gemini_executor_fails_closed_on_unlisted_model(self):
        node = o.Node("n01", "analyze", "gemini", [])
        registry = {
            "gemini": {
                "free_tier": True,
                "default_model": "gemini-3.8-flash",
                "free_models": ["gemini-3.8-flash"],
                "required_env": "GEMINI_API_KEY",
            }
        }
        with patch.dict(o.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "GEMINI_MODEL": "gemini-3.8-pro",
            "GEMINI_API_KEY": "key",
        }, clear=False), patch.object(o, "load_registry", return_value=registry), patch.object(o, "http_json") as http:
            with self.assertRaisesRegex(RuntimeError, "not allowed by the free-only model registry"):
                o.execute_gemini(node, "analyze")
        http.assert_not_called()

    def test_paid_gemini_planner_override_falls_back_without_api_call(self):
        registry = {
            "capability:research": {
                "default_tool": "research_bundle",
                "fallback_tools": ["wikipedia"],
            },
            "research_bundle": {"free_tier": True, "side_effects": []},
            "wikipedia": {"free_tier": True, "side_effects": []},
            "gemini": {
                "free_tier": True,
                "default_model": "gemini-3.8-flash",
                "free_models": ["gemini-3.8-flash"],
                "required_env": "GEMINI_API_KEY",
            },
        }
        import sys
        from types import SimpleNamespace

        planner_called = {"value": False}

        def forbidden_plan(*args, **kwargs):
            planner_called["value"] = True
            raise AssertionError("paid Gemini planner override must be blocked")

        fake_planner = SimpleNamespace(plan_goal=forbidden_plan)
        with patch.dict(o.os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "ORCHESTRATOR_LLM_PLANNER": "true",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_PLANNER_MODEL": "gemini-3.8-pro",
        }, clear=False), patch.object(o, "load_registry", return_value=registry), patch.dict(
            sys.modules, {"llm_planner": fake_planner}
        ):
            workflow = o.create_workflow("research AI safety", live=False)

        self.assertFalse(planner_called["value"])
        self.assertEqual(workflow["nodes"][0]["tool"], "research_bundle")

    def test_artifact_verifier_rejects_private_dns_target(self):
        with patch.object(
            o.socket,
            "getaddrinfo",
            return_value=[
                (
                    o.socket.AF_INET,
                    o.socket.SOCK_STREAM,
                    6,
                    "",
                    ("127.0.0.1", 443),
                )
            ],
        ):
            with self.assertRaisesRegex(
                RuntimeError, "non-public IP"
            ):
                o.safe_public_https_json("https://example.test/health")

    def test_artifact_verifier_rejects_redirects(self):
        class FakeSocket:
            def sendall(self, data):
                self.sent = data

            def makefile(self, *args, **kwargs):
                return io.BytesIO()

            def close(self):
                pass

        class FakeTLS(FakeSocket):
            pass

        class FakeContext:
            def wrap_socket(self, sock, server_hostname):
                return FakeTLS()

        class FakeResponse:
            status = 302

            def begin(self):
                pass

            def close(self):
                pass

        with patch.object(
            o.socket,
            "getaddrinfo",
            return_value=[
                (
                    o.socket.AF_INET,
                    o.socket.SOCK_STREAM,
                    6,
                    "",
                    ("93.184.216.34", 443),
                )
            ],
        ), patch.object(o.socket, "create_connection", return_value=FakeSocket()), patch.object(
            o.ssl, "create_default_context", return_value=FakeContext()
        ), patch.object(
            o.http.client, "HTTPResponse", return_value=FakeResponse()
        ):
            with self.assertRaisesRegex(RuntimeError, "redirects are disabled"):
                o.safe_public_https_json("https://example.test/redirect")

    def test_artifact_verifier_pins_request_and_uses_safe_fetch(self):
        class FakeSocket:
            def __init__(self):
                self.sent = b""

            def sendall(self, data):
                self.sent += data

            def close(self):
                pass

        class FakeContext:
            def __init__(self, tls):
                self.tls = tls

            def wrap_socket(self, sock, server_hostname):
                self.tls.sock = sock
                self.tls.server_hostname = server_hostname
                return self.tls

        class FakeTLS:
            def __init__(self):
                self.sock = None
                self.server_hostname = None

            def sendall(self, data):
                self.sock.sendall(data)

            def makefile(self, *args, **kwargs):
                return io.BytesIO()

            def close(self):
                pass

        class FakeResponse:
            status = 200

            def begin(self):
                pass

            def read(self, size=-1):
                return b'{"ok":true}'

            def close(self):
                pass

        fake_socket = FakeSocket()
        fake_tls = FakeTLS()
        with patch.object(
            o.socket,
            "getaddrinfo",
            return_value=[
                (
                    o.socket.AF_INET,
                    o.socket.SOCK_STREAM,
                    6,
                    "",
                    ("93.184.216.34", 443),
                )
            ],
        ), patch.object(o.socket, "create_connection", return_value=fake_socket), patch.object(
            o.ssl, "create_default_context", return_value=FakeContext(fake_tls)
        ), patch.object(
            o.http.client, "HTTPResponse", return_value=FakeResponse()
        ):
            result = o.safe_public_https_json(
                "https://example.test/api?x=1"
            )

        self.assertEqual(result["status_code"], 200)
        self.assertIn(b"GET /api?x=1 HTTP/1.1\r\n", fake_socket.sent)
        self.assertIn(b"Host: example.test\r\n", fake_socket.sent)

    def test_artifact_verifier_uses_safe_fetch(self):
        node = o.Node(
            "n01",
            "verify",
            "artifact_verifier",
            [],
            input={
                "artifacts": [
                    {
                        "type": "url",
                        "url": "https://example.test/artifact",
                        "contains": "ok",
                    }
                ]
            },
        )
        with patch.object(
            o, "safe_public_https_json",
            return_value={"status_code": 200, "data": {"ok": True}},
        ) as fetch:
            result = o.execute_artifact_verifier(node, "verify artifact")

        self.assertTrue(result["passed"])
        fetch.assert_called_once_with(
            "https://example.test/artifact", timeout=30
        )

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

    def test_one_step_fails_closed_for_stale_side_effecting_federation(self):
        node = o.Node(
            id="n01-dangerous",
            capability="execute",
            tool="github",
            input={"instruction": "must never replay remotely"},
        )
        node.status = "delegated"
        workflow = {
            "id": "wf-stale-side-effect",
            "goal": "blocked recovery",
            "live": True,
            "status": "waiting_agents",
            "nodes": [o.asdict(node)],
            "federation": {
                "id": "fed-dangerous",
                "status": "dispatched",
                "task_count": 1,
                "created_at": (
                    datetime.now(timezone.utc) - timedelta(
                        seconds=o.FEDERATION_STALE_SECONDS + 1
                    )
                ).isoformat(),
                "tasks": [{"task_id": "n01-dangerous"}],
                "artifact_id": None,
                "artifact_digest": None,
            },
        }

        with patch.object(
            o,
            "load_registry",
            return_value={"github": {"side_effects": ["issue_write"]}},
        ), patch.object(
            o,
            "persist_workflow",
        ), patch.object(
            o,
            "append_event",
        ):
            result = o._run_one_step_inner(workflow)

        self.assertEqual(result, "failed")
        self.assertEqual(workflow["status"], "failed")
        self.assertEqual(
            workflow["error"]["type"],
            "federation_recovery_safety_error",
        )
        self.assertEqual(workflow["nodes"][0]["status"], "delegated")

    def test_one_step_rearms_stale_safe_federation_in_direct_step_path(self):
        node = o.Node(
            id="n01-federated",
            capability="execute",
            tool="noop",
            input={"instruction": "complete delegated safe task"},
        )
        node.status = "delegated"
        workflow = {
            "id": "wf-stale-federation-step",
            "goal": "recover stale federation",
            "live": False,
            "status": "waiting_agents",
            "max_attempts": 8,
            "attempts_used": 1,
            "nodes": [o.asdict(node)],
            "federation": {
                "id": "fed-stale",
                "status": "dispatched",
                "task_count": 1,
                "created_at": (
                    datetime.now(timezone.utc) - timedelta(
                        seconds=o.FEDERATION_STALE_SECONDS + 1
                    )
                ).isoformat(),
                "tasks": [{"task_id": "n01-federated"}],
                "artifact_id": None,
                "artifact_digest": None,
            },
        }

        with patch.dict(
            o.os.environ,
            {"ORCHESTRATOR_FEDERATION_ENABLED": "false"},
            clear=False,
        ), patch.object(
            o,
            "load_registry",
            return_value={},
        ), patch.object(
            o,
            "persist_workflow",
        ), patch.object(
            o,
            "append_event",
        ), patch.object(
            o,
            "execute_node",
            return_value={
                "simulated": True,
                "tool": "noop",
                "capability": "execute",
            },
        ):
            result = o._run_one_step_inner(workflow)

        self.assertEqual(result, "completed")
        self.assertEqual(workflow["status"], "completed")
        self.assertEqual(workflow["federation"]["status"], "abandoned")
        self.assertEqual(workflow["nodes"][0]["status"], "completed")
        self.assertEqual(workflow["attempts_used"], 1)

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



    def test_retry_consumes_attempt_budget_once_per_execution(self):
        workflow = {
            "id": "wf_budget_retry",
            "goal": "retry",
            "live": False,
            "max_attempts": 2,
            "attempts_used": 0,
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
        self.assertEqual(workflow["attempts_used"], 2)
        self.assertEqual(workflow["nodes"][0]["retry_count"], 1)

    def test_retry_is_blocked_when_no_attempt_budget_remains(self):
        workflow = {
            "id": "wf_budget_exhausted",
            "goal": "retry",
            "live": False,
            "max_attempts": 1,
            "attempts_used": 0,
            "nodes": [o.asdict(o.Node("n01", "execute", "noop", [], max_retries=2))],
        }

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={}),                  patch.object(o, "execute_node", side_effect=RuntimeError("transient")):
                result = o.run_one_step(workflow)

        self.assertEqual(result, "failed")
        self.assertEqual(workflow["attempts_used"], 1)
        self.assertEqual(workflow["nodes"][0]["error"]["type"], "attempt_budget_exhausted")

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
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False), \
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
    def test_reconciliation_applied_uses_validating_state(self):
        self.assertNotIn("completed", o.TRANSITIONS["reconciling"])
        self.assertIn("validating", o.TRANSITIONS["reconciling"])

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

    def test_resume_scheduler_does_not_rewrite_stale_state_snapshot(self):
        older = {
            "id": "wf-old", "goal": "old", "live": False,
            "status": "running", "updated_at": "2026-10-02T00:00:00+00:00",
            "nodes": [o.asdict(o.Node("n01", "execute", "noop"))],
        }
        newer = {
            "id": "wf-new", "goal": "new", "live": False,
            "status": "running", "updated_at": "2026-10-02T01:00:00+00:00",
            "nodes": [o.asdict(o.Node("n01", "execute", "noop"))],
        }
        state = {"version": CURRENT_STATE_VERSION, "workflows": {"wf-old": older, "wf-new": newer}}
        with patch.object(o, "run_one_step", side_effect=lambda wf, approve_high_risk=False: (wf.update({"status": "completed"}) or "completed")), \
             patch.object(o, "persist_workflow"), patch.object(o, "save_state") as save_state:
            count = o.resume_pending_workflows(state, step=True)
        self.assertEqual(count, 1)
        save_state.assert_not_called()

    def test_deterministic_ingress_workflow_id_is_stable_and_namespaced(self):
        first = o.deterministic_ingress_workflow_id("evt-1", None)
        second = o.deterministic_ingress_workflow_id("evt-1", None)
        idem = o.deterministic_ingress_workflow_id(None, "evt-1")
        self.assertEqual(first, second)
        self.assertNotEqual(first, idem)
        self.assertTrue(first.startswith("wf-ingress-event-"))
        self.assertTrue(idem.startswith("wf-ingress-idempotency-"))

    def test_goal_ingress_uses_targeted_identity_before_global_state(self):
        workflow = {"id": "wf-ingress-event-123", "goal": "duplicate", "live": False, "status": "running", "nodes": [], "event_id": "evt-123"}
        with patch.object(o, "load_workflow", return_value=workflow) as load_workflow, \
             patch.object(o, "load_state", side_effect=AssertionError("legacy full scan should not run")), \
             patch.object(o, "print_summary"), patch.object(o, "notify_execution_callback"), \
             patch.object(sys, "argv", ["orchestrator", "--goal", "duplicate", "--step"]), \
             patch.dict(o.os.environ, {"ORCHESTRATOR_EVENT_ID": "evt-123"}, clear=False):
            with patch.object(o, "deterministic_ingress_workflow_id", return_value="wf-ingress-event-123"):
                result = o.main()
        self.assertEqual(result, 0)
        load_workflow.assert_called_once_with("wf-ingress-event-123")

    def test_idempotency_identity_conflict_fails_closed(self):
        workflow = {
            "id": "wf-ingress-idempotency-123",
            "status": "running",
            "nodes": [],
            "idempotency_key": "idem-1",
            "intent_fingerprint": "a" * 64,
        }
        with patch.object(o, "load_workflow", return_value=workflow), \
             patch.object(sys, "argv", ["orchestrator", "--goal", "conflict"]), \
             patch.dict(o.os.environ, {"ORCHESTRATOR_IDEMPOTENCY_KEY": "idem-1", "ORCHESTRATOR_INTENT_FINGERPRINT": "b" * 64}, clear=False):
            with self.assertRaises(SystemExit) as ctx:
                o.main()
        self.assertIn("ingress identity conflict", str(ctx.exception))

    def test_repository_event_workflow_id_routes_to_targeted_execution(self):
        workflow = {"id": "wf-cont", "goal": "continue", "status": "running", "nodes": []}
        with patch.object(o, "load_workflow", return_value=workflow) as load_workflow, \
             patch.object(o, "run_one_step", return_value="completed") as run_one_step, \
             patch.object(o, "print_summary"), patch.dict(
                 o.os.environ,
                 {"ORCHESTRATOR_EVENT_WORKFLOW_ID": "wf-cont"},
                 clear=False,
             ), patch.object(sys, "argv", ["orchestrator", "--goal", "continue", "--step"]):
            result = o.main()
        self.assertEqual(result, 0)
        load_workflow.assert_called_once_with("wf-cont")
        run_one_step.assert_called_once()

    def test_main_goal_ingress_deduplicates_existing_event_id(self):
        existing = {
            "id": "wf-existing",
            "goal": "duplicate",
            "status": "running",
            "nodes": [],
            "event_id": "evt-123",
        }
        state = {"version": CURRENT_STATE_VERSION, "workflows": {"wf-existing": existing}, "last_workflow_id": "wf-existing"}
        with patch.object(o, "load_state", return_value=state) as load_state, \
             patch.object(o, "create_workflow", side_effect=AssertionError("dedup must stop before create_workflow")), \
             patch.object(o, "print_summary"), patch.object(o, "notify_execution_callback"), \
             patch.object(sys, "argv", ["orchestrator", "--goal", "duplicate",]), \
             patch.dict(o.os.environ, {"ORCHESTRATOR_EVENT_ID": "evt-123"}, clear=False):
            result = o.main()
        self.assertEqual(result, 0)
        load_state.assert_called_once()

    def test_main_persists_new_workflow_before_execution(self):
        workflow = {"id": "wf-new", "goal": "build", "status": "planning", "live": False, "nodes": []}
        calls = []
        state = {"version": CURRENT_STATE_VERSION, "workflows": {}, "last_workflow_id": None}
        def fake_run(wf, approve_high_risk=False):
            calls.append("run")
            wf["status"] = "completed"
        with patch.object(o, "load_state", return_value=state), \
             patch.object(o, "create_workflow", return_value=workflow), \
             patch.object(o, "persist_workflow", side_effect=lambda wf: calls.append("persist")), \
             patch.object(o, "append_event"), patch.object(o, "run_workflow", side_effect=fake_run), \
             patch.object(o, "save_state") as save_state, patch.object(o, "notify_execution_callback"), \
             patch.object(o, "print_summary"), patch.object(sys, "argv", ["orchestrator", "--goal", "build"]):
            result = o.main()
        self.assertEqual(result, 0)
        self.assertEqual(calls[:2], ["persist", "run"])
        save_state.assert_not_called()
    def test_persist_workflow_writes_only_its_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_file = root / "state.json"
            state_file.write_text(json.dumps({"version": CURRENT_STATE_VERSION, "storage_format": "sharded-v1", "workflows": {}, "last_workflow_id": None}), encoding="utf-8")
            workflow = {"id": "wf-shard", "status": "running", "nodes": []}
            with patch.object(o, "STATE_DIR", root), patch.object(o, "STATE_FILE", state_file):
                before = state_file.read_bytes()
                o.persist_workflow(workflow)
                self.assertEqual(before, state_file.read_bytes())
                shard = o.workflow_shard_path("wf-shard")
                self.assertTrue(shard.exists())
                loaded = o.load_state()
                self.assertEqual(loaded["workflows"]["wf-shard"]["status"], "running")

    def test_load_state_prefers_canonical_shard_over_legacy_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_file = root / "state.json"
            legacy = {"id": "wf-legacy", "status": "running", "updated_at": "2026-10-02T00:00:00+00:00", "nodes": []}
            state_file.write_text(json.dumps({"version": CURRENT_STATE_VERSION, "workflows": {"wf-legacy": legacy}}), encoding="utf-8")
            shard = root / "workflows" / (o.hashlib.sha256(b"wf-legacy").hexdigest() + ".json")
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.write_text(json.dumps({**legacy, "status": "completed", "updated_at": "2026-10-02T01:00:00+00:00"}), encoding="utf-8")
            with patch.object(o, "STATE_DIR", root), patch.object(o, "STATE_FILE", state_file):
                loaded = o.load_state()
            self.assertEqual(loaded["workflows"]["wf-legacy"]["status"], "completed")

    def test_load_state_recomputes_latest_workflow_from_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_file = root / "state.json"
            state_file.write_text(json.dumps({"version": CURRENT_STATE_VERSION, "storage_format": "sharded-v1", "workflows": {}, "last_workflow_id": "stale-id"}), encoding="utf-8")
            for workflow_id, updated in (("wf-old", "2026-10-02T00:00:00+00:00"), ("wf-new", "2026-10-02T01:00:00+00:00")):
                shard = root / "workflows" / (o.hashlib.sha256(workflow_id.encode()).hexdigest() + ".json")
                shard.parent.mkdir(parents=True, exist_ok=True)
                shard.write_text(json.dumps({"id": workflow_id, "status": "completed", "updated_at": updated, "nodes": []}), encoding="utf-8")
            with patch.object(o, "STATE_DIR", root), patch.object(o, "STATE_FILE", state_file):
                loaded = o.load_state()
            self.assertEqual(loaded["last_workflow_id"], "wf-new")

    def test_load_state_rejects_unknown_storage_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_file = root / "state.json"
            state_file.write_text(json.dumps({"version": CURRENT_STATE_VERSION, "storage_format": "future-v99", "workflows": {}}), encoding="utf-8")
            with patch.object(o, "STATE_DIR", root), patch.object(o, "STATE_FILE", state_file):
                with self.assertRaisesRegex(RuntimeError, "unsupported orchestrator storage format"):
                    o.load_state()

    def test_load_state_rejects_mismatched_workflow_shard_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_dir = root / "workflows"
            shard_dir.mkdir(parents=True, exist_ok=True)
            (shard_dir / ("0" * 64 + ".json")).write_text(json.dumps({"id": "wf-other", "nodes": []}), encoding="utf-8")
            state_file = root / "state.json"
            state_file.write_text(json.dumps({"version": CURRENT_STATE_VERSION, "storage_format": "sharded-v1", "workflows": {}}), encoding="utf-8")
            with patch.object(o, "STATE_DIR", root), patch.object(o, "STATE_FILE", state_file):
                with self.assertRaisesRegex(RuntimeError, "workflow shard identity mismatch"):
                    o.load_state()

    def test_save_state_migrates_legacy_workflows_to_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_file = root / "state.json"
            workflow = {"id": "wf-migrate", "status": "completed", "nodes": []}
            with patch.object(o, "STATE_DIR", root), patch.object(o, "STATE_FILE", state_file):
                o.save_state({"version": CURRENT_STATE_VERSION, "workflows": {"wf-migrate": workflow}, "last_workflow_id": "wf-migrate"})
                compact = json.loads(state_file.read_text(encoding="utf-8"))
                self.assertEqual(compact["storage_format"], o.STATE_STORAGE_FORMAT)
                self.assertEqual(compact["workflows"], {})
                self.assertTrue(o.workflow_shard_path("wf-migrate").exists())

    def test_load_workflow_uses_single_canonical_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_file = root / "state.json"
            state_file.write_text(json.dumps({"version": CURRENT_STATE_VERSION, "storage_format": "sharded-v1", "workflows": {}, "last_workflow_id": None}), encoding="utf-8")
            for workflow_id, status in (("wf-target", "running"), ("wf-other", "completed")):
                shard = root / "workflows" / (o.hashlib.sha256(workflow_id.encode()).hexdigest() + ".json")
                shard.parent.mkdir(parents=True, exist_ok=True)
                shard.write_text(json.dumps({"id": workflow_id, "status": status, "nodes": []}), encoding="utf-8")
            with patch.object(o, "STATE_DIR", root), patch.object(o, "STATE_FILE", state_file), patch.object(o, "_load_workflow_shards", side_effect=AssertionError("global hydration must not run")):
                workflow = o.load_workflow("wf-target")
            self.assertEqual(workflow["status"], "running")

    def test_load_workflow_falls_back_to_legacy_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_file = root / "state.json"
            workflow = {"id": "wf-legacy-only", "status": "waiting_approval", "nodes": []}
            state_file.write_text(json.dumps({"version": CURRENT_STATE_VERSION, "workflows": {"wf-legacy-only": workflow}}), encoding="utf-8")
            with patch.object(o, "STATE_DIR", root), patch.object(o, "STATE_FILE", state_file):
                loaded = o.load_workflow("wf-legacy-only")
            self.assertEqual(loaded["status"], "waiting_approval")

    def test_compact_terminal_workflow_preserves_identity_and_hashes(self):
        workflow = {
            "id": "wf-terminal",
            "goal": "private-ish goal",
            "status": "completed",
            "created_at": "2026-08-01T00:00:00+00:00",
            "updated_at": "2026-08-01T00:00:00+00:00",
            "event_id": "evt-1",
            "idempotency_key": "idem-1",
            "intent_fingerprint": "a" * 64,
            "input_digest": "b" * 64,
            "plan_fingerprint": "c" * 64,
            "nodes": [{"id": "n01", "capability": "research", "tool": "noop", "status": "completed", "input": {"secret": "large"}, "output": {"large": "x" * 10000}, "error": {}, "depends_on": []}],
            "evidence": {"large": "y" * 10000},
            "reconciliations": {"large": "z" * 10000},
        }
        changed = o.compact_terminal_workflow(
            workflow,
            now=o.datetime.fromisoformat("2026-09-15T00:00:00+00:00"),
            retention_days=30,
        )
        self.assertTrue(changed)
        self.assertEqual(workflow["id"], "wf-terminal")
        self.assertEqual(workflow["status"], "completed")
        self.assertEqual(workflow["event_id"], "evt-1")
        self.assertEqual(workflow["idempotency_key"], "idem-1")
        self.assertEqual(workflow["intent_fingerprint"], "a" * 64)
        self.assertEqual(workflow["input_digest"], "b" * 64)
        self.assertNotIn("goal", workflow)
        self.assertTrue(workflow["goal_sha256"])
        self.assertTrue(workflow["original_state_sha256"])
        self.assertTrue(workflow["original_evidence_sha256"])
        self.assertTrue(workflow["original_reconciliation_sha256"])
        self.assertNotIn("output", workflow["nodes"][0])

    def test_compact_terminal_workflow_skips_recent_and_nonterminal(self):
        recent = {"id": "wf-recent", "status": "completed", "updated_at": "2026-09-10T00:00:00+00:00", "nodes": []}
        failed = {"id": "wf-failed", "status": "failed", "updated_at": "2026-08-01T00:00:00+00:00", "nodes": []}
        now = o.datetime.fromisoformat("2026-09-15T00:00:00+00:00")
        self.assertFalse(o.compact_terminal_workflow(recent, now=now, retention_days=30))
        self.assertFalse(o.compact_terminal_workflow(failed, now=now, retention_days=30))

    def test_compact_terminal_workflows_is_idempotent(self):
        state = {"workflows": {"wf-terminal": {"id": "wf-terminal", "status": "completed", "updated_at": "2026-07-01T00:00:00+00:00", "nodes": []}}}
        now = o.datetime.fromisoformat("2026-09-15T00:00:00+00:00")
        first = o.compact_terminal_workflows(state, now=now, retention_days=30)
        second = o.compact_terminal_workflows(state, now=now, retention_days=30)
        self.assertEqual(first, ["wf-terminal"])
        self.assertEqual(second, [])
        self.assertEqual(state["workflows"]["wf-terminal"]["terminal_compaction_version"], 1)

    def test_persist_workflow_redacts_sensitive_connector_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = {
                "id": "wf-secret",
                "status": "running",
                "nodes": [{
                    "id": "n01",
                    "capability": "execute",
                    "tool": "connector_bridge",
                    "status": "completed",
                    "depends_on": [],
                    "error": {},
                    "input": {"workflow_id": "wf-secret"},
                    "output": {"bridge_job_id": "job-1", "access_token": "TOP-SECRET"},
                    "contract": {},
                    "agent_role": "operator",
                }],
            }
            with patch.object(o, "STATE_DIR", root), patch.object(o, "STATE_FILE", root / "state.json"):
                o.persist_workflow(workflow)
                raw = o.workflow_shard_path("wf-secret").read_text(encoding="utf-8")
            self.assertIn("job-1", raw)
            self.assertNotIn("TOP-SECRET", raw)
            self.assertNotIn('"access_token": "TOP-SECRET"', raw)

    def test_checkpoint_redacts_sensitive_connector_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = o.Node(
                id="n01",
                capability="execute",
                tool="connector_bridge",
                status="completed",
                input={"workflow_id": "wf-checkpoint"},
                output={"bridge_job_id": "job-2", "api_key": "TOP-SECRET-2"},
            )
            workflow = {"id": "wf-checkpoint", "evidence": {}, "nodes": [o.asdict(node)]}
            checkpoint_dir = root / "checkpoints"
            with patch.object(o, "ROOT", root), patch.object(o, "CHECKPOINT_DIR", checkpoint_dir):
                o.node_success_checkpoint(workflow, node)
            raw = next(checkpoint_dir.glob("*.json")).read_text(encoding="utf-8")
            self.assertIn("job-2", raw)
            self.assertNotIn("TOP-SECRET-2", raw)
    def test_state_migration_is_idempotent(self):
        source = {
            "version": 2,
            "workflows": {
                "wf_1": {
                    "id": "wf_1",
                    "schema_version": 2,
                    "nodes": [
                        {"id": "n01", "capability": "execute", "tool": "noop"}
                    ],
                }
            },
        }
        first = o.migrate_state(source)
        snapshot = json.loads(json.dumps(first))
        second = o.migrate_state(first)
        self.assertEqual(second, snapshot)
        self.assertEqual(second["version"], CURRENT_STATE_VERSION)
        self.assertEqual(
            second["workflows"]["wf_1"]["schema_version"],
            CURRENT_WORKFLOW_SCHEMA_VERSION,
        )

    def test_future_state_version_fails_closed(self):
        with self.assertRaises(o.StateSchemaError):
            o.migrate_state({
                "version": CURRENT_STATE_VERSION + 1,
                "workflows": {},
            })

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

    def test_new_workflow_ids_are_collision_resistant(self):
        ids = {o.new_id("wf") for _ in range(256)}
        self.assertEqual(len(ids), 256)
        self.assertTrue(all(item.startswith("wf_") for item in ids))
        self.assertTrue(all(len(item.split("_", 1)[1]) == 32 for item in ids))

    def test_workflow_creation_is_persistable(self):
        workflow = o.create_workflow("build a small website", live=False)
        self.assertTrue(workflow["id"].startswith("wf_"))
        self.assertEqual(workflow["status"], "planning")
        self.assertEqual(workflow["execution_mode"], "dry-run")
        self.assertEqual(workflow["schema_version"], CURRENT_WORKFLOW_SCHEMA_VERSION)
        self.assertGreaterEqual(len(workflow["nodes"]), 4)

    def test_federation_disabled_by_default(self):
        with patch.dict(
            o.os.environ,
            {"ORCHESTRATOR_FEDERATION_ENABLED": "false", "GITHUB_ACTIONS": "true"},
            clear=False,
        ):
            self.assertFalse(o.federation_enabled())

    def test_federation_clamps_large_parallelism_to_protocol_batch_cap(self):
        workflow = {
            "id": "wf-test",
            "goal": "research test",
            "live": False,
            "status": "ready",
            "max_parallel": 8,
            "attempts_used": 0,
            "max_attempts": 64,
            "max_federation_batches": 4,
            "max_federation_tasks": 16,
            "federation_batches_used": 0,
            "federation_tasks_used": 0,
            "nodes": [],
        }
        nodes = [
            o.Node(
                f"n{i:02d}",
                "research",
                "research_bundle",
                risk="low",
                agent_role="researcher",
                input={"instruction": f"task {i}"},
            )
            for i in range(1, 7)
        ]
        workflow["nodes"] = [o.asdict(node) for node in nodes]
        registry = {
            "research_bundle": {"free_tier": True, "side_effects": []},
            "capability:research": {"default_tool": "research_bundle", "fallback_tools": []},
        }
        captured = {}
        with patch.dict(
            o.os.environ,
            {"ORCHESTRATOR_FEDERATION_ENABLED": "true", "GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "token"},
            clear=False,
        ), patch.object(o, "build_node_context", return_value={}), \
             patch.object(o, "new_id", side_effect=["fed_test"]), \
             patch.object(o, "persist_workflow"), \
             patch.object(o, "dispatch_federation", side_effect=lambda manifest: captured.setdefault("manifest", manifest)):
            budget = o.AttemptBudget(workflow)
            result = o.delegate_ready_agents(workflow, nodes, registry, budget)
        self.assertEqual(result, "fed_test")
        self.assertEqual(len(captured["manifest"]["tasks"]), 4)

    def test_dispatch_federation_emits_deterministic_slot(self):
        manifest = {
            "protocol_version": 2,
            "federation_id": "fed_example",
            "workflow_id": "wf_example",
            "tasks": [],
        }
        captured = {}
        with patch.object(o, "github_repository", return_value="owner/repo"),              patch.object(o, "github_headers", return_value={"Authorization": "Bearer x"}),              patch.object(o, "http_json", side_effect=lambda *args, **kwargs: captured.setdefault("body", kwargs.get("body"))):
            o.dispatch_federation(manifest)
        self.assertEqual(
            captured["body"]["client_payload"]["slot"],
            o.federation_scheduler.federation_slot("fed_example")
            if hasattr(o, "federation_scheduler")
            else __import__("federation_scheduler").federation_slot("fed_example"),
        )

    def test_delegate_ready_agents_persists_and_dispatches_safe_batch(self):
        workflow = {
            "id": "wf-test",
            "goal": "research test",
            "live": False,
            "status": "ready",
            "max_parallel": 4,
            "attempts_used": 0,
            "max_attempts": 64,
            "nodes": [],
        }
        nodes = [
            o.Node("n01", "research", "research_bundle", risk="low", agent_role="researcher", input={"instruction": "collect evidence"}),
            o.Node("n02", "research", "research_bundle", risk="low", agent_role="skeptic", input={"instruction": "check counterevidence"}),
        ]
        workflow["nodes"] = [o.asdict(node) for node in nodes]
        registry = {
            "research_bundle": {"free_tier": True, "side_effects": []},
            "capability:research": {"default_tool": "research_bundle", "fallback_tools": []},
        }
        captured = {}
        events = []
        with patch.dict(
            o.os.environ,
            {"ORCHESTRATOR_FEDERATION_ENABLED": "true", "GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "token"},
            clear=False,
        ), patch.object(o, "build_node_context", return_value={}), \
             patch.object(o, "new_id", return_value="fed_test"), \
             patch.object(o, "persist_workflow"), \
             patch.object(o, "append_event", side_effect=lambda event_type, payload: events.append((event_type, payload))), \
             patch.object(o, "dispatch_federation", side_effect=lambda manifest: captured.setdefault("manifest", manifest)):
            budget = o.AttemptBudget(workflow)
            result = o.delegate_ready_agents(workflow, nodes, registry, budget)
        if result is None:
            self.fail(f"federation prepare failed: {events}")
        self.assertEqual(result, "fed_test")
        self.assertEqual(workflow["status"], "waiting_agents")
        self.assertEqual(workflow["federation"]["task_count"], 2)
        self.assertEqual(workflow["attempts_used"], 2)
        self.assertTrue(all(node.status == "delegated" for node in nodes))
        self.assertEqual(len(captured["manifest"]["tasks"]), 2)

    def test_delegate_failure_refunds_reserved_attempts(self):
        workflow = {
            "id": "wf-test",
            "goal": "research test",
            "live": False,
            "status": "ready",
            "max_parallel": 4,
            "attempts_used": 0,
            "max_attempts": 64,
            "nodes": [],
        }
        nodes = [
            o.Node("n01", "research", "research_bundle", risk="low", agent_role="researcher", input={"instruction": "collect evidence"}),
            o.Node("n02", "research", "research_bundle", risk="low", agent_role="skeptic", input={"instruction": "check counterevidence"}),
        ]
        workflow["nodes"] = [o.asdict(node) for node in nodes]
        registry = {
            "research_bundle": {"free_tier": True, "side_effects": []},
            "capability:research": {"default_tool": "research_bundle", "fallback_tools": []},
        }
        with patch.dict(
            o.os.environ,
            {"ORCHESTRATOR_FEDERATION_ENABLED": "true", "GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "token"},
            clear=False,
        ), patch.object(o, "build_node_context", return_value={}), \
             patch.object(o, "new_id", return_value="fed_test"), \
             patch.object(o, "persist_workflow"), \
             patch.object(o, "dispatch_federation", side_effect=RuntimeError("dispatch down")):
            budget = o.AttemptBudget(workflow)
            result = o.delegate_ready_agents(workflow, nodes, registry, budget)
        self.assertIsNone(result)
        self.assertEqual(workflow["attempts_used"], 0)
        self.assertEqual(workflow["status"], "ready")
        self.assertTrue(all(node.status == "ready" for node in nodes))

    def test_min_sources_uses_independent_evidence_not_provider_bucket_count(self):
        node = o.Node(
            "n01",
            "research",
            "research_bundle",
            contract={"min_sources": 3},
        )
        output = {
            "sources": {
                "wikipedia": {},
                "arxiv": {},
                "crossref": {},
                "semantic_scholar": {},
            },
            "evidence_records": [
                {"canonical_id": "doi:10.1000/a", "independence_key": "doi:10.1000/a"},
                {"canonical_id": "doi:10.1000/b", "independence_key": "doi:10.1000/b"},
            ],
            "independent_source_count": 2,
        }
        with self.assertRaisesRegex(RuntimeError, "at least 3 sources"):
            o.validate_node_output(node, output)

    def test_local_validator_uses_independent_evidence_count(self):
        node = o.Node(
            "n02",
            "validate",
            "local_validator",
            contract={"min_sources": 3},
            input={"context": {
                "dependencies": {
                    "n01": {
                        "status": "completed",
                        "output": {
                            "sources": {"a": {}, "b": {}, "c": {}},
                            "evidence_records": [
                                {"canonical_id": "doi:10.1000/a"},
                                {"canonical_id": "doi:10.1000/a"},
                                {"canonical_id": "doi:10.1000/b"},
                            ],
                            "independent_source_count": 2,
                        },
                    }
                }
            }},
        )
        result = o.execute_local_validator(node, "validate research")
        minimum = [
            item for item in result["checks"]
            if item.get("check") == "contract:min_sources"
        ]
        self.assertEqual(len(minimum), 1)
        self.assertEqual(minimum[0]["actual"], 2)
        self.assertFalse(minimum[0]["passed"])

    def test_deterministic_research_plan_uses_multi_agent_scatter_gather(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "REGISTRY_FILE", Path(tmp) / "missing.json"):
                nodes = o.deterministic_plan("research AI safety", {})
        self.assertEqual(nodes[0].agent_role, "researcher")
        self.assertEqual(nodes[0].input["budget"], "balanced")
        self.assertEqual(nodes[1].agent_role, "skeptic")
        self.assertEqual(nodes[1].input["budget"], "balanced")
        self.assertEqual(nodes[1].depends_on, [])
        self.assertEqual(nodes[2].agent_role, "analyst")
        self.assertEqual(
            nodes[2].depends_on,
            ["n01-research", "n02-skeptic"],
        )
        self.assertEqual(nodes[4].agent_role, "critic")

    def test_deterministic_research_plan_diversifies_skeptic_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "REGISTRY_FILE", Path(tmp) / "missing.json"):
                nodes = o.deterministic_plan("research AI safety", {})
        self.assertEqual(nodes[0].input["research_focus"], "primary_evidence")
        self.assertEqual(nodes[1].input["research_focus"], "counterevidence")

    def test_skeptic_research_uses_counterevidence_query_variant(self):
        node = o.Node(
            id="n02-skeptic",
            capability="research",
            tool="research_bundle",
            input={
                "query": "AI safety",
                "research_focus": "counterevidence",
            },
            agent_role="skeptic",
        )
        with patch("research_bundle.research_bundle", return_value={"ok": True}) as research:
            result = o.execute_research_bundle(node, "unused goal")
        self.assertEqual(result, {"ok": True})
        query = research.call_args.args[0]
        self.assertIn("AI safety", query)
        self.assertIn("counterevidence", query)
        self.assertIn("contradictions", query)
        self.assertIn("limitations", query)
        self.assertIn("alternative findings", query)

    def test_workflow_creation_persists_agent_team_manifest(self):
        with patch.dict(o.os.environ, {}, clear=True):
            workflow = o.create_workflow("research AI safety", live=False)
        self.assertEqual(workflow["schema_version"], CURRENT_WORKFLOW_SCHEMA_VERSION)
        self.assertEqual(workflow["agent_team"]["supervisor"], "orchestrator")
        self.assertEqual(
            workflow["agent_team"]["pattern"],
            "parallel_deliberation",
        )
        self.assertTrue(workflow["agent_team"]["members"])

    def test_plan_is_acyclic(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "REGISTRY_FILE", Path(tmp) / "missing.json"):
                nodes = o.deterministic_plan("build and deploy a web app", {})
        o.validate_dag(nodes)
        self.assertEqual(len(nodes), 9)
        self.assertEqual(nodes[-1].capability, "notify")

    def test_validate_dag_rejects_unsafe_and_oversized_node_ids(self):
        with self.assertRaisesRegex(ValueError, "unsafe node id"):
            o.validate_dag([
                o.Node("../escape", "execute", "noop", []),
            ])
        with self.assertRaisesRegex(ValueError, "exceeds 100"):
            o.validate_dag([
                o.Node("n" * 101, "execute", "noop", []),
            ])

    def test_cycle_is_rejected(self):
        a = o.Node("a", "x", "noop", ["b"])
        b = o.Node("b", "x", "noop", ["a"])
        with self.assertRaises(ValueError):
            o.validate_dag([a, b])

    def test_http_requires_https(self):
        with self.assertRaises(RuntimeError):
            o.http_json("http://example.com")

    def test_callback_includes_idempotency_key(self):
        workflow = {
            "id": "wf-callback",
            "execution_id": "e" * 64,
            "status": "completed",
            "external_attempt": 1,
        }
        captured = {}

        class FakeResponse:
            status = 204
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        def fake_urlopen(request, timeout=30):
            captured["headers"] = dict(request.headers)
            return FakeResponse()

        with patch.dict(o.os.environ, {
            "ORCHESTRATOR_CALLBACK_URL": "https://callback.example/hook",
            "ORCHESTRATOR_CALLBACK_SECRET": "callback-secret",
        }, clear=False),              patch.object(o.urllib.request, "urlopen", side_effect=fake_urlopen),              patch.object(o, "append_event"):
            self.assertTrue(o.notify_execution_callback(workflow))

        self.assertEqual(
            captured["headers"].get("Idempotency-key"),
            workflow["execution_id"],
        )

    def test_http_response_is_bounded(self):
        class Response:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return None
            def read(self, limit=None):
                return b"x" * (limit + 1)

        with patch.object(o.urllib.request, "urlopen", return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, "HTTP response exceeds 2048 bytes"):
                o.http_json(
                    "https://example.test",
                    max_response_bytes=2048,
                )

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

    def test_checkpoint_filename_is_bounded_and_path_safe(self):
        workflow = {
            "id": "../wf/../../evil",
            "nodes": [],
            "evidence": {},
        }
        node = o.Node(
            "../node/../../escape",
            "execute",
            "noop",
            [],
            status="running",
            input={},
            output={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / ".orchestrator" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            with patch.object(o, "CHECKPOINT_DIR", checkpoint_dir):
                o.node_success_checkpoint(workflow, node)
            files = list(checkpoint_dir.iterdir())
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].suffix, ".json")
            self.assertRegex(files[0].name, r"^[0-9a-f]{64}\.json$")
            self.assertEqual(files[0].parent.resolve(), checkpoint_dir.resolve())

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
        with patch.object(o, "safe_public_https_json", return_value={"status_code": 200, "data": {"ok": True}}):
            result = o.execute_artifact_verifier(node, "verify")
        self.assertTrue(result["checks"][0]["passed"])

    def test_llm_retry_consumes_second_budget_slot(self):
        node = o.Node(
            "n01", "analyze", "gemini", [],
            status="running",
            max_retries=1,
        )
        workflow = {"id": "wf_llm_retry", "llm_calls_used": 1}
        budget = o.AttemptBudget({
            "attempts_used": 1,
            "max_attempts": 4,
        })
        with patch.object(
            o,
            "execute_node",
            side_effect=[
                RuntimeError("transient"),
                {"candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                 },
            ],
        ), patch.object(
            o,
            "execution_failure_policy",
            return_value=("transient", True, False),
        ), patch.dict(
            o.os.environ,
            {"ORCHESTRATOR_FREE_ONLY": "true", "ORCHESTRATOR_MAX_LLM_CALLS": "2"},
            clear=False,
        ):
            ok, error = o.execute_with_retries(
                node,
                "goal",
                False,
                attempt_budget=budget,
                initial_attempt_reserved=True,
                llm_budget_workflow=workflow,
            )
        self.assertTrue(ok)
        self.assertIsNone(error)
        self.assertEqual(workflow["llm_calls_used"], 2)
        self.assertEqual(budget.used, 2)

    def test_llm_retry_fails_closed_when_budget_is_exhausted(self):
        node = o.Node(
            "n01", "analyze", "gemini", [],
            status="running",
            max_retries=1,
        )
        workflow = {"id": "wf_llm_retry_limit", "llm_calls_used": 1}
        budget = o.AttemptBudget({
            "attempts_used": 1,
            "max_attempts": 4,
        })
        with patch.object(
            o,
            "execute_node",
            side_effect=RuntimeError("transient"),
        ) as execute, patch.object(
            o,
            "execution_failure_policy",
            return_value=("transient", True, False),
        ), patch.dict(
            o.os.environ,
            {"ORCHESTRATOR_FREE_ONLY": "true", "ORCHESTRATOR_MAX_LLM_CALLS": "1"},
            clear=False,
        ):
            ok, error = o.execute_with_retries(
                node,
                "goal",
                False,
                attempt_budget=budget,
                initial_attempt_reserved=True,
                llm_budget_workflow=workflow,
            )
        self.assertFalse(ok)
        self.assertEqual(error["type"], "llm_call_budget_exhausted")
        self.assertEqual(workflow["llm_calls_used"], 1)
        self.assertEqual(budget.used, 1)
        self.assertEqual(execute.call_count, 1)

    def test_free_only_llm_call_budget_is_bounded_and_persisted(self):
        node = o.Node("n01", "analyze", "gemini", [])
        workflow = {"id": "wf_llm_budget", "llm_calls_used": 11}
        with patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            self.assertTrue(o.reserve_llm_call(workflow, node, live=True))
            self.assertEqual(workflow["llm_calls_used"], 12)
            self.assertEqual(workflow["llm_call_limit"], 12)
            self.assertFalse(o.reserve_llm_call(workflow, node, live=True))
        self.assertEqual(node.error["type"], "llm_call_budget_exhausted")

    def test_llm_budget_exhaustion_survives_resume_state(self):
        node = o.Node("n01", "analyze", "gemini", [])
        workflow = {
            "id": "wf_llm_resume_budget",
            "goal": "analyze",
            "live": True,
            "llm_calls_used": 12,
            "nodes": [o.asdict(node)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                o.os.environ,
                {"ORCHESTRATOR_FREE_ONLY": "true"},
                clear=False,
            ), patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={"gemini": {"free_tier": True}}),                  patch.object(o, "execute_node") as execute:
                result = o.run_one_step(workflow)
                self.assertEqual(result, "failed")
                execute.assert_not_called()
                shard = o.workflow_shard_path(workflow["id"])
                persisted = json.loads(shard.read_text(encoding="utf-8"))
        self.assertEqual(persisted["llm_calls_used"], 12)
        self.assertEqual(
            persisted["nodes"][0]["error"]["type"],
            "llm_call_budget_exhausted",
        )

    def test_dry_run_does_not_consume_llm_call_budget(self):
        node = o.Node("n01", "analyze", "gemini", [])
        workflow = {"id": "wf_llm_dry"}
        self.assertTrue(o.reserve_llm_call(workflow, node, live=False))
        self.assertNotIn("llm_calls_used", workflow)

    def test_simulated_output_still_enforces_deterministic_contracts(self):
        node = o.Node(
            "n01", "execute", "noop", [],
            contract={"required_fields": ["answer"], "min_sources": 1},
        )
        with self.assertRaisesRegex(RuntimeError, "contract field missing: answer"):
            o.validate_node_output(
                node,
                {"simulated": True},
            )

    def test_simulated_epistemic_output_defers_only_epistemic_validation(self):
        node = o.Node(
            "n01", "execute", "gemini", [],
            contract={"epistemic": True, "min_coverage": 0.8},
        )
        result = o.validate_node_output(
            node,
            {"simulated": True, "candidates": []},
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"][0]["epistemic_deferred"])

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

    def test_approval_accepts_only_matching_node_fingerprint(self):
        node = o.Node(
            "n01-publish", "publish", "webhook", [], risk="high", status="waiting_approval",
            input={"approval_issue": 42, "approval_granted": False},
        )
        node.input["approval_fingerprint"] = o.fingerprint_nodes([node])
        workflow = {"id": "wf_approval_bind", "status": "waiting_approval", "nodes": [o.asdict(node)]}
        with patch.dict(o.os.environ, {"GITHUB_ACTOR": "reviewer"}, clear=False), \
             patch.object(o, "get_issue_labels", return_value={"orchestrator-approved"}):
            o.refresh_approvals(workflow, [node])
        self.assertTrue(node.input["approval_granted"])
        self.assertEqual(node.input["approval_actor"], "reviewer")
        self.assertTrue(node.input["approval_approved_at"])
        self.assertEqual(node.status, "ready")
        self.assertEqual(workflow["status"], "running")

    def test_approval_fingerprint_ignores_runtime_metadata_but_binds_semantic_fields(self):
        node = o.Node(
            "n01", "publish", "webhook", [],
            risk="high",
            input={
                "workflow_id": "wf-a",
                "instruction": "publish artifact",
                "context": {"volatile": 1},
                "repair_feedback": {"attempt": 2},
                "approval_issue": 7,
                "approval_granted": False,
                "approval_actor": "reviewer",
                "approval_approved_at": "2026-10-02T12:00:00+00:00",
                "approval_fingerprint": "old",
                "retry_jitter_seed": "seed-a",
                "payload": {"artifact": "build.zip"},
            },
        )
        baseline = o.fingerprint_nodes([node])
        node.input["context"] = {"volatile": 999}
        node.input["repair_feedback"] = {"attempt": 99}
        node.input["approval_actor"] = "other-reviewer"
        node.input["approval_approved_at"] = "2026-10-03T12:00:00+00:00"
        node.input["retry_jitter_seed"] = "seed-b"
        self.assertEqual(baseline, o.fingerprint_nodes([node]))

        node.input["payload"] = {"artifact": "different.zip"}
        self.assertNotEqual(baseline, o.fingerprint_nodes([node]))

    def test_stale_approval_is_rearmed_instead_of_accepted(self):
        node = o.Node(
            "n01-publish", "publish", "webhook", [], risk="high", status="waiting_approval",
            input={"approval_issue": 42, "approval_granted": False},
        )
        node.input["approval_fingerprint"] = o.fingerprint_nodes([node])
        node.input["instruction"] = "changed after approval request"
        workflow = {"id": "wf_stale_approval", "status": "waiting_approval", "nodes": [o.asdict(node)]}
        with patch.object(o, "get_issue_labels", return_value={"orchestrator-approved"}):
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


    def test_execution_ledger_stores_digest_not_full_output(self):
        workflow = {"id": "wf_ledger", "executions": {}}
        payload = {"result": "x" * 1000}
        o.mark_execution_completed(workflow, "exec-1", payload)
        record = workflow["executions"]["exec-1"]
        self.assertEqual(record["status"], "completed")
        self.assertIn("output_sha256", record)
        self.assertNotIn("output", record)

    def test_find_federation_artifact_filters_exact_name_and_unexpired(self):
        with patch.object(
            o,
            "github_repository",
            return_value="owner/repo",
        ), patch.object(
            o,
            "github_headers",
            return_value={"Authorization": "Bearer token"},
        ), patch.object(
            o,
            "http_json",
            return_value={
                "data": {
                    "artifacts": [
                        {"id": 10, "name": "federation-results-fed-bad", "expired": False},
                        {"id": 11, "name": "federation-results-fed-good", "expired": True},
                        {"id": 12, "name": "federation-results-fed-good", "expired": False, "digest": "sha256:ok"},
                    ]
                }
            },
        ):
            self.assertEqual(
                o.find_federation_artifact("fed-good"),
                (12, "sha256:ok"),
            )

    def test_run_one_step_does_not_reexecute_waiting_federation(self):
        workflow = {
            "id": "wf-test",
            "goal": "research test",
            "status": "waiting_agents",
            "live": False,
            "nodes": [],
            "federation": {"id": "fed-test", "status": "dispatched"},
        }
        with patch.object(o, "load_registry", return_value={}):
            result = o.run_one_step(workflow)
        self.assertEqual(result, "waiting_agents")
        self.assertEqual(workflow["status"], "waiting_agents")

    def test_tool_health_is_sharded_by_tool(self):
        node = o.Node(
            "n01",
            "research",
            "research_bundle",
            risk="low",
            input={"instruction": "collect evidence"},
        )
        registry = {"research_bundle": {"side_effects": [], "free_tier": True}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(o, "STATE_DIR", root),                  patch.object(o, "append_event"):
                updated = o.update_tool_health(node, False, registry)
                shard = o.tool_health_path("research_bundle")
                self.assertTrue(shard.exists())
                self.assertEqual(
                    o.load_tool_health()["research_bundle"]["status"],
                    updated["status"],
                )
                self.assertFalse((root / "tool_health.json").exists())

    def test_tool_health_legacy_file_is_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "tool_health.json"
            legacy.write_text(
                '{"research_bundle":{"status":"degraded","failure_streak":1}}',
                encoding="utf-8",
            )
            with patch.object(o, "STATE_DIR", root):
                self.assertEqual(
                    o.load_tool_health()["research_bundle"]["failure_streak"],
                    1,
                )

    def test_workflow_events_are_sharded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(o, "EVENT_DIR", root / "events"),                  patch.object(o, "EVENT_FILE", root / "legacy-events.jsonl"):
                o.append_event("workflow.started", {"workflow_id": "wf_a"})
                o.append_event("workflow.started", {"workflow_id": "wf_b"})
                a = root / "events" / o.hashlib.sha256(b"wf_a").hexdigest()
                b = root / "events" / o.hashlib.sha256(b"wf_b").hexdigest()
                self.assertTrue((a.with_suffix(".jsonl")).exists())
                self.assertTrue((b.with_suffix(".jsonl")).exists())
                self.assertNotEqual(a, b)
                self.assertFalse((root / "legacy-events.jsonl").exists())

    def test_event_without_workflow_keeps_legacy_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(o, "EVENT_DIR", root / "events"),                  patch.object(o, "EVENT_FILE", root / "legacy-events.jsonl"):
                o.append_event("planner.fallback", {"goal": "test"})
                self.assertTrue((root / "legacy-events.jsonl").exists())
                self.assertFalse(list((root / "events").glob("*.jsonl")) if (root / "events").exists() else [])

    def test_event_log_redacts_durable_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = Path(tmp) / "events"
            with patch.object(o, "EVENT_FILE", Path(tmp) / "legacy-events.jsonl"),                  patch.object(o, "EVENT_DIR", event_dir):
                payload = {
                    "workflow_id": "wf-event-redact",
                    "error": {
                        "message": "Authorization: Bearer TOPSECRET",
                        "traceback": "Traceback ... api_key=TRACESECRET",
                    },
                }
                o.append_event("node.failed", payload)
                event_path = o._event_file(payload)
            content = event_path.read_text(encoding="utf-8")

        self.assertNotIn("TOPSECRET", content)
        self.assertNotIn("TRACESECRET", content)
        event = json.loads(content)
        self.assertEqual(
            event["payload"]["error"]["traceback"]["reason"],
            "diagnostic_trace",
        )
        self.assertTrue(event["payload"]["error"]["traceback"]["redacted"])

    def test_event_payload_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_path = Path(tmp) / "events.jsonl"
            with patch.object(o, "EVENT_FILE", event_path):
                o.append_event("test.large", {"payload": ["x" * 4096] * 8})
            entry = json.loads(event_path.read_text(encoding="utf-8"))
        self.assertTrue(entry["payload"]["truncated"])
        self.assertEqual(len(entry["payload"]["sha256"]), 64)

    def test_workflow_attempt_budget_blocks_execution(self):
        node = o.Node("n01", "execute", "noop", [], max_retries=999)
        workflow = {
            "id": "wf_budget",
            "goal": "budget",
            "live": False,
            "attempts_used": 2,
            "max_attempts": 2,
            "nodes": [o.asdict(node)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value={}),                  patch.object(o, "execute_node") as execute:
                result = o.run_one_step(workflow)
        self.assertEqual(result, "failed")
        self.assertEqual(execute.call_count, 0)
        self.assertEqual(workflow["nodes"][0]["error"]["type"], "attempt_budget_exhausted")

    def test_research_bundle_is_not_globally_quota_serialized(self):
        node = o.Node("n01-research", "research", "research_bundle", [])
        self.assertFalse(o.quota_sensitive(node))

    def test_quota_sensitive_nodes_are_serialized(self):
        nodes = [
            o.Node("n01-llm-a", "analyze", "gemini", []),
            o.Node("n02-llm-b", "analyze", "gemini", []),
        ]
        workflow = {
            "id": "wf_quota_serial",
            "goal": "analysis",
            "live": True,
            "max_parallel": 2,
            "nodes": [o.asdict(n) for n in nodes],
        }
        active = 0
        maximum = 0
        lock = threading.Lock()

        def fake_execute(node, goal, dry_run):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return {"output": "ok"}

        registry = {
            "gemini": {"free_tier": True, "free_models": ["test-model"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)),                  patch.object(o, "EVENT_FILE", Path(tmp) / "events.jsonl"),                  patch.object(o, "CHECKPOINT_DIR", Path(tmp) / "checkpoints"),                  patch.object(o, "load_registry", return_value=registry),                  patch.object(o, "execute_node", side_effect=fake_execute),                  patch.dict(o.os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
                o.run_workflow(workflow)
        self.assertEqual(workflow["status"], "completed")
        self.assertEqual(maximum, 1)

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
        error = o.ConnectorRequestError("timeout", uncertain=True, idempotent=False)
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
            o.ConnectorRequestError("timeout-1", uncertain=True, idempotent=True),
            o.ConnectorRequestError("timeout-2", uncertain=True, idempotent=True),
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

    def test_epistemic_contract_accepts_supported_claims(self):
        node = o.Node(
            "n01", "draft", "gemini", [],
            contract={"epistemic": True, "min_coverage": 1.0},
        )
        output = {
            "candidates": [{
                "content": {"parts": [{
                    "text": json.dumps({
                        "result": "answer",
                        "claims": [{
                            "claim_id": "c1",
                            "statement": "fact",
                            "material": True,
                            "status": "SUPPORTED_DIRECT",
                            "evidence_refs": ["doi:10.1/a"],
                        }],
                        "evidence_records": [{"canonical_id": "doi:10.1/a"}],
                        "risks": [],
                        "next_action": "done",
                    })
                }]}
            }]
        }
        result = o.validate_node_output(node, output)
        self.assertTrue(result["passed"])

    def test_epistemic_contract_rejects_unknown_evidence_reference(self):
        node = o.Node(
            "n01", "draft", "gemini", [],
            contract={"epistemic": True},
        )
        output = {
            "candidates": [{
                "content": {"parts": [{
                    "text": json.dumps({
                        "result": "answer",
                        "claims": [{
                            "claim_id": "c1",
                            "statement": "fact",
                            "material": True,
                            "status": "SUPPORTED_DIRECT",
                            "evidence_refs": ["missing"],
                        }],
                        "evidence_records": [{"canonical_id": "doi:10.1/a"}],
                        "risks": [],
                        "next_action": "revise",
                    })
                }]}
            }]
        }
        with self.assertRaisesRegex(RuntimeError, "epistemic validation failed"):
            o.validate_node_output(node, output)

    def test_epistemic_gemini_prompt_requires_claim_evidence_contract(self):
        node = o.Node(
            "n01", "draft", "gemini", [],
            contract={"epistemic": True},
            agent_role="analyst",
            input={"instruction":"draft from evidence", "workflow_id":"wf-ep"},
        )
        captured = {}
        def fake_http(url, method="GET", body=None, headers=None, timeout=60, max_response_bytes=o.MAX_GENERIC_HTTP_RESPONSE_BYTES):
            captured["body"] = body
            return {"status_code": 200, "data": {"ok": True}}
        registry = {"gemini": {
            "default_model":"gemini-3.8-flash",
            "free_models":["gemini-3.8-flash"],
            "free_tier":True,
        }}
        with patch.dict(o.os.environ, {"GEMINI_API_KEY":"x","ORCHESTRATOR_FREE_ONLY":"true"}, clear=True), \
             patch.object(o, "load_registry", return_value=registry), \
             patch.object(o, "http_json", side_effect=fake_http):
            o.execute_gemini(node, "goal")
        prompt = captured["body"]["contents"][0]["parts"][0]["text"]
        self.assertIn("claims", prompt)
        self.assertIn("evidence_records", prompt)
        self.assertIn("evidence_refs", prompt)

    def test_side_effect_failure_does_not_switch_adapter_without_equivalence_contract(self):
        node = o.Node(
            "n01-publish", "publish", "connector_bridge", [],
            risk="high", max_retries=0,
        )
        node.status = "failed"
        node.error = {
            "type": "ConnectorRequestError",
            "message": "definitive upstream rejection",
            "failure_class": "permanent",
        }
        workflow = {
            "id": "wf-side-effect-replan",
            "goal": "publish",
            "live": True,
            "nodes": [o.asdict(node)],
            "replan_count": 0,
            "attempts_used": 1,
            "max_attempts": 8,
        }
        registry = {
            "capability:publish": {
                "default_tool": "connector_bridge",
                "fallback_tools": ["webhook"],
            },
            "connector_bridge": {
                "free_tier": True,
                "side_effects": ["external_request"],
            },
            "webhook": {
                "free_tier": True,
                "side_effects": ["external_request"],
            },
        }
        self.assertFalse(o.replan_after_failure(workflow, [node], node, registry))
        self.assertEqual(node.tool, "connector_bridge")


    def test_simulated_research_bundle_defers_provider_contract(self):
        node = o.Node(
            "n01", "research", "research_bundle", [],
            contract={},
        )
        output = {
            "simulated": True,
            "tool": "research_bundle",
            "capability": "research",
        }
        # Dry-run must still enforce deterministic shape checks, but must not
        # require live provider evidence that was intentionally not fetched.
        result = o.validate_node_output(node, output)
        self.assertTrue(result["passed"])
        self.assertTrue(any(
            check.get("check") == "adapter_contract"
            and check.get("deferred") is True
            for check in result["checks"]
        ))


    def test_live_workflow_uses_distributed_control_plane_authority(self):
        with patch.dict(
            o.os.environ,
            {
                "ORCHESTRATOR_LLM_PLANNER": "false",
                "ORCHESTRATOR_CONTROL_PLANE_URL": "https://cp.example.test",
                "ORCHESTRATOR_CONTROL_PLANE_SECRET": "test-secret",
            },
            clear=False,
        ), patch.object(o, "load_registry", return_value={}):
            workflow = o.create_workflow(
                "authority test",
                live=True,
                workflow_id="wf-authority-cp",
            )
        self.assertEqual(
            workflow["authority_mode"],
            o.AUTHORITY_DISTRIBUTED_CONTROL_PLANE,
        )

    def test_workflow_without_control_plane_uses_git_authority(self):
        with patch.dict(
            o.os.environ,
            {
                "ORCHESTRATOR_LLM_PLANNER": "false",
                "ORCHESTRATOR_CONTROL_PLANE_URL": "",
                "ORCHESTRATOR_CONTROL_PLANE_SECRET": "",
            },
            clear=False,
        ), patch.object(o, "load_registry", return_value={}):
            workflow = o.create_workflow(
                "authority test",
                live=True,
                workflow_id="wf-authority-git",
            )
        self.assertEqual(workflow["authority_mode"], o.AUTHORITY_GIT_DURABLE)

    def test_legacy_control_plane_state_migrates_to_distributed_authority(self):
        migrated = o.migrate_state({
            "version": CURRENT_STATE_VERSION,
            "workflows": {
                "wf-legacy-cp": {
                    "id": "wf-legacy-cp",
                    "live": True,
                    "control_plane": {"enabled": True},
                    "nodes": [],
                }
            },
        })
        self.assertEqual(
            migrated["workflows"]["wf-legacy-cp"]["authority_mode"],
            o.AUTHORITY_DISTRIBUTED_CONTROL_PLANE,
        )

    def test_distributed_workflow_fails_closed_when_control_plane_is_missing(self):
        workflow = {
            "id": "wf-authority-block",
            "goal": "authority block",
            "live": True,
            "authority_mode": o.AUTHORITY_DISTRIBUTED_CONTROL_PLANE,
            "status": "ready",
            "nodes": [],
        }
        with patch.dict(
            o.os.environ,
            {
                "ORCHESTRATOR_CONTROL_PLANE_URL": "",
                "ORCHESTRATOR_CONTROL_PLANE_SECRET": "",
            },
            clear=False,
        ), patch.object(o, "_run_one_step_inner", return_value="inner-ran") as inner:
            with self.assertRaises(o.ControlPlaneError):
                o.run_one_step(workflow)
            inner.assert_not_called()

    def test_distributed_authority_rejects_missing_remote_state(self):
        workflow = {
            "id": "wf-authority-missing",
            "goal": "authority missing",
            "live": True,
            "authority_mode": o.AUTHORITY_DISTRIBUTED_CONTROL_PLANE,
            "status": "running",
            "nodes": [],
            "control_plane_state_version": 4,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)), patch.dict(
                o.os.environ,
                {
                    "ORCHESTRATOR_CONTROL_PLANE_URL": "https://cp.example.test",
                    "ORCHESTRATOR_CONTROL_PLANE_SECRET": "test-secret",
                },
                clear=False,
            ), patch.object(
                o.ControlPlaneClient,
                "from_env",
            ) as from_env:
                client = from_env.return_value
                client.get_workflow_state.return_value = None
                o._write_workflow_shard(workflow)
                with self.assertRaises(RuntimeError):
                    o.load_workflow(workflow["id"])

    def test_git_authority_does_not_switch_to_control_plane_when_config_appears(self):
        workflow = {
            "id": "wf-authority-git-load",
            "goal": "git authority",
            "live": True,
            "authority_mode": o.AUTHORITY_GIT_DURABLE,
            "status": "ready",
            "nodes": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)), patch.dict(
                o.os.environ,
                {
                    "ORCHESTRATOR_CONTROL_PLANE_URL": "https://cp.example.test",
                    "ORCHESTRATOR_CONTROL_PLANE_SECRET": "test-secret",
                },
                clear=False,
            ), patch.object(
                o.ControlPlaneClient,
                "from_env",
                side_effect=AssertionError("Git-authority workflow must not query control plane"),
            ):
                o._write_workflow_shard(workflow)
                loaded = o.load_workflow(workflow["id"])
        self.assertEqual(loaded["authority_mode"], o.AUTHORITY_GIT_DURABLE)

    def test_run_one_step_uses_shared_control_plane_session(self):
        class Session:
            def __enter__(self):
                return (None, None)

            def __exit__(self, exc_type, exc, tb):
                return False

        workflow = {
            "id": "wf-session",
            "goal": "session",
            "live": False,
            "status": "ready",
            "nodes": [],
        }
        with patch.object(o, "control_plane_session", return_value=Session()) as session,              patch.object(o, "_run_one_step_inner", return_value="ok") as inner:
            result = o.run_one_step(workflow)
        self.assertEqual(result, "ok")
        session.assert_called_once_with(workflow)
        inner.assert_called_once()

    def test_create_workflow_persists_idempotency_key(self):
        with patch.dict(
            o.os.environ,
            {"ORCHESTRATOR_LLM_PLANNER": "false"},
            clear=False,
        ), patch.object(o, "load_registry", return_value={}):
            workflow = o.create_workflow(
                "idempotency test",
                live=False,
                idempotency_key="idem-123",
            )
        self.assertEqual(workflow["idempotency_key"], "idem-123")


if __name__ == "__main__":
    unittest.main()