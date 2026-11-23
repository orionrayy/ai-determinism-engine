from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

try:
    from . import orchestrator as engine_module
    from .robin_mcp import (
        MAX_MCP_MESSAGE_BYTES,
        MCPStdioClient,
        RobinMCPError,
        build_docker_command,
        validate_investigation_input,
        _sanitize_report,
        cleanup_robin_volume,
    )
except ImportError:
    import orchestrator as engine_module
    from robin_mcp import (
        MAX_MCP_MESSAGE_BYTES,
        MCPStdioClient,
        RobinMCPError,
        build_docker_command,
        validate_investigation_input,
        _sanitize_report,
    )


ROOT = Path(__file__).resolve().parents[1]
ROBIN_COMMIT = "b5d4f5f509fac39d65d3a28fd1d78b56efbeeaf4"
VALID_GOAL = (
    "I explicitly selected Robin preset threat_intel. Use it to investigate "
    "APT example indicators for defensive incident response. "
    "Authorization basis: authorized SOC incident response for our organization's "
    "publicly disclosed incident."
)


def valid_payload() -> dict:
    return {
        "action": "investigate",
        "query": "APT example indicators",
        "preset": "threat_intel",
        "user_chose": True,
        "authorization_basis": (
            "authorized SOC incident response for our organization's publicly disclosed incident"
        ),
        "custom_instructions": "Correlate domains and malware indicators.",
    }


def mcp_server_script(mode: str = "normal") -> str:
    return r"""
import json
import sys

mode = %r
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    request_id = request.get("id")
    if method == "initialize":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "robin-test", "version": "1.0"}
            }
        }), flush=True)
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        if mode == "malformed":
            print("{not-json", flush=True)
            break
        if mode == "oversized":
            print(json.dumps({"jsonrpc": "2.0", "id": request_id,
                "result": {"padding": "x" * (524288 + 10)}}), flush=True)
            break
        print(json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progress": 1}
        }), flush=True)
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": [{"name": "robin_investigate"}]}
        }), flush=True)
    elif method == "tools/call":
        if mode == "hang":
            continue
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": "status: ok\nreport: grounded result"}],
                "isError": False
            }
        }), flush=True)
"""


class RobinInputValidationTests(unittest.TestCase):
    def test_accepts_explicit_defensive_investigation(self):
        request = validate_investigation_input(valid_payload(), VALID_GOAL)
        self.assertEqual(request["preset"], "threat_intel")
        self.assertEqual(request["query"], "APT example indicators")
        self.assertTrue(request["user_chose"])
        self.assertIn("exclude credentials", request["custom_instructions"].lower())

    def test_rejects_missing_explicit_preset_choice(self):
        payload = valid_payload()
        payload["user_chose"] = False
        with self.assertRaisesRegex(RobinMCPError, "user_chose"):
            validate_investigation_input(payload, VALID_GOAL)

    def test_rejects_preset_not_explicitly_present_in_original_goal(self):
        goal = VALID_GOAL.replace("threat_intel", "general research")
        with self.assertRaisesRegex(RobinMCPError, "preset"):
            validate_investigation_input(valid_payload(), goal)

    def test_rejects_authorization_basis_not_present_in_original_goal(self):
        payload = valid_payload()
        payload["authorization_basis"] = "authorized testing for a different company and purpose"
        with self.assertRaisesRegex(RobinMCPError, "authorization"):
            validate_investigation_input(payload, VALID_GOAL)

    def test_rejects_preset_mentioned_without_explicit_user_selection(self):
        goal = (
            "Use Robin preset threat_intel to investigate APT example indicators. "
            "Authorization basis: authorized SOC incident response for our organization's "
            "publicly disclosed incident."
        )
        with self.assertRaisesRegex(RobinMCPError, "explicit.*select|user.*select"):
            validate_investigation_input(valid_payload(), goal)

    def test_rejects_authorization_basis_without_explicit_label(self):
        goal = (
            "I explicitly selected Robin preset threat_intel. "
            "APT example indicators are within defensive SOC incident response. "
            + valid_payload()["authorization_basis"]
        )
        with self.assertRaisesRegex(RobinMCPError, "Authorization basis"):
            validate_investigation_input(valid_payload(), goal)

    def test_rejects_personal_identity_and_corporate_espionage_presets(self):
        for preset in ("personal_identity", "corporate_espionage"):
            payload = valid_payload()
            payload["preset"] = preset
            goal = VALID_GOAL + " " + preset
            with self.subTest(preset=preset), self.assertRaisesRegex(RobinMCPError, "preset"):
                validate_investigation_input(payload, goal)

    def test_rejects_query_not_present_in_original_goal(self):
        payload = valid_payload()
        payload["query"] = "another unrelated target"
        with self.assertRaisesRegex(RobinMCPError, "query"):
            validate_investigation_input(payload, VALID_GOAL)

    def test_bounds_query_and_custom_instruction_size(self):
        payload = valid_payload()
        payload["query"] = "x" * 501
        with self.assertRaisesRegex(RobinMCPError, "query"):
            validate_investigation_input(payload, VALID_GOAL)

        payload = valid_payload()
        payload["custom_instructions"] = "x" * 3001
        long_goal = VALID_GOAL + " " + ("x" * 3100)
        with self.assertRaisesRegex(RobinMCPError, "instructions"):
            validate_investigation_input(payload, long_goal)


class RobinReportSanitizationTests(unittest.TestCase):
    def test_redacts_credential_fields_api_tokens_and_personal_emails(self):
        report = (
            "password: hunter2 api_key=sk-123456789012345678901234 "
            "owner jane.doe@example.com access token ghp_abcdefghijklmnopqrstuvwx"
        )
        sanitized = _sanitize_report(report)
        for secret in (
            "hunter2",
            "sk-123456789012345678901234",
            "jane.doe@example.com",
            "ghp_abcdefghijklmnopqrstuvwx",
        ):
            self.assertNotIn(secret, sanitized)
        self.assertIn("[REDACTED", sanitized)


class RobinVolumeCleanupTests(unittest.TestCase):
    def test_rejects_untrusted_volume_names_without_invoking_docker(self):
        with patch("robin_mcp.shutil.which") as which, patch("robin_mcp._run_docker_check") as run:
            self.assertFalse(cleanup_robin_volume("random-volume"))
            self.assertFalse(cleanup_robin_volume("robin-../../etc/passwd"))
            which.assert_not_called()
            run.assert_not_called()

    def test_removes_only_a_validated_robin_volume(self):
        from unittest.mock import Mock
        completed = Mock(returncode=0)
        with patch("robin_mcp.shutil.which", return_value="/usr/bin/docker"), patch(
            "robin_mcp._run_docker_check", return_value=completed
        ) as run:
            self.assertTrue(cleanup_robin_volume("robin-" + "a" * 24))
        run.assert_called_once_with(
            ["docker", "volume", "rm", "--force", "robin-" + "a" * 24],
            timeout=15.0,
        )


class RobinDockerCommandTests(unittest.TestCase):
    def test_command_is_argument_vector_and_forwards_only_named_secret(self):
        command = build_docker_command(
            "ai-orchestrator-robin:" + ROBIN_COMMIT[:12],
            "robin-investigation-123",
        )
        self.assertEqual(command[0:3], ["docker", "run", "--rm"])
        self.assertIn("-i", command)
        self.assertIn("-e", command)
        self.assertIn("GOOGLE_API_KEY", command)
        self.assertIn("ROBIN_MODEL", command)
        self.assertIn("ai-orchestrator-robin:" + ROBIN_COMMIT[:12], command)
        self.assertEqual(command[-1], "mcp")
        self.assertFalse(any("secret-value" in item for item in command))
        self.assertFalse(any("|" in item or ";" in item for item in command))


class RobinMCPProtocolTests(unittest.TestCase):
    def make_client(self, mode: str = "normal", timeout: float = 2.0):
        script = mcp_server_script(mode) % mode
        return MCPStdioClient(
            [sys.executable, "-u", "-c", script],
            env=os.environ.copy(),
            timeout=timeout,
        )

    def test_initializes_lists_tools_and_calls_tool_over_stdio(self):
        with self.make_client() as client:
            initialized = client.initialize()
            self.assertEqual(initialized["protocolVersion"], "2024-11-05")
            tools = client.list_tools()
            self.assertEqual([item["name"] for item in tools], ["robin_investigate"])
            result = client.call_tool("robin_investigate", {"query": "APT example"})
            self.assertIn("grounded result", result["content"][0]["text"])

    def test_rejects_malformed_jsonrpc_message(self):
        with self.make_client("malformed") as client:
            client.initialize()
            with self.assertRaisesRegex(RobinMCPError, "JSON"):
                client.list_tools()

    def test_rejects_oversized_jsonrpc_message(self):
        with self.make_client("oversized") as client:
            client.initialize()
            with self.assertRaisesRegex(RobinMCPError, "size"):
                client.list_tools()

    def test_tool_call_timeout_is_marked_uncertain(self):
        with self.make_client("hang", timeout=0.15) as client:
            client.initialize()
            client.list_tools()
            with self.assertRaises(RobinMCPError) as caught:
                client.call_tool("robin_investigate", {"query": "APT example"})
            self.assertTrue(caught.exception.uncertain)


class RobinEngineIntegrationTests(unittest.TestCase):
    def test_registry_exposes_high_risk_non_retryable_osint_capability(self):
        registry = json.loads((ROOT / "orchestrator" / "tools.json").read_text())
        self.assertEqual(
            registry["capability:osint"]["default_tool"],
            "robin_mcp",
        )
        spec = registry["robin_mcp"]
        self.assertEqual(spec["required_env"], "GEMINI_API_KEY")
        self.assertEqual(spec["enabled_env"], "ROBIN_ENABLE_TOKEN")
        self.assertEqual(spec["allowed_actions"], ["investigate"])
        self.assertEqual(spec["risk"], "high")
        effect = spec["effect_contracts"]["investigate"]
        self.assertEqual(effect["identity"], "engine")
        self.assertEqual(effect["retry"], "blocked")
        self.assertEqual(effect["reconciliation"], "none")
        self.assertEqual(effect["fencing"], "engine")

    def test_robin_is_unavailable_without_exact_enable_token(self):
        registry = json.loads((ROOT / "orchestrator" / "tools.json").read_text())
        with patch.dict(os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "GEMINI_API_KEY": "configured",
            "ROBIN_ENABLE_TOKEN": "false",
        }, clear=False):
            self.assertFalse(engine_module.tool_available("robin_mcp", registry))
        with patch.dict(os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "GEMINI_API_KEY": "configured",
        }, clear=False):
            os.environ.pop("ROBIN_ENABLE_TOKEN", None)
            self.assertFalse(engine_module.tool_available("robin_mcp", registry))

    def test_robin_is_available_only_when_enabled_and_gemini_key_exists(self):
        registry = json.loads((ROOT / "orchestrator" / "tools.json").read_text())
        with patch.dict(os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "GEMINI_API_KEY": "configured",
            "ROBIN_ENABLE_TOKEN": "true",
        }, clear=False):
            self.assertTrue(engine_module.tool_available("robin_mcp", registry))
        with patch.dict(os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "ROBIN_ENABLE_TOKEN": "true",
        }, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            self.assertFalse(engine_module.tool_available("robin_mcp", registry))

    def test_upstream_submodule_is_pinned_to_audited_commit(self):
        gitmodules = (ROOT / ".gitmodules").read_text()
        self.assertIn("path = vendor/robin", gitmodules)
        self.assertIn("https://github.com/apurvsinghgautam/robin.git", gitmodules)
        entry = subprocess.check_output(
            ["git", "ls-tree", "HEAD", "vendor/robin"],
            cwd=ROOT,
            text=True,
        ).strip()
        self.assertIn("160000 commit " + ROBIN_COMMIT, entry)

    def test_runner_build_is_opt_in_and_uses_pinned_image(self):
        workflow = (ROOT / ".github" / "workflows" / "orchestrator.yml").read_text()
        self.assertIn("vars.ORCHESTRATOR_ROBIN_ENABLED == 'true'", workflow)
        self.assertIn("ai-orchestrator-robin:" + ROBIN_COMMIT[:12], workflow)
        self.assertIn("ROBIN_ENABLE_TOKEN:", workflow)


if __name__ == "__main__":
    unittest.main()
