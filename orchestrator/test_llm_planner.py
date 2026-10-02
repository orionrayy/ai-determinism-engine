import json
import os
import unittest
from unittest.mock import patch

import llm_planner as lp


class FakeNode:
    def __init__(self, id, capability, tool, depends_on=None, risk="low", input=None, contract=None):
        self.id = id
        self.capability = capability
        self.tool = tool
        self.depends_on = depends_on or []
        self.risk = risk
        self.input = input or {}
        self.contract = contract or {}


def fake_validate(nodes):
    if not nodes:
        raise ValueError("empty")


class PlannerTests(unittest.TestCase):
    def test_extracts_plain_json(self):
        value = lp._object_from_text('{"nodes": []}')
        self.assertEqual(value, {"nodes": []})

    def test_extracts_fenced_json(self):
        value = lp._object_from_text('```json\n{"nodes": [{"id": "n01"}]}\n```')
        self.assertEqual(value["nodes"][0]["id"], "n01")


    def test_planner_rejects_oversized_goal_before_api_call(self):
        registry = {
            "gemini": {
                "free_tier": True,
                "default_model": "gemini-3.8-flash",
                "free_models": ["gemini-3.8-flash"],
            }
        }
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "planner-key",
        }, clear=True), patch.object(lp, "_post") as post:
            with self.assertRaisesRegex(ValueError, "planner goal exceeds 48 KiB"):
                lp.plan_goal("x" * (lp.MAX_PLANNER_GOAL_BYTES + 1), registry, FakeNode, fake_validate)
        post.assert_not_called()

    def test_free_only_rejects_unlisted_planner_model(self):
        registry = {
            "gemini": {
                "free_tier": True,
                "default_model": "gemini-3.8-flash",
                "free_models": ["gemini-3.8-flash"],
            }
        }
        with patch.dict(os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "GEMINI_API_KEY": "planner-key",
            "GEMINI_PLANNER_MODEL": "gemini-3.8-pro",
        }, clear=True), patch.object(lp, "_post") as post:
            with self.assertRaisesRegex(RuntimeError, "not allowed by the free-only model registry"):
                lp.plan_goal("analyze", registry, FakeNode, fake_validate)
        post.assert_not_called()

    def test_free_only_accepts_registry_free_planner_model(self):
        registry = {
            "gemini": {
                "free_tier": True,
                "default_model": "gemini-3.8-flash",
                "free_models": ["gemini-3.8-flash"],
            },
            "capability:analyze": {
                "default_tool": "gemini",
                "fallback_tools": [],
            },
        }
        planner_response = {"candidates": [{"content": {"parts": [{"text":
            json.dumps({"nodes": [{"id": "n01", "capability": "analyze", "tool": "gemini",
            "depends_on": [], "risk": "low", "instruction": "analyze",
            "contract": {}, "artifacts": []}]})
        }]}}]}
        with patch.dict(os.environ, {
            "ORCHESTRATOR_FREE_ONLY": "true",
            "GEMINI_API_KEY": "planner-key",
        }, clear=True), patch.object(lp, "_post", return_value=planner_response) as post:
            nodes = lp.plan_goal("analyze", registry, FakeNode, fake_validate)
        self.assertEqual(nodes[0].tool, "gemini")
        self.assertIn("gemini-3.8-flash", post.call_args.args[0])

    def test_live_planner_passes_connector_inventory_and_node_fields(self):
        planner_response = {"candidates": [{"content": {"parts": [{"text": json.dumps({"nodes": [{"id": "n01", "capability": "publish", "tool": "connector_bridge", "depends_on": [], "risk": "high", "instruction": "publish", "contract": {}, "artifacts": [], "connector": "notion", "action": "create_page", "payload": {"title": "Hello"}}]})}]}}]}
        registry = {"capability:publish": {"default_tool": "connector_bridge", "fallback_tools": []}, "connector_bridge": {"free_tier": True}, "gemini": {"free_tier": True, "default_model": "gemini-3.8-flash", "free_models": ["gemini-3.8-flash"]}}
        inventory = {"notion": {"actions": ["create_page"], "capabilities": ["publish"], "configured": True, "action_specs": {"create_page": {"required": ["title"], "types": {"title": "string"}, "idempotent": True}}}}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "planner-key", "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example/api/bridge"}, clear=True), patch.object(lp, "discover_capabilities", return_value=inventory), patch.object(lp, "_post", return_value=planner_response) as post:
            nodes = lp.plan_goal("publish a page", registry, FakeNode, fake_validate, live=True)
        self.assertEqual(nodes[0].input["connector"], "notion")
        self.assertEqual(nodes[0].input["action"], "create_page")
        self.assertEqual(nodes[0].input["payload"]["title"], "Hello")
        prompt = post.call_args.args[1]["contents"][0]["parts"][0]["text"]
        self.assertIn("notion", prompt)

    def test_post_bounds_response_body(self):
        class FakeResponse:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, size=-1):
                return b"x" * (size if size > 0 else 1)

        with patch.object(lp.urllib.request, "urlopen", return_value=FakeResponse()):
            with self.assertRaisesRegex(RuntimeError, "planner response exceeds 512 KiB"):
                lp._post("https://planner.example", {}, "key")

    def test_rejects_non_json(self):
        with self.assertRaises(ValueError):
            lp._object_from_text('no json here')


if __name__ == '__main__':
    unittest.main()