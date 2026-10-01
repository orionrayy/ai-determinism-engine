import json
import os
import unittest
from unittest.mock import patch

from llm_planner import _object_from_text


class PlannerTests(unittest.TestCase):
    def test_extracts_plain_json(self):
        value = _object_from_text('{"nodes": []}')
        self.assertEqual(value, {"nodes": []})

    def test_extracts_fenced_json(self):
        value = _object_from_text('```json\n{"nodes": [{"id": "n01"}]}\n```')
        self.assertEqual(value["nodes"][0]["id"], "n01")


    def test_live_planner_passes_connector_inventory_and_node_fields(self):
        planner_response = {"candidates": [{"content": {"parts": [{"text": json.dumps({"nodes": [{"id": "n01", "capability": "publish", "tool": "connector_bridge", "depends_on": [], "risk": "high", "instruction": "publish", "contract": {}, "artifacts": [], "connector": "notion", "action": "create_page", "payload": {"title": "Hello"}}]})}]}}]}
        registry = {"capability:publish": {"default_tool": "connector_bridge", "fallback_tools": []}, "connector_bridge": {"free_tier": True}}
        inventory = {"notion": {"actions": ["create_page"], "capabilities": ["publish"], "configured": True}}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "planner-key", "ORCHESTRATOR_CONNECTOR_BRIDGE_URL": "https://bridge.example/api/bridge"}, clear=True), patch.object(lp, "discover_capabilities", return_value=inventory), patch.object(lp, "_post", return_value=planner_response) as post:
            nodes = lp.plan_goal("publish a page", registry, FakeNode, fake_validate, live=True)
        self.assertEqual(nodes[0].input["connector"], "notion")
        self.assertEqual(nodes[0].input["action"], "create_page")
        self.assertEqual(nodes[0].input["payload"]["title"], "Hello")
        prompt = post.call_args.args[1]["contents"][0]["parts"][0]["text"]
        self.assertIn("notion", prompt)

    def test_rejects_non_json(self):
        with self.assertRaises(ValueError):
            _object_from_text('no json here')


if __name__ == '__main__':
    unittest.main()