from __future__ import annotations

import base64
import json
import unittest
import urllib.error
from unittest.mock import patch

import orchestrator as o


class GitHubEffectReconciliationTests(unittest.TestCase):
    def node(self, action: str, **extra):
        node = o.Node(
            "n01",
            "publish",
            "github",
            [],
            risk="high",
            input={"action": action, "workflow_id": "wf-gh", **extra},
        )
        return node

    def test_create_issue_reconciles_from_marker(self):
        node = self.node("create_issue", title="Task", body="Hello")
        execution_id = o.execution_key({"id": "wf-gh"}, node)
        with patch.object(
            o,
            "github_repository",
            return_value="orionrayy/ai-determinism-engine",
        ), patch.object(
            o,
            "github_headers",
            return_value={},
        ), patch.object(
            o,
            "http_json",
            return_value={
                "data": {"items": [{"number": 77, "title": "Task"}]}
            },
        ) as request:
            result = o.reconcile_github_execution(node, {"id": "wf-gh"}, execution_id)
        self.assertEqual(result["state"], "applied")
        self.assertEqual(result["issue"]["number"], 77)
        self.assertIn("/search/issues", request.call_args.args[0])

    def test_file_reconciliation_requires_exact_content_match(self):
        node = self.node(
            "create_or_update_file",
            path="out.txt",
            branch="main",
            content="hello",
        )
        encoded = base64.b64encode(b"hello").decode("ascii")
        with patch.object(o, "github_repository", return_value="repo"),              patch.object(o, "github_headers", return_value={}),              patch.object(
                 o,
                 "http_json",
                 return_value={"data": {"content": encoded, "sha": "abc"}},
             ):
            result = o.reconcile_github_execution(
                node,
                {"id": "wf-gh"},
                o.execution_key({"id": "wf-gh"}, node),
            )
        self.assertEqual(result["state"], "applied")

    def test_delete_reconciliation_proves_absence_only(self):
        node = self.node("delete_file", path="gone.txt", branch="main")
        missing = urllib.error.HTTPError(
            "https://api.github.com",
            404,
            "Not Found",
            {},
            None,
        )
        with patch.object(o, "github_repository", return_value="repo"),              patch.object(o, "github_headers", return_value={}),              patch.object(o, "http_json", side_effect=missing):
            result = o.reconcile_github_execution(
                node,
                {"id": "wf-gh"},
                o.execution_key({"id": "wf-gh"}, node),
            )
        self.assertEqual(result["state"], "applied")
        self.assertTrue(result["deleted"])

    def test_dispatch_reconciliation_remains_fail_closed(self):
        node = self.node("dispatch_workflow", workflow="target.yml", ref="main")
        result = o.reconcile_github_execution(
            node,
            {"id": "wf-gh"},
            o.execution_key({"id": "wf-gh"}, node),
        )
        self.assertEqual(result["state"], "unknown")
        self.assertIn("no_durable_reconciliation", result["reason"])


if __name__ == "__main__":
    unittest.main()
