import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestrator import TRACE_SCHEMA_VERSION, _trace_envelope, append_event


class TraceContractTests(unittest.TestCase):
    def test_workflow_trace_is_deterministic(self):
        first = _trace_envelope("workflow.started", {"workflow_id": "wf-a"})
        second = _trace_envelope("workflow.started", {"workflow_id": "wf-a"})
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], TRACE_SCHEMA_VERSION)
        self.assertEqual(first["span_kind"], "workflow")
        self.assertIsNone(first["parent_span_id"])

    def test_node_trace_has_workflow_parent(self):
        value = _trace_envelope(
            "node.started",
            {"workflow_id": "wf-a", "node_id": "n01"},
        )
        self.assertEqual(value["span_kind"], "node")
        self.assertIsNotNone(value["parent_span_id"])
        self.assertNotEqual(value["span_id"], value["parent_span_id"])

    def test_agent_trace_is_child_of_node(self):
        value = _trace_envelope(
            "agent.completed",
            {
                "workflow_id": "wf-a",
                "node_id": "n01",
                "agent_id": "agent-a",
            },
        )
        self.assertEqual(value["span_kind"], "agent")
        expected_parent = _trace_envelope(
            "node.completed",
            {"workflow_id": "wf-a", "node_id": "n01"},
        )["span_id"]
        self.assertEqual(value["parent_span_id"], expected_parent)

    def test_event_persists_trace_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("orchestrator.EVENT_DIR", root / "events"), patch(
                "orchestrator.EVENT_FILE", root / "events.jsonl"
            ):
                append_event(
                    "node.completed",
                    {"workflow_id": "wf-a", "node_id": "n01"},
                )
            files = list((root / "events").glob("*.jsonl"))
            self.assertEqual(len(files), 1)
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["trace"]["schema_version"], 1)
            self.assertEqual(payload["trace"]["span_kind"], "node")
            self.assertEqual(payload["payload"]["node_id"], "n01")


if __name__ == "__main__":
    unittest.main()
