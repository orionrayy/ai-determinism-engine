from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import orchestrator as o
from state_schema import StateSchemaError, migrate_state, MAX_NODES


class PersistenceInvariantTests(unittest.TestCase):
    def test_state_schema_rejects_node_overflow(self):
        nodes = [{"id": f"n{i:02d}"} for i in range(MAX_NODES + 1)]
        with self.assertRaises(StateSchemaError):
            migrate_state(
                {"version": 4, "workflows": {"wf": {"id": "wf", "nodes": nodes}}}
            )

    def test_workflow_writer_enforces_reader_size_limit(self):
        node = {"id": "n1", "capability": "research", "tool": "noop"}
        workflow = {"id": "wf", "nodes": [node]}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, "STATE_DIR", Path(tmp)), patch.object(
                o, "MAX_WORKFLOW_SHARD_BYTES", 100
            ):
                with self.assertRaisesRegex(RuntimeError, "exceeds 100 bytes"):
                    o._write_workflow_shard(workflow)

    def test_oversized_event_preserves_trace_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {
                "workflow_id": "wf-trace",
                "node_id": "n01",
                "agent_id": "agent-1",
                "large": "x" * (o.MAX_EVENT_PAYLOAD_BYTES + 1024),
            }
            with patch.object(o, "EVENT_DIR", root / "events"), patch.object(
                o, "EVENT_FILE", root / "events.jsonl"
            ):
                o.append_event("node.completed", payload)

            files = list((root / "events").glob("*.jsonl"))
            self.assertEqual(len(files), 1)
            entry = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertTrue(entry["payload"]["truncated"])
            self.assertEqual(entry["trace"]["span_kind"], "agent")
            self.assertEqual(
                entry["trace"]["parent_span_id"],
                o._trace_envelope("node.completed", payload)["parent_span_id"],
            )
            self.assertTrue(entry["trace"]["trace_id"])


if __name__ == "__main__":
    unittest.main()
