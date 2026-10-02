import json
import tempfile
import unittest
from pathlib import Path

from agent_protocol import build_result, build_task, digest
from aggregate_agent_results import aggregate, FederationProtocolError


class AgentAggregationTests(unittest.TestCase):
    def task(self, task_id="n01"):
        return build_task(
            federation_id="fed-a",
            workflow_id="wf-a",
            task_id=task_id,
            role="researcher",
            capability="research",
            tool="research_bundle",
            risk="low",
            instruction="collect evidence",
            context={"query": "test"},
            contract={},
        )

    def test_aggregate_accepts_exact_matching_results(self):
        tasks = [self.task("n01"), self.task("n02")]
        manifest = {
            "protocol_version": 2,
            "federation_id": "fed-a",
            "workflow_id": "wf-a",
            "tasks": [task.to_dict() for task in tasks],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            results = root / "results"
            for task in tasks:
                folder = results / ("agent-" + task.task_id)
                folder.mkdir(parents=True)
                result = build_result(
                    task,
                    status="completed",
                    output={"task_id": task.task_id},
                ).to_dict()
                (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
            value = aggregate(root / "manifest.json", results)
            self.assertEqual(value["status"], "completed")
            self.assertEqual(value["completed"], 2)

    def test_aggregate_rejects_tampered_result(self):
        task = self.task()
        manifest = {
            "protocol_version": 2,
            "federation_id": "fed-a",
            "workflow_id": "wf-a",
            "tasks": [task.to_dict()],
        }
        result = build_result(task, status="completed", output={"ok": True}).to_dict()
        result["output"] = {"ok": False}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            folder = root / "results" / "agent-n01"
            folder.mkdir(parents=True)
            (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaises(FederationProtocolError):
                aggregate(root / "manifest.json", root / "results")

    def test_aggregate_requires_every_task_result(self):
        tasks = [self.task("n01"), self.task("n02")]
        manifest = {
            "protocol_version": 2,
            "federation_id": "fed-a",
            "workflow_id": "wf-a",
            "tasks": [task.to_dict() for task in tasks],
        }
        result = build_result(tasks[0], status="completed", output={}).to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            folder = root / "results" / "agent-n01"
            folder.mkdir(parents=True)
            (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaises(FederationProtocolError):
                aggregate(root / "manifest.json", root / "results")


if __name__ == "__main__":
    unittest.main()
