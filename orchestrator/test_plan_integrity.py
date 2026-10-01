import unittest
from types import SimpleNamespace

from plan_integrity import fingerprint_nodes


class PlanIntegrityTests(unittest.TestCase):
    def node(self, tool="noop", payload=None):
        return SimpleNamespace(
            id="n01",
            capability="execute",
            tool=tool,
            depends_on=[],
            risk="low",
            contract={"required_fields": ["ok"]},
            input={
                "goal": "demo",
                "instruction": "do it",
                "payload": payload or {"x": 1},
                "workflow_id": "wf-a",
                "context": {"volatile": "value"},
                "repair_feedback": {"old": "failure"},
                "approval_granted": True,
            },
            status="ready",
            retry_count=2,
            output={"ok": True},
            error={"message": "old"},
        )

    def test_fingerprint_is_stable_and_ignores_runtime_fields(self):
        first = fingerprint_nodes([self.node()])
        second_node = self.node()
        second_node.status = "running"
        second_node.retry_count = 9
        second_node.output = {"changed": True}
        second_node.error = {"different": True}
        second_node.input["context"] = {"different": "volatile"}
        self.assertEqual(first, fingerprint_nodes([second_node]))

    def test_approval_fingerprint_and_audit_metadata_are_runtime_fields(self):
        first = fingerprint_nodes([self.node()])
        second_node = self.node()
        second_node.input["approval_fingerprint"] = "different-digest"
        second_node.input["approval_actor"] = "reviewer"
        second_node.input["approval_approved_at"] = "2026-10-01T19:00:00+00:00"
        self.assertEqual(first, fingerprint_nodes([second_node]))
    def test_fingerprint_changes_when_plan_definition_changes(self):
        base = fingerprint_nodes([self.node()])
        different_tool = fingerprint_nodes([self.node(tool="connector_bridge")])
        different_input = fingerprint_nodes([self.node(payload={"x": 2})])
        self.assertNotEqual(base, different_tool)
        self.assertNotEqual(base, different_input)


if __name__ == "__main__":
    unittest.main()
