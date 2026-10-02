import unittest

from agent_protocol import (
    FEDERATION_PROTOCOL_VERSION,
    FederationProtocolError,
    build_manifest,
    build_result,
    build_task,
    digest,
    task_from_dict,
    validate_result,
)


class AgentProtocolTests(unittest.TestCase):
    def task(self, **overrides):
        values = {
            "federation_id": "fed-a",
            "workflow_id": "wf-a",
            "task_id": "n01",
            "role": "researcher",
            "capability": "research",
            "risk": "low",
            "instruction": "collect evidence",
            "context": {"query": "test"},
            "contract": {"min_sources": 2},
            "attempt": 1,
        }
        values.update(overrides)
        return build_task(**values)

    def test_task_identity_and_digest_are_stable(self):
        first = self.task()
        second = task_from_dict(first.to_dict())
        self.assertEqual(first.agent_id, second.agent_id)
        self.assertEqual(first.input_digest, second.input_digest)
        self.assertEqual(first.protocol_version, FEDERATION_PROTOCOL_VERSION)

    def test_side_effect_roles_are_rejected(self):
        with self.assertRaises(FederationProtocolError):
            self.task(role="operator", capability="execute", risk="medium")

    def test_high_risk_federation_is_rejected(self):
        with self.assertRaises(FederationProtocolError):
            self.task(risk="high")

    def test_tampered_task_digest_is_rejected(self):
        value = self.task().to_dict()
        value["instruction"] = "tampered"
        with self.assertRaises(FederationProtocolError):
            task_from_dict(value)

    def test_result_is_bound_to_exact_task(self):
        task = self.task()
        result = build_result(task, status="completed", output={"items": ["a", "b"]})
        validate_result(result, expected_task=task)
        self.assertEqual(result.output_sha256, digest(result.output))

    def test_tampered_result_is_rejected(self):
        task = self.task()
        result = build_result(task, status="completed", output={"ok": True})
        value = result.to_dict()
        value["output"] = {"ok": False}
        broken = result.__class__(**value)
        with self.assertRaises(FederationProtocolError):
            validate_result(broken, expected_task=task)

    def test_manifest_is_bounded_and_single_workflow(self):
        tasks = [self.task(task_id="n01"), self.task(task_id="n02", role="skeptic", capability="research")]
        manifest = build_manifest("fed-a", tasks)
        self.assertEqual(manifest["workflow_id"], "wf-a")
        self.assertEqual(len(manifest["tasks"]), 2)

    def test_manifest_rejects_duplicate_tasks(self):
        with self.assertRaises(FederationProtocolError):
            build_manifest("fed-a", [self.task(), self.task()])

    def test_oversized_context_is_rejected(self):
        with self.assertRaises(FederationProtocolError):
            self.task(context={"data": "x" * 9000})


if __name__ == "__main__":
    unittest.main()
