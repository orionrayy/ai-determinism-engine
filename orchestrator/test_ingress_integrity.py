import unittest

from orchestrator import build_ingress_intent_digest, validate_ingress_identity


class IngressIntegrityTests(unittest.TestCase):
    def digest(self, **overrides):
        values = {
            "goal": "execute a deterministic workload",
            "live": True,
            "external_workflow_id": "external-1",
            "external_domain": "notion",
            "external_operation": "create_page",
            "intent_fingerprint": "a" * 64,
            "input_digest": "b" * 64,
            "private_input_ref": "c" * 64,
        }
        values.update(overrides)
        return build_ingress_intent_digest(**values)

    def test_digest_changes_when_semantic_intent_changes(self):
        first = self.digest()
        self.assertNotEqual(first, self.digest(goal="different workload"))
        self.assertNotEqual(first, self.digest(external_operation="update_page"))
        self.assertNotEqual(first, self.digest(input_digest="d" * 64))

    def test_digest_ignores_attempt_number_because_it_is_not_semantic(self):
        first = self.digest()
        self.assertEqual(first, self.digest())

    def test_ingress_rejects_digest_conflict(self):
        workflow = {
            "goal": "execute a deterministic workload",
            "live": True,
            "external_workflow_id": "external-1",
            "external_domain": "notion",
            "external_operation": "create_page",
            "intent_fingerprint": "a" * 64,
            "input_digest": "b" * 64,
            "private_input_ref": "c" * 64,
        }
        validate_ingress_identity(
            workflow,
            supplied_intent_digest=self.digest(),
        )
        with self.assertRaisesRegex(RuntimeError, "request intent"):
            validate_ingress_identity(
                workflow,
                supplied_intent_digest=self.digest(goal="tampered"),
            )

    def test_legacy_workflow_is_checked_against_derived_digest(self):
        workflow = {
            "goal": "legacy request",
            "live": False,
            "external_workflow_id": "",
            "external_domain": "",
            "external_operation": "",
            "intent_fingerprint": "",
            "input_digest": "",
            "private_input_ref": "",
        }
        expected = build_ingress_intent_digest(
            goal="legacy request",
            live=False,
        )
        validate_ingress_identity(
            workflow,
            supplied_intent_digest=expected,
        )
        with self.assertRaises(RuntimeError):
            validate_ingress_identity(
                workflow,
                supplied_intent_digest=build_ingress_intent_digest(
                    goal="tampered",
                    live=False,
                ),
            )


if __name__ == "__main__":
    unittest.main()
