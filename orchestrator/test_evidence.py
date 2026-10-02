import unittest

from evidence import build_evidence, sha256_value


class EvidenceTests(unittest.TestCase):
    def test_hash_is_deterministic(self):
        self.assertEqual(
            sha256_value({"b": 2, "a": 1}),
            sha256_value({"a": 1, "b": 2}),
        )

    def test_evidence_summary_redacts_sensitive_keys_but_keeps_raw_hash(self):
        output = {
            "bridge_job_id": "job-1",
            "access_token": "TOP-SECRET",
            "nested": {"api_key": "SECRET-2", "value": 7},
        }
        item = build_evidence(
            "wf1", "n1", "execute", "connector_bridge", output,
            {"passed": True}, artifacts=[],
        )
        self.assertEqual(item["output_sha256"], sha256_value(output))
        self.assertIn("job-1", item["output_summary"])
        self.assertNotIn("TOP-SECRET", item["output_summary"])
        self.assertNotIn("SECRET-2", item["output_summary"])
        self.assertIn("redacted", item["output_summary"])

    def test_durable_trace_is_redacted_and_secret_patterns_are_removed(self):
        from evidence import sanitize_for_durable

        value = {
            "message": "request failed Authorization: Bearer TOPSECRET",
            "traceback": "Traceback ... api_key=TRACESECRET",
            "nested": {"secret": "VALUE"},
        }
        sanitized = sanitize_for_durable(value)
        self.assertNotIn("TOPSECRET", str(sanitized))
        self.assertNotIn("TRACESECRET", str(sanitized))
        self.assertEqual(sanitized["traceback"]["reason"], "diagnostic_trace")
        self.assertEqual(sanitized["nested"]["secret"]["redacted"], True)

    def test_evidence_contains_output_and_record_hashes(self):
        item = build_evidence(
            "wf1",
            "n1",
            "research",
            "noop",
            {"answer": 42},
            {"passed": True},
            artifacts=[],
        )
        self.assertEqual(len(item["output_sha256"]), 64)
        self.assertEqual(len(item["evidence_sha256"]), 64)
        self.assertIn("answer", item["output_summary"])


if __name__ == "__main__":
    unittest.main()
