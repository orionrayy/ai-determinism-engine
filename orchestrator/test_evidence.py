import unittest

from evidence import build_evidence, sha256_value


class EvidenceTests(unittest.TestCase):
    def test_hash_is_deterministic(self):
        self.assertEqual(
            sha256_value({"b": 2, "a": 1}),
            sha256_value({"a": 1, "b": 2}),
        )

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
