import unittest
from context_budget import ContextBudgetError, pack_node_context

class ContextBudgetTests(unittest.TestCase):
    def test_total_context_is_bounded(self):
        deps = {
            f"n{i:02d}": {
                "capability": "analyze",
                "tool": "gemini",
                "status": "completed",
                "output": {"text": "x" * 10000},
                "error": {},
                "evidence_sha256": "a" * 64,
            }
            for i in range(20)
        }
        ctx = pack_node_context(
            goal="goal",
            dependencies=deps,
            contract={"required_fields": ["result"]},
            repair_feedback={},
        )
        self.assertLessEqual(
            len(pack_node_context.__globals__["canonical_json"](ctx)),
            48 * 1024,
        )
        self.assertTrue(ctx["context_budget"]["omitted_dependencies"] or any(
            item.get("output_truncated") or item.get("omitted")
            for item in ctx["dependencies"].values()
        ))

    def test_context_digest_is_stable_and_excludes_mutable_byte_counter(self):
        deps = {
            "n01": {
                "capability": "research",
                "tool": "wikipedia",
                "status": "completed",
                "output": {"answer": "42"},
                "error": {},
                "evidence_sha256": "b" * 64,
            }
        }
        ctx = pack_node_context(
            goal="verify digest",
            dependencies=deps,
            contract={"required_fields": ["answer"]},
            repair_feedback={},
        )
        stored = ctx["context_digest"]
        digestable = dict(ctx)
        digestable.pop("context_digest", None)
        budget = dict(digestable["context_budget"])
        budget.pop("used_bytes", None)
        digestable["context_budget"] = budget
        actual = pack_node_context.__globals__["digest"](digestable)
        self.assertEqual(stored, actual)
        self.assertLessEqual(
            ctx["context_budget"]["used_bytes"],
            ctx["context_budget"]["max_bytes"],
        )

    def test_too_small_budget_fails_closed(self):
        with self.assertRaises(ContextBudgetError):
            pack_node_context(
                goal="x",
                dependencies={},
                contract={},
                repair_feedback={},
                max_bytes=1024,
            )
    def test_evidence_records_are_compacted_as_valid_structure(self):
        from context_budget import pack_node_context
        output = {
            "query": "topic",
            "independent_source_count": 2,
            "evidence_records": [
                {"canonical_id": "doi:10.1000/a", "title": "A", "providers": ["openalex"]},
                {"canonical_id": "doi:10.1000/b", "title": "B", "providers": ["crossref"]},
            ],
            "errors": {"core": "unavailable"},
        }
        packed = pack_node_context(
            goal="research",
            dependencies={"n1": {
                "capability": "research",
                "tool": "research_bundle",
                "status": "completed",
                "output": output,
            }},
            contract={},
            repair_feedback={},
            dependency_bytes=700,
        )
        value = packed["dependencies"]["n1"]["output"]
        self.assertIsInstance(value, dict)
        self.assertIn("evidence_records", value)
        self.assertIsInstance(value["evidence_records"], list)
        self.assertIn("independent_source_count", value)


if __name__ == "__main__":
    unittest.main()