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

    def test_too_small_budget_fails_closed(self):
        with self.assertRaises(ContextBudgetError):
            pack_node_context(
                goal="x",
                dependencies={},
                contract={},
                repair_feedback={},
                max_bytes=1024,
            )

if __name__ == "__main__":
    unittest.main()
