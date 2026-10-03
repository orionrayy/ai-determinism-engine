import unittest

from epistemic_metrics import record_node_metrics


class EpistemicMetricsTests(unittest.TestCase):
    def test_metrics_recompute_deterministically_from_node_rows(self):
        workflow = {}
        record_node_metrics(
            workflow,
            "research",
            research_output={
                "evidence_records": [{"canonical_id": "a"}, {"canonical_id": "b"}],
                "independent_source_count": 2,
                "research_budget": {"name": "fast"},
                "extended_providers_used": ["semantic_scholar"],
            },
        )
        result = record_node_metrics(
            workflow,
            "analysis",
            verdict={
                "claims": [
                    {
                        "claim_id": "c1",
                        "statement": "supported",
                        "material": True,
                        "status": "SUPPORTED_DIRECT",
                        "evidence_refs": ["a"],
                    },
                    {
                        "claim_id": "c2",
                        "statement": "contested",
                        "material": True,
                        "status": "CONTESTED",
                        "evidence_refs": ["b"],
                    },
                ],
                "evidence_records": [
                    {"canonical_id": "a"},
                    {"canonical_id": "b"},
                ],
            },
        )
        self.assertEqual(result["research_nodes"], 1)
        self.assertEqual(result["epistemic_nodes"], 1)
        self.assertEqual(result["material_claims"], 2)
        self.assertEqual(result["supported_material_claims"], 1)
        self.assertEqual(result["evidence_linked_material_claims"], 2)
        self.assertEqual(result["supported_coverage"], 0.5)
        self.assertEqual(result["evidence_coverage"], 1.0)
        self.assertEqual(result["independent_source_count_max"], 2)


if __name__ == "__main__":
    unittest.main()
