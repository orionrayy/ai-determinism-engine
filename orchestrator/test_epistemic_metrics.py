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
        self.assertEqual(result["distinct_evidence_work_count_max"], 2)
        self.assertFalse(result["calibration"]["available"])


    def test_independence_is_recomputed_from_records(self):
        workflow = {}
        result = record_node_metrics(
            workflow,
            "analysis",
            verdict={
                "claims": [],
                "independent_source_count": 999,
                "evidence_records": [
                    {"doi": "10.1234/x", "provider": "openalex"},
                    {"doi": "10.1234/x", "provider": "semantic_scholar"},
                    {"doi": "10.5678/y", "provider": "crossref"},
                ],
            },
        )
        self.assertEqual(result["independent_source_count_max"], 2)

    def test_explicit_outcome_enables_calibration(self):
        workflow = {}
        result = record_node_metrics(
            workflow,
            "analysis",
            verdict={
                "claims": [],
                "confidence": 0.9,
                "correct": True,
                "evidence_records": [],
            },
        )
        self.assertTrue(result["calibration"]["available"])
        self.assertEqual(result["calibration"]["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
