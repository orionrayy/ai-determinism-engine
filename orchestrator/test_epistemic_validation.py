import unittest
from epistemic_validation import validate_claims, claim_coverage

class EpistemicValidationTests(unittest.TestCase):
    def test_claims_must_reference_existing_evidence(self):
        evidence=[{"canonical_id":"doi:10.1/a"},{"canonical_id":"arxiv:1234.5678"}]
        result=validate_claims([
            {"claim_id":"c1","statement":"supported","evidence_refs":["doi:10.1/a"],"status":"SUPPORTED_DIRECT"},
            {"claim_id":"c2","statement":"unknown","evidence_refs":["missing"],"status":"UNKNOWN"},
        ], evidence)
        self.assertFalse(result["passed"])
        self.assertEqual(result["invalid_claim_ids"], ["c2"])

    def test_coverage_ignores_unknown_and_counts_material_supported_claims(self):
        result=claim_coverage([
            {"claim_id":"c1","material":True,"status":"SUPPORTED_DIRECT"},
            {"claim_id":"c2","material":True,"status":"CONTESTED"},
            {"claim_id":"c3","material":False,"status":"UNSUPPORTED"},
        ])
        self.assertEqual(result["material_claims"],2)
        self.assertEqual(result["supported_material_claims"],1)
        self.assertEqual(result["coverage"],0.5)

if __name__ == "__main__":
    unittest.main()
