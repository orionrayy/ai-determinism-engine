import unittest

from epistemic_validation import (
    claim_coverage,
    validate_claims,
    validate_evidence_records,
    validate_epistemic_output,
)


class EpistemicValidationTests(unittest.TestCase):
    def test_claims_must_reference_existing_evidence(self):
        evidence = [{"canonical_id": "doi:10.1/a"}, {"canonical_id": "arxiv:1234.5678"}]
        result = validate_claims([
            {
                "claim_id": "c1",
                "statement": "supported",
                "evidence_refs": ["doi:10.1/a"],
                "status": "SUPPORTED_DIRECT",
            },
            {
                "claim_id": "c2",
                "statement": "unknown",
                "evidence_refs": ["missing"],
                "status": "UNKNOWN",
            },
        ], evidence)
        self.assertFalse(result["passed"])
        self.assertEqual(result["invalid_claim_ids"], ["c2"])

    def test_evidence_records_require_unique_canonical_ids(self):
        result = validate_evidence_records([
            {"canonical_id": "doi:10.1/a"},
            {"canonical_id": "doi:10.1/a"},
        ])
        self.assertFalse(result["passed"])
        self.assertEqual(result["invalid_indexes"], [1])

    def test_claim_structure_rejects_duplicate_ids_and_blank_statements(self):
        result = validate_claims([
            {
                "claim_id": "c1",
                "statement": "first",
                "material": True,
                "evidence_refs": [],
                "status": "UNKNOWN",
            },
            {
                "claim_id": "c1",
                "statement": "",
                "material": "yes",
                "evidence_refs": [],
                "status": "UNKNOWN",
            },
        ], [{"canonical_id": "source:a"}])
        self.assertFalse(result["passed"])
        self.assertEqual(result["invalid_claim_ids"], ["c1"])

    def test_coverage_reports_supported_and_evidence_linkage_separately(self):
        result = claim_coverage([
            {
                "claim_id": "c1",
                "material": True,
                "status": "SUPPORTED_DIRECT",
                "evidence_refs": ["a"],
            },
            {
                "claim_id": "c2",
                "material": True,
                "status": "CONTESTED",
                "evidence_refs": ["b"],
            },
            {
                "claim_id": "c3",
                "material": False,
                "status": "UNSUPPORTED",
                "evidence_refs": [],
            },
        ])
        self.assertEqual(result["material_claims"], 2)
        self.assertEqual(result["supported_material_claims"], 1)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["evidence_coverage"], 1.0)

    def test_contested_claim_with_evidence_counts_toward_min_coverage(self):
        result = validate_epistemic_output(
            {
                "claims": [
                    {
                        "claim_id": "c1",
                        "statement": "disputed",
                        "material": True,
                        "status": "CONTESTED",
                        "evidence_refs": ["doi:10.1/a"],
                    },
                    {
                        "claim_id": "c2",
                        "statement": "unknown",
                        "material": True,
                        "status": "UNKNOWN",
                        "evidence_refs": [],
                    },
                ],
                "evidence_records": [{"canonical_id": "doi:10.1/a"}],
            },
            min_coverage=0.5,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["coverage"]["coverage"], 0.0)
        self.assertEqual(result["coverage"]["evidence_coverage"], 0.5)
        self.assertEqual(result["min_coverage_basis"], "evidence_coverage")


if __name__ == "__main__":
    unittest.main()
