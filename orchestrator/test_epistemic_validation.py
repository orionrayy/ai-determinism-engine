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

    def test_claim_coverage_exposes_supported_coverage(self):
        result = claim_coverage([
            {
                "claim_id": "c1",
                "material": True,
                "status": "SUPPORTED_DIRECT",
                "evidence_refs": ["a"],
            },
        ])
        self.assertEqual(result["supported_coverage"], 1.0)

    def test_trusted_evidence_boundary_rejects_untrusted_claim_reference(self):
        result = validate_epistemic_output(
            {
                "confidence": 0.9,
                "claims": [{
                    "claim_id": "c1",
                    "statement": "supported",
                    "material": True,
                    "status": "SUPPORTED_DIRECT",
                    "evidence_refs": ["doi:untrusted"],
                }],
                "evidence_records": [{"canonical_id": "doi:untrusted"}],
            },
            trusted_evidence_records=[
                {"canonical_id": "doi:trusted"},
            ],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["evidence"]["reason"],
            "evidence_refs_crossed_trust_boundary",
        )

    def test_trusted_evidence_binding_canonicalizes_agent_record_metadata(self):
        result = validate_epistemic_output(
            {
                "confidence": 0.9,
                "claims": [{
                    "claim_id": "c1",
                    "statement": "supported",
                    "material": True,
                    "status": "SUPPORTED_DIRECT",
                    "evidence_refs": ["doi:trusted"],
                }],
                "evidence_records": [{
                    "canonical_id": "doi:trusted",
                    "authority_score": 0.01,
                }],
            },
            trusted_evidence_records=[
                {
                    "canonical_id": "doi:trusted",
                    "authority_score": 0.95,
                    "publication_status": "normal",
                },
            ],
        )
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["bound_evidence_records"][0]["authority_score"],
            0.95,
        )

    def test_epistemic_output_limits_are_enforced(self):
        evidence = [
            {"canonical_id": f"doi:10.1/{index}"}
            for index in range(65)
        ]
        result = validate_epistemic_output({
            "claims": [],
            "evidence_records": evidence,
        })
        self.assertFalse(result["passed"])
        self.assertIn("exceeds limit", result["reason"])

    def test_high_confidence_sufficient_evidence_does_not_abstain(self):
        result = validate_epistemic_output({
            "confidence": 0.95,
            "claims": [{
                "claim_id": "c1",
                "statement": "supported",
                "material": True,
                "status": "SUPPORTED_DIRECT",
                "evidence_refs": ["doi:10.1/a"],
            }],
            "evidence_records": [{"canonical_id": "doi:10.1/a"}],
        })
        self.assertTrue(result["passed"])
        self.assertFalse(result["selective_abstention"])

    def test_selective_abstention_blocks_validation(self):
        result = validate_epistemic_output({
            "confidence": 0.95,
            "claims": [{
                "claim_id": "c1",
                "statement": "unsupported",
                "material": True,
                "status": "UNKNOWN",
                "evidence_refs": [],
            }],
            "evidence_records": [{"canonical_id": "doi:10.1/a"}],
        })
        self.assertTrue(result["selective_abstention"])
        self.assertFalse(result["passed"])

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
