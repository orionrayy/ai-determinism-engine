import unittest

from claim_integrity import validate_truth_lock


class ClaimIntegrityTests(unittest.TestCase):
    def test_truth_lock_accepts_lineaged_supported_claim(self):
        source = {
            "claims": [{
                "claim_id": "c1",
                "statement": "Supported claim.",
                "status": "SUPPORTED_DIRECT",
                "evidence_refs": ["e1"],
            }]
        }
        draft = {
            "claims": [{
                "claim_id": "d1",
                "statement": "Supported claim.",
                "status": "SUPPORTED_INDIRECT",
                "material": True,
                "derives_from_claims": ["c1"],
                "evidence_refs": ["e1"],
            }]
        }
        self.assertTrue(validate_truth_lock(draft, source)["passed"])

    def test_truth_lock_rejects_unknown_to_supported_upgrade(self):
        source = {
            "claims": [{
                "claim_id": "c1",
                "status": "UNKNOWN",
                "evidence_refs": [],
            }]
        }
        draft = {
            "claims": [{
                "claim_id": "d1",
                "status": "SUPPORTED_DIRECT",
                "material": True,
                "derives_from_claims": ["c1"],
                "evidence_refs": ["e1"],
            }]
        }
        result = validate_truth_lock(draft, source)
        self.assertFalse(result["passed"])
        self.assertEqual(result["violations"][0]["reason"], "uncertainty_upgrade")

    def test_truth_lock_runtime_shape_can_be_extracted_before_validation(self):
        from orchestrator import extract_first_llm_json

        wrapper = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"claims":[{"claim_id":"c1","status":"SUPPORTED_DIRECT","evidence_refs":["e1"]}]}'
                    }]
                }
            }]
        }
        verdict = extract_first_llm_json(wrapper)
        self.assertEqual(verdict["claims"][0]["claim_id"], "c1")

    def test_truth_lock_rejects_new_material_claim_without_lineage(self):
        source = {
            "claims": [{
                "claim_id": "c1",
                "status": "SUPPORTED_DIRECT",
                "evidence_refs": ["e1"],
            }]
        }
        draft = {
            "claims": [{
                "claim_id": "d1",
                "status": "SUPPORTED_DIRECT",
                "material": True,
                "evidence_refs": ["e1"],
            }]
        }
        result = validate_truth_lock(draft, source)
        self.assertFalse(result["passed"])
        self.assertEqual(result["violations"][0]["reason"], "missing_claim_lineage")


if __name__ == "__main__":
    unittest.main()
