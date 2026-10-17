import unittest

from source_qualification_gate_v1 import (
    ALL_EVIDENCE_FIELDS,
    RESEARCH_EVIDENCE_FIELDS,
    evaluate_source_qualification,
)


def complete_evidence(value=True):
    return {field: value for field in ALL_EVIDENCE_FIELDS}


def complete_refs():
    return {field: "https://evidence.example/audit/" + field for field in ALL_EVIDENCE_FIELDS}


class SourceQualificationGateTests(unittest.TestCase):
    def test_missing_provenance_and_rights_are_reference_only(self):
        report = evaluate_source_qualification("xauusd-community-primary")
        self.assertEqual(report["research_qualification_status"], "REFERENCE_ONLY")
        self.assertEqual(report["execution_qualification_status"], "BLOCKED")
        self.assertFalse(report["source_promotion_allowed"])
        self.assertFalse(report["live_execution_authorized"])
        self.assertEqual(report["decision_output"], "NO_TRADE")
        self.assertIn("provenance_verified", report["research_missing_evidence"])
        self.assertIn("intended_use_rights_verified", report["research_missing_evidence"])

    def test_truth_requires_a_stable_reference_for_each_field(self):
        report = evaluate_source_qualification(
            "feed-a", complete_evidence(), evidence_refs={}
        )
        self.assertEqual(report["research_qualification_status"], "REFERENCE_ONLY")
        self.assertEqual(set(report["research_missing_evidence"]), set(RESEARCH_EVIDENCE_FIELDS))
        self.assertTrue(all(
            check["verified"] is False for check in report["checks"].values()
        ))
        self.assertTrue(all(
            field + "_evidence_reference_missing" in report["blockers"]
            for field in ALL_EVIDENCE_FIELDS
        ))

    def test_research_evidence_does_not_imply_execution_qualification(self):
        evidence = complete_evidence(False)
        refs = complete_refs()
        for field in RESEARCH_EVIDENCE_FIELDS:
            evidence[field] = True
        report = evaluate_source_qualification("feed-a", evidence, refs)
        self.assertEqual(
            report["research_qualification_status"], "QUALIFIED_FOR_RESEARCH_REVIEW"
        )
        self.assertEqual(report["execution_qualification_status"], "BLOCKED")
        self.assertTrue(report["source_promotion_allowed"])
        self.assertFalse(report["live_execution_authorized"])
        self.assertEqual(len(report["execution_missing_evidence"]), 3)

    def test_all_referenced_evidence_is_only_eligible_for_separate_review(self):
        report = evaluate_source_qualification(
            "broker-feed", complete_evidence(), complete_refs()
        )
        self.assertEqual(
            report["research_qualification_status"], "QUALIFIED_FOR_RESEARCH_REVIEW"
        )
        self.assertEqual(
            report["execution_qualification_status"], "ELIGIBLE_FOR_SEPARATE_EXECUTION_REVIEW"
        )
        self.assertFalse(report["live_execution_authorized"])
        self.assertEqual(report["decision_output"], "NO_TRADE")

    def test_truthy_strings_and_numbers_are_rejected(self):
        for invalid in ("true", "yes", 1, None):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "provenance_verified_must_be_boolean"):
                    evaluate_source_qualification(
                        "feed-a", {"provenance_verified": invalid}
                    )

    def test_unreferenced_or_non_https_reference_does_not_qualify(self):
        for reference in ("", "source-name", "http://insecure.example", 123, None):
            with self.subTest(reference=reference):
                evidence = complete_evidence()
                refs = complete_refs()
                refs["provenance_verified"] = reference
                report = evaluate_source_qualification("feed-a", evidence, refs)
                self.assertFalse(report["checks"]["provenance_verified"]["verified"])
                self.assertIn("provenance_verified", report["research_missing_evidence"])

    def test_unknown_fields_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown_evidence_fields:performance_verified"):
            evaluate_source_qualification("feed-a", {"performance_verified": True})

    def test_empty_identity_and_invalid_container_types_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "source_id_required"):
            evaluate_source_qualification(" ")
        with self.assertRaisesRegex(ValueError, "evidence_must_be_mapping"):
            evaluate_source_qualification("feed-a", ["provenance_verified"])
        with self.assertRaisesRegex(ValueError, "evidence_refs_must_be_mapping"):
            evaluate_source_qualification("feed-a", {}, [])

    def test_every_known_source_evidence_field_has_a_check(self):
        report = evaluate_source_qualification("feed-a", {}, {})
        self.assertEqual(set(report["checks"]), set(ALL_EVIDENCE_FIELDS))
        self.assertEqual(len(report["research_missing_evidence"]), 5)
        self.assertEqual(len(report["execution_missing_evidence"]), 3)


if __name__ == "__main__":
    unittest.main()
