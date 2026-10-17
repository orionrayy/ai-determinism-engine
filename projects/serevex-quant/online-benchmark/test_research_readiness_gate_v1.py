"""Regression tests for Serevex research readiness policy."""
import unittest

import research_readiness_gate_v1 as gate


class ResearchReadinessGateTests(unittest.TestCase):
    def setUp(self):
        self.statuses = dict(
            pipeline_status="PASS",
            data_status="QUALIFIED",
            statistical_status="SUPPORTED",
            economic_status="SUPPORTED",
            operations_status="PASS",
        )
        self.refs = {name: f"evidence://ci/{name}" for name in self.statuses}

    def evaluate(self, **overrides):
        args = dict(self.statuses)
        args.update(overrides)
        return gate.evaluate_research_readiness(**args, evidence_refs=self.refs)

    def test_complete_evidence_is_only_eligible_for_separate_review(self):
        r = self.evaluate()
        self.assertEqual(r["status"], "PASS")
        self.assertEqual(r["promotion_status"], "ELIGIBLE_FOR_SEPARATE_RELEASE_REVIEW")
        self.assertFalse(r["validated_model"])

    def test_complete_statuses_do_not_enable_live_execution(self):
        r = gate.evaluate_research_readiness(**self.statuses, evidence_refs=self.refs,
            release_review_status="APPROVED", release_review_reference="evidence://review/approved")
        self.assertEqual(r["promotion_status"], "RELEASE_REVIEW_APPROVED_DEPLOYMENT_STILL_DISABLED")
        self.assertEqual(r["decision_output"], "NO_TRADE")
        self.assertFalse(r["live_signal_enabled"])
        self.assertFalse(r["live_order_execution_enabled"])

    def test_missing_pipeline_evidence_blocks(self):
        refs = dict(self.refs)
        refs.pop("pipeline_status")
        r = gate.evaluate_research_readiness(**self.statuses, evidence_refs=refs)
        self.assertEqual(r["status"], "BLOCKED")
        self.assertIn("pipeline_status_evidence_reference_missing_or_invalid", r["blockers"])

    def test_reference_only_data_blocks(self):
        r = self.evaluate(data_status="REFERENCE_ONLY")
        self.assertEqual(r["status"], "BLOCKED")
        self.assertIn("data_status_reference_only", r["blockers"])

    def test_inconclusive_statistics_block_promotion(self):
        r = self.evaluate(statistical_status="INCONCLUSIVE")
        self.assertEqual(r["promotion_status"], "BLOCKED")

    def test_negative_economics_block_promotion(self):
        r = self.evaluate(economic_status="NEGATIVE")
        self.assertEqual(r["promotion_status"], "BLOCKED")

    def test_failed_pipeline_blocks(self):
        r = self.evaluate(pipeline_status="FAIL")
        self.assertEqual(r["status"], "BLOCKED")

    def test_operations_not_run_blocks(self):
        r = self.evaluate(operations_status="NOT_RUN")
        self.assertEqual(r["status"], "BLOCKED")

    def test_truthy_strings_are_not_accepted_as_statuses(self):
        with self.assertRaises(ValueError):
            self.evaluate(pipeline_status=True)

    def test_unknown_status_is_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(data_status="PASS")

    def test_unknown_evidence_key_is_rejected(self):
        with self.assertRaises(ValueError):
            gate.evaluate_research_readiness(**self.statuses,
                evidence_refs={**self.refs, "model_accuracy": "evidence://fake"})

    def test_malformed_http_reference_is_not_evidence(self):
        refs = dict(self.refs)
        refs["statistical_status"] = "https://user:password@example.invalid/evidence"
        r = gate.evaluate_research_readiness(**self.statuses, evidence_refs=refs)
        self.assertEqual(r["status"], "BLOCKED")
        self.assertIn("statistical_status_evidence_reference_missing_or_invalid", r["blockers"])

    def test_reference_must_be_nonempty_and_stable(self):
        refs = dict(self.refs)
        refs["economic_status"] = "not-a-reference"
        r = gate.evaluate_research_readiness(**self.statuses, evidence_refs=refs)
        self.assertEqual(r["status"], "BLOCKED")

    def test_release_approval_requires_reference(self):
        r = gate.evaluate_research_readiness(**self.statuses, evidence_refs=self.refs,
            release_review_status="APPROVED")
        self.assertEqual(r["promotion_status"], "BLOCKED")
        self.assertIn("release_review_approval_reference_missing_or_invalid", r["blockers"])

    def test_pending_review_does_not_claim_production_readiness(self):
        r = self.evaluate()
        self.assertNotIn("production_ready", r)
        self.assertFalse(r["deployment_calibrated"])

    def test_explicit_rejection_is_preserved(self):
        r = gate.evaluate_research_readiness(**self.statuses, evidence_refs=self.refs,
            release_review_status="REJECTED")
        self.assertEqual(r["promotion_status"], "REJECTED_BY_RELEASE_REVIEW")

    def test_all_outputs_keep_no_trade_boundary(self):
        for update in ({}, {"pipeline_status": "FAIL"}, {"economic_status": "INCONCLUSIVE"}):
            r = self.evaluate(**update)
            self.assertEqual(r["decision_output"], "NO_TRADE")
            self.assertFalse(r["live_signal_enabled"])
            self.assertFalse(r["live_order_execution_enabled"])


if __name__ == "__main__":
    unittest.main()
