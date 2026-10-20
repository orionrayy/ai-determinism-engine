import unittest
import serevex_readiness_audit_v1 as audit

def reports():
    metrics = {"brier": 0.20, "log_loss": 0.60, "ece": 0.01, "positive_rate": 0.35, "rows": 500}
    model = {
        "source_qualification": dict.fromkeys(audit.MODEL_SOURCE_FIELDS, True),
        "baseline_reproduction_audit": {"verified": True, "maximum_absolute_error": 0.0},
        "model_selection_gate": {"passed": True, "status": "PASS"},
        "calibration_gate": {"passed": True, "status": "PASS"},
        "final_holdout": {"policy": "hierarchical_candidate", "baseline_metrics": dict(metrics),
                          "policy_metrics": dict(metrics), "final_holdout_can_change_candidate_choice": False},
        "decision_output": "NO_TRADE", "model_validated": False, "deployment_calibrated": False,
        "live_signal_enabled": False, "live_order_execution_enabled": False,
    }
    source_gates = {n: {"research_qualification_status": "QUALIFIED", "source_promotion_allowed": True}
                    for n in ("primary_price", "secondary_price", "tick_bidask")}
    return {
        "model": model,
        "online_dataset": {"source_qualification_gates": source_gates},
        "tick_bidask": {"source_qualification_gates": {n: {"research_qualification_status": "QUALIFIED"} for n in source_gates},
                        "data_lineage_diagnostic": {"status": "PASS", "source_independence_verified": True}},
        "secondary_crosscheck": {"status": "PASS", "timestamp_semantics_verified": True,
                                 "provider_provenance_independently_verified": True, "intended_use_rights_verified": True},
        "external_context": {"temporal_feature_coverage": {k: {"eligible_rows": 1000, "total_rows": 1000, "coverage_fraction": 1.0}
                                                             for k in ("news_1h", "news_24h", "macro")}},
        "execution_replay": {"status": "VALIDATED", "validated_model": True, "validated_for_profitability": True,
                             "live_execution_authorized": False},
    }

class ReadinessAuditTests(unittest.TestCase):
    def test_complete_research_evidence_never_authorizes_production(self):
        result = audit.audit_reports(reports())
        self.assertEqual(result["overall_status"], "RESEARCH_CANDIDATE_ELIGIBLE_FOR_INDEPENDENT_REVIEW")
        self.assertTrue(result["research_candidate_review_eligible"])
        self.assertFalse(result["production_authorized"])
        self.assertEqual(result["decision_output"], "NO_TRADE")

    def test_failed_gate_must_fall_back_to_baseline(self):
        data = reports()
        data["model"]["model_selection_gate"] = {"passed": False, "status": "DUAL_SCORE_GATE_FAILED"}
        data["model"]["calibration_gate"] = {"passed": False, "status": "NOT_RUN_MODEL_SELECTION_GATE_FAILED"}
        data["model"]["final_holdout"]["policy"] = audit.BASELINE
        data["model"]["source_qualification"]["provenance_verified"] = False
        result = audit.audit_reports(data)
        self.assertEqual(result["model_candidate_status"], "BASELINE_ONLY")
        self.assertFalse(result["research_candidate_review_eligible"])
        self.assertEqual(result["integrity_violation_count"], 0)
        self.assertTrue(any("model_selection_gate_failed" in x for x in result["blockers_by_area"]["model"]))

    def test_failed_gate_cannot_promote_candidate_policy(self):
        data = reports()
        data["model"]["model_selection_gate"] = {"passed": False, "status": "FAIL"}
        data["model"]["calibration_gate"] = {"passed": False, "status": "NOT_RUN_MODEL_SELECTION_GATE_FAILED"}
        result = audit.audit_reports(data)
        self.assertIn("candidate_policy_promoted_without_both_research_gates", result["integrity_violations"])

    def test_final_holdout_is_locked(self):
        data = reports()
        data["model"]["final_holdout"]["final_holdout_can_change_candidate_choice"] = True
        result = audit.audit_reports(data)
        self.assertIn("final_holdout_is_not_locked_against_candidate_selection", result["integrity_violations"])

    def test_policy_metric_must_equal_baseline_on_fallback(self):
        data = reports()
        data["model"]["model_selection_gate"] = {"passed": False, "status": "FAIL"}
        data["model"]["calibration_gate"] = {"passed": False, "status": "NOT_RUN_MODEL_SELECTION_GATE_FAILED"}
        data["model"]["final_holdout"]["policy"] = audit.BASELINE
        data["model"]["final_holdout"]["policy_metrics"]["brier"] += 0.001
        result = audit.audit_reports(data)
        self.assertIn("fallback_policy_differs_from_baseline:brier", result["integrity_violations"])

    def test_live_authority_is_rejected(self):
        data = reports()
        data["model"]["live_signal_enabled"] = True
        result = audit.audit_reports(data)
        self.assertTrue(any("live_signal_enabled" in x for x in result["integrity_violations"]))
        self.assertFalse(result["production_authorized"])

    def test_data_disagreement_and_sparse_context_are_blockers(self):
        data = reports()
        data["online_dataset"]["source_qualification_gates"]["primary_price"]["research_qualification_status"] = "REFERENCE_ONLY"
        data["tick_bidask"]["data_lineage_diagnostic"]["source_independence_verified"] = False
        data["secondary_crosscheck"]["status"] = "SOURCE_DISAGREEMENT_REVIEW_REQUIRED"
        data["external_context"]["temporal_feature_coverage"]["news_1h"]["coverage_fraction"] = 0.02
        result = audit.audit_reports(data)
        self.assertTrue(any("primary_price" in x for x in result["blockers_by_area"]["price_data"]))
        self.assertTrue(any("source_independence_unverified" in x for x in result["blockers_by_area"]["price_data"]))
        self.assertTrue(any("SOURCE_DISAGREEMENT_REVIEW_REQUIRED" in x for x in result["blockers_by_area"]["price_data"]))
        self.assertTrue(any("news_1h" in x for x in result["blockers_by_area"]["external_context"]))

    def test_missing_reports_fail_closed(self):
        result = audit.audit_reports({})
        self.assertEqual(result["overall_status"], "RESEARCH_ONLY_BLOCKED")
        self.assertFalse(result["research_candidate_review_eligible"])
        self.assertFalse(result["production_authorized"])
        self.assertGreater(result["blocker_count"], 0)

if __name__ == "__main__":
    unittest.main()
