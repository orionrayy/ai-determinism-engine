import json
import tempfile
import unittest
from pathlib import Path
import serevex_readiness_audit_v1 as audit

def reports():
    metrics = {"brier": 0.20, "log_loss": 0.60, "ece": 0.01, "positive_rate": 0.35, "rows": 500}
    dataset_hash = "a" * 64
    secondary_hash = "b" * 64
    model = {
        "source_qualification": {**dict.fromkeys(audit.MODEL_SOURCE_FIELDS, True), "dataset_sha256": dataset_hash},
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
        "online_dataset": {
            "source": {"sha256": dataset_hash},
            "source_qualification_gates": source_gates,
            "secondary_source_crosscheck": {"dataset_sha256": secondary_hash},
        },
        "tick_bidask": {"primary_dataset": {"sha256": dataset_hash},
                        "secondary_dataset": {"sha256": secondary_hash},
                        "source_qualification_gates": {n: {"research_qualification_status": "QUALIFIED", "source_promotion_allowed": True} for n in source_gates},
                        "data_lineage_diagnostic": {"status": "PASS", "source_independence_verified": True}},
        "secondary_crosscheck": {"status": "PASS", "dataset_sha256": secondary_hash,
                                 "timestamp_semantics_verified": True,
                                 "provider_provenance_independently_verified": True, "intended_use_rights_verified": True},
        "external_context": {
            "temporal_feature_coverage": {k: {"eligible_rows": 1000, "total_rows": 1000, "coverage_fraction": 1.0, "status": "PASS"}
                                         for k in ("news_1h", "news_24h", "macro")},
            "decision_gate": "NO_TRADE", "validated": False,
            "deployment_calibrated": False, "live_signal_enabled": False,
            "ablation_benchmark": {"ablations": {"price_only": {
                "status": "SCORED", "scored_rows_sha256": "c" * 64,
                "labels_sha256": "d" * 64, "effective_context_families": [],
                "omitted_context_families": [],
            }}},
        },
        "execution_replay": {"status": "VALIDATED", "validated_model": True, "validated_for_profitability": True,
                             "decision_output": "NO_TRADE", "live_execution_authorized": False},
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
        data["external_context"]["temporal_feature_coverage"]["news_1h"]["status"] = "PARTIAL"
        result = audit.audit_reports(data)
        self.assertTrue(any("primary_price" in x for x in result["blockers_by_area"]["price_data"]))
        self.assertTrue(any("source_independence_unverified" in x for x in result["blockers_by_area"]["price_data"]))
        self.assertTrue(any("SOURCE_DISAGREEMENT_REVIEW_REQUIRED" in x for x in result["blockers_by_area"]["price_data"]))
        self.assertTrue(any("news_1h" in x for x in result["blockers_by_area"]["external_context"]))

    def test_candidate_cannot_be_promoted_with_unqualified_price_source(self):
        data = reports()
        data["model"]["source_qualification"]["provenance_verified"] = False
        result = audit.audit_reports(data)
        self.assertIn(
            "candidate_policy_promoted_while_price_sources_unqualified",
            result["integrity_violations"],
        )

    def test_null_metrics_do_not_satisfy_fallback_identity(self):
        data = reports()
        data["model"]["model_selection_gate"] = {"passed": False, "status": "FAIL"}
        data["model"]["calibration_gate"] = {
            "passed": False, "status": "NOT_RUN_MODEL_SELECTION_GATE_FAILED"
        }
        data["model"]["final_holdout"]["policy"] = audit.BASELINE
        data["model"]["final_holdout"]["policy_metrics"]["brier"] = None
        data["model"]["final_holdout"]["baseline_metrics"]["brier"] = None
        result = audit.audit_reports(data)
        self.assertIn(
            "fallback_metric_non_numeric:brier",
            result["integrity_violations"],
        )

    def test_artifact_shaped_baseline_only_result_is_a_blocked_research_result(self):
        data = reports()
        model = data["model"]
        model["status"] = "BASELINE_ONLY_MODEL_SELECTION_GATE_FAILED"
        model["source_qualification"] = {
            "provenance_verified": False,
            "data_rights_verified": False,
            "timestamp_semantics_verified": False,
            "historical_feature_availability_verified": False,
            "broker_quote_equivalence_verified": False,
            "dataset_sha256": "a" * 64,
        }
        model["model_selection_gate"] = {"passed": False, "status": "DUAL_SCORE_GATE_FAILED"}
        model["calibration_gate"] = {"passed": False, "status": "NOT_RUN_MODEL_SELECTION_GATE_FAILED"}
        model["final_holdout"]["policy"] = "0.1b"
        model["final_holdout"]["policy_metrics"] = dict(model["final_holdout"]["baseline_metrics"])
        data["online_dataset"]["source_qualification_gates"]["primary_price"] = {
            "research_qualification_status": "REFERENCE_ONLY",
            "execution_qualification_status": "BLOCKED",
            "source_promotion_allowed": False,
        }
        data["online_dataset"]["source_qualification_gates"]["secondary_price"] = {
            "research_qualification_status": "REFERENCE_ONLY",
            "execution_qualification_status": "BLOCKED",
            "source_promotion_allowed": False,
        }
        for gate in data["tick_bidask"]["source_qualification_gates"].values():
            gate["research_qualification_status"] = "REFERENCE_ONLY"
            gate["source_promotion_allowed"] = False
        data["tick_bidask"]["data_lineage_diagnostic"] = {
            "status": "POTENTIAL_SHARED_LINEAGE_OR_PRICE_BASIS_MISLABEL_REVIEW_REQUIRED",
            "source_independence_verified": False,
        }
        data["secondary_crosscheck"] = {
            "status": "SOURCE_DISAGREEMENT_REVIEW_REQUIRED",
            "timestamp_semantics_verified": False,
            "provider_provenance_independently_verified": False,
            "intended_use_rights_verified": False,
        }
        for key, fraction in (("news_1h", 205 / 19894), ("news_24h", 100 / 19894), ("macro", 0.0)):
            data["external_context"]["temporal_feature_coverage"][key] = {
                "coverage_fraction": fraction,
                "eligible_rows": round(fraction * 19894),
                "total_rows": 19894,
                "status": "PARTIAL" if fraction else "NOT_MET",
            }
        data["execution_replay"] = {
            "status": "DIAGNOSTIC_ONLY",
            "validated_model": False,
            "validated_for_profitability": False,
            "live_execution_authorized": False,
        }
        result = audit.audit_reports(data)
        self.assertEqual(result["overall_status"], "RESEARCH_ONLY_BLOCKED")
        self.assertEqual(result["model_candidate_status"], "BASELINE_ONLY")
        self.assertEqual(result["integrity_violation_count"], 0)
        self.assertFalse(result["production_authorized"])
        self.assertFalse(result["research_candidate_review_eligible"])
        self.assertTrue(any("SOURCE_DISAGREEMENT_REVIEW_REQUIRED" in x for x in result["blockers_by_area"]["price_data"]))

    def test_secondary_dataset_hashes_must_bind_across_reports(self):
        data = reports()
        data["tick_bidask"]["secondary_dataset"]["sha256"] = "c" * 64
        result = audit.audit_reports(data)
        self.assertIn(
            "secondary_dataset_hash_mismatch_across_reports",
            result["integrity_violations"],
        )

    def test_dataset_hash_mismatch_is_an_integrity_violation(self):
        data = reports()
        data["model"]["source_qualification"]["dataset_sha256"] = "b" * 64
        result = audit.audit_reports(data)
        self.assertIn(
            "model_ledger_dataset_hash_differs_from_online_benchmark",
            result["integrity_violations"],
        )
        self.assertEqual(result["overall_status"], "INTEGRITY_VIOLATION")

    def test_candidate_holdout_score_bundles_must_use_same_rows_and_labels(self):
        data = reports()
        data["model"]["final_holdout"]["policy_metrics"]["rows"] = 499
        data["model"]["final_holdout"]["policy_metrics"]["positive_rate"] = 0.34
        result = audit.audit_reports(data)
        self.assertIn("holdout_row_count_mismatch", result["integrity_violations"])
        self.assertIn("holdout_label_rate_mismatch", result["integrity_violations"])

    def test_context_ablation_cannot_score_a_sparse_feature_family(self):
        data = reports()
        data["external_context"]["ablation_benchmark"]["ablations"]["price_news"] = {
            "status": "SCORED", "scored_rows_sha256": "c" * 64,
            "labels_sha256": "d" * 64, "effective_context_families": ["news"],
            "omitted_context_families": [],
            "context_family_support_by_split": {
                "news": {
                    split: {"total_rows": 1000, "eligible_values_by_feature": {"feature_news_count_1h": 10}}
                    for split in ("development", "calibration", "holdout")
                }
            },
            "paired_moving_block_ci_vs_price_only": {
                "comparison_reference": "price_only", "lower_95": -0.01,
                "upper_95": 0.02, "replicates": 40,
            },
        }
        result = audit.audit_reports(data)
        self.assertIn(
            "context_ablation_scored_below_feature_coverage_floor:price_news:news:holdout",
            result["integrity_violations"],
        )

    def test_coverage_counts_and_fraction_must_reconcile(self):
        data = reports()
        data["external_context"]["temporal_feature_coverage"]["news_1h"]["coverage_fraction"] = 0.5
        result = audit.audit_reports(data)
        self.assertIn("temporal_coverage_fraction_mismatch:news_1h", result["integrity_violations"])
        self.assertTrue(any("temporal_coverage_not_passed:news_1h" in x
                            for x in result["blockers_by_area"]["external_context"]))

    def test_execution_report_cannot_enable_live_authority(self):
        data = reports()
        data["execution_replay"]["live_execution_authorized"] = True
        result = audit.audit_reports(data)
        self.assertIn("unsafe_execution_authority_state", result["integrity_violations"])
        self.assertFalse(result["production_authorized"])

    def test_missing_required_source_names_are_blockers(self):
        data = reports()
        data["online_dataset"]["source_qualification_gates"] = {
            "primary_price": {"research_qualification_status": "QUALIFIED", "source_promotion_allowed": True}
        }
        data["tick_bidask"]["source_qualification_gates"].pop("tick_bidask")
        result = audit.audit_reports(data)
        self.assertTrue(any("online_required_source_gate_missing:secondary_price" == x for x in result["blockers_by_area"]["price_data"]))
        self.assertTrue(any("tick_required_source_gate_missing:tick_bidask" == x for x in result["blockers_by_area"]["price_data"]))

    def test_malformed_artifact_is_recorded_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as root:
            benchmark_dir = Path(root) / "input"
            output_dir = Path(root) / "output"
            benchmark_dir.mkdir()
            (benchmark_dir / "online_dataset_benchmark.json").write_text("{not json", encoding="utf-8")
            result = audit.run_audit(benchmark_dir, output_dir)
            self.assertEqual(result["input_errors"]["online_dataset"], "invalid_json_or_encoding")
            self.assertTrue(any("benchmark_artifact_unreadable:online_dataset" in x for x in result["blockers_by_area"]["price_data"]))
            self.assertTrue((output_dir / "serevex_readiness_audit.json").is_file())
            self.assertTrue((output_dir / "readiness_summary.md").is_file())

    def test_missing_reports_fail_closed(self):
        result = audit.audit_reports({})
        self.assertEqual(result["overall_status"], "RESEARCH_ONLY_BLOCKED")
        self.assertFalse(result["research_candidate_review_eligible"])
        self.assertFalse(result["production_authorized"])
        self.assertGreater(result["blocker_count"], 0)

if __name__ == "__main__":
    unittest.main()
