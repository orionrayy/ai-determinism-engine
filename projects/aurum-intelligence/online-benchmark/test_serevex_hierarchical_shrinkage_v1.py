import copy
import random
import unittest

import serevex_hierarchical_shrinkage_v1 as shrinkage


PARTITION_ENDS = {
    "development": 2500,
    "model_selection": 3500,
    "calibration": 4000,
    "final_holdout": 5000,
}


def make_ledger(n=5000, informative=True):
    rng = random.Random(912037)
    records, labels = [], {}
    for i in range(n - 12):
        group_a = (i // 24) % 2 == 0
        regime = "trend_up" if group_a else "trend_down"
        volatility = "high" if group_a else "low"
        truth = 0.80 if group_a else 0.20
        y = int(rng.random() < (truth if informative else 0.50))
        labels[i] = y
        prior = [
            (j, labels[j]) for j in range(i)
            if j in labels and j + 6 < i
        ]
        global_labels = [label for _, label in prior]
        rolling_labels = global_labels[-288:]
        regime_labels = [
            label for j, label in prior
            if ("trend_up" if (j // 24) % 2 == 0 else "trend_down") == regime
        ]
        volatility_labels = [
            label for j, label in prior
            if ("high" if (j // 24) % 2 == 0 else "low") == volatility
        ]
        def laplace(values):
            return (sum(values) + 1.0) / (len(values) + 2.0)
        baseline = laplace(global_labels)
        records.append({
            "decision_id": f"d{i}", "decision_index": i, "label_available_index": i + 6,
            "label_available_time": i + 6, "label": y, "label_mature": True,
            "partition": ("development" if i < int(n * 0.50) else
                          "model_selection" if i < int(n * 0.70) else
                          "calibration" if i < int(n * 0.80) else "final_holdout"),
            "baseline_probability": baseline,
            "matured_label_count_before_decision": len(global_labels),
            "conditional_regime_support_before_decision": len(regime_labels),
            "conditional_volatility_support_before_decision": len(volatility_labels),
            "regime_id": regime, "state_tags": {"volatility_bucket": volatility},
            "expert_probabilities": {
                "rolling_matured_rate_288": laplace(rolling_labels),
                "regime_matured_rate": laplace(regime_labels) if len(regime_labels) >= 20 else baseline,
                "volatility_matured_rate": laplace(volatility_labels) if len(volatility_labels) >= 20 else baseline,
            },
        })
    return {"metadata": {"dataset_sha256": "synthetic-test-only"}, "records": records}


class HierarchicalShrinkageTests(unittest.TestCase):
    def test_development_selects_then_two_disjoint_gates_precede_holdout(self):
        report = shrinkage.evaluate_ledger(
            make_ledger(informative=True), as_of_index=5000,
            partition_end_indices=PARTITION_ENDS, bootstrap_replicates=200,
        )
        self.assertEqual(report["candidate_grid"]["candidate_count"], 18)
        self.assertEqual(report["development_selection"]["selection_partition"], "development")
        self.assertFalse(report["development_selection"]["holdout_labels_used_for_selection"])
        self.assertFalse(report["development_selection"]["model_selection_labels_used_for_selection"])
        self.assertFalse(report["development_selection"]["calibration_labels_used_for_selection"])
        self.assertTrue(report["causal_contracts"]["baseline_reproduced_exactly"])
        audit = report["causal_contracts"]["baseline_reproduction_audit"]
        self.assertTrue(audit["verified"])
        self.assertEqual(audit["rows_checked"], len(report_ledger["records"]) if False else 4988)
        self.assertEqual(report["decision_output"], "NO_TRADE")
        self.assertFalse(report["validation"]["model_validated"] if "validation" in report else report["model_validated"])
        self.assertEqual(report["final_holdout"]["rows"], report["partition_rows"]["final_holdout"])
        self.assertIn(report["promotion"]["final_holdout_policy"], ("0.1", "hierarchical_shrinkage_v1"))

    def test_uninformative_experts_fail_closed_to_exact_baseline(self):
        report = shrinkage.evaluate_ledger(
            make_ledger(informative=False), as_of_index=5000,
            partition_end_indices=PARTITION_ENDS, bootstrap_replicates=200,
        )
        self.assertFalse(report["promotion"]["frozen_candidate_authorized_for_research_holdout_comparison"])
        self.assertEqual(report["promotion"]["final_holdout_policy"], "0.1")
        self.assertEqual(report["final_holdout"]["policy_metrics"]["brier"],
                         report["final_holdout"]["baseline_metrics"]["brier"])
        self.assertEqual(report["final_holdout"]["policy_metrics"]["log_loss"],
                         report["final_holdout"]["baseline_metrics"]["log_loss"])

    def test_future_partition_label_changes_cannot_change_development_selection(self):
        original = make_ledger(informative=True)
        changed = copy.deepcopy(original)
        for row in changed["records"]:
            if row["partition"] != "development":
                row["label"] = 1 - row["label"]
        a = shrinkage.evaluate_ledger(original, as_of_index=5000,
                                      partition_end_indices=PARTITION_ENDS, bootstrap_replicates=200)
        b = shrinkage.evaluate_ledger(changed, as_of_index=5000,
                                      partition_end_indices=PARTITION_ENDS, bootstrap_replicates=200)
        self.assertEqual(a["development_selection"]["selected_candidate"],
                         b["development_selection"]["selected_candidate"])
        self.assertEqual(a["development_selection"]["development_selection_sha256"],
                         b["development_selection"]["development_selection_sha256"])
        self.assertNotEqual(a["report_sha256"], b["report_sha256"])

    def test_final_holdout_mutation_does_not_change_any_pre_holdout_gate(self):
        original = make_ledger(informative=True)
        changed = copy.deepcopy(original)
        for row in changed["records"]:
            if row["partition"] == "final_holdout":
                row["label"] = 1 - row["label"]
        a = shrinkage.evaluate_ledger(original, as_of_index=5000,
                                      partition_end_indices=PARTITION_ENDS, bootstrap_replicates=200)
        b = shrinkage.evaluate_ledger(changed, as_of_index=5000,
                                      partition_end_indices=PARTITION_ENDS, bootstrap_replicates=200)
        self.assertEqual(a["promotion"], b["promotion"])
        self.assertEqual(a["model_selection_gate"], b["model_selection_gate"])
        self.assertEqual(a["calibration_gate"], b["calibration_gate"])
        self.assertNotEqual(a["final_holdout"]["frozen_candidate_shadow_metrics"]["brier"],
                            b["final_holdout"]["frozen_candidate_shadow_metrics"]["brier"])

    def test_invalid_maturity_and_partition_boundaries_fail_closed(self):
        ledger = make_ledger(n=500)
        ledger["records"][10]["label_available_index"] = 10
        with self.assertRaises(ValueError):
            shrinkage.evaluate_ledger(ledger, as_of_index=500, partition_end_indices={
                "development": 250, "model_selection": 350, "calibration": 400, "final_holdout": 500,
            }, bootstrap_replicates=200)
        ledger = make_ledger(n=5000)
        with self.assertRaises(ValueError):
            shrinkage.evaluate_ledger(ledger, as_of_index=5000, partition_end_indices={
                "development": 3500, "model_selection": 2500, "calibration": 4000, "final_holdout": 5000,
            }, bootstrap_replicates=200)

    def test_all_candidate_probabilities_are_bounded_and_baseline_is_reference(self):
        ledger = make_ledger(n=500)
        report = shrinkage.evaluate_ledger(ledger, as_of_index=500, partition_end_indices={
            "development": 250, "model_selection": 350, "calibration": 400, "final_holdout": 500,
        }, bootstrap_replicates=200)
        self.assertEqual(report["candidate_grid"]["experts"], list(shrinkage.EXPERT_SUPPORT))
        self.assertEqual(report["candidate_grid"]["k_grid"], [40, 160])
        self.assertEqual(report["candidate_grid"]["max_weight_grid"], [0.2, 0.4, 0.6])
        self.assertEqual(report["promotion"]["reason"],
                         "both independent confirmation gates must pass; otherwise emit exact baseline")
        self.assertFalse(report["source_qualification"]["provenance_verified"])
        self.assertFalse(report["source_qualification"]["historical_feature_availability_verified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
