import copy
import random
import unittest

import serevex_baseline_refinement_v0_1a_v5_0a as refinement


def make_ledger(n=2400, informative=True):
    rng = random.Random(912037)
    records = []
    for i in range(n):
        group_a = (i // 24) % 2 == 0
        regime = "trend_up" if group_a else "trend_down"
        volatility = "high" if group_a else "low"
        if informative:
            y = int(rng.random() < (0.90 if group_a else 0.10))
            specialist = 0.90 if group_a else 0.10
        else:
            y = int(rng.random() < 0.5)
            specialist = 0.50
        records.append({
            "decision_id": f"d{i}", "decision_index": i, "label_available_index": i + 12,
            "label_available_time": i + 12, "label": y, "label_mature": True,
            "partition": "development" if i < int(n * 0.5) else "final_holdout",
            "baseline_probability": 0.55 if informative else 0.50,
            "matured_label_count_before_decision": min(i, 288),
            "conditional_regime_support_before_decision": min(i // 2, 500),
            "conditional_volatility_support_before_decision": min(i // 2, 500),
            "regime_id": regime, "state_tags": {"volatility_bucket": volatility},
            "expert_probabilities": {
                "rolling_matured_rate_288": specialist,
                "regime_matured_rate": specialist,
                "volatility_matured_rate": specialist,
            },
        })
    return {"metadata": {"dataset_sha256": "synthetic-test-only"}, "records": records}


class BaselineRefinementTests(unittest.TestCase):
    def test_version_reset_maps_legacy_v45_recipe_to_v01a(self):
        ledger = make_ledger(n=1000)
        row = refinement._valid_records(ledger["records"], 1000)[500]
        predictions = refinement._candidate_row(row)
        self.assertEqual(refinement.VERSION_ORDER[0], "0.1a")
        self.assertEqual(refinement.VERSION_ORDER[-1], "5.0a")
        self.assertEqual(len(refinement.VERSION_ORDER), 50)
        self.assertAlmostEqual(refinement.LEGACY_V45_SCALE, 19.0 / 36.0)
        self.assertIn("0.1a", predictions)
        self.assertEqual(refinement._version_spec(1)["renamed_from"], "legacy v4.5")
        self.assertEqual(refinement._version_spec(1)["promotion_status"], "exploratory_anchor_not_validated")

    def test_reproduces_causal_reference_and_no_trade_contract(self):
        report = refinement.evaluate_ledger(make_ledger(n=1200, informative=False), as_of_index=1200, bootstrap_replicates=100)
        self.assertEqual(report["version_order"], [f"{i / 10:.1f}a" for i in range(1, 51)])
        self.assertEqual(len(report["versions"]), 50)
        self.assertEqual(report["anchors"][refinement.CAUSAL_BASELINE]["brier"], 0.25)
        self.assertEqual(report["validation"]["baseline_reproduction"], "PASS")
        self.assertEqual(report["validation"]["decision_output"], "NO_TRADE")
        self.assertFalse(report["validation"]["model_validated"])
        self.assertFalse(report["validation"]["live_signal_enabled"])
        self.assertFalse(report["validation"]["live_order_execution_enabled"])
        self.assertFalse(report["prequential_governor"]["future_label_used_for_own_decision"])
        self.assertIs(report["prequential_governor"]["candidate_gate"]["final_holdout_used_for_static_parameter_selection"], False)

    def test_strong_causal_specialist_can_pass_governor_on_synthetic_data(self):
        report = refinement.evaluate_ledger(make_ledger(n=2400, informative=True), as_of_index=2400, bootstrap_replicates=100)
        selected = report["prequential_governor"]["selected_version_counts"]
        self.assertGreater(sum(count for version, count in selected.items() if version in refinement.CANDIDATE_VERSIONS), 0)
        self.assertGreater(report["prequential_governor"]["incremental_audit"]["selection_look_count"], 0)

    def test_harmful_specialists_fall_back_to_exact_causal_baseline(self):
        ledger = make_ledger(n=2400, informative=True)
        for row in ledger["records"]:
            group_a = (row["decision_index"] // 24) % 2 == 0
            wrong = 0.10 if group_a else 0.90
            row["expert_probabilities"] = {key: wrong for key in refinement.EXPERT_KEYS}
        report = refinement.evaluate_ledger(ledger, as_of_index=2400, bootstrap_replicates=100)
        selected = report["prequential_governor"]["selected_version_counts"]
        self.assertNotIn("0.1a", selected)
        self.assertGreater(selected.get(refinement.CAUSAL_BASELINE, 0), 0)
        metrics = report["versions"][refinement.GOVERNED_VERSION]
        baseline = report["anchors"][refinement.CAUSAL_BASELINE]
        self.assertAlmostEqual(metrics["brier"], baseline["brier"], places=12)
        self.assertAlmostEqual(metrics["log_loss"], baseline["log_loss"], places=12)

    def test_future_label_mutation_cannot_change_completed_prefix_trace(self):
        original = make_ledger(n=1800, informative=True)
        changed = copy.deepcopy(original)
        cutoff = 1000
        for row in changed["records"]:
            if row["decision_index"] >= cutoff:
                row["label"] = 1 - row["label"]
        report_a = refinement.evaluate_ledger(original, as_of_index=1800, bootstrap_replicates=100)
        report_b = refinement.evaluate_ledger(changed, as_of_index=1800, bootstrap_replicates=100)
        traces_a = report_a["prequential_governor"]["selection_trace_block_sha256"]
        traces_b = report_b["prequential_governor"]["selection_trace_block_sha256"]
        prefix_blocks = [key for key in traces_a if (int(key) + 1) * refinement.BLOCK_PERIOD <= cutoff - refinement.EMBARGO_BARS]
        self.assertTrue(prefix_blocks)
        for key in prefix_blocks:
            self.assertEqual(traces_a[key], traces_b[key], f"future label mutation changed earlier selection block {key}")

    def test_nonfinite_expert_and_duplicate_indices_fail_closed(self):
        ledger = make_ledger(n=50)
        ledger["records"][4]["expert_probabilities"][refinement.EXPERT_KEYS[0]] = float("nan")
        with self.assertRaises(ValueError):
            refinement.evaluate_ledger(ledger, as_of_index=50, bootstrap_replicates=100)
        ledger = make_ledger(n=50)
        ledger["records"][5]["decision_index"] = ledger["records"][4]["decision_index"]
        with self.assertRaises(ValueError):
            refinement.evaluate_ledger(ledger, as_of_index=50, bootstrap_replicates=100)

    def test_gate_is_cached_without_new_block_evidence(self):
        governor = refinement._Governor({})
        first = governor.select()
        second = governor.select()
        self.assertIs(first, second)
        self.assertEqual(governor.selection_look_count, 0)
        self.assertEqual(first["selected"], refinement.CAUSAL_BASELINE)

    def test_paired_bootstrap_labels_full_and_source_window_estimands_separately(self):
        rows, reference, candidate = [], [], []
        for index in range(240):
            source_window = index % refinement.BLOCK_PERIOD < refinement.BLOCK_SPAN_BARS
            label = 0 if source_window else 1
            rows.append({"decision_index": index, "label": label})
            reference.append(0.50)
            # Slight harm in scored source windows, strong gain in embargo windows:
            # the full-cohort sign intentionally differs from the bootstrap estimand.
            candidate.append(0.60 if source_window else 0.99)
        report = refinement._paired_block_bootstrap(
            rows, reference, candidate, replicates=100, seed=20261020,
        )
        self.assertEqual(report["scored_rows_full_cohort"], 240)
        self.assertEqual(report["source_window_rows"], 120)
        self.assertEqual(report["source_window_blocks"], 10)
        self.assertGreater(report["full_cohort_brier_mean_improvement"], 0.0)
        self.assertLess(report["brier_mean_improvement"], 0.0)
        self.assertLess(report["brier_improvement_ci95"][1], 0.0)
        self.assertEqual(report["bootstrap_point_estimand"], "equal_weight_mean_of_non_embargo_source_window_block_means")
        self.assertEqual(report["bootstrap_resampling"], "shared_paired_block_indices_for_brier_and_log_loss")
        self.assertEqual(report["blocks"], 10)
        self.assertEqual(report["replicates"], 100)

    def test_calibration_diagnostics_are_emitted_for_multiple_slices(self):
        report = refinement.evaluate_ledger(make_ledger(n=1200, informative=True), as_of_index=1200, bootstrap_replicates=100)
        self.assertIn("multi_calibration_audit", report)
        self.assertIn("regime", report["multi_calibration_audit"]["scores_by_regime_and_volatility"])
        self.assertIn("paired_block_interval_vs_causal_baseline", report["versions"]["4.5a"])
        self.assertEqual(report["versions"]["4.5a"]["paired_block_interval_vs_causal_baseline"]["replicates"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
