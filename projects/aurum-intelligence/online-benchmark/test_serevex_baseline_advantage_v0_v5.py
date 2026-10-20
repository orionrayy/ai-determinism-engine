import copy
import random
import unittest

import serevex_baseline_advantage_v0_v5 as evolution


def make_ledger(n=800, informative=False):
    rows = []
    rng = random.Random(73019)
    for i in range(n):
        group_a = (i % 50) < 25
        if informative:
            specialist = 0.64 if group_a else 0.36
            positive = rng.random() < specialist
        else:
            positive = (i % 25) < (16 if group_a else 9)
            specialist = 0.5
        rows.append({
            "decision_id": f"d{i}", "decision_index": i,
            "label_available_index": i + 12, "label_available_time": i + 12,
            "label": int(positive), "label_mature": True,
            "partition": "development" if i < 400 else "final_holdout",
            "baseline_probability": 0.55 if informative else 0.5,
            "matured_label_count_before_decision": min(i, 288),
            "conditional_regime_support_before_decision": min(i // 2, 200),
            "conditional_volatility_support_before_decision": min(i // 2, 200),
            "regime_id": "A" if group_a else "B",
            "state_tags": {"volatility_bucket": "high" if i % 3 == 0 else "low"},
            "expert_probabilities": {
                "rolling_matured_rate_288": specialist,
                "regime_matured_rate": specialist,
                "volatility_matured_rate": specialist,
            },
        })
    return {"metadata": {}, "records": rows}


class BaselineAdvantageTests(unittest.TestCase):
    def test_versions_are_ordered_and_baseline_reproduces_exactly(self):
        report = evolution.evaluate_ledger(make_ledger(), as_of_index=800, bootstrap_replicates=200)
        self.assertEqual(report["version_order"], [f"{i / 10:.1f}" for i in range(1, 51)])
        self.assertEqual(len(report["version_order"]), 50)
        self.assertEqual(report["version_order"][0], "0.1")
        self.assertEqual(report["version_order"][-1], "5.0")
        self.assertEqual(report["versions"]["0.1"]["brier"], 0.25)
        self.assertEqual(report["validation"]["baseline_reproduction"], "PASS")
        self.assertFalse(report["prequential_governor"]["future_label_used_for_own_decision"])
        self.assertFalse(report["validation"]["live_signal_enabled"])
        self.assertEqual(report["validation"]["decision_output"], "NO_TRADE")

    def test_stable_calibrated_specialist_can_be_selected(self):
        report = evolution.evaluate_ledger(make_ledger(informative=True), as_of_index=800, bootstrap_replicates=200)
        selected = report["prequential_governor"]["selected_version_counts"]
        self.assertGreater(sum(selected.get(v, 0) for v in evolution.CANDIDATE_VERSIONS), 0)
        self.assertLess(report["versions"]["5.0"]["brier"], report["versions"]["0.1"]["brier"])

    def test_bad_specialist_falls_back_exactly_to_baseline(self):
        rows = make_ledger(informative=False)
        for row in rows["records"]:
            row["expert_probabilities"] = {
                "rolling_matured_rate_288": 0.99 if row["label"] == 0 else 0.01,
                "regime_matured_rate": 0.99 if row["label"] == 0 else 0.01,
                "volatility_matured_rate": 0.99 if row["label"] == 0 else 0.01,
            }
        report = evolution.evaluate_ledger(rows, as_of_index=800, bootstrap_replicates=200)
        self.assertEqual(report["versions"]["5.0"]["brier"], report["versions"]["0.1"]["brier"])
        self.assertEqual(report["versions"]["5.0"]["log_loss"], report["versions"]["0.1"]["log_loss"])

    def test_future_label_mutations_cannot_change_prefix_selection(self):
        original = make_ledger(informative=True)
        changed = copy.deepcopy(original)
        cutoff = 500
        for row in changed["records"]:
            if row["decision_index"] >= cutoff:
                row["label"] = 1 - row["label"]
        full_a = evolution.evaluate_ledger(original, as_of_index=800, bootstrap_replicates=200)
        full_b = evolution.evaluate_ledger(changed, as_of_index=800, bootstrap_replicates=200)
        hashes_a = full_a["prequential_governor"]["selection_trace_block_sha256"]
        hashes_b = full_b["prequential_governor"]["selection_trace_block_sha256"]
        prefix_blocks = [key for key in hashes_a if int(key) * 24 + 24 <= cutoff]
        self.assertTrue(prefix_blocks)
        for key in prefix_blocks:
            self.assertEqual(hashes_a[key], hashes_b[key], f"future label mutation leaked into earlier block {key}")

    def test_invalid_probabilities_and_duplicate_indices_fail_closed(self):
        rows = make_ledger(n=50)
        rows["records"][5]["expert_probabilities"]["rolling_matured_rate_288"] = float("nan")
        with self.assertRaises(ValueError):
            evolution.evaluate_ledger(rows, as_of_index=50, bootstrap_replicates=200)
        rows = make_ledger(n=50)
        rows["records"][6]["decision_index"] = 5
        with self.assertRaises(ValueError):
            evolution.evaluate_ledger(rows, as_of_index=50, bootstrap_replicates=200)

    def test_report_integrity_hash_binds_scored_labels(self):
        original = make_ledger(informative=True)
        altered = copy.deepcopy(original)
        for row in altered["records"]:
            if row["partition"] == "final_holdout" and row["label_available_index"] < 800:
                row["label"] = 1 - row["label"]
        report_a = evolution.evaluate_ledger(original, as_of_index=800, bootstrap_replicates=200)
        report_b = evolution.evaluate_ledger(altered, as_of_index=800, bootstrap_replicates=200)
        self.assertNotEqual(report_a["score_cohort_sha256"], report_b["score_cohort_sha256"])
        self.assertNotEqual(report_a["report_sha256"], report_b["report_sha256"])

    def test_label_maturity_cannot_precede_decision(self):
        rows = make_ledger(n=50)
        rows["records"][10]["label_available_index"] = 9
        with self.assertRaises(ValueError):
            evolution.evaluate_ledger(rows, as_of_index=50, bootstrap_replicates=200)

    def test_sequential_alpha_spending_tightens_for_later_looks(self):
        z1, alpha1, _ = evolution._sequential_critical_z(1, len(evolution.CANDIDATE_VERSIONS))
        z2, alpha2, _ = evolution._sequential_critical_z(2, len(evolution.CANDIDATE_VERSIONS))
        z10, _, _ = evolution._sequential_critical_z(10, len(evolution.CANDIDATE_VERSIONS))
        self.assertGreater(z2, z1)
        self.assertGreater(z10, z2)
        self.assertGreater(alpha1, alpha2)
        spent = sum(0.05 / (k * (k + 1.0)) for k in range(1, 10000))
        self.assertLessEqual(spent, 0.05)
        self.assertGreater(spent, 0.0499)

    def test_selection_gate_is_cached_until_new_block_evidence(self):
        governor = evolution._Governor({})
        first = governor.select()
        second = governor.select()
        self.assertIs(first, second)
        self.assertEqual(governor.selection_look_count, 0)
        self.assertEqual(first["selected"], "0.1")

    def test_multi_calibration_and_embargoed_bootstrap_are_present(self):
        report = evolution.evaluate_ledger(make_ledger(informative=True), as_of_index=800, bootstrap_replicates=200)
        self.assertIn("multi_calibration_audit", report)
        self.assertIn("regime", report["multi_calibration_audit"]["scores_by_regime_and_volatility"])
        self.assertIn("paired_block_interval_vs_baseline", report["versions"]["5.0"])
        self.assertEqual(report["versions"]["5.0"]["paired_block_interval_vs_baseline"]["replicates"], 200)
        self.assertFalse(report["validation"]["model_validated"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
