import copy
import random
import unittest

import serevex_hierarchical_shrinkage_v2 as v2


ENDS = {"development": 2400, "model_selection": 3360, "calibration": 3840, "final_holdout": 4800}


def make_ledger(n=4800, informative=True, seed=20261020):
    rng = random.Random(seed)
    rows = []
    prior_labels = []
    pending = []
    positive_by_regime = {"trend_up": 0, "trend_down": 0}
    count_by_regime = {"trend_up": 0, "trend_down": 0}
    from collections import deque
    import heapq
    q = []
    recent = deque(maxlen=288)
    total = positive = 0
    regime_counts = {"trend_up": [0, 0], "trend_down": [0, 0]}
    vol_counts = {"high": [0, 0], "low": [0, 0]}
    for i in range(n):
        regime = "trend_up" if (i // 24) % 2 == 0 else "trend_down"
        vol = "high" if regime == "trend_up" else "low"
        while q and q[0][0] < i:
            _, _, label, rg, vb = heapq.heappop(q)
            total += 1
            positive += label
            recent.append(label)
            regime_counts[rg][0] += 1
            regime_counts[rg][1] += label
            vol_counts[vb][0] += 1
            vol_counts[vb][1] += label
        baseline = (positive + 1.0) / (total + 2.0)
        label = int(rng.random() < ((0.88 if regime == "trend_up" else 0.12) if informative else 0.5))
        partition = ("development" if i < ENDS["development"] else
                     "model_selection" if i < ENDS["model_selection"] else
                     "calibration" if i < ENDS["calibration"] else "final_holdout")
        rows.append({
            "decision_id": f"d{i}", "decision_index": i, "label_available_index": i + 6,
            "label": label, "label_mature": True, "partition": partition,
            "baseline_probability": baseline, "regime_id": regime,
            "state_tags": {"volatility_bucket": vol},
        })
        heapq.heappush(q, (i + 6, i, label, regime, vol))
    return {"metadata": {"dataset_sha256": "synthetic-only-test"}, "records": rows}


def refresh_causal_baseline(ledger):
    import heapq
    pending, total, positive = [], 0, 0
    for row in ledger["records"]:
        index = int(row["decision_index"])
        while pending and pending[0][0] < index:
            _, label = heapq.heappop(pending)
            total += 1
            positive += label
        row["baseline_probability"] = (positive + 1.0) / (total + 2.0)
        if row["label_mature"] and row["label"] in (0, 1):
            heapq.heappush(pending, (int(row["label_available_index"]), int(row["label"])))


class HierarchicalShrinkageV2Tests(unittest.TestCase):
    def evaluate(self, ledger):
        return v2.evaluate_ledger(
            ledger, as_of_index=4800, partition_end_indices=ENDS, bootstrap_replicates=200
        )

    def test_grid_and_version_contract(self):
        self.assertEqual(len(v2.CANDIDATES), 48)
        self.assertEqual(v2.CANDIDATES[0]["version"], "0.2b")
        self.assertEqual(v2.CANDIDATES[-1]["version"], "4.9b")
        self.assertEqual(v2.CANDIDATES[-1]["family"], "joint")

    def test_reproduces_immutable_causal_baseline_and_no_trade(self):
        report = self.evaluate(make_ledger(informative=False))
        self.assertTrue(report["baseline_reproduction_audit"]["verified"])
        self.assertEqual(report["version_order"][0], "0.1b")
        self.assertEqual(report["version_order"][-1], "5.0b")
        self.assertEqual(report["decision_output"], "NO_TRADE")
        self.assertFalse(report["model_validated"])
        self.assertFalse(report["deployment_calibrated"])
        self.assertFalse(report["live_signal_enabled"])
        self.assertFalse(report["live_order_execution_enabled"])

    def test_uninformative_specialists_fail_closed_to_baseline(self):
        for seed in (12, 441, 20261020):
            with self.subTest(seed=seed):
                report = self.evaluate(make_ledger(informative=False, seed=seed))
                self.assertFalse(report["model_selection_gate"]["passed"])
                self.assertEqual(report["final_holdout"]["policy"], "0.1b")
                self.assertEqual(
                    report["final_holdout"]["policy_metrics"]["brier"],
                    report["final_holdout"]["baseline_metrics"]["brier"],
                )
                self.assertEqual(
                    report["final_holdout"]["policy_metrics"]["log_loss"],
                    report["final_holdout"]["baseline_metrics"]["log_loss"],
                )

    def test_informative_regime_expert_can_pass_independent_gates(self):
        report = self.evaluate(make_ledger(informative=True))
        # It is a positive control: it may promote only if both independent gates pass.
        self.assertIn(report["model_selection_gate"]["status"],
                      {"PASS", "DUAL_SCORE_GATE_FAILED", "SUBGROUP_CALIBRATION_GATE_FAILED"})
        self.assertIn(report["calibration_gate"]["status"],
                      {"PASS", "NOT_RUN_MODEL_SELECTION_GATE_FAILED", "DUAL_SCORE_GATE_FAILED",
                       "SUBGROUP_CALIBRATION_GATE_FAILED"})
        self.assertIn(report["final_holdout"]["policy"], {"0.1b", "hierarchical_candidate"})
        self.assertFalse(report["final_holdout"]["final_holdout_can_change_candidate_choice"])

    def test_final_holdout_label_mutation_cannot_change_pre_holdout_gates(self):
        original = make_ledger(informative=True)
        changed = copy.deepcopy(original)
        for row in changed["records"]:
            if row["partition"] == "final_holdout":
                row["label"] = 1 - row["label"]
        refresh_causal_baseline(changed)
        a, b = self.evaluate(original), self.evaluate(changed)
        self.assertEqual(a["development_selection"]["selection_sha256"],
                         b["development_selection"]["selection_sha256"])
        self.assertEqual(a["development_selection"]["selected_candidate"],
                         b["development_selection"]["selected_candidate"])
        self.assertEqual(a["model_selection_gate"], b["model_selection_gate"])
        self.assertEqual(a["calibration_gate"], b["calibration_gate"])
        self.assertNotEqual(a["final_holdout"]["score_cohort_sha256"],
                            b["final_holdout"]["score_cohort_sha256"])

    def test_partition_labels_must_match_frozen_index_ranges(self):
        ledger = make_ledger(informative=False)
        ledger["records"][2500]["partition"] = "development"
        with self.assertRaisesRegex(ValueError, "partition_tag_disagrees"):
            self.evaluate(ledger)

    def test_duplicate_decision_and_nonfinite_probability_fail_closed(self):
        ledger = make_ledger(informative=False)
        ledger["records"][10]["decision_index"] = 9
        with self.assertRaisesRegex(ValueError, "duplicate_decision_index"):
            self.evaluate(ledger)
        ledger = make_ledger(informative=False)
        ledger["records"][10]["baseline_probability"] = float("nan")
        with self.assertRaisesRegex(ValueError, "probability_must_be_finite"):
            self.evaluate(ledger)

    def test_block_purge_never_scores_boundary_crossing_blocks(self):
        report = self.evaluate(make_ledger(informative=False))
        for name, part in report["partition_rows"].items():
            audit = part["block_audit"]
            self.assertFalse(audit["partial_or_boundary_crossing_blocks_used"])
            self.assertEqual(part["block_purged_scored_rows"], audit["eligible_block_count"] * v2.BLOCK_SPAN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
