import copy
import unittest

import serevex_alpha_model_v1 as alpha


def make_events(n=400):
    boundaries = {"development": 0, "model_selection": 200, "calibration": 280, "final_holdout": 320}
    events = []
    for i in range(n):
        momentum = ((i % 17) - 8) / 4.0
        trend = 1.0 if momentum > 0 else -1.0
        label = int((momentum > 0) == ((i % 11) not in (0, 1)))
        if i < boundaries["model_selection"]:
            partition = "development"
        elif i < boundaries["calibration"]:
            partition = "model_selection"
        elif i < boundaries["final_holdout"]:
            partition = "calibration"
        else:
            partition = "final_holdout"
        events.append({
            "decision_id": "event-" + str(i),
            "decision_index": i,
            "partition": partition,
            "label_available_index": i + 1,
            "label_available_time": i + 1,
            "label": label,
            "label_mature": True,
            "baseline_probability": 0.4 + (i % 3) * 0.05,
            "alpha_features": {
                "atr_pct": 0.01 + (i % 7) * 0.001,
                "trend_efficiency": max(-1.0, min(1.0, trend * min(abs(momentum) / 4.0, 1.0))),
                "momentum_1_atr": momentum,
                "momentum_5_atr": momentum * 0.7,
                "candle_body_to_range": max(-1.0, min(1.0, momentum / 8.0)),
                "close_location_in_range": max(-1.0, min(1.0, momentum / 7.0)),
                "bar_range_to_atr": 1.0 + (i % 5) * 0.2,
                "volume_log_ratio": ((i % 9) - 4) / 3.0,
            },
        })
    return events


class SerevexAlphaModelTests(unittest.TestCase):
    def train(self, events=None):
        return alpha.train_alpha_model(
            events or make_events(),
            partition_start_indices={
                "development": 0,
                "model_selection": 200,
                "calibration": 280,
                "final_holdout": 320,
            },
            as_of_index=400,
        )

    def test_training_is_deterministic_and_hashes_the_fitted_artifact(self):
        a = self.train()
        b = self.train()
        self.assertEqual(a["model_artifact_sha256"], b["model_artifact_sha256"])
        self.assertEqual(a["predictions_by_decision_index"], b["predictions_by_decision_index"])
        self.assertEqual(a["status"], "ALPHA_RESEARCH_TRAINED")
        self.assertEqual(len(a["model_artifact_sha256"]), 64)
        self.assertEqual(a["decision_output"], "NO_TRADE")
        self.assertFalse(a["model_validated"])
        self.assertFalse(a["live_signal_enabled"])

    def test_final_holdout_labels_cannot_change_fit_hash_or_holdout_predictions(self):
        rows = make_events()
        original = self.train(rows)
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["partition"] == "final_holdout":
                row["label"] = 1 - row["label"]
        mutated = self.train(changed)
        self.assertEqual(original["model_artifact_sha256"], mutated["model_artifact_sha256"])
        for index in range(320, 400):
            self.assertEqual(
                original["predictions_by_decision_index"][str(index)],
                mutated["predictions_by_decision_index"][str(index)],
            )

    def test_training_and_calibration_labels_obey_strict_maturity_cutoffs(self):
        result = self.train()
        policy = result["training_policy"]
        self.assertLess(policy["max_final_training_label_available_index"], 280)
        self.assertEqual(policy["strict_training_maturity_cutoff_index"], 280)
        self.assertEqual(policy["calibration_label_cutoff_index"], 320)
        self.assertFalse(policy["final_holdout_labels_used_for_fit_or_calibration"])
        self.assertTrue(policy["final_holdout_predictions_are_frozen"])
        stages = result["forecast_stage_by_decision_index"]
        self.assertEqual(stages["5"], "DEVELOPMENT_COLD_START_BASELINE")
        self.assertEqual(stages["205"], "DEVELOPMENT_FIT_SCOUT_FIXED_BEFORE_MODEL_SELECTION")
        self.assertEqual(stages["285"], "REFIT_BEFORE_CALIBRATION_UNCALIBRATED")
        self.assertEqual(stages["325"], "FROZEN_FINAL_MODEL_CALIBRATOR_AND_BASELINE_GOVERNOR")

    def test_probability_governor_defaults_to_exact_baseline_when_alpha_loses(self):
        rows = [
            {"decision_id": f"bad-{i}", "decision_index": i, "label_mature": True,
             "label": 0, "baseline_probability": 0.2}
            for i in range(120)
        ]
        alpha_probabilities = {i: 0.95 for i in range(120)}
        result = alpha._choose_baseline_shrinkage_weight(rows, alpha_probabilities)
        self.assertEqual(result["alpha_weight"], 0.0)
        self.assertEqual(result["status"], "BASELINE_ONLY_NO_CONFIRMED_CALIBRATION_GAIN")

    def test_probability_governor_selects_alpha_when_gain_is_large_and_stable(self):
        rows = [
            {"decision_id": f"good-{i}", "decision_index": i, "label_mature": True,
             "label": int(i % 2 == 0), "baseline_probability": 0.5}
            for i in range(240)
        ]
        alpha_probabilities = {i: (0.98 if i % 2 == 0 else 0.02) for i in range(240)}
        result = alpha._choose_baseline_shrinkage_weight(rows, alpha_probabilities)
        self.assertEqual(result["alpha_weight"], 1.0)
        self.assertEqual(result["status"], "CALIBRATION_BLOCK_SELECTED_ALPHA_WEIGHT_PASSES_CONSERVATIVE_SCREEN")

    def test_probability_governor_fails_closed_on_insufficient_blocks(self):
        rows = [
            {"decision_id": f"short-{i}", "decision_index": i, "label_mature": True,
             "label": int(i % 2 == 0), "baseline_probability": 0.5}
            for i in range(60)
        ]
        alpha_probabilities = {i: (0.99 if i % 2 == 0 else 0.01) for i in range(60)}
        result = alpha._choose_baseline_shrinkage_weight(rows, alpha_probabilities)
        self.assertEqual(result["alpha_weight"], 0.0)
        self.assertEqual(result["status"], "BASELINE_ONLY_INSUFFICIENT_CALIBRATION_SUPPORT")

    def test_holdout_reports_paired_block_interval_and_reliability(self):
        rows = make_events()
        result = self.train(rows)
        metrics = result["final_holdout_metrics"]
        self.assertIn("paired_brier_improvement_ci_vs_causal_baseline", metrics)
        self.assertIn("reliability_diagnostics", metrics)
        self.assertEqual(metrics["reliability_diagnostics"]["rows"], metrics["rows"])
        self.assertEqual(result["probability_governor"]["alpha_weight"], 0.0)  # calibration slice has too few independent blocks

    def test_invalid_partition_timeline_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "strictly_chronological"):
            alpha.train_alpha_model(
                make_events(),
                partition_start_indices={
                    "development": 0, "model_selection": 280,
                    "calibration": 200, "final_holdout": 320,
                },
                as_of_index=400,
            )


if __name__ == "__main__":
    unittest.main()
