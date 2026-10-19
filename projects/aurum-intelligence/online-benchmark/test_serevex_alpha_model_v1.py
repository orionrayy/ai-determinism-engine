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
        self.assertEqual(stages["325"], "FROZEN_FINAL_MODEL_AND_CALIBRATOR")

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
