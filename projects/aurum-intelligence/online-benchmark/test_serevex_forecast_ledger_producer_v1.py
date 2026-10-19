import hashlib
import math
import unittest
from datetime import datetime, timedelta, timezone

import serevex_forecast_ledger_producer_v1 as producer
import serevex_stage123_integrated_v1 as stage123


def synthetic_rows(n=1800, *, change_at=None, naive=False):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        trend = 0.025 * i
        if change_at is not None and i >= change_at:
            wave = 1.8 * math.sin(i / 4.5) + 1.2 * math.sin(i / 11.0)
        else:
            wave = 1.8 * math.sin(i / 7.0) + 0.8 * math.sin(i / 23.0)
        close = 2000.0 + trend + wave
        j = i - 1
        if change_at is not None and j >= change_at:
            previous_wave = 1.8 * math.sin(j / 4.5) + 1.2 * math.sin(j / 11.0)
        else:
            previous_wave = 1.8 * math.sin(j / 7.0) + 0.8 * math.sin(j / 23.0)
        open_price = 2000.0 + 0.025 * max(0, i - 1) + previous_wave
        high = max(open_price, close) + 0.28 + (i % 3) * 0.015
        low = min(open_price, close) - 0.28 - (i % 4) * 0.012
        timestamp = start + timedelta(minutes=5 * i)
        if naive:
            timestamp = timestamp.replace(tzinfo=None)
        rows.append({
            "timestamp": timestamp,
            "open": open_price, "high": high, "low": low, "close": close,
            "volume": 100.0 + (i % 13),
        })
    return rows


def digest(raw=b"fixture dataset"):
    return hashlib.sha256(raw).hexdigest()


class SerevexForecastLedgerProducerTests(unittest.TestCase):
    def build(self, rows=None, **kwargs):
        return producer.build_forecast_ledger(
            rows or synthetic_rows(),
            dataset_sha256=digest(),
            source_uri="https://example.invalid/reference/xau.csv",
            rule="last_return_sign",
            side="long",
            evidence_kind="synthetic_contract_test",
            **kwargs,
        )

    def test_producer_output_passes_integrated_contract(self):
        ledger = self.build()
        checked = stage123.validate_forecast_ledger(ledger)
        self.assertEqual(checked["contract_status"], "PASS")
        self.assertEqual(checked["record_count"], len(ledger["records"]))
        self.assertEqual(ledger["metadata"]["registered_experts"], list(producer.EXPERT_SPECS))
        self.assertFalse(checked["all_source_verification_flags_true"])
        self.assertFalse(checked["source_authenticity_independently_verified"])
        self.assertEqual(ledger["metadata"]["source_verification"]["broker_feed_match_verified"], False)

    def test_ledger_contains_trained_alpha_model_and_causal_features(self):
        ledger = self.build()
        metadata = ledger["metadata"]
        alpha_report = metadata["alpha_model_training"]
        self.assertIn("alpha_logistic_v1", metadata["registered_experts"])
        self.assertEqual(alpha_report["model_version"], "serevex-alpha-logistic-v4-embargoed-dual-score-ood")
        self.assertEqual(alpha_report["status"], "ALPHA_RESEARCH_TRAINED")
        self.assertFalse(alpha_report["training_policy"]["final_holdout_labels_used_for_fit_or_calibration"])
        self.assertTrue(alpha_report["training_policy"]["final_holdout_predictions_are_frozen"])
        self.assertGreater(alpha_report["final_holdout_metrics"]["rows"], 0)
        self.assertIn("probability_governor", alpha_report)
        self.assertIn("out_of_domain_fallback_rate", alpha_report["final_holdout_metrics"])
        for large_field in ("predictions_by_decision_index", "raw_predictions_by_decision_index",
                            "calibrated_alpha_predictions_by_decision_index", "out_of_domain_fractions_by_decision_index",
                            "forecast_policy_by_decision_index", "forecast_stage_by_decision_index"):
            self.assertNotIn(large_field, alpha_report)
        self.assertIn(alpha_report["probability_governor"]["status"], (
            "BASELINE_ONLY_INSUFFICIENT_CALIBRATION_SUPPORT",
            "BASELINE_ONLY_NO_CONFIRMED_DUAL_SCORE_GAIN",
            "CALIBRATION_BLOCK_SELECTED_ALPHA_WEIGHT_PASSES_DUAL_SCORE_STABILITY_SCREEN",
        ))
        for row in ledger["records"]:
            self.assertIn("alpha_logistic_v1", row["expert_probabilities"])
            self.assertEqual(len(row["feature_facts"]), len(producer.alpha_model.FEATURE_NAMES))
            self.assertEqual(row["alpha_model_artifact_sha256"], alpha_report["model_artifact_sha256"])
            if row["partition"] == "final_holdout":
                self.assertIn(row["alpha_model_forecast_policy"], ("ALPHA_ELIGIBLE", "BASELINE_FALLBACK_OUT_OF_DOMAIN_FEATURES"))
                self.assertTrue(0.0 <= row["alpha_model_out_of_domain_fraction"] <= 1.0)
                if row["alpha_model_forecast_policy"] == "BASELINE_FALLBACK_OUT_OF_DOMAIN_FEATURES":
                    self.assertEqual(
                        row["expert_probabilities"]["alpha_calibrated_logistic_v1"],
                        row["baseline_probability"],
                    )
            else:
                self.assertEqual(row["alpha_model_forecast_policy"], "NOT_APPLICABLE_PRE_FINAL_HOLDOUT")
                self.assertIsNone(row["alpha_model_out_of_domain_fraction"])
                # Do not back-apply the final calibrator to its own calibration-fit
                # rows; pre-holdout policy history must retain the forecast as issued.
                self.assertEqual(
                    row["expert_probabilities"]["alpha_calibrated_logistic_v1"],
                    row["expert_probabilities"]["alpha_logistic_v1"],
                )
        checked = stage123.validate_forecast_ledger(ledger)
        self.assertEqual(checked["contract_status"], "PASS")
        self.assertFalse(checked["predictive_skill_proven"] if "predictive_skill_proven" in checked else False)

    def test_estimators_always_emit_finite_probabilities(self):
        ledger = self.build()
        for row in ledger["records"]:
            self.assertTrue(0 <= row["baseline_probability"] <= 1)
            self.assertTrue(math.isfinite(row["baseline_probability"]))
            self.assertEqual(set(row["expert_probabilities"]), set(producer.EXPERT_SPECS))
            for probability in row["expert_probabilities"].values():
                self.assertTrue(math.isfinite(probability))
                self.assertTrue(0 <= probability <= 1)

    def test_partition_boundaries_are_chronological(self):
        ledger = self.build()
        order = {"development": 0, "model_selection": 1, "calibration": 2, "final_holdout": 3}
        partitions = [row["partition"] for row in ledger["records"]]
        values = [order[partition] for partition in partitions]
        self.assertEqual(values, sorted(values))
        self.assertIn("development", partitions)
        self.assertIn("model_selection", partitions)
        self.assertIn("calibration", partitions)
        self.assertIn("final_holdout", partitions)

    def test_prediction_before_maturity_does_not_use_future_label(self):
        rows = synthetic_rows(1800)
        original = self.build(rows)
        altered_rows = synthetic_rows(1800, change_at=900)
        altered = self.build(altered_rows)
        left = {r["decision_index"]: r for r in original["records"]}
        right = {r["decision_index"]: r for r in altered["records"]}
        safe_cutoff = 900 - producer.HORIZON_BARS - 2
        common = sorted(set(left) & set(right))
        checked = 0
        for index in common:
            if index >= safe_cutoff:
                continue
            self.assertEqual(left[index]["baseline_probability"], right[index]["baseline_probability"])
            self.assertEqual(left[index]["expert_probabilities"], right[index]["expert_probabilities"])
            checked += 1
        self.assertGreater(checked, 100)

    def test_naive_timestamps_require_explicit_diagnostic_policy(self):
        with self.assertRaisesRegex(ValueError, "naive_timestamp_requires_explicit"):
            self.build(synthetic_rows(1800, naive=True))
        ledger = self.build(synthetic_rows(1800, naive=True), naive_timestamp_policy="assume_utc_for_diagnostic_only")
        self.assertEqual(ledger["metadata"]["source_timezone_policy"], "naive_timestamp_assumed_utc_for_diagnostic_only")
        self.assertFalse(ledger["metadata"]["source_verification"]["provider_provenance_independently_verified"])

    def test_future_state_tags_and_features_are_past_or_point_in_time(self):
        ledger = self.build()
        for row in ledger["records"]:
            decision = datetime.fromisoformat(row["decision_time"].replace("Z", "+00:00"))
            self.assertLessEqual(datetime.fromisoformat(row["state_tags_available_time"].replace("Z", "+00:00")), decision)
            for fact in row["feature_facts"]:
                self.assertLessEqual(datetime.fromisoformat(fact["available_at"].replace("Z", "+00:00")), decision)
                self.assertLessEqual(datetime.fromisoformat(fact["observed_at"].replace("Z", "+00:00")), decision)
                self.assertEqual(fact["quality_status"], "REFERENCE_ONLY_SOURCE_TIME_AND_AVAILABILITY_UNVERIFIED")

    def test_label_maturity_indices_are_after_the_decision(self):
        ledger = self.build()
        for row in ledger["records"]:
            self.assertGreater(row["label_available_index"], row["decision_index"])
            if row["label_mature"]:
                self.assertIn(row["label"], (0, 1))
            else:
                self.assertIsNone(row["label"])

    def test_no_events_fails_closed_for_registered_cohort(self):
        with self.assertRaisesRegex(ValueError, "rule_must_be_registered_directional_rule"):
            producer.build_forecast_ledger(
                synthetic_rows(), dataset_sha256=digest(),
                source_uri="https://example.invalid/data.csv", rule="flat_no_trade", side="long",
            )

    def test_end_to_end_producer_to_stages_1_3(self):
        ledger, report = producer.run_end_to_end(
            synthetic_rows(),
            dataset_sha256=digest(),
            source_uri="https://example.invalid/reference/xau.csv",
            rule="last_return_sign",
            side="long",
            as_of_index=1800,
            evidence_kind="synthetic_contract_test",
            atlas_bootstrap_replicates=200,
        )
        self.assertEqual(report["status"], "PASS_CONTRACT_DIAGNOSTIC_ONLY")
        self.assertEqual(report["stage_1_ledger_contract"], "PASS")
        self.assertEqual(report["decision_output"], "NO_TRADE")
        self.assertFalse(report["predictive_skill_proven"])
        self.assertEqual(report["stage_3_delayed_feedback_mixer"]["rows_at_or_after_as_of_excluded"], 0)
        self.assertEqual(len(report["report_sha256"]), 64)
        self.assertEqual(ledger["metadata"]["evidence_kind"], "synthetic_contract_test")


if __name__ == "__main__":
    unittest.main()
