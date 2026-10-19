import unittest

from forecastability_discovery_v1 import (
    assess_forecastability,
    combine_current_forecasts,
    _contiguous_runs,
)


def make_records(n=1200, *, models=("expert",), regimes=("all",), pattern="good"):
    rows = []
    for i in range(n):
        regime = regimes[0] if len(regimes) == 1 or i < n // 2 else regimes[1]
        signal = int((i % 17) < 8)
        label = signal
        for model in models:
            if pattern == "good":
                p = 0.9 if signal else 0.1
            elif pattern == "null":
                p = 0.5
            elif pattern == "regime_conditional":
                if regime == regimes[0]:
                    p = 0.9 if signal else 0.1
                else:
                    p = 0.1 if signal else 0.9
            else:
                raise AssertionError(pattern)
            rows.append({
                "decision_id": f"d-{i:05d}",
                "model_id": model,
                "decision_index": i,
                "label_available_index": i + 1,
                "regime_id": regime,
                "regime_available_index": i,
                "segment_id": "session-1" if i < n // 2 else "session-2",
                "probability": p,
                "baseline_probability": 0.5,
                "label": label,
                "label_mature": True,
            })
    return rows


class ForecastabilityDiscoveryTests(unittest.TestCase):
    def test_strong_synthetic_skill_is_replication_only_never_market_promoted(self):
        report = assess_forecastability(
            make_records(n=1200), as_of_index=1202,
            registered_models=["expert"], registered_regimes=["all"],
            min_matured_rows=240, block_size=12, min_independent_blocks=10,
            bootstrap_replicates=500, seed=7, evidence_kind="synthetic_contract_test",
        )
        entry = report["models"]["expert"]["all"]
        self.assertEqual(entry["status"], "ELIGIBLE_FOR_INDEPENDENT_REPLICATION")
        self.assertGreater(entry["lower_skill_bound"], 0)
        self.assertEqual(report["decision_output"], "NO_TRADE")
        self.assertFalse(report["market_promotion_allowed"])
        self.assertFalse(report["live_order_execution_enabled"])

    def test_null_skill_is_not_promoted(self):
        report = assess_forecastability(
            make_records(n=1200, pattern="null"), as_of_index=1202,
            registered_models=["expert"], registered_regimes=["all"],
            min_matured_rows=240, block_size=12, min_independent_blocks=10,
            bootstrap_replicates=300, seed=7,
        )
        entry = report["models"]["expert"]["all"]
        self.assertEqual(entry["status"], "SKILL_NOT_DEMONSTRATED")
        self.assertLessEqual(entry["lower_skill_bound"], 0)

    def test_future_and_same_index_labels_are_excluded(self):
        report = assess_forecastability(
            make_records(n=30), as_of_index=10,
            registered_models=["expert"], registered_regimes=["all"],
            min_matured_rows=20, block_size=2, min_independent_blocks=2,
            bootstrap_replicates=200,
        )
        self.assertEqual(report["matured_decision_count"], 9)
        self.assertEqual(report["excluded_unmatured_model_record_count"], 21)
        self.assertEqual(report["decision_output"], "NO_TRADE")

    def test_regime_conditioning_separates_good_and_bad_regimes(self):
        regimes = ("good", "bad")
        report = assess_forecastability(
            make_records(n=1200, regimes=regimes, pattern="regime_conditional"),
            as_of_index=1202, registered_models=["expert"], registered_regimes=list(regimes),
            min_matured_rows=240, block_size=12, min_independent_blocks=10,
            bootstrap_replicates=500, seed=11,
        )
        self.assertEqual(report["models"]["expert"]["good"]["status"], "ELIGIBLE_FOR_INDEPENDENT_REPLICATION")
        self.assertEqual(report["models"]["expert"]["bad"]["status"], "SKILL_NOT_DEMONSTRATED")

    def test_bootstrap_runs_never_cross_time_gaps_or_segments(self):
        items = [
            {"decision_index": 0, "segment_id": "a", "regime_id": "x", "skill_difference": 1.0},
            {"decision_index": 1, "segment_id": "a", "regime_id": "x", "skill_difference": 1.0},
            {"decision_index": 3, "segment_id": "a", "regime_id": "x", "skill_difference": 1.0},
            {"decision_index": 4, "segment_id": "b", "regime_id": "x", "skill_difference": 1.0},
        ]
        self.assertEqual(_contiguous_runs(items), [[1.0, 1.0], [1.0], [1.0]])

    def test_duplicate_model_decision_pair_is_rejected(self):
        rows = make_records(n=30)
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate_decision_model_pair"):
            assess_forecastability(
                rows, as_of_index=40, registered_models=["expert"], registered_regimes=["all"],
                min_matured_rows=20, block_size=2, min_independent_blocks=2, bootstrap_replicates=200,
            )

    def test_missing_model_on_any_decision_breaks_paired_cohort(self):
        rows = make_records(n=30, models=("expert", "expert2"))
        rows = [r for r in rows if not (r["decision_id"] == "d-00010" and r["model_id"] == "expert2")]
        with self.assertRaisesRegex(ValueError, "unaligned_model_cohort"):
            assess_forecastability(
                rows, as_of_index=40, registered_models=["expert", "expert2"], registered_regimes=["all"],
                min_matured_rows=20, block_size=2, min_independent_blocks=2, bootstrap_replicates=200,
            )

    def test_model_rows_must_share_exact_baseline_probabilities(self):
        rows = make_records(n=30, models=("expert", "expert2"))
        target = next(r for r in rows if r["decision_id"] == "d-00010" and r["model_id"] == "expert2")
        target["baseline_probability"] = 0.8
        with self.assertRaisesRegex(ValueError, "cross_model_cohort_mismatch.*baseline_probability"):
            assess_forecastability(
                rows, as_of_index=40, registered_models=["expert", "expert2"], registered_regimes=["all"],
                min_matured_rows=20, block_size=2, min_independent_blocks=2, bootstrap_replicates=200,
            )

    def test_future_dated_regime_is_rejected(self):
        rows = make_records(n=30)
        rows[4]["regime_available_index"] = 5
        with self.assertRaisesRegex(ValueError, "regime_must_be_available_as_of_decision"):
            assess_forecastability(
                rows, as_of_index=40, registered_models=["expert"], registered_regimes=["all"],
                min_matured_rows=20, block_size=2, min_independent_blocks=2, bootstrap_replicates=200,
            )

    def test_combiner_always_keeps_no_trade_authority(self):
        good = assess_forecastability(
            make_records(n=1200), as_of_index=1202,
            registered_models=["expert"], registered_regimes=["all"],
            min_matured_rows=240, block_size=12, min_independent_blocks=10,
            bootstrap_replicates=500, seed=7, evidence_kind="synthetic_contract_test",
        )
        forecast = combine_current_forecasts(good, current_regime="all", current_probabilities={"expert": 0.8})
        self.assertEqual(forecast["probability"], 0.8)
        self.assertEqual(forecast["decision_output"], "NO_TRADE")
        self.assertFalse(forecast["live_signal_enabled"])
        no_skill = assess_forecastability(
            make_records(n=1200, pattern="null"), as_of_index=1202,
            registered_models=["expert"], registered_regimes=["all"],
            min_matured_rows=240, block_size=12, min_independent_blocks=10,
            bootstrap_replicates=300, seed=7,
        )
        none = combine_current_forecasts(no_skill, current_regime="all", current_probabilities={"expert": 0.8})
        self.assertIsNone(none["probability"])
        self.assertEqual(none["forecast_status"], "NO_ELIGIBLE_FORECASTER")
        self.assertEqual(none["decision_output"], "NO_TRADE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
