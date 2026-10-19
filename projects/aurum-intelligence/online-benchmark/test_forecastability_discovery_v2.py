import unittest

from forecastability_discovery_v2 import (
    assess_forecastability_v2,
    combine_frozen_forecasts_v2,
    _runs,
)


def make_records(n=1200, models=("expert",), *, mode="good"):
    rows = []
    for i in range(n):
        label = int((i * 7 % 13) < 6)
        regime = "gold" if (i % 20) < 10 else "silver"
        segment = "session-a" if i < n // 2 else "session-b"
        for model in models:
            if mode == "good" or (mode == "regime-specific" and regime == "gold"):
                correct = True
            elif mode == "regime-specific" and regime == "silver":
                correct = False
            elif mode == "drift" and i < n - 240:
                correct = True
            elif mode == "drift":
                correct = False
            else:
                raise AssertionError(mode)
            confidence = 0.76 + 0.18 * ((i % 11) / 11)
            p = confidence if label == 1 else 1 - confidence
            if not correct:
                p = 1 - p
            baseline = 0.43 + 0.13 * ((i % 7) / 7)
            rows.append({
                "decision_id": f"d-{i:05d}",
                "model_id": model,
                "decision_index": i,
                "label_available_index": i + 1,
                "regime_id": regime,
                "regime_available_index": i,
                "segment_id": segment,
                "probability": p,
                "baseline_probability": baseline,
                "label": label,
                "label_mature": True,
            })
    return rows


def evaluate(rows, models=("expert",), regimes=("gold", "silver"), as_of=1202):
    return assess_forecastability_v2(
        rows,
        as_of_index=as_of,
        registered_models=list(models),
        registered_regimes=list(regimes),
        window_bars=(120, 240),
        block_sizes=(6, 12),
        min_rows_per_regime_window=30,
        min_independent_blocks=4,
        bootstrap_replicates=400,
        familywise_alpha=0.05,
        minimum_skill=0.0,
        seed=37,
        evidence_kind="synthetic_contract_test",
    )


class ForecastabilityDiscoveryV2Tests(unittest.TestCase):
    def test_strong_synthetic_expert_is_replication_only_and_no_trade(self):
        report = evaluate(make_records())
        self.assertEqual(report["models"]["expert"]["gold"]["status"], "ELIGIBLE_FOR_INDEPENDENT_REPLICATION")
        self.assertEqual(report["models"]["expert"]["silver"]["status"], "ELIGIBLE_FOR_INDEPENDENT_REPLICATION")
        self.assertFalse(report["market_promotion_allowed"])
        self.assertEqual(report["decision_output"], "NO_TRADE")
        self.assertFalse(report["live_order_execution_enabled"])

    def test_regime_conditional_gate_rejects_expert_where_skill_reverses(self):
        report = evaluate(make_records(mode="regime-specific"))
        self.assertEqual(report["models"]["expert"]["gold"]["status"], "ELIGIBLE_FOR_INDEPENDENT_REPLICATION")
        self.assertNotEqual(report["models"]["expert"]["silver"]["status"], "ELIGIBLE_FOR_INDEPENDENT_REPLICATION")
        self.assertGreater(report["models"]["expert"]["gold"]["weight"], 0)
        self.assertEqual(report["models"]["expert"]["silver"]["weight"], 0)

    def test_null_or_degenerate_model_cannot_be_promoted(self):
        rows = make_records()
        for row in rows:
            row["probability"] = row["baseline_probability"]
        report = evaluate(rows)
        self.assertNotEqual(report["models"]["expert"]["gold"]["status"], "ELIGIBLE_FOR_INDEPENDENT_REPLICATION")
        self.assertNotEqual(report["models"]["expert"]["silver"]["status"], "ELIGIBLE_FOR_INDEPENDENT_REPLICATION")

    def test_unmatured_or_at_cutoff_labels_are_excluded(self):
        rows = make_records(n=100)
        report = evaluate(rows, as_of=50)
        self.assertEqual(report["matured_decisions"], 49)

    def test_future_regime_metadata_fails_closed(self):
        rows = make_records(n=100)
        rows[8]["regime_available_index"] = 9
        with self.assertRaisesRegex(ValueError, "regime_not_available_as_of_decision"):
            evaluate(rows, as_of=102)

    def test_unaligned_model_cohorts_fail_closed(self):
        rows = make_records(n=100, models=("expert", "challenger"))
        rows = [row for row in rows if not (row["decision_id"] == "d-00020" and row["model_id"] == "challenger")]
        with self.assertRaisesRegex(ValueError, "unaligned_model_cohort"):
            evaluate(rows, models=("expert", "challenger"), as_of=102)

    def test_cross_model_label_and_baseline_must_match(self):
        rows = make_records(n=100, models=("expert", "challenger"))
        rows[-1]["baseline_probability"] = 0.99
        with self.assertRaisesRegex(ValueError, "cross_model_cohort_mismatch"):
            evaluate(rows, models=("expert", "challenger"), as_of=102)

    def test_duplicate_decision_model_is_rejected(self):
        rows = make_records(n=100)
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate_decision_model_pair"):
            evaluate(rows, as_of=102)

    def test_missing_and_future_labels_are_not_required_to_be_known(self):
        rows = make_records(n=100)
        for row in rows:
            if row["decision_index"] >= 50:
                row["label_mature"] = False
                row["label"] = None
        report = evaluate(rows, as_of=50)
        self.assertEqual(report["matured_decisions"], 49)
        self.assertEqual(report["decision_output"], "NO_TRADE")

    def test_bootstrap_runs_stop_at_index_gaps_and_session_edges(self):
        rows = [
            {"decision_index": 0, "segment_id": "a"},
            {"decision_index": 1, "segment_id": "a"},
            {"decision_index": 3, "segment_id": "a"},
            {"decision_index": 4, "segment_id": "b"},
        ]
        self.assertEqual(_runs(rows), [[0, 1], [2], [3]])

    def test_combiner_never_grants_trading_authority(self):
        report = evaluate(make_records())
        forecast = combine_frozen_forecasts_v2(
            report, current_regime="gold", current_probabilities={"expert": 0.8}
        )
        self.assertIsNotNone(forecast["probability"])
        self.assertEqual(forecast["decision_output"], "NO_TRADE")
        self.assertFalse(forecast["live_signal_enabled"])
        no_skill = evaluate([
            {**row, "probability": row["baseline_probability"]}
            for row in make_records()
        ])
        none = combine_frozen_forecasts_v2(
            no_skill, current_regime="gold", current_probabilities={"expert": 0.8}
        )
        self.assertIsNone(none["probability"])
        self.assertEqual(none["forecast_status"], "NO_ELIGIBLE_FORECASTER")
        self.assertEqual(none["decision_output"], "NO_TRADE")


    def test_simultaneous_max_t_bounds_are_reported_for_each_window_and_block_size(self):
        report = evaluate(make_records())
        self.assertIn("max-|t|", report["simultaneous_method"])
        entry = report["models"]["expert"]["gold"]
        self.assertEqual(set(entry["windows"]), {"120", "240"})
        for window in entry["windows"].values():
            self.assertEqual(set(window["block_sensitivity"]), {"6", "12"})
            for result in window["block_sensitivity"].values():
                self.assertIn("simultaneous_lower_bound", result)
                self.assertIn("simultaneous_upper_bound", result)
                self.assertGreaterEqual(result["replicates"], 200)
        self.assertEqual(report["decision_output"], "NO_TRADE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
