import copy
import unittest

import serevex_probability_evolution_v4_v9 as evo
import serevex_forecast_ledger_producer_v1 as producer
from test_serevex_forecast_ledger_producer_v1 import digest as fixture_digest, synthetic_rows as synthetic_ohlcv_rows


def make_rows(n=720, *, candidate_mode="strong", regimes=1):
    rows = []
    for i in range(n):
        label = int(i % 2 == 0)
        if candidate_mode == "strong":
            candidate = 0.97 if label else 0.03
        elif candidate_mode == "weak":
            candidate = 0.51 if label else 0.49
        elif candidate_mode == "adverse":
            candidate = 0.03 if label else 0.97
        else:
            candidate = None
        rows.append({
            "decision_id": f"row-{i}",
            "decision_index": i,
            "partition": "final_holdout",
            "target_id": "tp-before-sl-h12",
            "instrument": "XAUUSD",
            "horizon_bars": 12,
            "candidate_probability": candidate,
            "baseline_probability": 0.5,
            "label": label,
            "label_mature": True,
            "label_available_index": i + 1,
            "regime": f"regime-{i % regimes}",
            "regime_available_index": i,
            "feature_available_index": i,
        })
    return rows


class ProbabilityEvolutionTests(unittest.TestCase):
    def test_version_order_is_preregistered_and_complete(self):
        self.assertEqual(
            evo.VERSION_ORDER,
            ("4.0", "4.1", "4.2", "5.0", "6.0", "7.0", "8.0", "9.0"),
        )

    def test_v40_is_reference_passthrough_for_existing_v4_candidate(self):
        result = evo.evaluate_probability_evolution(make_rows(30), version="4.0")
        self.assertAlmostEqual(result["forecasts"][10]["probability"], 0.97)
        self.assertEqual(result["forecasts"][10]["alpha_weight"], 1.0)
        self.assertEqual(result["evidence"]["scored_rows"], 29)
        self.assertEqual(result["evidence"]["decision_output"], "NO_TRADE")

    def test_as_of_index_excludes_later_forecasts_and_post_cutoff_rows(self):
        rows = make_rows(40)
        result = evo.evaluate_probability_evolution(
            rows, version="4.0", as_of_index=20
        )
        forecasts = result["forecasts"]
        self.assertEqual(len(forecasts), 20)
        self.assertTrue(all(row["decision_index"] < 20 for row in forecasts))
        self.assertEqual(result["evidence"]["forecast_rows_emitted"], 20)
        self.assertEqual(result["evidence"]["rows_at_or_after_as_of_excluded"], 20)
        self.assertEqual(result["evidence"]["last_emitted_decision_index"], 19)
        self.assertEqual(result["evidence"]["scored_rows"], 19)
        self.assertEqual(
            result["evidence"]["decision_output"], "NO_TRADE"
        )

    def test_as_of_before_first_decision_emits_no_forecasts(self):
        result = evo.evaluate_probability_evolution(
            make_rows(10), version="9.0", as_of_index=0
        )
        self.assertEqual(result["forecasts"], [])
        self.assertEqual(result["evidence"]["forecast_rows_emitted"], 0)
        self.assertEqual(result["evidence"]["rows_at_or_after_as_of_excluded"], 10)
        self.assertIsNone(result["evidence"]["last_emitted_decision_index"])
        self.assertEqual(result["evidence"]["scored_rows"], 0)
        self.assertEqual(result["evidence"]["decision_output"], "NO_TRADE")

    def test_score_partition_filters_metrics_but_not_causal_forecast_stream(self):
        rows = make_rows(40)
        for row in rows:
            row["partition"] = "development" if row["decision_index"] < 20 else "final_holdout"
        result = evo.evaluate_probability_evolution(
            rows, version="4.0", as_of_index=40, score_partition="final_holdout"
        )
        self.assertEqual(len(result["forecasts"]), 40)
        self.assertEqual(result["evidence"]["forecast_rows_emitted"], 40)
        self.assertEqual(result["evidence"]["score_partition"], "final_holdout")
        self.assertEqual(result["evidence"]["scored_rows"], 19)
        scored_indices = [
            row["decision_index"] for row in rows
            if row["partition"] == "final_holdout"
            and row["label_mature"] and row["label_available_index"] < 40
        ]
        self.assertEqual(scored_indices, list(range(20, 39)))

    def test_future_feature_is_quarantined_and_baseline_is_used(self):
        rows = make_rows(10)
        rows[4]["feature_available_index"] = 5
        result = evo.evaluate_probability_evolution(rows, version="4.1")
        out = result["forecasts"][4]
        self.assertEqual(out["probability"], 0.5)
        self.assertIn("FUTURE_OR_INVALID_FEATURE_AVAILABILITY", out["reason_codes"])
        self.assertEqual(out["alpha_weight"], 0.0)

    def test_future_regime_is_replaced_with_unknown(self):
        rows = make_rows(10)
        rows[4]["regime_available_index"] = 9
        result = evo.evaluate_probability_evolution(rows, version="6.0")
        self.assertEqual(result["forecasts"][4]["regime"], evo.UNKNOWN_REGIME)
        self.assertIn("REGIME_NOT_AVAILABLE_AT_DECISION", result["forecasts"][4]["reason_codes"])

    def test_invalid_baseline_fails_closed(self):
        rows = make_rows(10)
        rows[3]["baseline_probability"] = float("nan")
        with self.assertRaisesRegex(evo.ProbabilityInputError, "invalid_or_missing_causal_baseline"):
            evo.evaluate_probability_evolution(rows, version="4.1")

    def test_duplicate_index_fails_closed(self):
        rows = make_rows(10)
        rows[3]["decision_index"] = 2
        with self.assertRaisesRegex(evo.ProbabilityInputError, "duplicate_decision_index"):
            evo.evaluate_probability_evolution(rows, version="4.0")

    def test_mixed_target_cohorts_are_rejected(self):
        rows = make_rows(10)
        rows[3]["target_id"] = "different-horizon"
        with self.assertRaisesRegex(evo.ProbabilityInputError, "mixed_forecast_cohorts"):
            evo.evaluate_probability_evolution(rows, version="4.0")

    def test_future_label_maturity_does_not_enter_current_history(self):
        rows = make_rows(20)
        rows[2]["label_available_index"] = 18
        normalized = evo._normalize_rows(rows)
        hist = evo._matured_history(normalized[:10], 10)
        self.assertNotIn("row-2", [row["decision_id"] for row in hist])
        self.assertNotIn(2, [row["decision_index"] for row in hist])

    def test_invalid_candidate_uses_exact_baseline(self):
        rows = make_rows(10)
        rows[5]["candidate_probability"] = None
        rows[5]["baseline_probability"] = 0.37
        result = evo.evaluate_probability_evolution(rows, version="9.0")
        out = result["forecasts"][5]
        self.assertAlmostEqual(out["probability"], 0.37)
        self.assertEqual(out["alpha_weight"], 0.0)
        self.assertIn("INVALID_OR_MISSING_CANDIDATE_PROBABILITY", out["reason_codes"])

    def test_v42_requires_paired_dual_score_evidence(self):
        rows = make_rows(30)
        result = evo.evaluate_probability_evolution(rows, version="4.2")
        self.assertAlmostEqual(result["forecasts"][-1]["probability"], 0.5)
        self.assertEqual(result["forecasts"][-1]["status"], "BASELINE_FALLBACK_DUAL_SCORE_NOT_ESTABLISHED")

    def test_all_adaptive_versions_fall_back_during_empty_matured_history(self):
        rows = make_rows(24)
        for version in ("5.0", "6.0", "7.0", "8.0", "9.0"):
            result = evo.evaluate_probability_evolution(rows, version=version)
            first = result["forecasts"][0]
            self.assertEqual(first["probability"], first["baseline_probability"])
            self.assertEqual(first["alpha_weight"], 0.0)
            self.assertTrue(
                "BASELINE" in first["status"] or "FALLBACK" in first["status"],
                first["status"],
            )

    def test_v50_selects_alpha_only_after_selection_and_later_gate(self):
        rows = make_rows(720)
        result = evo.evaluate_probability_evolution(rows, version="5.0")
        forecast = result["forecasts"][-1]
        self.assertGreater(forecast["alpha_weight"], 0.0)
        self.assertIn("CHRONOLOGICAL_DUAL_SCORE_GATE", forecast["status"])
        diag = forecast["diagnostics"]["global_weight_selection"]
        self.assertGreaterEqual(diag["selection_blocks"], 8)
        self.assertGreaterEqual(diag["gate_blocks"], 4)

    def test_selection_and_confirmation_use_disjoint_absolute_time_blocks(self):
        normalized = evo._normalize_rows(make_rows(720))
        sparse = [row for row in normalized if int(row["decision_index"]) % 7 not in (1, 2, 5)]
        choice = evo._choose_weight(sparse, config=evo._default_config(None), min_gate_blocks=4)
        self.assertEqual(choice["selection_cutoff_index"] % (evo.DEFAULT_BLOCK_SPAN + evo.DEFAULT_EMBARGO_BARS), 0)
        self.assertEqual(set(choice["selection_source_block_ids"]) & set(choice["gate_source_block_ids"]), set())
        self.assertGreaterEqual(choice["selection_blocks"], 8)

    def test_v60_reports_regime_shrinkage_or_safe_global_fallback(self):
        rows = make_rows(720, regimes=1)
        result = evo.evaluate_probability_evolution(rows, version="6.0")
        out = result["forecasts"][-1]
        regime_diag = out["diagnostics"]["regime_weight_selection"]
        self.assertIn(regime_diag["status"], {
            "REGIME_WEIGHT_SHRUNK_TO_GLOBAL",
            "GLOBAL_FALLBACK_REGIME_SUPPORT_LOW",
            "GLOBAL_FALLBACK_LOCAL_GATE_FAILED",
        })
        self.assertGreaterEqual(out["alpha_weight"], 0.0)
        self.assertLessEqual(out["alpha_weight"], 1.0)

    def test_v70_uses_a_disjoint_later_gate_without_recursive_baseline_deadlock(self):
        rows = make_rows(720)
        result = evo.evaluate_probability_evolution(rows, version="7.0")
        out = result["forecasts"][-1]
        self.assertGreater(out["alpha_weight"], 0.0)
        uncertainty = out["diagnostics"]["uncertainty_gate"]
        self.assertEqual(uncertainty["method"], "held_out_chronological_block_lower_screen")
        self.assertGreaterEqual(uncertainty["minimum_independent_blocks"], 8)

    def test_drift_alarm_uses_only_prior_issued_forecasts(self):
        rows = []
        for i in range(600):
            label = int(i % 2 == 0)
            row = {
                "decision_id": f"drift-{i}",
                "decision_index": i,
                "_baseline": 0.5,
                "_y": label,
                "label": label,
                "label_mature": True,
                "label_available_index": i + 1,
                "_issued_probability": (0.95 if label else 0.05) if i < 400 else (0.05 if label else 0.95),
            }
            rows.append(row)
        cfg = evo._default_config({
            "drift_recent_bars": 192,
            "drift_min_rows": 300,
        })
        diagnostic = evo._drift_diagnostic(rows[:-1], config=cfg)
        self.assertIn("alarm", diagnostic)
        self.assertTrue(diagnostic["alarm"])
        self.assertEqual(diagnostic["status"], "DRIFT_ALARM_BASELINE_FALLBACK")

    def test_failed_selection_diagnostics_are_retained_and_confirmation_gate_is_marked_not_run(self):
        rows = make_rows(720, candidate_mode="adverse")
        result = evo.evaluate_probability_evolution(rows, version="7.0")
        out = result["forecasts"][-1]
        self.assertEqual(out["alpha_weight"], 0.0)
        uncertainty = out["diagnostics"]["uncertainty_gate"]
        self.assertEqual(uncertainty["status"], "NOT_RUN_SELECTION_GATE_FAILED")
        self.assertIsNone(uncertainty["evidence"])
        global_trace = [
            item for item in result["policy_trace"]
            if item.get("pool") == "global"
        ][-1]
        self.assertEqual(global_trace["status"], "BASELINE_ONLY_SELECTION_DUAL_SCORE_FAILED")
        diagnostics = global_trace["selection_candidate_diagnostics"]
        self.assertEqual(len(diagnostics), len(evo.BLEND_GRID) - 1)
        self.assertTrue(all(item["eligible"] is False for item in diagnostics))
        self.assertTrue(all(item["brier_lower"] is not None for item in diagnostics))
        self.assertEqual(global_trace["confirmation_gate_status"], "NOT_RUN_SELECTION_GATE_FAILED")

    def test_v80_exposes_drift_diagnostic_and_never_authorizes_trading(self):
        rows = make_rows(720)
        result = evo.evaluate_probability_evolution(rows, version="8.0")
        out = result["forecasts"][-1]
        self.assertIn("drift_guard", out["diagnostics"])
        self.assertEqual(out["decision_output"], "NO_TRADE")
        self.assertFalse(out["live_order_execution_enabled"])

    def test_v90_emits_stable_hashes_and_explicit_nonpromotion_state(self):
        rows = make_rows(720)
        first = evo.evaluate_probability_evolution(rows, version="9.0")
        second = evo.evaluate_probability_evolution(rows, version="9.0")
        self.assertEqual(first["evidence"]["forecast_manifest_sha256"], second["evidence"]["forecast_manifest_sha256"])
        self.assertEqual(first["evidence"]["evaluation_report_sha256"], second["evidence"]["evaluation_report_sha256"])
        self.assertFalse(first["evidence"]["model_validated"])
        self.assertFalse(first["evidence"]["deployment_calibrated"])
        self.assertEqual(first["evidence"]["decision_output"], "NO_TRADE")

    def test_all_versions_use_identical_scoring_cohort_and_no_auto_promotion(self):
        result = evo.evaluate_all_probability_versions(make_rows(720))
        self.assertEqual(tuple(result["versions"]), evo.VERSION_ORDER)
        scored = {entry["scored_rows"] for entry in result["comparison"].values()}
        self.assertEqual(len(scored), 1)
        self.assertEqual(result["status"], "RESEARCH_COMPARISON_NOT_PROMOTION")
        self.assertFalse(result["model_validated"])
        self.assertEqual(result["decision_output"], "NO_TRADE")

    def test_unknown_version_and_unknown_config_fail_closed(self):
        with self.assertRaisesRegex(evo.ProbabilityInputError, "unsupported_probability_version"):
            evo.evaluate_probability_evolution(make_rows(10), version="10.0")
        with self.assertRaisesRegex(evo.ProbabilityInputError, "unknown_policy_config"):
            evo.evaluate_probability_evolution(make_rows(10), config={"allow_live": True})
        with self.assertRaisesRegex(evo.ProbabilityInputError, "unsupported_score_partition"):
            evo.evaluate_probability_evolution(make_rows(10), score_partition="live_trading")

    def test_non_mapping_rows_use_typed_input_error(self):
        with self.assertRaisesRegex(evo.ProbabilityInputError, "forecast_row_not_mapping"):
            evo.evaluate_probability_evolution([None], version="4.0")


    def test_matured_label_cannot_mature_on_or_before_its_decision(self):
        rows = make_rows(20)
        rows[5]["label_available_index"] = 5
        with self.assertRaisesRegex(evo.ProbabilityInputError, "label_maturity_not_after_decision"):
            evo.evaluate_probability_evolution(rows, version="4.0")

    def test_canonical_ledger_adapter_preserves_source_qualification_boundary(self):
        ledger = {
            "schema_version": "serevex.forecast-ledger.v1",
            "metadata": {
                "series_id": "XAUUSD:OHLCV:momentum_5:long:tp-before-sl",
                "target_id": "xauusd:tp-before-sl:momentum_5:long:h12",
                "target_spec": {"horizon_bars_from_next_open": 12},
                "dataset_sha256": "a" * 64,
                "source_lineage_sha256": "b" * 64,
                "source_verification": {
                    "provider_provenance_independently_verified": False,
                    "data_rights_verified": False,
                    "broker_feed_match_verified": False,
                },
                "source_timestamp_semantics_verified": False,
                "historical_feature_availability_verified": False,
                "broker_quote_basis_verified": False,
            },
            "records": [
                {
                    "decision_id": f"ledger-{i}",
                    "decision_index": i,
                    "partition": "final_holdout",
                    "label_available_index": i + 1,
                    "label": int(i % 2 == 0),
                    "label_mature": True,
                    "regime_id": "regime-a",
                    "regime_available_index": i,
                    "state_tags_available_index": i,
                    "baseline_probability": 0.5,
                    "expert_probabilities": {
                        "alpha_logistic_v1": 0.95 if i % 2 == 0 else 0.05,
                        "alpha_calibrated_logistic_v1": 0.91 if i % 2 == 0 else 0.09,
                    },
                }
                for i in range(40)
            ],
        }
        rows, source = evo.adapt_forecast_ledger(ledger)
        self.assertEqual(len(rows), 40)
        self.assertEqual(rows[0]["candidate_probability"], 0.91)
        self.assertEqual(rows[0]["target_id"], ledger["metadata"]["target_id"])
        self.assertEqual(
            [(row["decision_id"], row["decision_index"]) for row in rows],
            [(row["decision_id"], row["decision_index"]) for row in ledger["records"]],
        )
        self.assertFalse(source["source_timestamp_semantics_verified"])
        report = evo.evaluate_forecast_ledger_versions(ledger, as_of_index=40)
        self.assertEqual(tuple(report["versions"]), evo.VERSION_ORDER)
        self.assertEqual(report["score_partition"], "final_holdout")
        self.assertEqual(report["source_evidence"]["dataset_sha256"], "a" * 64)
        self.assertEqual(report["source_evidence"]["candidate_expert"], "alpha_calibrated_logistic_v1")
        self.assertFalse(report["research_boundary"]["source_qualified_for_training"])
        self.assertEqual(report["decision_output"], "NO_TRADE")
        self.assertEqual(
            [row["decision_id"] for row in report["versions"]["9.0"]["forecasts"]],
            [row["decision_id"] for row in ledger["records"]],
        )
        self.assertEqual(report["versions"]["9.0"]["evidence"]["scored_rows"], 39)

    def test_synthetic_producer_to_v4_v9_end_to_end_ledger_reconciliation(self):
        # Exercise the real ledger producer -> adapter -> all governance versions
        # without network/data-source dependencies. This proves the wire contract,
        # not any predictive skill on market data.
        bars = synthetic_ohlcv_rows(1800)
        ledger = producer.build_forecast_ledger(
            bars,
            dataset_sha256=fixture_digest(b"v4-v9-end-to-end-synthetic-contract"),
            source_uri="https://example.invalid/synthetic-contract-only",
            rule="last_return_sign",
            side="long",
            evidence_kind="synthetic_contract_test",
            license_reference="SYNTHETIC_FIXTURE_NOT_EXTERNAL_DATA",
        )
        report = evo.evaluate_forecast_ledger_versions(
            ledger,
            candidate_expert="alpha_logistic_v1",
            as_of_index=len(bars),
            score_partition="final_holdout",
        )
        expected_identity = [
            (row["decision_id"], row["decision_index"])
            for row in sorted(ledger["records"], key=lambda row: row["decision_index"])
        ]
        self.assertEqual(report["score_partition"], "final_holdout")
        self.assertEqual(tuple(report["versions"]), evo.VERSION_ORDER)
        row_counts = {
            result["scored_rows"] for result in report["comparison"].values()
        }
        self.assertEqual(len(row_counts), 1)
        self.assertGreater(next(iter(row_counts)), 0)
        for version, result in report["versions"].items():
            evidence = result["evidence"]
            self.assertEqual(evidence["score_partition"], "final_holdout", version)
            self.assertEqual(evidence["decision_output"], "NO_TRADE", version)
            self.assertFalse(evidence["model_validated"], version)
            self.assertFalse(evidence["deployment_calibrated"], version)
            self.assertEqual(evidence["forecast_rows_emitted"], len(ledger["records"]), version)
            self.assertEqual(evidence["rows_at_or_after_as_of_excluded"], 0, version)
            actual_identity = [
                (row["decision_id"], row["decision_index"])
                for row in result["forecasts"]
            ]
            self.assertEqual(actual_identity, expected_identity, version)
        self.assertFalse(report["research_boundary"]["source_qualified_for_training"])
        self.assertEqual(report["decision_output"], "NO_TRADE")

    def test_adapter_rejects_unknown_schema_and_missing_expert_falls_back(self):
        with self.assertRaisesRegex(evo.ProbabilityInputError, "unsupported_forecast_ledger_schema"):
            evo.adapt_forecast_ledger({"schema_version": "unknown", "metadata": {}, "records": []})
        ledger = {
            "schema_version": "serevex.forecast-ledger.v1",
            "metadata": {
                "series_id": "XAUUSD:OHLCV:rule:long",
                "target_id": "target",
                "target_spec": {"horizon_bars_from_next_open": 12},
            },
            "records": [{
                "decision_id": "only",
                "decision_index": 0,
                "label": None,
                "label_mature": False,
                "label_available_index": 1,
                "baseline_probability": 0.41,
                "expert_probabilities": {},
                "state_tags_available_index": 0,
                "regime_available_index": 0,
            }],
        }
        result = evo.evaluate_forecast_ledger_versions(ledger, as_of_index=2)
        self.assertEqual(result["versions"]["9.0"]["forecasts"][0]["probability"], 0.41)
        self.assertEqual(result["versions"]["9.0"]["forecasts"][0]["alpha_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
