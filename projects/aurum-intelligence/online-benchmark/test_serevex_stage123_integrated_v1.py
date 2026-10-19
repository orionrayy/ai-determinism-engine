import unittest
from datetime import datetime, timedelta, timezone

from serevex_stage123_integrated_v1 import (
    validate_forecast_ledger,
    build_predictability_atlas,
    run_delayed_feedback_mixer,
    run_serevex_stage123,
)

def digest(label):
    import hashlib
    return hashlib.sha256(label.encode("utf-8")).hexdigest()

def make_ledger(n=600, *, leak_at=None, weak_second=False, partitions=None):
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = []
    for i in range(n):
        y = int((i % 10) < 4)
        regime = "trend" if i < n // 2 else "range"
        p_good = 0.9 if y == 1 else 0.1
        p_other = 0.55 if y == 1 else 0.45
        if weak_second:
            p_good, p_other = p_other, p_good
        decision_time = base_time + timedelta(minutes=i)
        if leak_at is not None and i == leak_at:
            available_time = decision_time + timedelta(seconds=1)
            observed_time = available_time
        else:
            available_time = decision_time - timedelta(seconds=5)
            observed_time = decision_time - timedelta(seconds=2)
        split = (partitions[i] if partitions else "final_holdout")
        records.append({
            "decision_id": "xau-m1-" + str(i),
            "decision_index": i,
            "decision_time": decision_time.isoformat().replace("+00:00", "Z"),
            "partition": split,
            "label_available_index": i + 1,
            "label_available_time": (decision_time + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
            "label": y,
            "label_mature": True,
            "regime_id": regime,
            "regime_available_index": i,
            "regime_available_time": (decision_time - timedelta(seconds=3)).isoformat().replace("+00:00", "Z"),
            "session_id": "london" if (i // 120) % 2 == 0 else "new_york",
            "baseline_probability": 0.4,
            "expert_probabilities": {"trend_expert": p_good, "alt_expert": p_other},
            "state_tags": {
                "volatility_bucket": "high" if i % 3 else "low",
                "spread_bucket": "normal" if i % 5 else "wide",
                "event_risk_bucket": "scheduled" if i % 7 == 0 else "quiet",
            },
            "state_tags_available_index": i,
            "state_tags_available_time": (decision_time - timedelta(seconds=4)).isoformat().replace("+00:00", "Z"),
            "feature_facts": [{
                "feature_id": "ema_fast_gap",
                "available_at": available_time.isoformat().replace("+00:00", "Z"),
                "observed_at": observed_time.isoformat().replace("+00:00", "Z"),
                "source_revision": "feed-v1",
                "source_sha256": digest("source-xau"),
                "transform_id": "ema-gap-v1",
                "transform_sha256": digest("ema-transform"),
                "semantic_type": "price_derived_feature",
                "quote_basis": "mid",
                "quality_status": "REFERENCE_ONLY",
                "value": float((i % 7) / 10),
            }],
        })
    return {
        "schema_version": "serevex.forecast-ledger.v1",
        "metadata": {
            "series_id": "XAUUSD:M1:direction_5bar",
            "target_id": "xau-direction-next-5-bars",
            "target_definition_sha256": digest("binary-target-v1"),
            "dataset_sha256": digest("dataset-xau-m1"),
            "source_lineage_sha256": digest("source-lineage"),
            "model_artifact_sha256_by_expert": {
                "trend_expert": digest("model-trend"),
                "alt_expert": digest("model-alt"),
            },
            "registered_experts": ["trend_expert", "alt_expert"],
            "registered_regimes": ["trend", "range"],
            "atlas_category_registry": {
                "regime_id": ["trend", "range"],
                "session_id": ["london", "new_york"],
                "volatility_bucket": ["high", "low", "not_observed_in_holdout"],
                "spread_bucket": ["normal", "wide"],
                "event_risk_bucket": ["scheduled", "quiet"],
            },
            "evidence_kind": "synthetic_contract_test",
            "source_verification": {
                "provider_provenance_independently_verified": False,
                "data_rights_verified": False,
                "broker_feed_match_verified": False,
            },
            "decision_time_basis": "UTC close of completed M1 bar",
            "label_definition": "synthetic binary target for contract tests only",
        },
        "records": records,
    }

class SerevexStage123Tests(unittest.TestCase):
    def test_ledger_passes_and_digest_is_deterministic(self):
        ledger = make_ledger(80)
        a = validate_forecast_ledger(ledger)
        b = validate_forecast_ledger(ledger)
        self.assertEqual(a["contract_status"], "PASS")
        self.assertEqual(a["ledger_sha256"], b["ledger_sha256"])
        self.assertEqual(a["point_in_time_feature_fact_count"], 80)
        self.assertFalse(a["all_source_verification_flags_true"])
        self.assertFalse(a["source_authenticity_independently_verified"])

    def test_future_feature_fact_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "future_feature_leakage"):
            validate_forecast_ledger(make_ledger(20, leak_at=7))

    def test_timezone_naive_decision_time_is_rejected(self):
        ledger = make_ledger(20)
        ledger["records"][4]["decision_time"] = "2026-01-01T00:04:00"
        with self.assertRaisesRegex(ValueError, "timezone_aware"):
            validate_forecast_ledger(ledger)

    def test_future_regime_is_rejected(self):
        ledger = make_ledger(20)
        ledger["records"][4]["regime_available_index"] = 5
        with self.assertRaisesRegex(ValueError, "regime_not_available"):
            validate_forecast_ledger(ledger)

    def test_label_timestamp_cannot_postdate_its_declared_maturity_index(self):
        ledger = make_ledger(20)
        decision_time = datetime.fromisoformat(ledger["records"][0]["decision_time"].replace("Z", "+00:00"))
        ledger["records"][0]["label_available_time"] = (decision_time + timedelta(minutes=2)).isoformat().replace("+00:00", "Z")
        with self.assertRaisesRegex(ValueError, "label_time_after_declared_maturity_index"):
            validate_forecast_ledger(ledger)

    def test_future_state_tags_time_is_rejected(self):
        ledger = make_ledger(20)
        decision_time = datetime.fromisoformat(ledger["records"][4]["decision_time"].replace("Z", "+00:00"))
        ledger["records"][4]["state_tags_available_time"] = (decision_time + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
        with self.assertRaisesRegex(ValueError, "future_state_tags"):
            validate_forecast_ledger(ledger)

    def test_partition_order_cannot_go_backward(self):
        ledger = make_ledger(4, partitions=["development", "calibration", "model_selection", "final_holdout"])
        with self.assertRaisesRegex(ValueError, "chronologically_ordered"):
            validate_forecast_ledger(ledger)

    def test_atlas_finds_synthetic_skill_but_never_authorizes_promotion(self):
        ledger = validate_forecast_ledger(make_ledger(600))
        result = build_predictability_atlas(
            ledger["normalized_ledger"]["records"],
            registered_experts=["trend_expert", "alt_expert"],
            as_of_index=610,
            registered_category_registry=ledger["normalized_ledger"]["metadata"]["atlas_category_registry"],
            min_rows=40, block_size=6, min_independent_blocks=4,
            bootstrap_replicates=200, axes=["regime_id"],
        )
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertTrue(any(x["expert_id"] == "trend_expert" and x["status"] == "ELIGIBLE_FOR_INDEPENDENT_REPLICATION" for x in result["results"]))
        missing_cell = next(x for x in result["results"] if x["category"] == "not_observed_in_holdout")
        self.assertEqual(missing_cell["status"], "INSUFFICIENT_SUPPORT")
        self.assertFalse(result["model_validated"])
        self.assertFalse(result["live_signal_enabled"])
        self.assertEqual(result["decision_output"], "NO_TRADE")

    def test_mixer_uses_baseline_until_matured_support_exists(self):
        ledger = validate_forecast_ledger(make_ledger(80))
        result = run_delayed_feedback_mixer(
            ledger["normalized_ledger"]["records"],
            registered_experts=["trend_expert", "alt_expert"],
            as_of_index=100, min_global_matured=10, min_regime_matured=1000,
        )
        forecasts = result["forecast_ledger"]
        self.assertEqual(forecasts[0]["forecast_source"], "BASELINE_FALLBACK_INSUFFICIENT_MATURED_HISTORY")
        self.assertAlmostEqual(forecasts[0]["probability"], forecasts[0]["baseline_probability"])
        self.assertEqual(result["decision_output"], "NO_TRADE")

    def test_mixer_updates_only_after_strict_maturity(self):
        ledger = validate_forecast_ledger(make_ledger(30))
        result = run_delayed_feedback_mixer(
            ledger["normalized_ledger"]["records"],
            registered_experts=["trend_expert", "alt_expert"],
            as_of_index=40, min_global_matured=1, min_regime_matured=1000,
            fixed_share_rate=0.0,
        )
        forecasts = result["forecast_ledger"]
        self.assertEqual(forecasts[1]["matured_labels_seen_globally_before_decision"], 0)
        self.assertEqual(forecasts[2]["matured_labels_seen_globally_before_decision"], 1)
        self.assertTrue(result["updates_only_after_strict_label_maturity"])

    def test_mixer_does_not_process_rows_after_as_of_cutoff(self):
        ledger = validate_forecast_ledger(make_ledger(80))
        records = ledger["normalized_ledger"]["records"]
        kwargs = dict(
            registered_experts=["trend_expert", "alt_expert"], as_of_index=40,
            min_global_matured=5, min_regime_matured=10, eta=2.0, fixed_share_rate=0.02,
        )
        full = run_delayed_feedback_mixer(records, **kwargs)
        prefix = run_delayed_feedback_mixer(records[:40], **kwargs)
        self.assertEqual(full["forecast_ledger"], prefix["forecast_ledger"])
        self.assertEqual(full["final_global_weights"], prefix["final_global_weights"])
        self.assertEqual(full["final_global_weights_sha256"], prefix["final_global_weights_sha256"])
        self.assertEqual(full["rows_at_or_after_as_of_excluded"], 40)
        self.assertEqual(full["last_processed_decision_index"], 39)
        self.assertEqual(full["decision_output"], "NO_TRADE")

    def test_integrated_run_keeps_authority_boundary_and_report_hash(self):
        result = run_serevex_stage123(
            make_ledger(600), as_of_index=610, atlas_axes=["regime_id"],
            atlas_min_rows=40, atlas_block_size=6, atlas_min_independent_blocks=4,
            atlas_bootstrap_replicates=200, min_global_matured=20, min_regime_matured=30,
        )
        self.assertEqual(result["status"], "PASS_CONTRACT_DIAGNOSTIC_ONLY")
        self.assertEqual(result["stage_1_ledger_contract"], "PASS")
        self.assertEqual(result["decision_output"], "NO_TRADE")
        self.assertFalse(result["predictive_skill_proven"])
        self.assertFalse(result["model_validated"])
        self.assertEqual(len(result["report_sha256"]), 64)

    def test_invalid_expert_probability_fails_closed(self):
        ledger = make_ledger(20)
        ledger["records"][3]["expert_probabilities"]["trend_expert"] = 1.01
        with self.assertRaisesRegex(ValueError, "finite_probability"):
            validate_forecast_ledger(ledger)

if __name__ == "__main__":
    unittest.main()
