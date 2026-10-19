import unittest
from datetime import datetime, timedelta, timezone

from temporal_semantic_contract_v1 import (
    CONTRACT_VERSION,
    validate_candle_series,
    validate_feature_batch,
    validate_feature_fact,
    validate_feature_lineage,
    validate_ohlc_candle,
)

UTC = timezone.utc


class TemporalSemanticContractTests(unittest.TestCase):
    def fact(self):
        return {
            "schema_version": CONTRACT_VERSION,
            "feature_id": "xau.close.m1",
            "source_id": "feed-a",
            "instrument": "XAUUSD",
            "semantic_type": "market_bar",
            "event_time_role": "bar_close",
            "value_type": "number",
            "value": 2350.25,
            "unit": "USD_per_troy_ounce",
            "quote_basis": "bid",
            "event_time": "2026-01-01T00:04:00Z",
            "published_at": "2026-01-01T00:04:01Z",
            "available_at": "2026-01-01T00:04:02Z",
            "observed_at": "2026-01-01T00:04:03Z",
            "source_record_id": "feed-a:XAUUSD:M1:2026-01-01T00:04:00Z:bid",
            "source_revision": "rev-1",
            "source_sha256": "a" * 64,
            "transform_id": "identity",
            "transform_version": "1.0.0",
            "transform_sha256": "b" * 64,
            "timezone_semantics_verified": True,
            "availability_semantics_verified": True,
            "revision_semantics_verified": True,
            "provenance_status": "VERIFIED",
            "rights_status": "VERIFIED",
            "quality_status": "PASS",
        }

    def candle(self, start="2026-01-01T00:00:00Z", end="2026-01-01T00:01:00Z"):
        return {
            "instrument": "XAUUSD",
            "source_id": "feed-a",
            "session_id": "session-1",
            "bar_start": start,
            "bar_end": end,
            "open": 2350.0,
            "high": 2351.0,
            "low": 2349.5,
            "close": 2350.5,
            "volume": 12.0,
        }

    def test_complete_verified_feature_fact_is_eligible_for_research_only(self):
        report = validate_feature_fact(
            self.fact(),
            decision_time="2026-01-01T00:05:00Z",
            max_age_seconds=120,
        )
        self.assertTrue(report["eligible_at_decision_time"], report)
        self.assertEqual(report["status"], "ELIGIBLE_FOR_RESEARCH_FEATURE_SET")
        self.assertEqual(report["information_age_seconds"], 58.0)
        self.assertEqual(report["promotion_status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(report["decision_output"], "NO_TRADE")
        self.assertFalse(report["live_signal_enabled"])
        self.assertFalse(report["live_order_execution_enabled"])

    def test_available_after_decision_is_rejected_even_if_event_time_is_old(self):
        fact = self.fact()
        fact["available_at"] = "2026-01-01T00:05:01Z"
        fact["observed_at"] = "2026-01-01T00:05:02Z"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("unavailable_at_decision_time", report["blockers"])
        self.assertFalse(report["eligible_at_decision_time"])

    def test_known_ahead_scheduled_event_can_be_eligible_when_metadata_proves_availability(self):
        fact = self.fact()
        fact["semantic_type"] = "scheduled_event"
        fact["event_time_role"] = "scheduled_future_event"
        fact["event_time"] = "2026-01-01T12:00:00Z"
        fact["feature_id"] = "calendar.rate-decision"
        fact["value_type"] = "category"
        fact["value"] = "scheduled"
        fact["quote_basis"] = "not_applicable"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertTrue(report["eligible_at_decision_time"], report)

    def test_future_bar_close_is_rejected_even_if_other_timestamps_look_valid(self):
        fact = self.fact()
        fact["event_time"] = "2026-01-01T00:06:00Z"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("event_time_after_decision_for_past_only_role", report["blockers"])

    def test_naive_timestamp_fails_closed(self):
        fact = self.fact()
        fact["available_at"] = "2026-01-01T00:04:02"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("available_at_timezone_unverified", report["blockers"])

    def test_published_available_observed_order_is_enforced(self):
        fact = self.fact()
        fact["available_at"] = "2026-01-01T00:03:59Z"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("available_before_published_timestamp", report["blockers"])
        fact = self.fact()
        fact["observed_at"] = "2026-01-01T00:04:01Z"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("observed_before_available_timestamp", report["blockers"])

    def test_staleness_is_policy_driven_and_never_silently_ignored(self):
        report = validate_feature_fact(
            self.fact(),
            decision_time="2026-01-01T00:05:00Z",
            max_age_seconds=30,
        )
        self.assertIn("feature_exceeds_maximum_age", report["blockers"])
        self.assertFalse(report["eligible_at_decision_time"])

    def test_unverified_rights_provenance_revision_and_hashes_block(self):
        fact = self.fact()
        fact["rights_status"] = "UNKNOWN"
        fact["provenance_status"] = "UNVERIFIED"
        fact["revision_semantics_verified"] = False
        fact["source_sha256"] = "not-a-hash"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        for reason in (
            "source_usage_rights_not_verified",
            "source_provenance_not_verified",
            "unverified_semantics:revision_semantics_verified",
            "invalid_sha256:source_sha256",
        ):
            self.assertIn(reason, report["blockers"])

    def test_value_type_and_quote_basis_cannot_be_ambiguous(self):
        fact = self.fact()
        fact["value_type"] = "number"
        fact["value"] = True
        fact["quote_basis"] = "unknown"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("value_does_not_match_declared_type", report["blockers"])
        self.assertIn("market_feature_requires_explicit_quote_basis", report["blockers"])

    def test_invalid_contract_arguments_raise_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "decision_time_timezone_unverified"):
            validate_feature_fact(self.fact(), decision_time="2026-01-01T00:05:00")
        with self.assertRaisesRegex(ValueError, "max_age_seconds"):
            validate_feature_fact(self.fact(), decision_time="2026-01-01T00:05:00Z", max_age_seconds=float("nan"))

    def test_valid_closed_candle_passes_without_repair_or_promotion(self):
        report = validate_ohlc_candle(
            self.candle(),
            decision_time="2026-01-01T00:01:00Z",
            expected_interval_seconds=60,
            timestamp_semantics_verified=True,
            quote_basis="bid",
        )
        self.assertTrue(report["eligible_for_closed_bar_research"], report)
        self.assertFalse(report["price_repair_applied"])
        self.assertFalse(report["source_promoted"])
        self.assertFalse(report["live_signal_enabled"])

    def test_unclosed_bar_and_naive_time_fail_closed(self):
        report = validate_ohlc_candle(
            self.candle(end="2026-01-01T00:02:00Z"),
            decision_time="2026-01-01T00:01:00Z",
            expected_interval_seconds=60,
            timestamp_semantics_verified=True,
            quote_basis="bid",
        )
        self.assertIn("bar_not_closed_by_decision_time", report["blockers"])
        report = validate_ohlc_candle(
            self.candle(start="2026-01-01T00:00:00", end="2026-01-01T00:01:00Z"),
            decision_time="2026-01-01T00:01:00Z",
            expected_interval_seconds=60,
            timestamp_semantics_verified=True,
            quote_basis="bid",
        )
        self.assertIn("bar_start_timezone_unverified", report["blockers"])

    def test_ohlc_geometry_prices_volume_and_quote_basis_are_enforced(self):
        candle = self.candle()
        candle["high"] = 2349.0
        candle["volume"] = -1
        report = validate_ohlc_candle(
            candle,
            decision_time="2026-01-01T00:01:00Z",
            expected_interval_seconds=60,
            timestamp_semantics_verified=True,
            quote_basis="unknown",
        )
        self.assertIn("high_violates_ohlc_geometry", report["blockers"])
        self.assertIn("volume_must_be_finite_nonnegative", report["blockers"])
        self.assertIn("candle_quote_basis_not_explicit", report["blockers"])

    def test_explicit_session_gap_is_allowed_without_fabricating_bars(self):
        first = self.candle()
        second = self.candle("2026-01-01T06:00:00Z", "2026-01-01T06:01:00Z")
        second["session_id"] = "session-2"
        report = validate_candle_series(
            [first, second],
            decision_time="2026-01-01T06:01:00Z",
            expected_interval_seconds=60,
            timestamp_semantics_verified=True,
            quote_basis="bid",
        )
        self.assertEqual(report["status"], "PASS", report)
        self.assertFalse(report["missing_bars_synthesized"])

    def test_implicit_gap_inside_session_is_rejected(self):
        first = self.candle()
        second = self.candle("2026-01-01T00:02:00Z", "2026-01-01T00:03:00Z")
        report = validate_candle_series(
            [first, second],
            decision_time="2026-01-01T00:03:00Z",
            expected_interval_seconds=60,
            timestamp_semantics_verified=True,
            quote_basis="bid",
        )
        self.assertIn("row_1:implicit_gap_within_session", report["blockers"])

    def test_overlapping_bars_and_recurring_session_ids_are_rejected(self):
        first = self.candle()
        overlap = self.candle("2026-01-01T00:00:30Z", "2026-01-01T00:01:30Z")
        report = validate_candle_series(
            [first, overlap],
            decision_time="2026-01-01T00:02:00Z",
            expected_interval_seconds=60,
            timestamp_semantics_verified=True,
            quote_basis="bid",
        )
        self.assertTrue(any("bars_overlap" in item for item in report["blockers"]))
        one = self.candle()
        two = self.candle("2026-01-01T06:00:00Z", "2026-01-01T06:01:00Z")
        two["session_id"] = "session-2"
        three = self.candle("2026-01-01T12:00:00Z", "2026-01-01T12:01:00Z")
        report = validate_candle_series(
            [one, two, three],
            decision_time="2026-01-01T12:01:00Z",
            expected_interval_seconds=60,
            timestamp_semantics_verified=True,
            quote_basis="bid",
        )
        self.assertIn("row_2:session_id_recurred_after_boundary", report["blockers"])



    def test_semantic_type_and_event_time_role_must_match(self):
        fact = self.fact()
        fact["event_time_role"] = "scheduled_future_event"
        fact["event_time"] = "2026-01-01T12:00:00Z"
        fact["published_at"] = "2026-01-01T00:04:01Z"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("semantic_type_event_time_role_mismatch", report["blockers"])

    def test_malformed_enum_types_fail_closed_without_type_error(self):
        fact = self.fact()
        fact["semantic_type"] = {"malformed": True}
        fact["event_time_role"] = []
        fact["value_type"] = {}
        fact["quote_basis"] = []
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertFalse(report["eligible_at_decision_time"])
        self.assertIn("semantic_type_unknown", report["blockers"])
        self.assertIn("event_time_role_unknown", report["blockers"])
        self.assertIn("value_type_unknown", report["blockers"])
        self.assertIn("quote_basis_unknown", report["blockers"])

    def test_feature_batch_requires_slots_and_blocks_duplicate_records(self):
        second = self.fact()
        second["feature_id"] = "macro.cpi.yoy"
        second["semantic_type"] = "macro_observation"
        second["event_time_role"] = "observation_period"
        second["quote_basis"] = "not_applicable"
        second["value"] = 3.1
        second["unit"] = "percent"
        second["source_record_id"] = "macro:cpi:2025-12"
        report = validate_feature_batch(
            [self.fact(), second],
            decision_time="2026-01-01T00:05:00Z",
            required_feature_ids=["xau.close.m1", "macro.cpi.yoy"],
        )
        self.assertEqual(report["status"], "PASS", report)
        self.assertTrue(report["eligible_for_research_feature_snapshot"])
        duplicate_report = validate_feature_batch(
            [self.fact(), self.fact()],
            decision_time="2026-01-01T00:05:00Z",
            required_feature_ids=["xau.close.m1"],
        )
        self.assertIn("row_1:duplicate_feature_source_record_revision", duplicate_report["blockers"])
        self.assertEqual(duplicate_report["status"], "BLOCKED")

    def test_feature_batch_blocks_when_required_semantic_slot_is_absent(self):
        report = validate_feature_batch(
            [self.fact()],
            decision_time="2026-01-01T00:05:00Z",
            required_feature_ids=["xau.close.m1", "macro.cpi.yoy"],
        )
        self.assertIn("macro.cpi.yoy", report["missing_required_feature_ids"])
        self.assertEqual(report["status"], "BLOCKED")


    def derived_fact(self, feature_id="xau.ema.5", source_record_id="derived:ema5"):
        fact = self.fact()
        fact.update({
            "feature_id": feature_id,
            "semantic_type": "derived_market_feature",
            "event_time_role": "bar_close",
            "value": 2350.75,
            "published_at": "2026-01-01T00:04:04Z",
            "available_at": "2026-01-01T00:04:05Z",
            "observed_at": "2026-01-01T00:04:06Z",
            "source_record_id": source_record_id,
            "source_revision": "calc-1",
            "source_sha256": "c" * 64,
            "transform_id": "ema",
            "transform_version": "1.0.0",
            "transform_sha256": "d" * 64,
            "parent_feature_refs": [{
                "feature_id": "xau.close.m1",
                "instrument": "XAUUSD",
                "source_id": "feed-a",
                "source_record_id": "feed-a:XAUUSD:M1:2026-01-01T00:04:00Z:bid",
                "source_revision": "rev-1",
                "source_sha256": "a" * 64,
            }],
            "transform_fit_scope": "stateless",
        })
        return fact

    def parent_fact(self):
        return self.fact()

    def test_valid_declared_lineage_passes_only_for_research_snapshot(self):
        report = validate_feature_lineage(
            [self.parent_fact(), self.derived_fact()],
            decision_time="2026-01-01T00:05:00Z",
        )
        self.assertEqual(report["status"], "PASS", report)
        self.assertTrue(report["lineage_valid_for_research_snapshot"])
        self.assertEqual(report["lineage_edge_count"], 1)
        self.assertEqual(report["promotion_status"], "DIAGNOSTIC_ONLY")
        self.assertFalse(report["live_signal_enabled"])
        self.assertFalse(report["live_order_execution_enabled"])

    def test_derived_feature_without_parents_or_causal_fit_scope_is_blocked(self):
        fact = self.derived_fact()
        fact.pop("parent_feature_refs")
        fact.pop("transform_fit_scope")
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("derived_feature_parent_lineage_required", report["blockers"])
        self.assertIn("derived_feature_fit_scope_not_causal_or_unknown", report["blockers"])

        fact = self.derived_fact()
        fact["transform_fit_scope"] = "global_dataset"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("derived_feature_fit_scope_not_causal_or_unknown", report["blockers"])

    def test_fitted_transform_requires_cutoff_hash_manifest_and_no_future_fit(self):
        fact = self.derived_fact()
        fact["transform_fit_scope"] = "expanding_past_only"
        fact["transform_fit_cutoff"] = "2026-01-01T00:04:01Z"
        fact["transform_fit_data_sha256"] = "e" * 64
        fact["transform_fit_manifest_ref"] = "repo://fit-manifests/model-a.json"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertNotIn("derived_feature_fit_scope_not_causal_or_unknown", report["blockers"])
        self.assertFalse(any(item.startswith("transform_fit_") for item in report["blockers"]), report)

        fact["transform_fit_cutoff"] = "2026-01-01T00:05:30Z"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("transform_fit_cutoff_after_decision", report["blockers"])

    def test_parent_missing_or_digest_mismatch_blocks_lineage(self):
        report = validate_feature_lineage(
            [self.derived_fact()],
            decision_time="2026-01-01T00:05:00Z",
        )
        self.assertIn("row_0:parent_0:parent_fact_not_found_in_snapshot", report["blockers"])

        fact = self.derived_fact()
        fact["parent_feature_refs"][0]["source_sha256"] = "f" * 64
        report = validate_feature_lineage(
            [self.parent_fact(), fact],
            decision_time="2026-01-01T00:05:00Z",
        )
        self.assertIn("row_1:parent_0:parent_digest_or_revision_mismatch", report["blockers"])

    def test_parent_available_after_derived_feature_is_rejected(self):
        parent = self.parent_fact()
        parent["available_at"] = "2026-01-01T00:04:50Z"
        parent["observed_at"] = "2026-01-01T00:04:55Z"
        fact = self.derived_fact()
        report = validate_feature_lineage(
            [parent, fact],
            decision_time="2026-01-01T00:05:00Z",
        )
        self.assertIn("row_1:parent_0:parent_available_after_derived_feature", report["blockers"])

    def test_cyclic_feature_dependencies_are_rejected(self):
        left = self.derived_fact("xau.derived.left", "derived:left")
        right = self.derived_fact("xau.derived.right", "derived:right")
        right["source_sha256"] = "f" * 64
        # Each derived value is declared as a parent of the other; a valid data
        # lineage graph must be a directed acyclic graph.
        def ref_for(item):
            return {
                "feature_id": item["feature_id"],
                "instrument": item["instrument"],
                "source_id": item["source_id"],
                "source_record_id": item["source_record_id"],
                "source_revision": item["source_revision"],
                "source_sha256": item["source_sha256"],
            }
        left["parent_feature_refs"] = [ref_for(right)]
        right["parent_feature_refs"] = [ref_for(left)]
        right["available_at"] = "2026-01-01T00:04:06Z"
        right["observed_at"] = "2026-01-01T00:04:07Z"
        left["available_at"] = "2026-01-01T00:04:08Z"
        left["observed_at"] = "2026-01-01T00:04:09Z"
        report = validate_feature_lineage(
            [left, right],
            decision_time="2026-01-01T00:05:00Z",
        )
        self.assertTrue(any("feature_dependency_cycle" in item for item in report["blockers"]), report)


    def test_observed_after_decision_is_rejected_even_if_available_at_is_old(self):
        fact = self.fact()
        fact["observed_at"] = "2026-01-01T00:05:01Z"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("observed_after_decision_time", report["blockers"])
        self.assertFalse(report["eligible_at_decision_time"])

    def test_future_macro_observation_period_is_not_misclassified_as_known_ahead_event(self):
        fact = self.fact()
        fact["semantic_type"] = "macro_observation"
        fact["event_time_role"] = "observation_period"
        fact["event_time"] = "2026-01-01T12:00:00Z"
        fact["quote_basis"] = "not_applicable"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("event_time_after_decision_for_past_only_role", report["blockers"])

    def test_parent_must_be_observed_before_derived_feature_is_available(self):
        parent = self.parent_fact()
        parent["observed_at"] = "2026-01-01T00:04:07Z"
        fact = self.derived_fact()
        fact["available_at"] = "2026-01-01T00:04:05Z"
        fact["observed_at"] = "2026-01-01T00:04:08Z"
        report = validate_feature_lineage(
            [parent, fact],
            decision_time="2026-01-01T00:05:00Z",
        )
        self.assertIn(
            "row_1:parent_0:parent_not_observed_before_derived_feature_available",
            report["blockers"],
        )

    def test_snapshot_integrates_lineage_and_fails_when_parent_is_missing(self):
        fact = self.derived_fact()
        report = validate_feature_batch(
            [fact],
            decision_time="2026-01-01T00:05:00Z",
            required_feature_ids=["xau.ema.5"],
        )
        self.assertEqual(report["lineage_status"], "BLOCKED")
        self.assertFalse(report["eligible_for_research_feature_snapshot"])
        self.assertTrue(any("parent_fact_not_found_in_snapshot" in item for item in report["blockers"]))

    def test_malformed_fit_scope_fails_closed_without_type_error(self):
        fact = self.derived_fact()
        fact["transform_fit_scope"] = {"bad": "scope"}
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("derived_feature_fit_scope_not_causal_or_unknown", report["blockers"])


    def test_malformed_fit_manifest_url_fails_closed_without_parser_exception(self):
        fact = self.derived_fact()
        fact["transform_fit_scope"] = "rolling_past_only"
        fact["transform_fit_cutoff"] = "2026-01-01T00:04:01Z"
        fact["transform_fit_data_sha256"] = "e" * 64
        fact["transform_fit_manifest_ref"] = "https://[malformed"
        report = validate_feature_fact(fact, decision_time="2026-01-01T00:05:00Z")
        self.assertIn("transform_fit_manifest_reference_invalid", report["blockers"])

if __name__ == "__main__":
    unittest.main()
