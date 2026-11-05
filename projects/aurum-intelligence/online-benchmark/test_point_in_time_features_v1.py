import unittest
from datetime import datetime, timedelta, timezone

import point_in_time_features_v1 as features
from external_context_schema_v1 import ExternalObservation


def obs(source_id, rid, available_at, value=1.0, quality="verified", attrs=None):
    return ExternalObservation(source_id=source_id,source_record_id=rid,
        dataset_or_series_id=source_id,schema_version="1.0",
        event_time=available_at-timedelta(minutes=5) if available_at else None,
        published_at=None,available_at=available_at,availability_quality=quality,
        retrieved_at=datetime(2026,3,5,tzinfo=timezone.utc),source_url="https://example.invalid",
        payload_sha256=(rid[:1].lower()*64 if rid else "a"*64),source_revision=None,
        value=value,unit="unit",attributes=attrs or {},rights_status="unknown",
        provenance_status="unverified",quality_flags=())


class PointInTimeFeatureTests(unittest.TestCase):
    def setUp(self):
        self.t=datetime(2026,3,1,12,tzinfo=timezone.utc)
        self.rows=[{"decision_id":"b","decision_time":self.t},
                   {"decision_id":"a","decision_time":self.t-timedelta(hours=1)}]

    def test_future_observation_never_changes_past_feature(self):
        past=features.build_asof_features(self.rows,[obs("cftc-cot","cot1",self.t-timedelta(minutes=10),value=3)])
        with_future=features.build_asof_features(self.rows,[obs("cftc-cot","cot1",self.t-timedelta(minutes=10),value=3),
            obs("cftc-cot","future",self.t+timedelta(minutes=1),value=900)])
        self.assertEqual(past,with_future)

    def test_unknown_availability_is_excluded_from_causal_features(self):
        out=features.visible_as_of([obs("cftc-cot","x",None,7,"unknown")],self.t)
        self.assertEqual(out,[])

    def test_conservative_proxy_is_separately_counted(self):
        row=features.build_asof_features([{"decision_id":"a","decision_time":self.t}],
            [obs("cftc-cot","x",self.t-timedelta(minutes=1),5,"conservative_proxy")])[0]
        self.assertEqual(row["availability_proxy_count"],1)

    def test_deduplicated_news_counts_first_available_time_only(self):
        a=obs("gdelt-gkg","news-a",self.t-timedelta(minutes=10),-1,attrs={"document_sha256":"d"*64,"themes":("ECON_GOLD",),"tone":-1})
        b=obs("gdelt-gkg","news-b",self.t-timedelta(minutes=5),-1,attrs={"document_sha256":"d"*64,"themes":("ECON_GOLD",),"tone":-1})
        row=features.build_asof_features(
            [{"decision_id":"a","decision_time":self.t}], [a,b],
            source_statuses={"news":"PASS"},
            news_coverage_intervals=[(self.t-timedelta(hours=24), self.t)],
        )[0]
        self.assertEqual(row["feature_news_count_24h"],1)
        self.assertEqual(row["news_duplicate_records_removed"],1)
        self.assertTrue(row["news_window_24h_fully_covered"])

    def test_sparse_news_archive_does_not_claim_zero_event_coverage(self):
        row=features.build_asof_features(
            [{"decision_id":"a","decision_time":self.t}], [],
            source_statuses={"news":"PASS"},
            news_coverage_intervals=[(self.t-timedelta(minutes=15), self.t)],
        )[0]
        self.assertIsNone(row["feature_news_count_1h"])
        self.assertIsNone(row["feature_news_count_24h"])
        self.assertEqual(row["coverage_news"], "incomplete_source_window")
        self.assertFalse(row["news_window_1h_fully_covered"])
        self.assertFalse(row["news_window_24h_fully_covered"])

    def test_complete_coverage_can_report_zero_news_events(self):
        row=features.build_asof_features(
            [{"decision_id":"a","decision_time":self.t}], [],
            source_statuses={"news":"PASS"},
            news_coverage_intervals=[(self.t-timedelta(hours=24), self.t)],
        )[0]
        self.assertEqual(row["feature_news_count_1h"], 0)
        self.assertEqual(row["feature_news_count_24h"], 0)
        self.assertEqual(row["coverage_news"], "available_zero_events")
        self.assertTrue(row["news_window_1h_fully_covered"])
        self.assertTrue(row["news_window_24h_fully_covered"])

    def test_future_archive_coverage_cannot_make_past_window_eligible(self):
        row=features.build_asof_features(
            [{"decision_id":"a","decision_time":self.t}], [],
            source_statuses={"news":"PASS"},
            news_coverage_intervals=[(self.t, self.t+timedelta(hours=24))],
        )[0]
        self.assertIsNone(row["feature_news_count_1h"])
        self.assertEqual(row["coverage_news"], "incomplete_source_window")
        self.assertEqual(row["news_coverage_intervals_visible"], 0)

    def test_future_contiguous_archive_does_not_hide_visible_prior_batch(self):
        row=features.build_asof_features(
            [{"decision_id":"a","decision_time":self.t}], [],
            source_statuses={"news":"PASS"},
            news_coverage_intervals=[
                (self.t-timedelta(minutes=15), self.t),
                (self.t, self.t+timedelta(minutes=15)),
            ],
        )[0]
        self.assertEqual(row["news_coverage_intervals_visible"], 1)
        self.assertEqual(row["coverage_news"], "incomplete_source_window")

    def test_family_coverage_is_reported_without_dropping_baseline_rows(self):
        rows=features.build_asof_features(self.rows,[],source_statuses={"news":"NOT_MET","cot":"PASS"})
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]["coverage_news"],"source_unavailable")
        self.assertEqual(rows[0]["coverage_cot"],"not_yet_available")

    def test_output_is_deterministic_and_sorted(self):
        one=features.build_asof_features(self.rows,[])
        two=features.build_asof_features(list(reversed(self.rows)),[])
        self.assertEqual(one,two)
        self.assertEqual([r["decision_id"] for r in one],["a","b"])

    def test_mixed_timezone_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            features.visible_as_of([],datetime(2026,3,1,12))


if __name__=="__main__":
    unittest.main()
