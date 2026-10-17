import unittest
from datetime import datetime, timezone

import gdelt_gkg_adapter_v1 as gdelt
from external_context_schema_v1 import SourceManifest, manifest_sha256


def manifest(raw=b"fixture"):
    return SourceManifest("gdelt-gkg", "historical-gkg", "https://data.gdeltproject.org/gdeltv2/",
        datetime(2026, 3, 1, tzinfo=timezone.utc), manifest_sha256(raw), len(raw), 0, 0,
        "GKG v2", "review-required", "official-host;rights-independent-review-required")


def row(record_id, doc_url="https://example.com/story/1", datefield="20260301120000"):
    fields=[""] * 11
    fields[0]=record_id
    fields[1]=datefield
    fields[2]="1"
    fields[3]="example.com"
    fields[4]=doc_url
    fields[5]=""
    fields[6]="ECON_GOLD;ECON_MARKET"
    fields[7]=""
    fields[8]=""
    fields[9]=""
    fields[10]="-1.2,2.3,0.4,1"
    return "\t".join(fields).encode()


class GdeltGkgTests(unittest.TestCase):
    def test_backdated_publication_is_not_available_before_archive_batch(self):
        raw=row("20260301120000-1")
        result=gdelt.parse_gkg_tsv(raw, manifest(raw))
        self.assertGreater(result[0].available_at, result[0].event_time)
        self.assertEqual(result[0].availability_quality, "conservative_proxy")

    def test_duplicate_article_across_snapshots_keeps_first_seen(self):
        early=gdelt.parse_gkg_tsv(row("20260301120000-1"),manifest())[0]
        later=gdelt.parse_gkg_tsv(row("20260301121500-1"),manifest())[0]
        result=gdelt.deduplicate_snapshots([later,early])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].source_record_id,early.source_record_id)

    def test_conflicting_duplicate_is_audit_warning(self):
        a=gdelt.parse_gkg_tsv(row("20260301120000-1"),manifest())[0]
        b=gdelt.parse_gkg_tsv(row("20260301120000-1",doc_url="https://elsewhere.test/different"),manifest())[0]
        result=gdelt.deduplicate_snapshots([a,b])
        self.assertTrue(any("conflicting_snapshot" in x for x in result[0].quality_flags))

    def test_bad_record_id_and_oversized_payload_fail_closed(self):
        with self.assertRaises(ValueError):
            gdelt.archive_available_at("not-a-record-id")
        with self.assertRaises(ValueError):
            gdelt.parse_gkg_tsv(b"x"*(gdelt.MAX_PAYLOAD_BYTES+1),manifest())

    def test_doc_api_rolling_window_is_not_labeled_long_history(self):
        self.assertEqual(gdelt.classify_doc_api_history(is_doc_api=True, claimed_multiyear=True),"NOT_MET")
        self.assertEqual(gdelt.classify_doc_api_history(is_doc_api=True, claimed_multiyear=False),"SMOKE_ONLY")

    def test_historical_master_list_limit_covers_current_official_listing(self):
        self.assertGreaterEqual(gdelt.MAX_MASTER_BYTES, 130 * 1024 * 1024)

    def test_official_http_archive_links_are_upgraded_to_https(self):
        source = "http://data.gdeltproject.org/gdeltv2/20230101120000.gkg.csv.zip"
        self.assertEqual(gdelt.normalize_archive_url(source),
            "https://data.gdeltproject.org/gdeltv2/20230101120000.gkg.csv.zip")
        with self.assertRaises(ValueError):
            gdelt.normalize_archive_url("http://example.com/gdeltv2/20230101120000.gkg.csv.zip")

    def test_master_list_selector_uses_url_column_and_keeps_historical_gkg_rows(self):
        listing = "\n".join([
            "6800000 abc http://data.gdeltproject.org/gdeltv2/20230101120000.gkg.csv.zip",
            "6800000 def http://data.gdeltproject.org/gdeltv2/20240101120000.gkg.csv.zip",
            "150000 ghi http://data.gdeltproject.org/gdeltv2/20240101120000.export.CSV.zip",
            "6800000 jkl http://attacker.invalid/gdeltv2/20240101120000.gkg.csv.zip",
            "6800000 mno http://data.gdeltproject.org/gdeltv2/20260101120000.gkg.csv.zip",
        ])
        selected, discovered = gdelt.select_historical_gkg_archives(
            listing,
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 6, 30, tzinfo=timezone.utc),
            max_archives=10,
        )
        self.assertEqual(discovered, 2)
        self.assertEqual([url for _, url in selected], [
            "https://data.gdeltproject.org/gdeltv2/20230101120000.gkg.csv.zip",
            "https://data.gdeltproject.org/gdeltv2/20240101120000.gkg.csv.zip",
        ])

    def test_budget_limit_after_prior_records_is_a_partial_stop(self):
        from unittest.mock import patch
        with patch.object(gdelt, "_bounded_get", side_effect=ValueError("content_length_exceeds_limit")):
            payload, warning = gdelt._download_archive_with_budget(
                "https://data.gdeltproject.org/gdeltv2/20240101120000.gkg.csv.zip",
                remaining_bytes=100, prior_observation_count=500,
            )
        self.assertIsNone(payload)
        self.assertEqual(warning, "aggregate_byte_cap_reached_before_next_archive")

    def test_budget_limit_before_any_records_fails_closed(self):
        from unittest.mock import patch
        with patch.object(gdelt, "_bounded_get", side_effect=ValueError("content_length_exceeds_limit")):
            with self.assertRaisesRegex(ValueError, "content_length_exceeds_limit"):
                gdelt._download_archive_with_budget(
                    "https://data.gdeltproject.org/gdeltv2/20240101120000.gkg.csv.zip",
                    remaining_bytes=100, prior_observation_count=0,
                )

    def test_no_article_body_is_persisted(self):
        raw=row("20260301120000-1")
        result=gdelt.parse_gkg_tsv(raw,manifest(raw))
        self.assertNotIn("story body",str(result[0].attributes).lower())
        self.assertNotIn("document_text",result[0].attributes)


if __name__ == "__main__":
    unittest.main()
