import unittest
import io
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone

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

def archive_zip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("fixture.csv", payload)
    return buffer.getvalue()


def archive_listing(*timestamps: str) -> bytes:
    return "\n".join(
        f"6800000 fixture http://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
        for stamp in timestamps
    ).encode("utf-8")


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

    def test_large_but_bounded_gkg_field_is_parsed(self):
        raw = row("20260301120000-1")
        fields = raw.decode("utf-8").split("\t")
        fields[5] = "x" * 150_000
        payload = "\t".join(fields).encode("utf-8")
        parsed = gdelt.parse_gkg_tsv(payload, manifest(payload))
        self.assertEqual(len(parsed), 1)
        self.assertLessEqual(len(payload), gdelt.MAX_PAYLOAD_BYTES)

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
        self.assertGreaterEqual(gdelt.MAX_ARCHIVES_PER_RUN, 480)

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

    def test_split_stratified_sampler_selects_contiguous_blocks_inside_each_split(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        stamps = [
            (start + timedelta(minutes=15 * i)).strftime("%Y%m%d%H%M%S")
            for i in range(72 * 4)
        ]
        listing = archive_listing(*stamps)
        boundaries = (start + timedelta(hours=24), start + timedelta(hours=48))
        centers = (start + timedelta(hours=12), start + timedelta(hours=36), start + timedelta(hours=60))
        selected, discovered = gdelt.select_historical_gkg_archives(
            listing.decode("utf-8"), start, start + timedelta(hours=72),
            max_archives=12, sampling_design="split_stratified_contiguous_blocks",
            split_boundaries=boundaries, block_centers=centers,
        )
        self.assertEqual(discovered, 72 * 4)
        self.assertEqual(len(selected), 12)
        edges = (start, boundaries[0], boundaries[1], start + timedelta(hours=72))
        for split_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            block = sorted(when for when, _ in selected
                           if left <= when < right or (split_index == 2 and when == right))
            self.assertEqual(len(block), 4)
            for previous, current in zip(block, block[1:]):
                self.assertEqual(current - previous, timedelta(minutes=15))

    def test_split_stratified_sampler_rejects_invalid_boundaries(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        listing = archive_listing("20240101000000", "20240101001500")
        with self.assertRaisesRegex(ValueError, "split_boundaries_must_be_inside_window"):
            gdelt.select_historical_gkg_archives(
                listing.decode("utf-8"), start, start + timedelta(hours=1), max_archives=1,
                sampling_design="split_stratified_contiguous_blocks",
                split_boundaries=(start, start + timedelta(minutes=30)),
            )

    def test_budget_limit_after_prior_records_is_a_partial_stop(self):
        from unittest.mock import patch
        with patch.object(gdelt, "_bounded_get", side_effect=ValueError("content_length_exceeds_limit")):
            payload, warning = gdelt._download_archive_with_budget(
                "https://data.gdeltproject.org/gdeltv2/20240101120000.gkg.csv.zip",
                remaining_bytes=100, prior_observation_count=500,
            )
        self.assertIsNone(payload)
        self.assertEqual(warning, "aggregate_byte_cap_reached_before_next_archive")

    def test_single_archive_size_limit_is_not_misreported_as_aggregate_exhaustion(self):
        from unittest.mock import patch
        with patch.object(gdelt, "_bounded_get", side_effect=ValueError("content_length_exceeds_limit")):
            payload, warning = gdelt._download_archive_with_budget(
                "https://data.gdeltproject.org/gdeltv2/20240101120000.gkg.csv.zip",
                remaining_bytes=gdelt.MAX_ARCHIVE_BYTES * 2, prior_observation_count=500,
            )
        self.assertIsNone(payload)
        self.assertEqual(warning, "archive_skipped_per_archive_byte_cap")

    def test_oversized_archive_does_not_stop_later_sampled_archives(self):
        from unittest.mock import patch

        urls = [
            "https://data.gdeltproject.org/gdeltv2/20240101000000.gkg.csv.zip",
            "https://data.gdeltproject.org/gdeltv2/20240102000000.gkg.csv.zip",
            "https://data.gdeltproject.org/gdeltv2/20240103000000.gkg.csv.zip",
        ]
        listing = archive_listing("20240101000000", "20240102000000", "20240103000000")

        def fake_get(url, cap):
            if url == gdelt.MASTER_URL:
                return listing
            if url == urls[0]:
                raise ValueError("content_length_exceeds_limit")
            if url == urls[1]:
                return archive_zip(row("20240102000000-1", "https://example.com/a"))
            if url == urls[2]:
                return archive_zip(row("20240103000000-1", "https://example.com/b"))
            raise AssertionError(f"unexpected URL: {url}")

        with patch.object(gdelt, "_bounded_get", side_effect=fake_get):
            result = gdelt.acquire_historical_gkg(
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 4, tzinfo=timezone.utc),
                target_records=2, max_archives=3, aggregate_cap=gdelt.MAX_AGGREGATE_BYTES,
            )

        self.assertEqual(len(result.observations), 2)
        self.assertEqual(result.status, "PARTIAL")
        self.assertTrue(any("archive_skipped_per_archive_byte_cap" in w for w in result.warnings))
        self.assertTrue(any("processed_sampled_archives=2" in w for w in result.warnings))

    def test_bounded_sample_can_pass_scale_gate_without_claiming_full_window_coverage(self):
        from unittest.mock import patch

        listing = archive_listing("20240101000000", "20240102000000")
        first_url = "https://data.gdeltproject.org/gdeltv2/20240101000000.gkg.csv.zip"

        def fake_get(url, cap):
            if url == gdelt.MASTER_URL:
                return listing
            if url == first_url:
                return archive_zip(row("20240101000000-1", "https://example.com/a"))
            raise AssertionError(f"unexpected URL: {url}")

        with patch.object(gdelt, "_bounded_get", side_effect=fake_get):
            result = gdelt.acquire_historical_gkg(
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 3, tzinfo=timezone.utc),
                target_records=1, max_archives=1, aggregate_cap=gdelt.MAX_AGGREGATE_BYTES,
            )

        self.assertEqual(result.status, "PASS")
        self.assertTrue(any("sample_completed=true" in w for w in result.warnings))
        self.assertTrue(any("archive_window_complete=false" in w for w in result.warnings))

    def test_budget_limit_before_any_records_fails_closed(self):
        from unittest.mock import patch
        with patch.object(gdelt, "_bounded_get", side_effect=ValueError("content_length_exceeds_limit")):
            with self.assertRaisesRegex(ValueError, "content_length_exceeds_limit"):
                gdelt._download_archive_with_budget(
                    "https://data.gdeltproject.org/gdeltv2/20240101120000.gkg.csv.zip",
                    remaining_bytes=100, prior_observation_count=0,
                )

    def test_aggregate_budget_is_large_enough_for_real_scale_target(self):
        self.assertGreaterEqual(gdelt.MAX_AGGREGATE_BYTES, 800 * 1024 * 1024)

    def test_archive_coverage_intervals_preserve_sampling_gaps(self):
        first = replace(
            manifest(),
            source_url="https://data.gdeltproject.org/gdeltv2/20240101000000.gkg.csv.zip",
        )
        later = replace(
            manifest(),
            source_url="https://data.gdeltproject.org/gdeltv2/20240102000000.gkg.csv.zip",
        )
        intervals = gdelt.archive_coverage_intervals([later, first])
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0][0], datetime(2024, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(intervals[0][1], intervals[0][0] + gdelt.BATCH_INTERVAL)
        self.assertGreater(intervals[1][0], intervals[0][1])

    def test_archive_coverage_summary_does_not_fill_unobserved_gaps(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(hours=1)
        summary = gdelt.summarize_archive_coverage(
            [(start, start + timedelta(minutes=15)),
             (start + timedelta(minutes=30), end)],
            start, end,
        )
        self.assertEqual(summary["status"], "PARTIAL")
        self.assertEqual(summary["covered_minutes"], 45.0)
        self.assertEqual(summary["coverage_fraction"], 0.75)

    def test_archive_coverage_summary_merges_adjacent_batches_and_deduplicates(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        middle = start + timedelta(minutes=15)
        end = middle + timedelta(minutes=15)
        summary = gdelt.summarize_archive_coverage(
            [(start, middle), (start, middle), (middle, end)], start, end
        )
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["observed_interval_count"], 2)
        self.assertEqual(summary["merged_coverage_interval_count"], 1)
        self.assertEqual(summary["covered_minutes"], 30.0)

    def test_archive_coverage_interval_rejects_untrusted_manifest_host(self):
        untrusted = replace(
            manifest(),
            source_url="https://evil.example/gdeltv2/20240101000000.gkg.csv.zip",
        )
        with self.assertRaisesRegex(ValueError, "gdelt_manifest_source_not_allowlisted"):
            gdelt.archive_coverage_intervals([untrusted])

    def test_same_gkg_record_from_different_archive_hash_is_not_false_conflict(self):
        raw = row("20260301120000-1")
        first = gdelt.parse_gkg_tsv(raw, manifest(b"archive-one"))[0]
        second = gdelt.parse_gkg_tsv(raw, manifest(b"archive-two"))[0]
        result = gdelt.deduplicate_snapshots([first, second])
        self.assertEqual(len(result), 1)
        self.assertFalse(any("conflicting_snapshot" in flag for flag in result[0].quality_flags))

    def test_no_article_body_is_persisted(self):
        raw=row("20260301120000-1")
        result=gdelt.parse_gkg_tsv(raw,manifest(raw))
        self.assertNotIn("story body",str(result[0].attributes).lower())
        self.assertNotIn("document_text",result[0].attributes)


if __name__ == "__main__":
    unittest.main()
