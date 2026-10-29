import json
import tempfile
import unittest
from pathlib import Path

import validate_corpus as vc


def record(**overrides):
    base = {
        "record_id": "doc-example-001",
        "source_family_id": "family-example-001",
        "source_url": "https://example.org/repository/item/1",
        "title": "Contoh laporan penelitian",
        "creator": "Penulis yang tercantum",
        "publisher": "Repositori Institusi",
        "genre": "research_report",
        "subgenre": "laporan penelitian",
        "publication_year": 2024,
        "retrieved_at": "2026-10-30",
        "language": "id",
        "format": "pdf",
        "access_basis": "open_license",
        "license": "CC BY 4.0 (per-item license verified)",
        "usage_scope": "full_text_analysis",
        "originality_status": "original_verified",
        "provenance_confidence": "high",
        "provenance_evidence": [
            "https://example.org/repository/item/1",
            "https://creativecommons.org/licenses/by/4.0/",
        ],
        "acquisition_method": "official_download",
        "access_status": "accessed",
        "anonymization_status": "not_applicable",
        "privacy_review": "not_required",
        "verification_status": "verified",
        "sha256": "a" * 64,
        "byte_size": 12345,
        "duplicate_of": None,
        "exclusion_reason": None,
    }
    base.update(overrides)
    return base


class CorpusValidationTests(unittest.TestCase):
    def write_manifest(self, directory: Path, rows):
        path = directory / "manifest.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_empty_manifest_is_insufficient_evidence_not_false_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text("\n", encoding="utf-8")
            report = vc.load_manifest(path, target=500)
        self.assertEqual(report["records_seen"], 0)
        self.assertEqual(report["eligible_unique_original_documents"], 0)
        self.assertEqual(report["target_status"], "INSUFFICIENT_EVIDENCE")
        self.assertFalse(report["target_reached"])
        self.assertEqual(report["errors"], [])

    def test_licensed_original_with_high_provenance_is_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = vc.load_manifest(self.write_manifest(Path(tmp), [record()]), target=1)
        self.assertEqual(report["eligible_unique_original_documents"], 1)
        self.assertTrue(report["target_reached"])
        self.assertEqual(report["by_genre"]["research_report"]["eligible_unique"], 1)

    def test_metadata_only_or_unclear_rights_never_counts(self):
        candidate = record(
            access_basis="metadata_only",
            usage_scope="metadata_only",
            access_status="metadata_only",
            license="unknown",
            verification_status="pending",
            originality_status="appears_original_unverified",
            provenance_confidence="medium",
        )
        with tempfile.TemporaryDirectory() as tmp:
            report = vc.load_manifest(self.write_manifest(Path(tmp), [candidate]), target=1)
        self.assertEqual(report["eligible_unique_original_documents"], 0)
        blockers = report["items"][0]["blockers"]
        self.assertIn("rights_or_access_basis_not_verified", blockers)
        self.assertIn("document_content_not_accessed", blockers)

    def test_partial_access_does_not_count_as_a_complete_corpus_sample(self):
        candidate = record(
            access_status="partial",
            verification_status="pending",
            originality_status="appears_original_unverified",
            usage_scope="feature_analysis_only",
            notes="Only openly licensed summary and introductory chapter are accessible; full document is restricted.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            report = vc.load_manifest(self.write_manifest(Path(tmp), [candidate]), target=1)
        self.assertEqual(report["eligible_unique_original_documents"], 0)
        self.assertEqual(report["target_status"], "INSUFFICIENT_EVIDENCE")
        self.assertIn("document_content_not_accessed", report["items"][0]["blockers"])

    def test_template_and_synthetic_documents_are_not_original_samples(self):
        rows = [
            record(record_id="doc-template-001", source_family_id="family-template-001",
                   source_url="https://example.org/templates/1", originality_status="official_template"),
            record(record_id="doc-synthetic-001", source_family_id="family-synthetic-001",
                   source_url="https://example.org/generated/1", originality_status="synthetic_or_ai"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = vc.load_manifest(self.write_manifest(Path(tmp), rows), target=1)
        self.assertEqual(report["eligible_unique_original_documents"], 0)

    def test_duplicate_urls_or_hashes_cannot_inflate_the_count(self):
        first = record()
        second = record(record_id="doc-example-002", source_family_id="family-example-001",
                        source_url="https://example.org/repository/item/1?utm_source=mirror",
                        sha256="b" * 64)
        third = record(record_id="doc-example-003", source_family_id="family-example-003",
                       source_url="https://example.org/repository/item/3",
                       sha256="a" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            report = vc.load_manifest(self.write_manifest(Path(tmp), [first, second, third]), target=1)
        self.assertEqual(report["eligible_unique_original_documents"], 1)
        self.assertTrue(any("duplicates" in warning for warning in report["warnings"]))

    def test_duplicate_record_ids_are_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = vc.load_manifest(self.write_manifest(Path(tmp), [record(), record()]), target=1)
        self.assertTrue(any("duplicate record_id" in error for error in report["errors"]))
        self.assertEqual(report["target_status"], "INSUFFICIENT_EVIDENCE")

    def test_invalid_json_is_reported_and_never_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text('{"record_id":\n', encoding="utf-8")
            report = vc.load_manifest(path, target=1)
        self.assertEqual(report["records_seen"], 1)
        self.assertEqual(report["eligible_unique_original_documents"], 0)
        self.assertIn("invalid JSON", report["errors"][0])

    def test_http_url_with_embedded_credentials_is_rejected(self):
        bad = record(source_url="https://person:secret@example.org/private")
        with tempfile.TemporaryDirectory() as tmp:
            report = vc.load_manifest(self.write_manifest(Path(tmp), [bad]), target=1)
        self.assertTrue(any("embedded credentials" in error for error in report["errors"]))

    def test_target_flag_returns_failure_when_samples_are_insufficient(self):
        # CLI accepts a valid empty manifest and distinguishes target failure from schema failure.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text("", encoding="utf-8")
            self.assertEqual(vc.main([str(path), "--target", "1", "--require-target"]), 3)


if __name__ == "__main__":
    unittest.main()
