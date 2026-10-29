import json
import tempfile
import unittest
from pathlib import Path

import validate_features as vf


def manifest_record(**overrides):
    base = {
        "record_id": "doc-example-001",
        "source_family_id": "doi:10.15294/example",
        "source_url": "https://example.org/article.pdf",
        "title": "Example article",
        "creator": "Author A",
        "publisher": "Example Press",
        "genre": "academic_paper",
        "subgenre": "original article",
        "publication_year": 2023,
        "retrieved_at": "2026-10-30",
        "language": "en",
        "format": "pdf",
        "access_basis": "open_license",
        "license": "CC BY 4.0",
        "usage_scope": "feature_analysis_only",
        "originality_status": "original_verified",
        "provenance_confidence": "high",
        "provenance_evidence": ["https://example.org/article", "https://creativecommons.org/licenses/by/4.0/"],
        "acquisition_method": "official_web_view",
        "access_status": "accessed",
        "anonymization_status": "not_applicable",
        "privacy_review": "not_required",
        "verification_status": "verified",
        "sha256": None,
        "byte_size": None,
        "duplicate_of": None,
        "exclusion_reason": None,
    }
    base.update(overrides)
    return base


def feature_record(**overrides):
    base = {
        "record_id": "doc-example-001",
        "source_family_id": "doi:10.15294/example",
        "analysis_version": "manual-structure-review-1.0.0",
        "reviewed_at": "2026-10-30",
        "evidence_urls": ["https://example.org/article"],
        "method": {
            "access": "official article page",
            "text_review": "headings examined",
            "visual_review": "cover and contents page inspected",
            "stored_content": "metadata and derived features only; no full text retained",
        },
        "observed_structure": {
            "front_matter": ["title", "author", "abstract"],
            "visual_features": ["single-column layout", "running page number"],
            "body_headings": ["I. Introduction", "II. Results"],
        },
        "review_flags": {
            "source_is_authored_publication": True,
            "factual_claims_independently_verified": False,
            "benchmark_role": "structure exemplar, not gold standard",
            "generalization_status": "single sample; no genre-level inference",
        },
    }
    base.update(overrides)
    return base


class FeatureTraceabilityTests(unittest.TestCase):
    def write_jsonl(self, folder, name, rows):
        path = folder / name
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
        return path

    def run_validation(self, folder, manifest_rows, feature_rows):
        manifest = self.write_jsonl(folder, "manifest.jsonl", manifest_rows)
        features = self.write_jsonl(folder, "features.jsonl", feature_rows)
        return vf.load_features(features, manifest, target=500)

    def test_valid_feature_record_links_to_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_validation(Path(tmp), [manifest_record()], [feature_record()])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["eligible_unique_documents"], 1)
        self.assertEqual(report["eligible_documents_with_analysis"], 1)
        self.assertEqual(report["manifest_eligible_by_genre"]["academic_paper"], 1)

    def test_feature_for_unknown_document_is_rejected(self):
        item = feature_record(record_id="doc-not-in-manifest")
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_validation(Path(tmp), [manifest_record()], [item])
        self.assertTrue(any("does not exist in the corpus manifest" in e for e in report["errors"]))

    def test_family_mismatch_is_rejected(self):
        item = feature_record(source_family_id="different-family")
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_validation(Path(tmp), [manifest_record()], [item])
        self.assertTrue(any("source_family_id does not match" in e for e in report["errors"]))

    def test_evidence_must_be_present_in_manifest_provenance(self):
        item = feature_record(evidence_urls=["https://unrelated.example/source"])
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_validation(Path(tmp), [manifest_record()], [item])
        self.assertTrue(any("evidence URL not recorded in manifest provenance" in e for e in report["errors"]))

    def test_duplicate_analysis_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_validation(Path(tmp), [manifest_record()], [feature_record(), feature_record()])
        self.assertTrue(any("duplicate analysis for record_id" in e for e in report["errors"]))

    def test_missing_visual_method_and_visual_features_are_rejected(self):
        item = feature_record(method={"access": "official page"}, observed_structure={"front_matter": ["Title"]})
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_validation(Path(tmp), [manifest_record()], [item])
        self.assertTrue(any("method.text_review is required" in e for e in report["errors"]))
        self.assertTrue(any("method.visual_review is required" in e for e in report["errors"]))
        self.assertTrue(any("observed_structure.visual_features is required" in e for e in report["errors"]))

    def test_source_quality_limits_are_reported_as_warnings_not_hidden(self):
        item = feature_record(review_flags={
            "source_is_authored_publication": True,
            "factual_claims_independently_verified": False,
            "benchmark_role": "sample",
            "generalization_status": "single sample; not universal",
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_validation(Path(tmp), [manifest_record()], [item])
        self.assertEqual(report["errors"], [])
        self.assertTrue(any("factual claims not independently verified" in w for w in report["warnings"]))
        self.assertTrue(any("limits generalization" in w for w in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
