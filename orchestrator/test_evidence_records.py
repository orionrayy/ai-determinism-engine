import unittest
from evidence_records import canonical_source_id, deduplicate_sources, count_independent_sources, normalize_source

class EvidenceRecordTests(unittest.TestCase):
    def test_same_work_from_multiple_providers_deduplicates(self):
        records = [
            normalize_source("crossref", {"DOI":"10.1000/ABC.1","title":"A Study","author":[{"family":"Doe"}],"published":{"date-parts":[[2026,1,1]]}}),
            normalize_source("openalex", {"doi":"https://doi.org/10.1000/abc.1","title":"A Study","authorships":[{"author":{"display_name":"Jane Doe"}}],"publication_year":2026}),
        ]
        merged = deduplicate_sources(records)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["canonical_id"], "doi:10.1000/abc.1")
        self.assertEqual(sorted(merged[0]["providers"]), ["crossref","openalex"])
        self.assertEqual(count_independent_sources(merged), 1)

    def test_fallback_identity_is_deterministic_and_cross_provider(self):
        a = {"provider":"openalex","title":"Novel Evidence","authors":[{"name":"Jane Doe"}],"publication_year":2025,"id":"https://openalex.org/W1"}
        b = {"provider":"semantic_scholar","title":"Novel Evidence","authors":[{"name":"Jane Doe"}],"year":2025,"paperId":"P1"}
        self.assertEqual(canonical_source_id(a), canonical_source_id(a))
        self.assertEqual(canonical_source_id(a), canonical_source_id(b))

    def test_normalized_record_has_credibility_signals_without_opaque_score(self):
        record = normalize_source("semantic_scholar", {
            "paperId":"p1","title":"Paper","year":2026,
            "authors":[{"name":"Jane Doe"}],"citationCount":12,
            "openAccessPdf":{"url":"https://example.org/p.pdf"}
        })
        self.assertEqual(record["provider"], "semantic_scholar")
        self.assertEqual(record["citation_count"], 12)
        self.assertTrue(record["open_access"])
        self.assertIn("authority_signals", record)
        self.assertNotIn("trust_score", record)

if __name__ == "__main__":
    unittest.main()