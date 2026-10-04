from __future__ import annotations

import unittest

from evidence_independence import count_distinct_evidence_works, independence_summary


class EvidenceIndependenceTests(unittest.TestCase):
    def test_cross_provider_same_doi_is_one_work(self):
        records = [
            {
                "doi": "10.1234/example",
                "provider": "openalex",
                "title": "Example",
            },
            {
                "doi": "10.1234/example",
                "provider": "semantic_scholar",
                "title": "Example",
            },
            {
                "doi": "10.5678/other",
                "provider": "crossref",
                "title": "Other",
            },
        ]
        summary = independence_summary(records)
        self.assertEqual(summary["distinct_work_count"], 2)
        self.assertEqual(summary["strong_work_count"], 2)
        self.assertTrue(summary["independence_proxy"])

    def test_arxiv_versions_are_one_work(self):
        records = [
            {"arxiv_id": "2601.1234v1", "provider": "arxiv"},
            {"arxiv_id": "2601.1234v2", "provider": "openalex"},
        ]
        self.assertEqual(count_distinct_evidence_works(records), 1)

    def test_bibliographic_identity_is_marked_heuristic(self):
        records = [
            {
                "title": "A Study",
                "authors": ["Alice Example", "Bob Example"],
                "year": 2026,
                "venue": "Example Journal",
                "provider": "openalex",
            },
            {
                "title": "A Study",
                "authors": ["Alice Example", "Bob Example"],
                "year": 2026,
                "venue": "Example Journal",
                "provider": "semantic_scholar",
            },
        ]
        summary = independence_summary(records)
        self.assertEqual(summary["distinct_work_count"], 1)
        self.assertEqual(summary["heuristic_work_count"], 1)
        self.assertEqual(summary["independence_proxy_confidence"], 0.8)

    def test_provider_identifier_is_not_called_strong(self):
        summary = independence_summary([
            {"provider": "example", "provider_id": "x"},
        ])
        self.assertEqual(summary["weak_work_count"], 1)
        self.assertEqual(summary["independence_proxy_confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
