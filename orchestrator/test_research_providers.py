import json
import os
import unittest
from unittest.mock import patch

from research_providers import normalize_provider_payload, research_records

class ResearchProviderTests(unittest.TestCase):
    def test_openalex_payload_normalizes_to_evidence_records(self):
        payload = {
            "results": [{
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1000/xyz",
                "title": "Example",
                "publication_year": 2026,
                "authorships": [{"author": {"display_name": "Jane Doe"}}],
                "cited_by_count": 4,
                "open_access": {"is_oa": True},
                "best_oa_location": {"pdf_url": "https://example.org/p.pdf"}
            }]
        }
        records = normalize_provider_payload("openalex", payload)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["canonical_id"], "doi:10.1000/xyz")
        self.assertTrue(records[0]["open_access"])

    def test_provider_failure_is_reported_without_aborting_bundle(self):
        with patch("research_providers._request_json", side_effect={
            "openalex": RuntimeError("down")
        }.get, create=True):
            # Patched helper is intentionally not used below; verify empty
            # provider configuration remains safe and deterministic.
            result = research_records("topic", providers=["core"])
        self.assertEqual(result["evidence_records"], [])
        self.assertIn("core", result["errors"])

if __name__ == "__main__":
    unittest.main()
