import unittest
import urllib.request
from unittest.mock import patch

import research_bundle as r


class ResearchBundleTests(unittest.TestCase):
    def test_crossref_normalization(self):
        payload = {
            'message': {
                'items': [{
                    'DOI': '10.1234/test',
                    'title': ['Example'],
                    'published-online': {'date-parts': [[2026, 1, 2]]},
                    'container-title': ['Journal'],
                }]
            }
        }
        with patch.object(r, '_request', return_value=payload):
            value = r.search_crossref('example')
        self.assertEqual(value['results'][0]['DOI'], '10.1234/test')

    def test_wikipedia_uses_search_endpoint_shape(self):
        payload = {'query': {'search': [{'title': 'AI'}]}}
        with patch.object(r, '_request', return_value=payload) as mocked:
            value = r.search_wikipedia('AI')
        self.assertEqual(value, payload)
        mocked.assert_called_once()

    def test_request_bounds_large_json_response(self):
        class FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, size=-1):
                return b"x" * (size if size > 0 else 1)

        with patch.object(urllib.request, "urlopen", return_value=FakeResponse()):
            with self.assertRaisesRegex(RuntimeError, "research response exceeds 512 KiB"):
                r._request("https://example.test", "test")

    def test_arxiv_response_is_bounded(self):
        class FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, size=-1):
                return b"x" * (size if size > 0 else 1)

        with patch.object(urllib.request, "urlopen", return_value=FakeResponse()):
            with self.assertRaisesRegex(RuntimeError, "research response exceeds 512 KiB"):
                r.search_arxiv("topic")

    def test_bundle_survives_partial_provider_failure(self):
        def fake(name):
            if name == 'wiki':
                raise RuntimeError('down')
            return {'ok': name}
        with patch.object(r, 'search_wikipedia', side_effect=lambda q: fake('wiki')), \
             patch.object(r, 'search_arxiv', side_effect=lambda q: fake('arxiv')), \
             patch.object(r, 'search_crossref', side_effect=lambda q: fake('crossref')):
            value = r.research_bundle('topic')
        self.assertIn('arxiv', value['sources'])
        self.assertIn('crossref', value['sources'])
        self.assertIn('wikipedia', value['errors'])

    def test_enriched_bundle_reports_independent_evidence_count(self):
        extended = {
            "evidence_records": [
                {"canonical_id": "doi:10.1000/x", "provider": "openalex", "providers": ["openalex"]},
                {"canonical_id": "doi:10.1000/x", "provider": "semantic_scholar", "providers": ["semantic_scholar"]},
                {"canonical_id": "arxiv:2601.0001", "provider": "arxiv", "providers": ["arxiv"]},
            ],
            "independent_source_count": 2,
            "provider_counts": {"openalex": 1, "semantic_scholar": 1, "arxiv": 1},
            "errors": {},
        }
        with patch.object(r, "search_wikipedia", return_value={"query": {"search": []}}), \
             patch.object(r, "search_arxiv", return_value={"results": []}), \
             patch.object(r, "search_crossref", return_value={"message": {"items": []}}), \
             patch("research_providers.research_records", return_value=extended):
            value = r.research_bundle("topic", include_extended=True)
        self.assertEqual(value["independent_source_count"], 2)
        self.assertEqual(len(value["evidence_records"]), 2)

    def test_extended_research_stops_when_legacy_evidence_meets_budget(self):
        with patch.object(r, "search_wikipedia", return_value={
            "query": {"search": [{"title": f"Source {i}"} for i in range(8)]}
        }), \
             patch.object(r, "search_arxiv", return_value={"results": []}), \
             patch.object(r, "search_crossref", return_value={"message": {"items": []}}), \
             patch("research_providers.research_records") as extended:
            value = r.research_bundle("topic", include_extended=True, budget="fast")
        extended.assert_not_called()
        self.assertGreaterEqual(value["independent_source_count"], 4)
        self.assertTrue(value["research_stopped_early"])

    def test_extended_research_can_rescue_legacy_outage(self):
        extended = {
            "evidence_records": [
                {"canonical_id": "doi:10.1000/rescue", "provider": "semantic_scholar"},
            ],
            "independent_source_count": 1,
            "provider_counts": {"semantic_scholar": 1},
            "errors": {},
        }
        with patch.object(r, "search_wikipedia", side_effect=RuntimeError("down")), \
             patch.object(r, "search_arxiv", side_effect=RuntimeError("down")), \
             patch.object(r, "search_crossref", side_effect=RuntimeError("down")), \
             patch("research_providers.research_records", return_value=extended):
            value = r.research_bundle("topic", include_extended=True, budget="fast")
        self.assertEqual(value["independent_source_count"], 1)
        self.assertEqual(len(value["evidence_records"]), 1)
        self.assertIn("research_budget", value)

    def test_extended_failure_preserves_legacy_evidence(self):
        with patch.object(r, "search_wikipedia", return_value={"query": {"search": [{"title": "Background"}]}}), \
             patch.object(r, "search_arxiv", return_value={"results": []}), \
             patch.object(r, "search_crossref", return_value={"message": {"items": []}}), \
             patch("research_providers.research_records", side_effect=RuntimeError("new provider unavailable")):
            value = r.research_bundle("topic", include_extended=True)
        self.assertGreaterEqual(len(value["evidence_records"]), 1)
        self.assertEqual(value["extended_errors"]["fabric"]["type"], "RuntimeError")


if __name__ == '__main__':
    unittest.main()