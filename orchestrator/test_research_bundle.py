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


if __name__ == '__main__':
    unittest.main()