import json
import os
import unittest
import urllib.error
import tempfile
from unittest.mock import patch

from research_providers import (
    _request_json,
    normalize_provider_payload,
    research_records,
)

class ResearchProviderTests(unittest.TestCase):
    def test_crossref_public_payload_normalizes_and_classifies_access(self):
        payload = {
            "message": {
                "items": [{
                    "DOI": "10.1000/crossref",
                    "title": ["Crossref Example"],
                    "author": [{"given": "Jane", "family": "Doe"}],
                    "published": {"date-parts": [[2026, 10, 4]]},
                    "type": "journal-article",
                    "abstract": "<jats:p>Structured abstract.</jats:p>",
                    "URL": "https://publisher.example/article",
                    "link": [{
                        "URL": "https://publisher.example/article.pdf",
                        "content-type": "application/pdf",
                    }],
                }]
            }
        }
        records = normalize_provider_payload("crossref", payload)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["canonical_id"], "doi:10.1000/crossref")
        self.assertEqual(record["access_level"], "L3")
        self.assertEqual(record["access_route"], "publisher_url")
        self.assertTrue(0.0 <= record["authority_score"] <= 1.0)
        self.assertTrue(record["authority_heuristic"])

    def test_crossref_is_available_in_free_only_mode(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False),              patch("research_providers._provider_search", return_value={"message": {"items": []}}) as search:
            result = research_records("topic")
        self.assertEqual(
            result["providers"],
            ["semantic_scholar", "europe_pmc", "crossref"],
        )
        self.assertEqual(search.call_count, 3)

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

    def test_all_disallowed_explicit_providers_fail_closed(self):
        with patch.dict(
            os.environ,
            {"ORCHESTRATOR_FREE_ONLY": "true"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "no requested research providers"):
                research_records(
                    "topic",
                    providers=("openalex", "core"),
                    max_results=2,
                )

    def test_explicit_metered_provider_is_blocked_in_free_only_mode(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False), \
             patch(
                 "research_providers._provider_search",
                 return_value={"results": []},
             ) as search:
            result = research_records(
                "topic",
                providers=("openalex", "semantic_scholar"),
                max_results=2,
            )
        self.assertEqual(result["providers"], ["semantic_scholar"])
        self.assertNotIn("openalex", result["providers"])
        self.assertEqual(search.call_count, 1)

    def test_provider_cache_avoids_repeated_network_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "ORCHESTRATOR_RESEARCH_CACHE": "true",
                    "ORCHESTRATOR_RESEARCH_CACHE_DIR": tmp,
                },
                clear=False,
            ), patch(
                "research_providers.search_semantic_scholar",
                return_value={"data": [{"paperId": "S1", "title": "Cached"}]},
            ) as search:
                first = research_records(
                    "cache topic",
                    providers=("semantic_scholar",),
                    max_results=2,
                )
                second = research_records(
                    "cache topic",
                    providers=("semantic_scholar",),
                    max_results=2,
                )
        self.assertEqual(search.call_count, 1)
        self.assertEqual(first["evidence_records"], second["evidence_records"])

    def test_provider_cache_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "ORCHESTRATOR_RESEARCH_CACHE": "false",
                    "ORCHESTRATOR_RESEARCH_CACHE_DIR": tmp,
                },
                clear=False,
            ), patch(
                "research_providers.search_semantic_scholar",
                return_value={"data": [{"paperId": "S1", "title": "Fresh"}]},
            ) as search:
                research_records(
                    "uncached topic",
                    providers=("semantic_scholar",),
                    max_results=2,
                )
                research_records(
                    "uncached topic",
                    providers=("semantic_scholar",),
                    max_results=2,
                )
        self.assertEqual(search.call_count, 2)

    def test_request_retries_429_with_bounded_backoff(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def read(self, size):
                self.size = size
                return b'{"ok": true}'

        rate_limited = urllib.error.HTTPError(
            "https://example.test",
            429,
            "Too Many Requests",
            {"Retry-After": "0"},
            None,
        )
        with patch(
            "research_providers.urllib.request.urlopen",
            side_effect=[rate_limited, Response()],
        ) as urlopen, patch("research_providers.time.sleep") as sleep:
            result = _request_json("https://example.test")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_not_called()

    def test_request_does_not_retry_non_transient_http_error(self):
        forbidden = urllib.error.HTTPError(
            "https://example.test",
            403,
            "Forbidden",
            {},
            None,
        )
        with patch(
            "research_providers.urllib.request.urlopen",
            side_effect=forbidden,
        ) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "HTTP 403"):
                _request_json("https://example.test")
        self.assertEqual(urlopen.call_count, 1)

    def test_default_provider_order_honors_free_only_mode(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False), \
             patch(
                 "research_providers._provider_search",
                 return_value={"data": []},
             ):
            self.assertEqual(
                research_records("topic")["providers"],
                ["semantic_scholar", "europe_pmc", "crossref"],
            )
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "false"}, clear=False), \
             patch(
                 "research_providers._provider_search",
                 return_value={"data": []},
             ):
            self.assertEqual(
                research_records("topic")["providers"],
                ["openalex", "semantic_scholar", "europe_pmc", "crossref", "core"],
            )

    def test_crossref_search_is_public_with_optional_polite_identity(self):
        with patch.dict(os.environ, {"CROSSREF_MAILTO": "test@example.invalid"}, clear=False),              patch("research_providers._request_json", return_value={"message": {"items": []}}) as request:
            from research_providers import search_crossref
            search_crossref("topic", max_results=3)
        url = request.call_args.args[0]
        self.assertIn("api.crossref.org/works?", url)
        self.assertIn("mailto=test%40example.invalid", url)
        self.assertIn("rows=3", url)

    def test_provider_requests_run_in_parallel_with_deterministic_output_order(self):
        def fake(provider, query, max_results):
            if provider == "openalex":
                return {
                    "results": [{
                        "id": "https://openalex.org/W1",
                        "title": "OpenAlex",
                        "publication_year": 2026,
                    }]
                }
            return {
                "data": [{
                    "paperId": "S1",
                    "title": "Semantic Scholar",
                    "year": 2026,
                }]
            }

        with patch.dict(
            os.environ,
            {"ORCHESTRATOR_FREE_ONLY": "false"},
            clear=False,
        ), patch("research_providers._provider_search", side_effect=fake):
            first = research_records(
                "topic",
                providers=("openalex", "semantic_scholar"),
                max_results=2,
            )
            second = research_records(
                "topic",
                providers=("openalex", "semantic_scholar"),
                max_results=2,
            )

        first_providers = [item["provider"] for item in first["evidence_records"]]
        second_providers = [item["provider"] for item in second["evidence_records"]]
        self.assertEqual(set(first_providers), {"openalex", "semantic_scholar"})
        self.assertEqual(first_providers, second_providers)

    def test_provider_failure_is_reported_without_aborting_bundle(self):
        with patch.dict(
            os.environ,
            {"ORCHESTRATOR_FREE_ONLY": "false"},
            clear=False,
        ), patch(
            "research_providers._provider_search",
            side_effect=RuntimeError("down"),
        ):
            result = research_records("topic", providers=("core",))
        self.assertEqual(result["evidence_records"], [])
        self.assertIn("core", result["errors"])

if __name__ == "__main__":
    unittest.main()
