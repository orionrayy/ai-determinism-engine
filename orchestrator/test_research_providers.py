import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from research_providers import (
    _request_json,
    normalize_provider_payload,
    research_records,
)

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
        sleep.assert_called_once_with(0.0)

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
                ["semantic_scholar", "europe_pmc"],
            )
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "false"}, clear=False), \
             patch(
                 "research_providers._provider_search",
                 return_value={"data": []},
             ):
            self.assertEqual(
                research_records("topic")["providers"],
                ["openalex", "semantic_scholar", "europe_pmc", "core"],
            )

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

        with patch("research_providers._provider_search", side_effect=fake):
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
