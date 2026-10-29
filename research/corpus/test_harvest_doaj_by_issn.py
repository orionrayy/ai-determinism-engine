import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import harvest_doaj_by_issn as harvest


def registry_file(folder: Path) -> Path:
    path = folder / "registry.json"
    path.write_text(json.dumps({
        "sources": [
            {
                "source_id": "target-a",
                "name": "Journal A",
                "source_type": "open_access_journal_directory",
                "source_url": "https://doaj.org/toc/2548-592X",
                "counts_as_document": False,
            },
            {
                "source_id": "target-b",
                "name": "Journal B",
                "source_type": "open_access_journal_directory",
                "source_url": "https://doaj.org/toc/2656-4602",
                "counts_as_document": False,
            },
            {
                "source_id": "catalog-only",
                "name": "Catalogue",
                "source_type": "official_public_data_catalogue",
                "source_url": "https://data.go.id/dataset",
                "counts_as_document": False,
            },
        ]
    }), encoding="utf-8")
    return path


def doaj_article(article_id="article-a", title="Example research article"):
    return {
        "id": article_id,
        "bibjson": {
            "title": title,
            "author": [{"name": "Author One"}, {"name": "Author Two"}],
            "year": "2024",
            "identifier": [{"type": "doi", "id": "10.1234/example.2024"}],
            "journal": {
                "title": "Journal A",
                "publisher": "Publisher A",
                "issns": ["2548-592X", "2656-4602"],
            },
            "link": [{"type": "fulltext", "url": "https://publisher.example/article.pdf"}],
            "abstract": "This abstract must never enter our candidate inventory.",
            "license": [{"title": "CC BY 4.0"}],
        },
    }


def payload(results, total=None):
    return json.dumps({
        "total": len(results) if total is None else total,
        "page": 1,
        "pageSize": 10,
        "results": results,
        "last": "https://doaj.org/api/v4/search/articles",
    }).encode("utf-8")


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class DoajByIssnTests(unittest.TestCase):
    def test_metadata_is_discovered_without_abstract_or_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = registry_file(Path(tmp))
            responses = {
                "2548-592X": payload([doaj_article()], total=127),
                "2656-4602": payload([], total=0),
            }
            requested = []

            def opener(request, timeout):
                requested.append((request.full_url, timeout))
                issn = "2548-592X" if "2548-592X" in request.full_url else "2656-4602"
                return FakeResponse(responses[issn])

            with patch.object(harvest, "time") as clock:
                clock.monotonic.side_effect = [0, 1, 2, 3]
                clock.sleep.return_value = None
                result = harvest.discover(
                    registry, per_issn=10, min_interval=1, opener=opener,
                    sleeper=lambda _delay: None,
                )

        self.assertEqual(result["summary"]["status"], "CANDIDATES_DISCOVERED")
        self.assertEqual(result["summary"]["target_journal_count"], 2)
        self.assertEqual(result["summary"]["successful_queries"], 2)
        self.assertEqual(result["summary"]["failed_queries"], 0)
        self.assertEqual(result["summary"]["api_records_reported_across_queries"], 127)
        self.assertEqual(len(result["candidates"]), 1)
        item = result["candidates"][0]
        self.assertEqual(item["title"], "Example research article")
        self.assertEqual(item["doi"], "10.1234/example.2024")
        self.assertEqual(item["matched_target_issns"], ["2548-592X"])
        self.assertEqual(item["article_license_status"], "not_verified_for_this_article")
        self.assertFalse(item["corpus_eligible"])
        self.assertFalse(item["abstract_saved"])
        self.assertFalse(item["full_text_downloaded"])
        self.assertNotIn("abstract", item)
        self.assertNotIn("license", item)
        self.assertNotIn('"abstract":', json.dumps(item).lower())
        self.assertNotIn("this abstract must never", json.dumps(item).lower())
        self.assertEqual(len(requested), 2)
        self.assertTrue(all("pageSize=10" in url for url, _ in requested))
        self.assertTrue(all(timeout == 15 for _, timeout in requested))

    def test_candidate_inventory_writes_atomically_and_summary_is_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            registry = registry_file(folder)
            output = folder / "candidates.jsonl"
            summary = folder / "summary.json"

            def opener(request, timeout):
                result = [doaj_article()] if "2548-592X" in request.full_url else []
                return FakeResponse(payload(result))

            with patch.object(harvest.time, "monotonic", side_effect=[0, 2, 4, 6]):
                result = harvest.discover(registry, opener=opener, sleeper=lambda _delay: None)

            harvest._atomic_write(output, "".join(json.dumps(row) + "\n" for row in result["candidates"]))
            harvest._atomic_write(summary, json.dumps(result["summary"]) + "\n")
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
            item = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            report = json.loads(summary.read_text(encoding="utf-8"))
            self.assertFalse(item["corpus_eligible"])
            self.assertEqual(report["status"], "CANDIDATES_DISCOVERED")
            self.assertFalse(report["candidate_inventory_is_corpus"])

    def test_duplicate_candidate_across_issns_is_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = registry_file(Path(tmp))
            requested = []

            def opener(request, timeout):
                requested.append(request.full_url)
                return FakeResponse(payload([doaj_article()]))

            with patch.object(harvest.time, "monotonic", side_effect=[0, 2, 4, 6]):
                result = harvest.discover(registry, opener=opener, sleeper=lambda _delay: None)

        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["matched_target_issns"], ["2548-592X", "2656-4602"])
        self.assertEqual(len(requested), 2)

    def test_one_failed_query_does_not_discard_other_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = registry_file(Path(tmp))

            def opener(request, timeout):
                if "2656-4602" in request.full_url:
                    raise OSError("temporary service failure")
                return FakeResponse(payload([doaj_article()]))

            with patch.object(harvest.time, "monotonic", side_effect=[0, 2, 4]):
                result = harvest.discover(registry, opener=opener, sleeper=lambda _delay: None)

        self.assertEqual(result["summary"]["status"], "PARTIAL")
        self.assertEqual(result["summary"]["successful_queries"], 1)
        self.assertEqual(result["summary"]["failed_queries"], 1)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["summary"]["errors"][0]["issn"], "2656-4602")

    def test_all_queries_fail_is_not_reported_as_zero_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = registry_file(Path(tmp))

            def opener(_request, _timeout):
                raise OSError("offline")

            with patch.object(harvest.time, "monotonic", side_effect=[0, 2, 4]):
                result = harvest.discover(registry, opener=opener, sleeper=lambda _delay: None)

        self.assertEqual(result["summary"]["status"], "LIVE_DISCOVERY_UNAVAILABLE")
        self.assertEqual(result["summary"]["successful_queries"], 0)
        self.assertEqual(result["summary"]["failed_queries"], 2)
        self.assertEqual(result["summary"]["unique_candidate_records_written"], 0)

    def test_response_size_bound_is_enforced(self):
        # Exercise the real 8 MiB read boundary rather than monkeypatching the
        # same constant that controls both the read size and the comparison.
        oversized = b"x" * (harvest.MAX_RESPONSE_BYTES + 1)
        with tempfile.TemporaryDirectory() as tmp:
            registry = registry_file(Path(tmp))

            def opener(_request, timeout):
                return FakeResponse(oversized)

            with patch.object(harvest.time, "monotonic", side_effect=[0, 2, 4]):
                result = harvest.discover(registry, opener=opener, sleeper=lambda _delay: None)

        self.assertEqual(result["summary"]["status"], "LIVE_DISCOVERY_UNAVAILABLE")
        self.assertEqual(len(result["summary"]["errors"]), 2)
        self.assertTrue(all("api_response_exceeded_8MiB_limit" in e["error"] for e in result["summary"]["errors"]), msg=repr(result["summary"]["errors"]))

    def test_invalid_limits_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = registry_file(Path(tmp))
            with self.assertRaisesRegex(ValueError, "min_interval_must_be_at_least"):
                harvest.discover(registry, min_interval=0.1, opener=lambda *_a, **_kw: None)
            with self.assertRaisesRegex(ValueError, "per_issn_must_be_between"):
                harvest.discover(registry, per_issn=51, opener=lambda *_a, **_kw: None)


if __name__ == "__main__":
    unittest.main()
