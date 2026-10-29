import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import harvest_doaj_candidates as harvest


OAI_XML = b'''<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
         xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <ListRecords>
    <record>
      <header><identifier>oai:doaj.org/article:abc123</identifier><datestamp>2020-01-01</datestamp></header>
      <metadata><oai_dc:dc>
        <dc:title>Original public policy article</dc:title>
        <dc:creator>Author A</dc:creator>
        <dc:identifier>1234-5679</dc:identifier>
        <dc:identifier>https://doi.org/10.1234/EXAMPLE.2020</dc:identifier>
        <dc:identifier>https://doaj.org/article/abc123</dc:identifier>
        <dc:relation>https://publisher.example/article.pdf</dc:relation>
        <dc:publisher>Official Journal Publisher</dc:publisher>
        <dc:date>2020-04-10</dc:date>
        <dc:language>en</dc:language>
        <dc:subject>public policy</dc:subject>
        <dc:description>This long abstract must not be copied into candidate metadata.</dc:description>
        <dc:type>article</dc:type>
      </oai_dc:dc></metadata>
    </record>
    <record>
      <header><identifier>oai:doaj.org/article:def456</identifier><datestamp>2021-01-01</datestamp></header>
      <metadata><oai_dc:dc>
        <dc:title>Different journal article</dc:title>
        <dc:identifier>9876-543X</dc:identifier>
        <dc:date>2021</dc:date>
      </oai_dc:dc></metadata>
    </record>
    <resumptionToken cursor="0" completeListSize="2">next%2Btoken</resumptionToken>
  </ListRecords>
</OAI-PMH>'''


class DoajCandidateHarvesterTests(unittest.TestCase):
    def test_issn_normalization_and_invalid_values(self):
        self.assertEqual(harvest.normalize_issn("2656-4602"), "2656-4602")
        self.assertEqual(harvest.normalize_issn("26564602"), "2656-4602")
        self.assertIsNone(harvest.normalize_issn("not-an-issn"))

    def test_invalid_check_digit_and_doi_path_do_not_match_journal_issn(self):
        self.assertIsNone(harvest.normalize_issn("1234-567X"))
        self.assertEqual(harvest._issns_in_values(["https://doi.org/10.9999/1234-5679"]), set())
        self.assertEqual(harvest._issns_in_values(["ISSN: 2656-4602"]), {"2656-4602"})

    def test_registry_issn_extraction(self):
        registry = {
            "sources": [
                {"source_url": "https://doaj.org/toc/2656-4602"},
                {"source_url": "https://data.go.id/dataset"},
                {"source_url": "https://doaj.org/toc/26564602/"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(json.dumps(registry), encoding="utf-8")
            found = harvest._registry_issns(path)
        self.assertEqual(found, {"2656-4602"})

    def test_parse_keeps_cc0_metadata_candidates_and_never_copies_abstract(self):
        records, token, info = harvest.parse_records(OAI_XML, {"1234-5679"})
        self.assertEqual(len(records), 1)
        item = records[0]
        self.assertEqual(item["title"], "Original public policy article")
        self.assertEqual(item["doi"], "10.1234/example.2020")
        self.assertEqual(item["journal_issns"], ["1234-5679"])
        self.assertEqual(item["content_license_status"], "not_verified_for_this_article")
        self.assertFalse(item["corpus_eligible"])
        self.assertNotIn("abstract", item)
        self.assertNotIn("description", item)
        self.assertNotIn("long abstract", json.dumps(item))
        self.assertEqual(token, "next%2Btoken")
        self.assertEqual(info["records_in_response"], 2)

    def test_metadata_ids_without_target_issn_are_excluded(self):
        records, token, _ = harvest.parse_records(OAI_XML, {"9999-9999"})
        self.assertEqual(records, [])
        self.assertIsNotNone(token)

    def test_bad_xml_and_oai_errors_fail_closed(self):
        with self.assertRaisesRegex(harvest.HarvestError, "invalid OAI XML"):
            harvest.parse_records(b"<malformed", {"1234-5679"})
        error_xml = b'''<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="badResumptionToken">expired</error></OAI-PMH>'''
        with self.assertRaisesRegex(harvest.HarvestError, "badResumptionToken"):
            harvest.parse_records(error_xml, {"1234-5679"})

    def test_no_records_match_is_not_an_exception(self):
        xml = b'''<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="noRecordsMatch">none</error></OAI-PMH>'''
        records, token, info = harvest.parse_records(xml, {"1234-5679"})
        self.assertEqual(records, [])
        self.assertIsNone(token)
        self.assertEqual(info["oai_error"], "noRecordsMatch")

    def test_merge_is_idempotent_and_never_qualifies_candidates_as_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "candidates.jsonl"
            with patch.object(harvest, "fetch_xml", return_value=OAI_XML):
                first = harvest.harvest(
                    output_path=out, target_issns={"1234-5679"}, max_pages=1,
                    max_records=50, min_interval=0,
                )
            self.assertEqual(first["candidate_inventory_unique"], 1)
            self.assertEqual(first["new_unique_candidates"], 1)
            existing_bytes = out.read_bytes()
            with patch.object(harvest, "fetch_xml", return_value=OAI_XML):
                second = harvest.harvest(
                    output_path=out, target_issns={"1234-5679"}, max_pages=1,
                    max_records=50, min_interval=0,
                )
            self.assertEqual(second["candidate_inventory_unique"], 1)
            self.assertEqual(second["new_unique_candidates"], 0)
            self.assertEqual(out.read_bytes(), existing_bytes)
            item = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(item["corpus_eligible"])
            self.assertEqual(second["target_500_status"], "INSUFFICIENT_EVIDENCE")

    def test_dry_run_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "candidates.jsonl"
            with patch.object(harvest, "fetch_xml", return_value=OAI_XML):
                report = harvest.harvest(
                    output_path=out, target_issns={"1234-5679"}, max_pages=1,
                    max_records=50, min_interval=0, dry_run=True,
                )
            self.assertTrue(report["dry_run"])
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
