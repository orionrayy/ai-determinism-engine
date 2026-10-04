import os
import unittest
from unittest.mock import patch

from claim_integrity import validate_truth_lock
from epistemic_deliberation import _evidence_strength
from epistemic_deliberation_runtime import (
    build_claim_challenges,
    validate_deliberation_responses,
)
from epistemic_validation import validate_epistemic_output
from context_budget import pack_node_context
from orchestrator import deterministic_plan
from evidence_records import normalize_source
from research_budget import default_available_providers
from research_providers import search_crossref, search_openalex, _provider_local_rate


class V79BoundaryQualityTests(unittest.TestCase):
    def test_openalex_is_available_in_hard_free_mode(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_FREE_ONLY": "true"}, clear=False):
            providers = default_available_providers()
        self.assertIn("openalex", providers)

    def test_openalex_does_not_receive_api_key_in_free_mode(self):
        with patch.dict(
            os.environ,
            {
                "ORCHESTRATOR_FREE_ONLY": "true",
                "OPENALEX_API_KEY": "should-not-be-sent",
            },
            clear=False,
        ):
            with patch("research_providers._request_json", return_value={}) as request:
                search_openalex("test query", max_results=2)
        url = request.call_args.args[0]
        self.assertNotIn("api_key=", url)

    def test_crossref_local_fallback_matches_pool_rate(self):
        with patch.dict(os.environ, {"CROSSREF_MAILTO": ""}, clear=False):
            self.assertEqual(_provider_local_rate("crossref"), 1.0)
        with patch.dict(os.environ, {"CROSSREF_MAILTO": "research@example.org"}, clear=False):
            self.assertEqual(_provider_local_rate("crossref"), 3.0)

    def test_crossref_can_use_polite_mailto_identity(self):
        with patch.dict(
            os.environ,
            {"CROSSREF_MAILTO": "research@example.org"},
            clear=False,
        ):
            with patch("research_providers._request_json", return_value={}) as request:
                search_crossref("test query", max_results=2)
        headers = request.call_args.kwargs.get("headers") or {}
        self.assertIn("mailto=research@example.org", headers.get("User-Agent", ""))

    def test_europe_pmc_full_text_route_is_normalized(self):
        record = normalize_source(
            "europe_pmc",
            {
                "pmcid": "PMC123",
                "title": "Example",
                "year": 2026,
                "fullTextUrlList": {
                    "fullTextUrl": [
                        {"url": "https://example.org/fulltext.xml", "documentStyle": "html"}
                    ]
                },
            },
        )
        self.assertEqual(record["full_text_url"], "https://example.org/fulltext.xml")
        self.assertEqual(record["access_level"], "L3")

    def test_retraction_signal_from_index_is_hard_negative(self):
        record = normalize_source(
            "openalex",
            {
                "id": "https://openalex.org/W1",
                "title": "Retracted Study",
                "publication_year": 2026,
                "is_retracted": True,
            },
        )
        self.assertEqual(record["publication_status"], "retracted")
        self.assertTrue(record["retraction_signal"])

    def test_truth_lock_rejects_numeric_detail_loss(self):
        source = {
            "claims": [{
                "claim_id": "c1",
                "statement": "Treatment A reduced symptoms by 20 percent.",
                "status": "SUPPORTED_DIRECT",
                "evidence_refs": ["doi:x"],
            }]
        }
        draft = {
            "claims": [{
                "claim_id": "d1",
                "statement": "Treatment A reduced symptoms.",
                "status": "SUPPORTED_DIRECT",
                "derives_from_claims": ["c1"],
                "evidence_refs": ["doi:x"],
            }]
        }
        result = validate_truth_lock(draft, source)
        self.assertFalse(result["passed"])

    def test_truth_lock_rejects_negation_flip(self):
        source = {
            "claims": [{
                "claim_id": "c1",
                "statement": "Treatment A did not improve survival.",
                "status": "CONTESTED",
                "evidence_refs": ["doi:x"],
            }]
        }
        draft = {
            "claims": [{
                "claim_id": "d1",
                "statement": "Treatment A improved survival.",
                "status": "CONTESTED",
                "derives_from_claims": ["c1"],
                "evidence_refs": ["doi:x"],
            }]
        }
        result = validate_truth_lock(draft, source)
        self.assertFalse(result["passed"])

    def test_rebuttal_must_use_challenge_evidence(self):
        challenges = [{
            "claim_id": "c1",
            "challenge_type": "claim_disagreement",
            "evidence_refs": ["doi:x"],
        }]
        verdict = {
            "evidence_records": [
                {"canonical_id": "doi:x"},
                {"canonical_id": "doi:y"},
            ],
            "deliberation_responses": [{
                "claim_id": "c1",
                "status": "resolved",
                "evidence_refs": ["doi:y"],
            }],
        }
        result = validate_deliberation_responses(
            verdict,
            {"challenges": challenges},
        )
        self.assertFalse(result["passed"])

    def test_rebuttal_requires_explicit_evidence_delta(self):
        challenges = [{
            "claim_id": "c1",
            "challenge_type": "claim_status_conflict",
            "evidence_refs": ["doi:x"],
        }]
        verdict = {
            "evidence_records": [{"canonical_id": "doi:x"}],
            "deliberation_responses": [{
                "claim_id": "c1",
                "status": "contested",
                "evidence_refs": ["doi:x"],
            }],
        }
        result = validate_deliberation_responses(
            verdict,
            {"challenges": challenges},
        )
        self.assertFalse(result["passed"])

    def test_strict_passage_must_belong_to_claim_evidence_refs(self):
        verdict = {
            "claims": [{
                "claim_id": "c1",
                "statement": "Treatment A improved survival.",
                "status": "SUPPORTED_DIRECT",
                "material": True,
                "evidence_refs": ["doi:x"],
                "evidence_passages": [{
                    "evidence_ref": "doi:y",
                    "text": "Treatment A improved survival.",
                }],
            }],
            "evidence_records": [
                {
                    "canonical_id": "doi:x",
                    "abstract": "Treatment A improved survival.",
                },
                {
                    "canonical_id": "doi:y",
                    "abstract": "Treatment A improved survival.",
                },
            ],
        }
        result = validate_epistemic_output(verdict, require_passages=True)
        self.assertFalse(result["passed"])

    def test_declared_access_does_not_score_as_verified_access(self):
        declared = _evidence_strength({
            "evidence_records": [{
                "canonical_id": "doi:x",
                "authority_score": 1.0,
                "independence_confidence": 1.0,
                "access_verification": "declared_only",
                "publication_status": "normal",
            }],
        })
        verified = _evidence_strength({
            "evidence_records": [{
                "canonical_id": "doi:x",
                "authority_score": 1.0,
                "independence_confidence": 1.0,
                "access_verification": "verified",
                "publication_status": "normal",
            }],
        })
        self.assertLess(declared, verified)

    def test_high_utility_evidence_gets_abstract_window(self):
        records = [{
            "canonical_id": f"doi:{index:02d}",
            "authority_score": 0.2,
            "independence_confidence": 0.2,
            "access_verification": "identifier_only",
            "publication_status": "normal",
            "year": 2016,
            "abstract": "",
        } for index in range(32)]
        records.append({
            "canonical_id": "doi:zz",
            "authority_score": 1.0,
            "independence_confidence": 1.0,
            "access_verification": "verified",
            "publication_status": "normal",
            "year": 2026,
            "abstract": "High utility evidence passage.",
        })
        packed = pack_node_context(
            goal="research",
            dependencies={},
            contract={"epistemic": True},
            repair_feedback={},
            trusted_evidence_records=records,
        )
        high = next(
            item for item in packed["trusted_evidence"]
            if item["canonical_id"] == "doi:zz"
        )
        self.assertEqual(high["abstract"], "High utility evidence passage.")

    def test_strict_passage_requires_verifiable_corpus(self):
        verdict = {
            "claims": [{
                "claim_id": "c1",
                "statement": "Treatment A improved survival.",
                "status": "SUPPORTED_DIRECT",
                "material": True,
                "evidence_refs": ["doi:x"],
                "evidence_passages": [{
                    "evidence_ref": "doi:x",
                    "text": "Treatment A improved survival.",
                }],
            }],
            "evidence_records": [{
                "canonical_id": "doi:x",
            }],
        }
        result = validate_epistemic_output(verdict, require_passages=True)
        self.assertFalse(result["passed"])
        self.assertGreater(result["passage_validation"]["missing_corpus"], 0)

    def test_trusted_context_preserves_access_verification(self):
        packed = pack_node_context(
            goal="research",
            dependencies={},
            contract={"epistemic": True},
            repair_feedback={},
            trusted_evidence_records=[{
                "canonical_id": "doi:x",
                "access_level": "L1",
                "access_route": "publisher_url",
                "access_verification": "metadata_only",
            }],
        )
        record = packed["trusted_evidence"][0]
        self.assertEqual(record["access_verification"], "metadata_only")
        self.assertEqual(record["access_level"], "L1")

    def test_strict_passage_length_is_bounded(self):
        long_text = "x" * 701
        verdict = {
            "claims": [{
                "claim_id": "c1",
                "statement": "Treatment A improved survival.",
                "status": "SUPPORTED_DIRECT",
                "material": True,
                "evidence_refs": ["doi:x"],
                "evidence_passages": [{
                    "evidence_ref": "doi:x",
                    "text": long_text,
                }],
            }],
            "evidence_records": [{
                "canonical_id": "doi:x",
                "abstract": long_text,
            }],
        }
        result = validate_epistemic_output(verdict, require_passages=True)
        self.assertFalse(result["passed"])

    def test_research_epistemic_nodes_require_passages(self):
        nodes = deterministic_plan("research on treatment outcomes", {}, live=False)
        epistemic_nodes = [
            node for node in nodes
            if node.contract.get("epistemic")
        ]
        self.assertTrue(epistemic_nodes)
        self.assertTrue(
            all(node.contract.get("require_passages") is True for node in epistemic_nodes)
        )

    def test_claim_challenge_is_deterministic(self):
        proposals = [
            {
                "claims": [{
                    "claim_id": "c1",
                    "statement": "A",
                    "status": "SUPPORTED_DIRECT",
                    "material": True,
                    "evidence_refs": ["doi:x"],
                }],
                "evidence_records": [{
                    "canonical_id": "doi:x",
                    "authority_score": 0.9,
                }],
            },
            {
                "claims": [{
                    "claim_id": "c1",
                    "statement": "B",
                    "status": "SUPPORTED_DIRECT",
                    "material": True,
                    "evidence_refs": ["doi:x"],
                }],
                "evidence_records": [{
                    "canonical_id": "doi:x",
                    "authority_score": 0.9,
                }],
            },
        ]
        first = build_claim_challenges(proposals)
        second = build_claim_challenges(list(reversed(proposals)))
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertEqual(first[0]["claim_id"], "c1")


if __name__ == "__main__":
    unittest.main()
