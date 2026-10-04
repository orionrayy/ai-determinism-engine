import unittest

from epistemic_deliberation import blind_challenge_view, debate_decision
from epistemic_deliberation_runtime import build_claim_challenges, deliberation_context, validate_deliberation_responses


class EpistemicDeliberationTests(unittest.TestCase):
    def test_debate_stops_when_proposals_agree_with_strong_evidence(self):
        proposals = [
            {
                "agent_id": "a1",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["x"],
                "evidence_records": [
                    {"doi": "10.1/a", "provider": "openalex"},
                    {"doi": "10.2/b", "provider": "openalex"},
                ],
            },
            {
                "agent_id": "a2",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["y"],
                "evidence_records": [
                    {"doi": "10.1/a", "provider": "crossref"},
                    {"doi": "10.2/b", "provider": "crossref"},
                ],
            },
        ]
        result = debate_decision(proposals)
        self.assertFalse(result["required"])
        self.assertEqual(result["reason"], "stable_consensus")
        self.assertEqual(result["challenge_mode"], "none")

    def test_debate_triggers_on_weak_independent_evidence_even_with_refs(self):
        proposals = [
            {
                "agent_id": "a1",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["x"],
                "independence": {"distinct_work_count": 1},
            },
            {
                "agent_id": "a2",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["y"],
                "independence": {"distinct_work_count": 1},
            },
        ]
        result = debate_decision(proposals)
        self.assertTrue(result["required"])
        self.assertEqual(result["reason"], "insufficient_evidence")
        self.assertEqual(result["challenge_mode"], "blind")

    def test_debate_triggers_on_disagreement_or_weak_evidence(self):
        proposals = [
            {
                "agent_id": "a1",
                "answer": "A",
                "confidence": 0.7,
                "evidence_refs": ["x"],
                "evidence_records": [
                    {"doi": "10.1/a", "provider": "openalex"},
                    {"doi": "10.2/b", "provider": "openalex"},
                ],
            },
            {
                "agent_id": "a2",
                "answer": "B",
                "confidence": 0.9,
                "evidence_refs": ["y"],
                "evidence_records": [
                    {"doi": "10.3/c", "provider": "crossref"},
                    {"doi": "10.4/d", "provider": "crossref"},
                ],
            },
        ]
        result = debate_decision(proposals)
        self.assertTrue(result["required"])
        self.assertEqual(result["reason"], "material_disagreement")
        self.assertEqual(result["max_rounds"], 2)

    def test_blind_view_removes_consensus_fields(self):
        view = blind_challenge_view([
            {
                "agent_id": "a2",
                "answer": "B",
                "vote_count": 1,
                "majority": True,
            },
            {
                "agent_id": "a1",
                "answer": "A",
                "vote_count": 2,
                "majority": False,
            },
        ])
        self.assertEqual([item["agent_id"] for item in view], ["a1", "a2"])
        self.assertNotIn("vote_count", view[0])
        self.assertNotIn("majority", view[0])


    def test_claim_challenge_targets_material_disagreement(self):
        proposals = [
            {
                "agent_id": "a1",
                "claims": [{
                    "claim_id": "c1",
                    "statement": "A is effective.",
                    "material": True,
                    "status": "SUPPORTED_DIRECT",
                    "evidence_refs": ["e1"],
                }],
                "evidence_records": [{
                    "canonical_id": "e1",
                    "authority_score": 0.9,
                }],
            },
            {
                "agent_id": "a2",
                "claims": [{
                    "claim_id": "c1",
                    "statement": "A is ineffective.",
                    "material": True,
                    "status": "CONTESTED",
                    "evidence_refs": ["e2"],
                }],
                "evidence_records": [{
                    "canonical_id": "e2",
                    "authority_score": 0.85,
                }],
            },
        ]
        challenges = build_claim_challenges(proposals)
        self.assertEqual(len(challenges), 1)
        self.assertEqual(challenges[0]["claim_id"], "c1")
        self.assertEqual(challenges[0]["challenge_type"], "claim_disagreement")
        self.assertTrue(challenges[0]["requires_rebuttal"])

    def test_deliberation_context_has_adaptive_rounds(self):
        proposals = [
            {
                "agent_id": "a1",
                "answer": "A",
                "confidence": 0.7,
                "claims": [{
                    "claim_id": "c1",
                    "statement": "A",
                    "material": True,
                    "status": "SUPPORTED_DIRECT",
                    "evidence_refs": ["e1"],
                }],
                "evidence_records": [{
                    "canonical_id": "e1",
                    "authority_score": 0.9,
                }],
            },
            {
                "agent_id": "a2",
                "answer": "B",
                "confidence": 0.7,
                "claims": [{
                    "claim_id": "c1",
                    "statement": "B",
                    "material": True,
                    "status": "CONTESTED",
                    "evidence_refs": ["e2"],
                }],
                "evidence_records": [{
                    "canonical_id": "e2",
                    "authority_score": 0.8,
                }],
            },
        ]
        context = deliberation_context(proposals)
        self.assertEqual(context["policy_version"], 2)
        self.assertTrue(context["challenges"])
        self.assertEqual(context["adaptive_rounds"][0]["round"], 1)
        self.assertEqual(context["adaptive_rounds"][1]["round"], 2)

    def test_deliberation_responses_require_each_claim_challenge(self):
        deliberation = {
            "challenges": [{
                "claim_id": "c1",
                "challenge_type": "claim_disagreement",
            }]
        }
        verdict = {
            "evidence_records": [{
                "canonical_id": "e1",
            }],
            "deliberation_responses": [{
                "claim_id": "c1",
                "status": "resolved",
                "evidence_refs": ["e1"],
                "justification": "Direct evidence resolves the conflict.",
            }],
        }
        result = validate_deliberation_responses(verdict, deliberation)
        self.assertTrue(result["passed"])
        self.assertEqual(result["response_count"], 1)

    def test_deliberation_responses_reject_unknown_evidence(self):
        deliberation = {
            "challenges": [{
                "claim_id": "c1",
                "challenge_type": "evidence_gap",
            }]
        }
        verdict = {
            "evidence_records": [{
                "canonical_id": "trusted",
            }],
            "deliberation_responses": [{
                "claim_id": "c1",
                "status": "resolved",
                "evidence_refs": ["forged"],
            }],
        }
        result = validate_deliberation_responses(verdict, deliberation)
        self.assertFalse(result["passed"])
        self.assertIn("c1", result["invalid"])


if __name__ == "__main__":
    unittest.main()
