import unittest

from epistemic_deliberation import blind_challenge_view, debate_decision


class EpistemicDeliberationTests(unittest.TestCase):
    def test_debate_stops_when_proposals_agree_with_strong_evidence(self):
        proposals = [
            {
                "agent_id": "a1",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["x"],
                "independent_source_count": 2,
            },
            {
                "agent_id": "a2",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["y"],
                "independent_source_count": 2,
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
                "independent_source_count": 1,
            },
            {
                "agent_id": "a2",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["y"],
                "independent_source_count": 1,
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
                "independent_source_count": 2,
            },
            {
                "agent_id": "a2",
                "answer": "B",
                "confidence": 0.9,
                "evidence_refs": ["y"],
                "independent_source_count": 2,
            },
        ]
        result = debate_decision(proposals)
        self.assertTrue(result["required"])
        self.assertEqual(result["reason"], "material_disagreement")
        self.assertEqual(result["max_rounds"], 2)

    def test_blind_view_is_identity_opaque(self):
        view = blind_challenge_view([
            {
                "agent_id": "a2",
                "agent_role": "skeptic",
                "answer": "B",
                "vote_count": 1,
                "majority": True,
            },
            {
                "agent_id": "a1",
                "agent_role": "analyst",
                "answer": "A",
                "vote_count": 2,
                "majority": False,
            },
        ])
        self.assertEqual(
            [item["candidate_id"] for item in view],
            ["candidate_1", "candidate_2"],
        )
        for item in view:
            self.assertNotIn("agent_id", item)
            self.assertNotIn("agent_role", item)
            self.assertNotIn("vote_count", item)
            self.assertNotIn("majority", item)

    def test_self_reported_independent_source_count_cannot_satisfy_evidence(self):
        proposals = [
            {
                "agent_id": "a1",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["x"],
                "independent_source_count": 999,
                "evidence_records": [],
            },
            {
                "agent_id": "a2",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["y"],
                "independent_source_count": 999,
                "evidence_records": [],
            },
        ]
        result = debate_decision(proposals)
        self.assertTrue(result["required"])
        self.assertEqual(result["reason"], "insufficient_evidence")

    def test_valid_evidence_records_satisfy_independence_threshold(self):
        evidence = [
            {"canonical_id": "doi:10.1/a"},
            {"canonical_id": "doi:10.1/b"},
        ]
        proposals = [
            {
                "agent_id": "a1",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["doi:10.1/a"],
                "independent_source_count": 0,
                "evidence_records": evidence,
            },
            {
                "agent_id": "a2",
                "answer": "A",
                "confidence": 0.9,
                "evidence_refs": ["doi:10.1/b"],
                "independent_source_count": 0,
                "evidence_records": evidence,
            },
        ]
        result = debate_decision(proposals)
        self.assertFalse(result["required"])
        self.assertEqual(result["reason"], "stable_consensus")


if __name__ == "__main__":
    unittest.main()
