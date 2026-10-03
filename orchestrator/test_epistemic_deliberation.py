import unittest

from epistemic_deliberation import debate_decision, blind_challenge_view


class EpistemicDeliberationTests(unittest.TestCase):
    def test_debate_stops_when_proposals_agree_with_strong_evidence(self):
        proposals = [
            {"agent_id":"a1","answer":"A","confidence":0.9,"evidence_refs":["s1","s2"]},
            {"agent_id":"a2","answer":"A","confidence":0.88,"evidence_refs":["s1","s2"]},
        ]
        result = debate_decision(proposals)
        self.assertFalse(result["required"])
        self.assertEqual(result["reason"], "stable_consensus")

    def test_debate_triggers_on_disagreement_or_weak_evidence(self):
        proposals = [
            {"agent_id":"a1","answer":"A","confidence":0.9,"evidence_refs":["s1"]},
            {"agent_id":"a2","answer":"B","confidence":0.55,"evidence_refs":[]},
        ]
        result = debate_decision(proposals)
        self.assertTrue(result["required"])
        self.assertIn(result["reason"], {"material_disagreement","low_confidence","insufficient_evidence"})

    def test_blind_challenge_does_not_expose_vote_counts(self):
        proposals = [
            {"agent_id":"a2","answer":"B","confidence":0.5,"vote_count":4},
            {"agent_id":"a1","answer":"A","confidence":0.7,"vote_count":7},
        ]
        view = blind_challenge_view(proposals)
        self.assertEqual([item["agent_id"] for item in view], ["a1","a2"])
        self.assertNotIn("vote_count", view[0])
        self.assertNotIn("vote_count", view[1])


if __name__ == "__main__":
    unittest.main()
