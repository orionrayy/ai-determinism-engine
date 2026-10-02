import unittest
from types import SimpleNamespace

from agent_fabric import (
    AGENT_PROTOCOL_VERSION,
    agent_id,
    assign_role,
    collaboration_mode,
    team_manifest,
    validate_role,
)


class AgentFabricTests(unittest.TestCase):
    def test_capability_roles_are_valid(self):
        node = SimpleNamespace(
            id="n01",
            capability="research",
            risk="low",
            agent_role="",
        )
        self.assertEqual(assign_role(node), "researcher")
        self.assertEqual(node.agent_role, "researcher")

    def test_invalid_role_capability_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_role("tester", "research")

    def test_agent_id_is_stable(self):
        first = agent_id("wf-a", "n01", "researcher")
        second = agent_id("wf-a", "n01", "researcher")
        self.assertEqual(first, second)
        self.assertNotEqual(first, agent_id("wf-a", "n02", "researcher"))

    def test_team_manifest_exposes_handoffs_and_deliberation(self):
        nodes = [
            SimpleNamespace(id="n1", capability="research", risk="low", agent_role="researcher", depends_on=[]),
            SimpleNamespace(id="n2", capability="research", risk="low", agent_role="skeptic", depends_on=[]),
            SimpleNamespace(id="n3", capability="analyze", risk="low", agent_role="analyst", depends_on=["n1", "n2"]),
            SimpleNamespace(id="n4", capability="validate", risk="low", agent_role="critic", depends_on=["n3"]),
        ]
        manifest = team_manifest("wf-a", nodes)
        self.assertEqual(manifest["protocol_version"], AGENT_PROTOCOL_VERSION)
        self.assertEqual(manifest["pattern"], "parallel_deliberation")
        self.assertEqual(manifest["consensus_nodes"], 0)
        self.assertEqual(len(manifest["members"]), 4)
        self.assertEqual(len(manifest["edges"]), 3)

    def test_consensus_role_is_detected(self):
        nodes = [
            SimpleNamespace(id="n1", capability="research", risk="low", agent_role="researcher", depends_on=[]),
            SimpleNamespace(id="n2", capability="research", risk="low", agent_role="skeptic", depends_on=[]),
            SimpleNamespace(id="n3", capability="validate", risk="low", agent_role="critic", depends_on=["n1", "n2"]),
        ]
        manifest = team_manifest("wf-a", nodes)
        self.assertEqual(manifest["pattern"], "parallel_deliberation")
        self.assertEqual(manifest["consensus_nodes"], 1)

    def test_collaboration_mode_is_deterministic(self):
        nodes = [
            SimpleNamespace(id="n1", capability="research", risk="low", agent_role="researcher", depends_on=[]),
            SimpleNamespace(id="n2", capability="research", risk="low", agent_role="skeptic", depends_on=[]),
        ]
        self.assertEqual(collaboration_mode(nodes), "scatter_gather")


if __name__ == "__main__":
    unittest.main()
