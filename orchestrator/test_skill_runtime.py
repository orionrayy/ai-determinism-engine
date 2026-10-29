import copy
import unittest

import skill_runtime as sr
from enterprise_runtime import SkillAuthorizationError, SkillRegistry


class FakeNode:
    def __init__(self, node_id, *, tool="gemini", capability="analyze", role="analyst", instruction=""):
        self.id = node_id
        self.tool = tool
        self.capability = capability
        self.agent_role = role
        self.input = {"instruction": instruction}
        self.contract = {}
        self.risk = "low"


class SkillRuntimeTests(unittest.TestCase):
    def test_catalogued_skills_have_valid_pinned_content_and_manifests(self):
        catalog = sr.load_skill_catalog()
        registry = SkillRegistry()
        self.assertEqual({item["skill_id"] for item in catalog["skills"]}, {"hallmark", "document-craft"})
        for entry in catalog["skills"]:
            content, blob_sha = sr._read_verified_skill(entry)
            self.assertTrue(content.strip())
            self.assertEqual(blob_sha, entry["git_blob_sha"])
            manifest = sr._build_manifest(entry)
            registry.register(manifest)
            self.assertEqual(registry.get(entry["skill_id"], entry["version"]).digest, manifest.digest)

    def test_selector_recognizes_indonesian_and_english_document_work(self):
        self.assertEqual(
            sr.selected_skill_ids("Tulis makalah dan laporan praktikum Bahasa Indonesia yang natural"),
            ("document-craft",),
        )
        self.assertEqual(sr.selected_skill_ids("Write a technical report from verified sources"), ("document-craft",))
        self.assertEqual(sr.selected_skill_ids("Build a responsive website landing page"), ("hallmark",))
        self.assertEqual(
            sr.selected_skill_ids("Build a website and prepare a humanized academic report"),
            ("document-craft", "hallmark"),
        )
        self.assertEqual(sr.selected_skill_ids("Explain the concept of a reward system"), ())

    def test_document_skill_is_authorized_and_added_to_writing_nodes(self):
        nodes = [
            FakeNode("research", capability="research", role="researcher", instruction="Find trustworthy sources"),
            FakeNode("draft", capability="draft", role="writer", instruction="Draft the paper"),
            FakeNode("review", capability="validate", role="critic", instruction="Review the citations and claims"),
            FakeNode("publish", tool="github", capability="write", role="publisher", instruction="Publish release"),
        ]
        report = sr.apply_skill_routing(
            "Buat makalah akademik dan laporan praktikum yang humanized dengan sumber yang valid",
            nodes,
            free_only=True,
        )
        self.assertEqual(report["selected_skill_ids"], ["document-craft"])
        attached = set(report["routes"][0]["attached_node_ids"])
        self.assertTrue({"research", "draft", "review"}.issubset(attached))
        self.assertNotIn("publish", attached)
        for node in nodes[:3]:
            self.assertIn("VERIFIED REPOSITORY SKILL CONTEXT: document-craft@1.0.0", node.input["instruction"])
            self.assertIn("Never invent sources", node.input["instruction"])
            self.assertEqual(len(node.input["skill_context"]), 1)
            self.assertTrue(node.input["skill_context"][0]["manifest_digest"].startswith("sha256:"))
        self.assertNotIn("skill_context", nodes[3].input)

        old_length = len(nodes[1].input["instruction"])
        sr.apply_skill_routing(
            "Buat makalah akademik dan laporan praktikum yang humanized dengan sumber yang valid",
            nodes,
            free_only=True,
        )
        self.assertEqual(len(nodes[1].input["instruction"]), old_length)
        self.assertEqual(len(nodes[1].input["skill_context"]), 1)

    def test_mixed_document_and_ui_goals_route_to_separate_nodes(self):
        nodes = [
            FakeNode("implement", capability="implement", role="implementer", instruction="Build the landing page"),
            FakeNode("ui-review", capability="validate", role="tester", instruction="Check responsive behavior and accessibility"),
            FakeNode("document-draft", capability="draft", role="writer", instruction="Write the report with accurate citations"),
            FakeNode("document-review", capability="validate", role="critic", instruction="Review the report and bibliography"),
        ]
        report = sr.apply_skill_routing(
            "Build a website landing page and prepare a humanized academic report about its operation",
            nodes,
            free_only=True,
        )
        routes = {route["skill_id"]: set(route["attached_node_ids"]) for route in report["routes"]}
        self.assertEqual(report["selected_skill_ids"], ["document-craft", "hallmark"])
        self.assertIn("document-draft", routes["document-craft"])
        self.assertIn("document-review", routes["document-craft"])
        self.assertNotIn("implement", routes["document-craft"])
        self.assertIn("implement", routes["hallmark"])
        self.assertIn("ui-review", routes["hallmark"])
        self.assertNotIn("document-draft", routes["hallmark"])

    def test_modified_skill_prompt_fails_closed(self):
        entry = copy.deepcopy(sr._entry_by_id(sr.load_skill_catalog(), "document-craft"))
        entry["git_blob_sha"] = "0" * 40
        with self.assertRaisesRegex(sr.SkillRoutingError, "integrity_mismatch"):
            sr._read_verified_skill(entry)

    def test_registry_rejects_paid_skill_when_free_only_is_enabled(self):
        entry = copy.deepcopy(sr._entry_by_id(sr.load_skill_catalog(), "document-craft"))
        entry["billing_class"] = "paid"
        manifest = sr._build_manifest(entry)
        registry = SkillRegistry()
        registry.register(manifest)
        envelope = sr._task_envelope("write a report", manifest)
        with self.assertRaisesRegex(SkillAuthorizationError, "hard_free_rejects_skill"):
            registry.authorize(
                envelope,
                trust_level="standard",
                network_class="none",
                free_only=True,
                granted_permissions={},
            )

    def test_planner_hint_is_compact_and_intent_specific(self):
        doc_hint = sr.routing_prompt_hint("Write a paper")
        self.assertIn("document-craft@1.0.0", doc_hint)
        self.assertIn("never fabricate", doc_hint)
        ui_hint = sr.routing_prompt_hint("Design a dashboard")
        self.assertIn("hallmark@2.0.0", ui_hint)
        self.assertNotIn("document-craft", ui_hint)


if __name__ == "__main__":
    unittest.main()
