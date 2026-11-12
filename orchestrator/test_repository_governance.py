from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"


class RepositoryGovernanceTests(unittest.TestCase):
    def read_workflow(self, name: str) -> str:
        path = WORKFLOWS / name
        self.assertTrue(path.is_file(), f"missing workflow: {path}")
        return path.read_text(encoding="utf-8")

    def test_orchestrator_workflows_do_not_write_runtime_state_to_main(self):
        for name in ("orchestrator.yml", "orchestrator-continuation.yml"):
            with self.subTest(workflow=name):
                content = self.read_workflow(name)
                self.assertNotIn("git push origin HEAD:main", content)
        orchestrator = self.read_workflow("orchestrator.yml")
        self.assertIn("node orchestrator/state_branch.mjs restore", orchestrator)
        self.assertIn("node orchestrator/state_branch.mjs persist", orchestrator)

    def test_every_state_reader_restores_before_reading_workflow_state(self):
        orchestrator = self.read_workflow("orchestrator.yml")
        continuation = self.read_workflow("orchestrator-continuation.yml")
        continuation_restore = continuation.index("node orchestrator/state_branch.mjs restore")
        continuation_decision = continuation.index("python3 -m orchestrator.continuation_runtime")
        self.assertLess(continuation_restore, continuation_decision)
        self.assertIn("node orchestrator/state_branch.mjs restore", orchestrator)
        self.assertIn("node orchestrator/state_branch.mjs persist", orchestrator)

    def test_path_independent_governance_check_is_present(self):
        content = self.read_workflow("repository-governance.yml")
        self.assertIn("pull_request:", content)
        self.assertIn("name: Repository Governance", content)
        self.assertIn("policy-gate:", content)
        self.assertNotRegex(content, r"pull_request:\\s*\\n\\s+paths:")
        self.assertIn("python -m unittest orchestrator.test_repository_governance", content)
        self.assertIn("node --test orchestrator/test_state_branch.mjs", content)


if __name__ == "__main__":
    unittest.main()
