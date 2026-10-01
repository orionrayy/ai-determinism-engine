import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ActionsConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orchestrator = (ROOT / '.github' / 'workflows' / 'orchestrator.yml').read_text()
        cls.continuation = (ROOT / '.github' / 'workflows' / 'orchestrator-continuation.yml').read_text()
        cls.tests = (ROOT / '.github' / 'workflows' / 'orchestrator-tests.yml').read_text()
        cls.bridge_deploy = (ROOT / '.github' / 'workflows' / 'bridge-deploy.yml').read_text()

    def test_approval_labels_trigger_worker(self):
        self.assertIn('types: [opened, edited, labeled]', self.orchestrator)
        self.assertIn('orchestrator-approved|orchestrator-rejected', self.orchestrator)
        self.assertIn('Authorize approval actor', self.orchestrator)
        self.assertIn('github.actor', self.orchestrator)

    def test_pending_runs_are_not_replaced(self):
        self.assertIn('queue: max', self.orchestrator)

    def test_workflow_target_is_preferred_for_approval(self):
        self.assertIn('ORCHESTRATOR_TARGET_WORKFLOW_ID', self.orchestrator)
        self.assertIn('args=(--workflow-id "$ORCHESTRATOR_TARGET_WORKFLOW_ID" --step)', self.orchestrator)

    def test_source_issue_is_propagated(self):
        self.assertIn('ORCHESTRATOR_TRIGGER_ISSUE', self.orchestrator)
        self.assertIn('ORCHESTRATOR_GITHUB_RUN_ID', self.orchestrator)

    def test_continuation_binds_to_originating_run(self):
        self.assertIn('workflow_run.id', self.continuation)
        self.assertIn("item.get('github_run_id')", self.continuation)

    def test_ci_watches_orchestrator_workflow(self):
        self.assertIn('orchestrator.yml', self.tests)


    def test_core_actions_are_pinned_to_node24_releases(self):
        expected = {
            "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
            "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
            "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020",
        }
        combined = "\n".join((self.orchestrator, self.continuation, self.tests, self.bridge_deploy))
        for ref in expected:
            self.assertIn(ref, combined)


if __name__ == '__main__':
    unittest.main()